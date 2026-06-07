"""Training loop for NanoMark.

Trains for a number of epochs over a HuggingFace dataset (image + text columns),
holding out a fraction for evaluation, with optional Weights & Biases logging.

Examples:
    # 3 epochs on the first 20k rows, eval every 500 steps, log to W&B
    python train.py --dataset some/ocr-dataset --epochs 3 --max-rows 20000 \
        --eval-every 500 --log-every 20 --wandb --wandb-project nanomark

    # quick pipeline sanity check: overfit a single batch
    python train.py --dataset some/ocr-dataset --overfit-one-batch
"""

import argparse
import math
import os
from dataclasses import asdict, replace
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from data import OCRDataset, collate_fn
from model import NanoMark
from optimizer import build_optimizers


def pick_device(requested):
    """Resolve the compute device.

    Args:
        requested: An explicit device string, or a falsy value to auto-detect.

    Returns:
        ``requested`` if given, else the best available of ``cuda``/``mps``/``cpu``.
    """
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_scale(step, warmup, max_steps):
    """Compute the learning-rate multiplier: linear warmup then cosine decay to 0.

    Args:
        step: Current step (0-indexed).
        warmup: Number of warmup steps (clamped to ``max_steps`` so short runs
            still reach full learning rate).
        max_steps: Total steps over which to decay.

    Returns:
        A multiplier in [0, 1] to apply to the base learning rates.
    """
    warmup = min(warmup, max_steps)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def to_device(batch, device):
    """Move every tensor in a collated batch dict to ``device``."""
    return {k: v.to(device) for k, v in batch.items()}


def forward_loss(model, batch):
    """Run the model and return the label-masked cross-entropy loss.

    Args:
        model: The NanoMark model.
        batch: A collated, on-device batch from :func:`data.collate_fn`.

    Returns:
        A scalar loss tensor (computed in fp32).
    """
    logits = model(batch["input_ids"], batch["patches"], batch["image_slots"],
                   batch["pos_h"], batch["pos_w"], batch["attn_mask"])
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        batch["labels"].reshape(-1),
        ignore_index=-100,
    )


def set_lr(muon, adamw, cfg, opt_step):
    """Set both optimizers' learning rates from the schedule for ``opt_step``.

    Returns the schedule multiplier (in [0, 1]).
    """
    scale = lr_scale(opt_step, cfg.warmup_steps, cfg.max_steps)
    for g in muon.param_groups:
        g["lr"] = cfg.lr_muon * scale
    for g in adamw.param_groups:
        g["lr"] = cfg.lr_adamw * scale
    return scale


def amp_ctx(use_amp):
    """Autocast context for the forward pass (bf16 on CUDA), else a no-op."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.enable_grad()


def train_step(model, batch, muon, adamw, cfg, step, use_amp):
    """Run one full optimization step (no accumulation) and return the loss.

    Used by the overfit-one-batch sanity mode. The main loop does its own
    gradient accumulation inline.

    Args:
        model: The NanoMark model (in train mode).
        batch: A collated, on-device batch from :func:`data.collate_fn`.
        muon: The Muon optimizer (block matmuls + patch projector).
        adamw: The AdamW optimizer (embedding/head + norms).
        cfg: Configuration (learning rates, schedule, grad clip).
        step: Current step index, for the LR schedule.
        use_amp: Whether to run the forward pass under bf16 autocast (CUDA only).

    Returns:
        The loss value as a Python float.
    """
    set_lr(muon, adamw, cfg, step)
    with amp_ctx(use_amp):
        loss = forward_loss(model, batch)
    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    muon.step()
    adamw.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Compute the mean eval loss over (up to ``max_batches``) eval batches.

    Args:
        model: The NanoMark model (switched to eval mode and back).
        loader: The eval ``DataLoader``.
        device: Device to run on.
        max_batches: Optional cap on the number of eval batches (None = all).

    Returns:
        The mean eval loss as a Python float (0.0 if the loader is empty).
    """
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        total += forward_loss(model, to_device(batch, device)).item()
        n += 1
    model.train()
    return total / max(1, n)


def parse_args():
    """Parse CLI args. Every default comes from Config, so you can set the run
    in ``config.py`` and just pass a flag to override one value for one run.
    """
    d = Config()
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--dataset", default=(d.dataset or None), required=(d.dataset == ""),
                    help="HF dataset name/path (overrides Config.dataset)")
    ap.add_argument("--split", default=d.split)
    ap.add_argument("--image-col", default=d.image_col)
    ap.add_argument("--text-col", default=d.text_col)
    ap.add_argument("--max-rows", type=int, default=d.max_rows)
    ap.add_argument("--eval-frac", type=float, default=d.eval_frac)
    ap.add_argument("--num-workers", type=int, default=d.num_workers,
                    help="DataLoader worker processes (0 = synchronous, stalls the GPU)")
    # schedule
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size, help="micro-batch per forward")
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum,
                    help="gradient accumulation steps (effective batch = batch-size * grad-accum)")
    ap.add_argument("--max-seq-len", type=int, default=d.max_seq_len)
    # logging / eval / checkpoints
    ap.add_argument("--log-every", type=int, default=d.log_every)
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--eval-batches", type=int, default=d.eval_batches)
    ap.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    ap.add_argument("--out", default=d.out_dir)
    # wandb
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=d.use_wandb,
                    help="enable Weights & Biases logging")
    ap.add_argument("--wandb-project", default=d.wandb_project)
    # misc
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--overfit-one-batch", action="store_true", help="sanity check: overfit one batch")
    return ap.parse_args()


def build_loaders(cfg):
    """Load the dataset, optionally subset it, and split into train/eval loaders.

    Reads ``cfg.dataset``, ``cfg.split``, ``cfg.max_rows``, ``cfg.eval_frac``,
    ``cfg.image_col``/``cfg.text_col``, ``cfg.batch_size``, and ``cfg.seed``.

    Returns:
        A tuple ``(train_loader, eval_loader)``.
    """
    from datasets import load_dataset

    raw = load_dataset(cfg.dataset, split=cfg.split)
    if cfg.max_rows is not None:
        raw = raw.select(range(min(cfg.max_rows, len(raw))))
    parts = raw.train_test_split(test_size=cfg.eval_frac, seed=cfg.seed)

    collate = partial(collate_fn, cfg=cfg)
    train_ds = OCRDataset(parts["train"], cfg, cfg.image_col, cfg.text_col, augment=cfg.augment)
    eval_ds = OCRDataset(parts["test"], cfg, cfg.image_col, cfg.text_col, augment=False)
    # Image preprocessing (decode/resize/patchify) runs in __getitem__ on the CPU;
    # with workers + prefetch it overlaps GPU compute instead of stalling it.
    pin_memory = cfg.pin_memory and torch.cuda.is_available()  # pinning only helps H2D copies to CUDA
    loader_kwargs = {}
    if cfg.num_workers > 0:
        loader_kwargs = {"persistent_workers": True, "prefetch_factor": cfg.prefetch_factor}
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=cfg.num_workers, pin_memory=pin_memory, **loader_kwargs)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate,
                             num_workers=cfg.num_workers, pin_memory=pin_memory, **loader_kwargs)
    print(f"train rows={len(train_ds)}  eval rows={len(eval_ds)}  steps/epoch={len(train_loader)}")
    return train_loader, eval_loader


def build_cfg(args):
    """Overlay CLI overrides onto Config() to get the runtime config."""
    return replace(
        Config(),
        dataset=args.dataset, split=args.split, image_col=args.image_col, text_col=args.text_col,
        max_rows=args.max_rows, eval_frac=args.eval_frac, num_workers=args.num_workers, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum, max_seq_len=args.max_seq_len,
        log_every=args.log_every, eval_every=args.eval_every,
        eval_batches=args.eval_batches, ckpt_every=args.ckpt_every, out_dir=args.out,
        use_wandb=args.wandb, wandb_project=args.wandb_project, device=args.device, seed=args.seed,
    )


def main():
    """Build the runtime config (Config + CLI overrides), then train with eval."""
    args = parse_args()
    cfg = build_cfg(args)

    device = pick_device(cfg.device)
    use_amp = device == "cuda"
    model = NanoMark(cfg).to(device)
    print(f"device={device}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    muon, adamw = build_optimizers(model, cfg)
    model.train()

    train_loader, eval_loader = build_loaders(cfg)

    wandb = None
    if cfg.use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    # --- sanity mode: overfit a single batch ---
    if args.overfit_one_batch:
        ocfg = replace(cfg, max_steps=200)
        batch = to_device(next(iter(train_loader)), device)
        for step in range(ocfg.max_steps):
            loss = train_step(model, batch, muon, adamw, ocfg, step, use_amp)
            if step % cfg.log_every == 0:
                print(f"step {step:4d}  loss {loss:.4f}")
        return

    # --- epoch-based training with gradient accumulation ---
    accum = cfg.grad_accum
    total_steps = max(1, (cfg.epochs * len(train_loader)) // accum)  # optimizer steps
    cfg = replace(cfg, max_steps=total_steps)  # so the cosine LR schedule spans the full run
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"epochs={cfg.epochs}  grad_accum={accum}  effective_batch={cfg.batch_size * accum}  "
          f"opt_steps={total_steps}")

    def save_ckpt(name, step):
        path = os.path.join(cfg.out_dir, name)
        torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, path)
        print(f"  saved {path}")

    opt_step, micro, running = 0, 0, 0.0
    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)
    for epoch in range(cfg.epochs):
        for batch in train_loader:
            set_lr(muon, adamw, cfg, opt_step)
            with amp_ctx(use_amp):
                loss = forward_loss(model, to_device(batch, device)) / accum
            loss.backward()
            running += loss.item()
            micro += 1

            if micro % accum != 0:
                continue  # keep accumulating

            # --- one optimizer step (every `accum` micro-batches) ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            muon.step()
            adamw.step()
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)

            if opt_step % cfg.log_every == 0:
                lr = cfg.lr_adamw * lr_scale(opt_step, cfg.warmup_steps, cfg.max_steps)
                print(f"epoch {epoch}  step {opt_step:6d}/{total_steps}  loss {running:.4f}  lr {lr:.2e}")
                if wandb:
                    wandb.log({"train/loss": running, "lr": lr, "epoch": epoch}, step=opt_step)
            if cfg.eval_every and opt_step > 0 and opt_step % cfg.eval_every == 0:
                eval_loss = evaluate(model, eval_loader, device, cfg.eval_batches)
                print(f"  [eval] step {opt_step}  eval_loss {eval_loss:.4f}")
                if wandb:
                    wandb.log({"eval/loss": eval_loss}, step=opt_step)
            if opt_step > 0 and opt_step % cfg.ckpt_every == 0:
                save_ckpt(f"step{opt_step}.pt", opt_step)

            opt_step += 1
            running = 0.0

    # flush a trailing partial accumulation window, if any
    if micro % accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        muon.step()
        adamw.step()

    # final eval + checkpoint
    eval_loss = evaluate(model, eval_loader, device, cfg.eval_batches)
    print(f"[final] step {opt_step}  eval_loss {eval_loss:.4f}")
    save_ckpt("final.pt", opt_step)
    if wandb:
        wandb.log({"eval/loss": eval_loss}, step=opt_step)
        wandb.finish()


if __name__ == "__main__":
    main()
