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

# The HF fast tokenizer is used inside forked DataLoader workers; disable its
# internal Rust parallelism so it can't deadlock or spam warnings on fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import asdict, replace
from functools import partial

import torch
from torch.utils.data import DataLoader

from config import Config
from data import OCRDataset, collate_fn
from model import NanoMark, is_vision_adapter
from qwen import load_qwen3


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

    Delegates to :meth:`model.NanoMark.loss`, which projects the LM head and the
    fp32 logits only at positions that carry a label (``labels != -100``) instead
    of over the whole padded sequence. The result is identical to a full-sequence
    ``cross_entropy(..., ignore_index=-100)`` but avoids materializing a
    [B*S, vocab] fp32 logits tensor for the image-patch and padding positions that
    contribute nothing to the loss.

    Args:
        model: The NanoMark model.
        batch: A collated, on-device batch from :func:`data.collate_fn`.

    Returns:
        A scalar loss tensor (computed in fp32).
    """
    return model.loss(batch["input_ids"], batch["patches"], batch["image_slots"],
                      batch["pos_h"], batch["pos_w"], batch["attn_mask"], batch["labels"],
                      batch["patch_pos"])


def build_optimizer(model, cfg):
    """Build the AdamW optimizer with weight-decay and LR groups.

    All parameters use AdamW (Muon was removed -- it is designed for from-scratch
    pretraining, whereas NanoMark fine-tunes a pretrained base).

    Weight decay: applied to 2D weight matrices (block matmuls, the patch
    projector, the token embedding / tied head), not to 1D params (RMSNorm gains).

    Learning-rate groups: params loaded from the pretrained base get their LR
    scaled by ``cfg.lr_mult_pretrained`` while the fresh vision adapter
    (``patch_proj``/``patch_norm``/``patch_pos_emb``) trains at the full base LR.
    The schedule in :func:`set_lr` reads each group's ``lr_mult``.

    Args:
        model: The :class:`model.NanoMark` (or any module) to optimize.
        cfg: Configuration providing learning rate, betas, and weight decay.

    Returns:
        A single ``torch.optim.AdamW``. Each param group carries an ``lr_mult`` key.
    """
    def lr_mult(name):
        # fresh vision-adapter params train at the full base LR; everything else is
        # loaded from the pretrained base and fine-tuned more gently.
        return 1.0 if is_vision_adapter(name) else cfg.lr_mult_pretrained

    # bucket params by (weight-decay, lr_mult) so each distinct combo is its own group
    decay, nodecay = {}, {}  # lr_mult -> [params]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = lr_mult(name)
        if p.ndim >= 2:
            decay.setdefault(m, []).append(p)     # matmuls, patch_proj, tok_emb (tied head)
        else:
            nodecay.setdefault(m, []).append(p)   # RMSNorm gains, biases

    groups = (
        [{"params": ps, "weight_decay": cfg.weight_decay, "lr_mult": m} for m, ps in decay.items()]
        + [{"params": ps, "weight_decay": 0.0, "lr_mult": m} for m, ps in nodecay.items()]
    )
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.adam_betas)


def set_lr(opt, cfg, opt_step):
    """Set the optimizer's learning rate from the schedule for ``opt_step``.

    Each param group's per-group ``lr_mult`` (1.0 for the fresh vision adapter,
    ``cfg.lr_mult_pretrained`` for base-loaded params) scales the base LR before
    the shared warmup/cosine schedule. Returns the schedule multiplier (in [0, 1]).
    """
    scale = lr_scale(opt_step, cfg.warmup_steps, cfg.max_steps)
    for g in opt.param_groups:
        g["lr"] = cfg.lr * g.get("lr_mult", 1.0) * scale
    return scale


def amp_ctx(use_amp):
    """Autocast context for the forward pass (bf16 on CUDA), else a no-op."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.enable_grad()


def train_step(model, batch, opt, cfg, step, use_amp):
    """Run one full optimization step (no accumulation) and return the loss.

    Used by the overfit-one-batch sanity mode. The main loop does its own
    gradient accumulation inline.

    Args:
        model: The NanoMark model (in train mode).
        batch: A collated, on-device batch from :func:`data.collate_fn`.
        opt: The AdamW optimizer.
        cfg: Configuration (learning rate, schedule, grad clip).
        step: Current step index, for the LR schedule.
        use_amp: Whether to run the forward pass under bf16 autocast (CUDA only).

    Returns:
        The loss value as a Python float.
    """
    set_lr(opt, cfg, step)
    with amp_ctx(use_amp):
        loss = forward_loss(model, batch)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if torch.isfinite(total_norm):  # skip the update on a NaN/Inf grad so it can't poison the weights
        opt.step()
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
    ap.add_argument("--run-name", default=d.wandb_run_name, help="W&B run display name (default: auto-generated)")
    # base model (the decoder is always initialized from this; shape is read from it too)
    ap.add_argument("--base-repo", default=d.base_repo, help="HF repo for the base tokenizer + weights + shape")
    # vision positional scheme: parameter-free 2D RoPE vs. a learned 2D table
    ap.add_argument("--image-pos-mode", default=d.image_pos_mode, choices=("rope2d", "learned"),
                    help="how image patches carry 2D position (default: %(default)s)")
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
    """Build the runtime config: model shape from the base model, CLI knobs on top.

    :meth:`Config.from_base` reads ``args.base_repo``'s HF config for the decoder
    shape (d_model/n_layers/...); the CLI overrides supply the data/schedule knobs.
    """
    return Config.from_base(
        args.base_repo,
        dataset=args.dataset, split=args.split, image_col=args.image_col, text_col=args.text_col,
        max_rows=args.max_rows, eval_frac=args.eval_frac, num_workers=args.num_workers, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum, max_seq_len=args.max_seq_len,
        image_pos_mode=args.image_pos_mode,
        log_every=args.log_every, eval_every=args.eval_every,
        eval_batches=args.eval_batches, ckpt_every=args.ckpt_every, out_dir=args.out,
        use_wandb=args.wandb, wandb_project=args.wandb_project, wandb_run_name=args.run_name,
        device=args.device, seed=args.seed,
    )


def main():
    """Build the runtime config (Config + CLI overrides), then train with eval."""
    args = parse_args()
    cfg = build_cfg(args)

    device = pick_device(cfg.device)
    use_amp = device == "cuda"
    if device == "cuda":
        # TF32 tensor cores for the fp32 matmuls (logits/loss path); free speedup on
        # A100 with no meaningful accuracy cost. The block matmuls already run in bf16
        # under autocast.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = NanoMark(cfg)
    load_qwen3(model, cfg.base_repo)   # fill the decoder from the base; vision adapter stays fresh
    model = model.to(device)
    print(f"device={device}  base={cfg.base_repo}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = build_optimizer(model, cfg)
    model.train()

    train_loader, eval_loader = build_loaders(cfg)

    wandb = None
    if cfg.use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))

    # --- sanity mode: overfit a single batch ---
    if args.overfit_one_batch:
        ocfg = replace(cfg, max_steps=200, warmup_steps=max(1, round(cfg.warmup_ratio * 200)))
        batch = to_device(next(iter(train_loader)), device)
        for step in range(ocfg.max_steps):
            loss = train_step(model, batch, opt, ocfg, step, use_amp)
            if step % cfg.log_every == 0:
                print(f"step {step:4d}  loss {loss:.4f}")
        return

    # --- epoch-based training with gradient accumulation ---
    accum = cfg.grad_accum
    total_steps = max(1, (cfg.epochs * len(train_loader)) // accum)  # optimizer steps
    warmup = max(1, round(cfg.warmup_ratio * total_steps))  # warmup as a fraction of the full run
    cfg = replace(cfg, max_steps=total_steps, warmup_steps=warmup)  # so the LR schedule spans the run
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"epochs={cfg.epochs}  grad_accum={accum}  effective_batch={cfg.batch_size * accum}  "
          f"opt_steps={total_steps}  warmup_steps={warmup}")

    def save_ckpt(name, step):
        path = os.path.join(cfg.out_dir, name)
        torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, path)
        print(f"  saved {path}")

    opt_step, micro, running = 0, 0, 0.0
    best_eval = float("inf")  # track the best eval loss so we can keep best.pt
    opt.zero_grad(set_to_none=True)
    for epoch in range(cfg.epochs):
        for batch in train_loader:
            set_lr(opt, cfg, opt_step)
            with amp_ctx(use_amp):
                loss = forward_loss(model, to_device(batch, device)) / accum
            loss.backward()
            running += loss.item()
            micro += 1

            if micro % accum != 0:
                continue  # keep accumulating

            # --- one optimizer step (every `accum` micro-batches) ---
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if torch.isfinite(total_norm):  # a NaN/Inf grad would clip to NaN and poison every later step
                opt.step()
            else:
                print(f"  skipping step {opt_step}: non-finite grad norm")
            opt.zero_grad(set_to_none=True)  # drop the (possibly bad) grads either way

            if opt_step % cfg.log_every == 0:
                lr = cfg.lr * lr_scale(opt_step, cfg.warmup_steps, cfg.max_steps)
                print(f"epoch {epoch}  step {opt_step:6d}/{total_steps}  loss {running:.4f}  lr {lr:.2e}")
                if wandb:
                    wandb.log({"train/loss": running, "lr": lr, "epoch": epoch}, step=opt_step)
            if cfg.eval_every and opt_step > 0 and opt_step % cfg.eval_every == 0:
                eval_loss = evaluate(model, eval_loader, device, cfg.eval_batches)
                print(f"  [eval] step {opt_step}  eval_loss {eval_loss:.4f}")
                if wandb:
                    wandb.log({"eval/loss": eval_loss}, step=opt_step)
                if eval_loss < best_eval:  # keep the best-eval checkpoint
                    best_eval = eval_loss
                    save_ckpt("best.pt", opt_step)
            if opt_step > 0 and opt_step % cfg.ckpt_every == 0:
                save_ckpt(f"step{opt_step}.pt", opt_step)

            opt_step += 1
            running = 0.0

    # flush a trailing partial accumulation window, if any
    if micro % accum != 0:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if torch.isfinite(total_norm):
            opt.step()

    # final eval + checkpoint
    eval_loss = evaluate(model, eval_loader, device, cfg.eval_batches)
    print(f"[final] step {opt_step}  eval_loss {eval_loss:.4f}  (best {min(best_eval, eval_loss):.4f})")
    save_ckpt("final.pt", opt_step)
    if eval_loss < best_eval:  # the final weights are also the best-eval weights
        best_eval = eval_loss
        save_ckpt("best.pt", opt_step)
    if wandb:
        wandb.log({"eval/loss": eval_loss}, step=opt_step)
        wandb.finish()


if __name__ == "__main__":
    main()
