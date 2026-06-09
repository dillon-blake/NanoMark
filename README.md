# NanoMark

A small **OCR model**: a Qwen3-0.6B-shaped decoder that reads a grayscale
document image and autoregressively transcribes its text. No vision encoder —
image patches are fed straight into the decoder via a single linear projection
(Fuyu / Gemma-4-style). The decoder is **always initialized from Qwen3 0.6B Base**
(so it already knows language/markdown) — and its shape is **read from the base
model's config** rather than hardcoded — while the linear vision adapter is the
only part learned from scratch.

## Architecture

- **Shape:** Qwen3 0.6B (28 layers, 1024 dim, 16 query heads, **head_dim 128 —
  decoupled** from `d_model`, Qwen3-style). RoPE base 1e6.
- **Encoder-free vision:** each 32×32 grayscale patch is linearly projected to
  `d_model` (then RMSNorm'd) and spliced into the sequence. No ViT.
- **Attention:** full (non-windowed) attention on every layer, GQA (8 KV heads),
  **QK-Norm** (per-head RMSNorm on q/k before RoPE, Qwen3-style — also stabilizes
  the mixed text/image softmax), PyTorch fused attention
  (`scaled_dot_product_attention`) with a naive fallback. Image patches attend
  **bidirectionally** (to BOS + real patches); text attends **causally**. Padding
  patches are masked out.
- **Positions:** 2D RoPE — image patches get `(row, col)`; text gets a 1D
  progression (`pos_h == pos_w`) resuming after the image. The frequencies are
  interleaved across the two axes using Qwen3's exact 1D layout, so for text the
  encoding reduces **bit-exactly** to stock Qwen3 RoPE (which is what the loaded
  attention weights expect) while image tokens still get a true 2D encoding.
- **Other:** RMSNorm, SwiGLU MLP, tied input/output embedding.
- **Tokenizer:** Qwen3 BPE (vocab 151936). The structural markers BOS/SOC/EOS
  reuse existing Qwen3 control tokens (`<|im_start|>`/`<|im_end|>`/`<|endoftext|>`),
  so their embedding rows start pretrained rather than random.
- **Optimizer:** AdamW for all params, with weight decay on 2D weights (matmuls,
  patch projector, embedding/head) and none on 1D norm gains. Pretrained weights
  use a gentler LR than the fresh vision adapter.

### Base-model initialization

The decoder is always loaded from the base model. `Config.from_base(base_repo)`
reads the base's HF config to set the model shape (d_model, n_layers, n_heads,
n_kv_heads, head_dim, mlp_hidden, rope_base, vocab) — so the shape always matches
the weights — then `qwen.load_qwen3` copies Qwen3 0.6B Base into every block (GQA +
QK-Norm + SwiGLU, embedding, final norm); only the vision adapter
(`patch_proj`/`patch_norm`/`patch_pos_emb`) stays at random init. Training uses
**two LR groups**: the fresh vision adapter trains at the full base LR, while the
pretrained weights are fine-tuned more gently (scaled by `lr_mult_pretrained`,
default 0.1).

### Sequence layout

```
[BOS] [image patches …] [SOC] [ocr text …] [EOS]
```

Loss is computed only on the ocr-text region + EOS.

### Image handling

Resize so the longest edge ≤ 1536 (**never upscaled** — handles varying
document sizes), pad to a square multiple of 32 with white, cut into a G×G grid
of patches in raster order. Sub-patch padding on edge patches is baked in;
whole white pad patches are masked.

> **Resolution ceiling:** 1536 + 32px patches comfortably handles typical
> single-page documents. For very dense / fine-print pages, raise `max_image_px`
> or add tiling.

## Files

| File | Purpose |
|------|---------|
| `config.py` | all hyperparameters (one dataclass) |
| `model.py` | RMSNorm, QK-Norm, 2D RoPE, GQA attention, SwiGLU, the model |
| `data.py` | image→patches, tokenize, sequence/mask/label assembly, collate |
| `qwen.py` | load Qwen3 0.6B Base weights into the decoder (name mapping + coverage check) |
| `train.py` | training loop + AdamW weight-decay grouping + fresh/pretrained LR groups |
| `inference.py` | transcribe a single image |
| `tests/` | pytest suite + `synthetic.py` (rendered text→image data for tests) |

## Setup

```bash
uv sync          # torch, numpy, pillow, transformers, safetensors, huggingface-hub, datasets, wandb, pytest
```

## Usage

```bash
# sanity check: overfit one batch to ~0 loss
uv run python train.py --dataset <name> --overfit-one-batch

# train from the Qwen3 0.6B base: 3 epochs, 90/10 split
# (the decoder shape is read from the base; --base-repo selects a different one)
uv run python train.py --dataset <name> \
    --image-col image --text-col text \
    --epochs 3 --eval-frac 0.1 --eval-every 500 --log-every 20

# ...optionally cap the rows loaded and log to Weights & Biases
uv run python train.py --dataset <name> --epochs 3 \
    --max-rows 20000 --wandb --wandb-project nanomark --eval-batches 50

# transcribe an image
uv run python inference.py --ckpt checkpoints/final.pt --image page.png

# run tests
uv run python -m pytest -q
```

All run settings live in `config.py`: `dataset`, `epochs`, `batch_size`
(micro-batch), `grad_accum` (**effective batch = batch_size × grad_accum**, default
8 × 8 = 64), `base_repo` / `lr_mult_pretrained` (base-model initialization; the
decoder shape is derived from `base_repo`), `max_seq_len` (default 6144; longer text is truncated to fit),
`max_rows` (subset the dataset), `eval_frac` (held-out fraction, default 0.1),
`eval_every` / `eval_batches`, `log_every`, `ckpt_every`, `out_dir`, `seed`, and
`use_wandb` / `wandb_project`. **Set them there to configure a run** — every CLI
flag simply defaults to its `Config` value, so flags are optional overrides.

## Notes / possible follow-ups

- At ~597M params (Qwen3 0.6B), the default `batch_size=8 × grad_accum=8` is
  memory-hungry — drop the micro-batch or add activation checkpointing if you OOM.
- The base LR (`lr`, AdamW) plus `lr_mult_pretrained` (gentler LR for loaded
  weights) is set up for fine-tuning; lower `lr` further if the pretrained features
  degrade early.
- Defaults assume fine-tuning a pretrained base: `epochs=2` (few passes + keep
  `best.pt`), `weight_decay=0.05`. Raise epochs only if you watch eval for overfit.
- Generation has **no KV cache** (correct but O(n²)) — the obvious speedup.
- Per-image square padding is efficient for small docs; switch to a fixed 1536
  canvas if you want fully static shapes.
