# NanoMark

A small, from-scratch **OCR model**: a GPT-2-small-shaped decoder that reads a
grayscale document image and autoregressively transcribes its text. No vision
encoder — image patches are fed straight into the decoder via a single linear
projection (Fuyu / Gemma-4-style).

## Architecture

- **Shape:** GPT-2 small (12 layers, 768 dim, 12 query heads, head_dim 64).
- **Encoder-free vision:** each 32×32 grayscale patch is linearly projected to
  `d_model` and spliced into the sequence. No ViT.
- **Attention:** full (non-windowed) attention on every layer, GQA (4 KV heads),
  PyTorch fused attention (`scaled_dot_product_attention`) with a naive fallback.
  Image patches attend **bidirectionally** (to BOS + real patches); text attends
  **causally**. Padding patches are masked out.
- **Positions:** 2D RoPE — image patches get `(row, col)`; text gets a 1D
  progression (`pos_h == pos_w`) resuming after the image.
- **Other:** RMSNorm, SwiGLU MLP, tied input/output embedding.
- **Optimizer:** Muon for the block matmuls + patch projector, AdamW for the
  embedding/head and norms (the standard hybrid).

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
| `model.py` | RMSNorm, 2D RoPE, GQA attention, SwiGLU, the model |
| `data.py` | image→patches, tokenize, sequence/mask/label assembly, collate |
| `optimizer.py` | Muon/AdamW param grouping (uses `torch.optim.Muon`) |
| `train.py` | training loop |
| `inference.py` | transcribe a single image |
| `tests/` | pytest suite + `synthetic.py` (rendered text→image data for tests) |

## Setup

```bash
uv sync          # installs torch, numpy, pillow, tiktoken, datasets, wandb, pytest
```

## Usage

```bash
# sanity check: overfit one batch to ~0 loss
uv run python train.py --dataset <name> --overfit-one-batch

# train: 3 epochs, 90/10 train/eval split, eval every 500 steps, log every 20
uv run python train.py --dataset <name> --image-col image --text-col text \
    --epochs 3 --eval-frac 0.1 --eval-every 500 --log-every 20

# ...optionally cap the rows loaded and log to Weights & Biases
uv run python train.py --dataset <name> --epochs 3 --max-rows 20000 \
    --wandb --wandb-project nanomark --eval-batches 50

# transcribe an image
uv run python inference.py --ckpt checkpoints/final.pt --image page.png

# run tests
uv run python -m pytest -q
```

All run settings live in `config.py`: `dataset`, `epochs`, `batch_size`
(micro-batch), `grad_accum` (**effective batch = batch_size × grad_accum**, default
4 × 32 = 128), `max_seq_len` (default 8192; longer text is truncated to fit),
`max_rows` (subset the dataset), `eval_frac` (held-out fraction, default 0.1),
`eval_every` / `eval_batches`, `log_every`, `ckpt_every`, `out_dir`, `seed`, and
`use_wandb` / `wandb_project`. **Set them there to configure a run** — every CLI
flag simply defaults to its `Config` value, so flags are optional overrides.

## Notes / possible follow-ups

- Generation has **no KV cache** (correct but O(n²)) — the obvious speedup.
- The patch projector is on Muon; flip it to AdamW in `optimizer.py` if it misbehaves.
- Per-image square padding is efficient for small docs; switch to a fixed 1536
  canvas if you want fully static shapes.
