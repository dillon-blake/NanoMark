"""All hyperparameters for NanoMark in one place.

NanoMark is a small, encoder-free OCR model: a GPT-2-small-shaped decoder that
reads a grayscale document image (linearly projected 32x32 patches, no ViT) and
autoregressively transcribes its text.

Sequence layout per sample:
    [BOS] [image patches ...] [SOC] [ocr text ...] [EOS]
Loss is computed only on the ocr-text region + EOS.
"""

from dataclasses import dataclass


# --- Special token ids -------------------------------------------------------
# GPT-2 BPE occupies ids 0..50256 (50256 is <|endoftext|>). We place our three
# special tokens just above the BPE range; the embedding table is padded to
# `padded_vocab` (a multiple of 64) for tensor-core efficiency.
BOS_ID = 50257  # beginning of sequence
SOC_ID = 50258  # start of ocr (separates image from text)
EOS_ID = 50259  # end of sequence


@dataclass(frozen=True)
class Config:
    """Immutable bundle of every NanoMark hyperparameter.

    Shared by the model, data pipeline, training, and inference so there is a
    single source of truth. Frozen, so instances are safe to pass around and to
    pickle into checkpoints. :meth:`__post_init__` validates the shape invariants
    at construction, turning a bad config into an immediate, clear error.
    """

    # --- Model shape (GPT-2 small) ---
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12          # query heads
    n_kv_heads: int = 4        # GQA: must divide n_heads
    head_dim: int = 64         # d_model // n_heads
    mlp_hidden: int = 3072     # SwiGLU hidden = 4 * d_model (Gemma-style; modern mainstream)
    rope_base: float = 10000.0

    # --- Vision ---
    patch_size: int = 32
    max_image_px: int = 1536   # longest edge; images are never upscaled
    img_channels: int = 1      # grayscale

    # --- Augmentation (train split only; eval is never augmented) ---
    # Light *photometric* jitter applied per __getitem__ so repeated epochs do not
    # see byte-identical inputs (the preprocessing is otherwise deterministic).
    # Geometry is untouched, so the patch grid / 2D-RoPE positions / masks stay
    # exactly valid. Set augment=False to reproduce the deterministic pipeline.
    augment: bool = True
    aug_contrast: float = 0.10      # contrast factor jitter, +/- this fraction
    aug_brightness: float = 0.05    # brightness shift, +/- this fraction of full scale
    aug_noise_std: float = 4.0      # additive Gaussian noise std, on the 0-255 scale

    # --- Vocab ---
    vocab_size: int = 50257    # GPT-2 BPE
    padded_vocab: int = 50304  # round up past the special tokens to a multiple of 64

    # --- Special tokens ---
    bos_id: int = BOS_ID
    soc_id: int = SOC_ID
    eos_id: int = EOS_ID

    # --- Optimization ---
    batch_size: int = 8         # micro-batch per forward; effective = batch_size * grad_accum
    grad_accum: int = 8         # gradient accumulation steps -> effective batch 64
    max_seq_len: int = 8192     # hard cap on sequence length (text is truncated to fit)
    lr_muon: float = 0.02
    lr_adamw: float = 3e-4
    weight_decay: float = 0.1
    adam_betas: tuple = (0.9, 0.95)
    warmup_ratio: float = 0.05  # fraction of total optimizer steps spent in LR warmup
    warmup_steps: int = 0       # resolved at runtime to warmup_ratio * max_steps (see train.main)
    max_steps: int = 10000     # set automatically at runtime from epochs * steps/epoch
    grad_clip: float = 1.0

    # --- Training run (data, length, logging, eval, checkpoints) ---
    # These are the knobs you tune per run; CLI flags in train.py default to them.
    dataset: str = "DBlake-BoxedLogic/Image-2-Markdown"  # HF dataset name/path
    split: str = "train"            # HF split to load before the train/eval re-split
    image_col: str = "image"
    text_col: str = "text"
    max_rows: int | None = None     # limit rows loaded from the dataset (None = all)
    eval_frac: float = 0.1          # held-out fraction for eval (90/10 default)
    # DataLoader: image preprocessing is CPU-bound, so workers + prefetch overlap
    # it with GPU compute. 0 workers = synchronous loading (stalls the GPU).
    num_workers: int = 8            # DataLoader worker processes (0 = main process)
    pin_memory: bool = True         # page-locked host buffers for faster H2D copies
    prefetch_factor: int = 4        # batches prefetched per worker (when num_workers>0)
    epochs: int = 6
    log_every: int = 50             # log train loss every N steps
    eval_every: int = 500            # run eval every N steps
    eval_batches: int | None = 50   # cap eval to N batches (None = full eval split)
    ckpt_every: int = 1000          # write a checkpoint every N steps
    out_dir: str = "checkpoints"
    use_wandb: bool = True
    wandb_project: str = "nanomark"
    device: str | None = None       # None = auto-detect (cuda > mps > cpu)
    seed: int = 0                   # seed for the train/eval split

    @property
    def patch_dim(self) -> int:
        """Flattened input size of one patch (patch_size**2 * channels)."""
        return self.patch_size * self.patch_size * self.img_channels

    def __post_init__(self):
        """Validate shape invariants; raise AssertionError on a bad config."""
        assert self.d_model == self.n_heads * self.head_dim, "d_model must equal n_heads * head_dim"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert (self.head_dim // 2) % 2 == 0, "head_dim/2 must be even (split into h/w RoPE halves)"
        assert self.padded_vocab > self.eos_id, "padded_vocab must cover all special tokens"
        assert 0.0 < self.eval_frac < 1.0, "eval_frac must be in (0, 1)"
        assert self.epochs >= 1, "epochs must be >= 1"
        assert self.grad_accum >= 1, "grad_accum must be >= 1"
        max_img_tokens = (self.max_image_px // self.patch_size) ** 2
        assert self.max_seq_len >= max_img_tokens + 3, "max_seq_len too small to fit a full image + BOS/SOC/EOS"
