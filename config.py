"""All hyperparameters for NanoMark in one place.

NanoMark is a small, encoder-free OCR model: a Granite-4.0-350M-shaped decoder
that reads a grayscale document image (linearly projected 32x32 patches, no ViT)
and autoregressively transcribes its text. The decoder is always initialized from
a pretrained base LM (its shape is read from the base too, see Config.from_base);
only the linear vision adapter is trained from scratch.

Sequence layout per sample:
    [BOS] [image patches ...] [SOC] [ocr text ...] [EOS]
Loss is computed only on the ocr-text region + EOS.
"""

import math
from dataclasses import dataclass


# --- Special token ids -------------------------------------------------------
# NanoMark is built on the Granite-4.0 tokenizer, so our three structural markers
# reuse existing Granite control tokens (all < vocab_size). Their embedding rows
# ship in the Granite checkpoint, so they start *pretrained* rather than random.
# They never appear in normal tokenized text; the data pipeline inserts them by
# hand. Granite's tokenizer aliases bos==eos (<|end_of_text|>), so we pick three
# *distinct* control tokens rather than its bos/eos/pad.
BOS_ID = 100264  # <|start_of_role|> -> beginning of sequence
SOC_ID = 100265  # <|end_of_role|>   -> start of ocr (separates image from text)
EOS_ID = 100257  # <|end_of_text|>   -> end of sequence

# Base model: single source of truth for the HF repo (tokenizer + optional weights).
BASE_REPO = "ibm-granite/granite-4.0-350m-base"


@dataclass(frozen=True)
class Config:
    """Immutable bundle of every NanoMark hyperparameter.

    Shared by the model, data pipeline, training, and inference so there is a
    single source of truth. Frozen, so instances are safe to pass around and to
    pickle into checkpoints. :meth:`__post_init__` validates the shape invariants
    at construction, turning a bad config into an immediate, clear error.
    """

    # --- Model shape ---
    # Resolved from the base model's HF config by :meth:`from_base` (the decoder is
    # always initialized from the base, so its shape must match exactly). Left None
    # here rather than hardcoded; tests pass explicit values for a tiny model.
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None          # query heads
    n_kv_heads: int | None = None       # GQA: must divide n_heads
    head_dim: int | None = None         # decoupled from d_model (n_heads*head_dim != d_model, Granite-style)
    mlp_hidden: int | None = None       # SwiGLU hidden (base shared_intermediate_size)
    rope_base: float | None = None      # base rope_theta
    rms_norm_eps: float = 1e-5          # RMSNorm epsilon (Granite uses 1e-5)

    # --- Granite muP-style multipliers (read from the base config by from_base) ---
    # Granite scales activations at four points instead of relying on the usual
    # 1/sqrt(d) conventions; the pretrained weights are only correct with these
    # applied. Defaults are the no-op values so a directly-constructed Config (e.g.
    # the tiny test model) behaves like a plain transformer.
    embedding_multiplier: float = 1.0          # input embeddings scaled by this
    attention_multiplier: float | None = None  # attention score scale (None -> 1/sqrt(head_dim))
    residual_multiplier: float = 1.0           # each sublayer output scaled before its residual add
    logits_scaling: float = 1.0                # logits divided by this

    # --- Vision ---
    patch_size: int = 32
    max_image_px: int = 1536   # longest edge; images are never upscaled
    img_channels: int = 1      # grayscale
    # How image patches carry their 2D position:
    #   "rope2d"  - parameter-free 2D RoPE: each patch is rotated by its (row, col)
    #               grid coordinate (the default; text rides the pos_h==pos_w diagonal).
    #   "learned" - Gemma-4-Unified-style: a factorized learned 2D positional
    #               embedding (per-axis tables summed) is added to the projected
    #               patch, and image tokens take ordinary sequential 1D RoPE
    #               positions like text. See model.NanoMark / data.build_sample.
    image_pos_mode: str = "rope2d"
    # When True (default) the image is a fully-observed bidirectional prefix: every
    # patch attends to BOS + all real patches. When False, image patches are causal
    # over the prefix (BOS + earlier-or-equal real patches only), like text.
    bidirectional_img: bool = True
    # Attention kernel. "sdpa" (default) builds a dense [B,1,S,S] bool mask and runs
    # F.scaled_dot_product_attention -- portable, used on CPU/tests and as the universal
    # fallback. "flex" uses torch FlexAttention with a BlockMask: fused + block-sparse
    # (skips the all-pad blocks), and on the flex path collate pads every batch to
    # max_seq_len so the compiled trunk keeps static shapes. CUDA-only; auto-falls back to
    # the dense path off CUDA (see data.resolve_attn_mask).
    attn_impl: str = "sdpa"

    # --- Base language model (the encoder-free decoder is always initialized from this) ---
    # NanoMark is a base LM fine-tuned with a bolted-on vision adapter: the whole
    # decoder is loaded from base_repo and only the vision adapter
    # (patch_proj/patch_norm/patch_pos_emb) is trained from scratch.
    base_repo: str = BASE_REPO    # HF repo for the tokenizer + weights + shape

    # --- Augmentation (train split only; eval is never augmented) ---
    # Light *photometric* jitter applied per __getitem__ so repeated epochs do not
    # see byte-identical inputs (the preprocessing is otherwise deterministic).
    # Geometry is untouched, so the patch grid / 2D-RoPE positions / masks stay
    # exactly valid. Set augment=False to reproduce the deterministic pipeline.
    augment: bool = True
    aug_contrast: float = 0.10      # contrast factor jitter, +/- this fraction
    aug_brightness: float = 0.05    # brightness shift, +/- this fraction of full scale
    aug_noise_std: float = 4.0      # additive Gaussian noise std, on the 0-255 scale

    # --- Vocab (resolved from the base model by from_base) ---
    # padded_vocab equals the base embedding row count so the tied tok_emb can be
    # copied from the base 1:1; vocab_size is the usable token range.
    vocab_size: int | None = None
    padded_vocab: int | None = None

    # --- Special tokens ---
    bos_id: int = BOS_ID
    soc_id: int = SOC_ID
    eos_id: int = EOS_ID

    # --- Optimization ---
    # Sized for an 80GB H100. Peak activation memory ~= batch_size * max_seq_len;
    # max_seq_len 6144 covers the measured sequence distribution (p99 ~4.9k, max ~5.6k,
    # image prefix included) with ~no truncation. batch_size 8 is the known-safe fit
    # (it was the 597M base's A100 fit, and Granite-4.0-350M is smaller, so it runs with
    # headroom to spare). Batch size needn't be a power of two -- the GEMM M-dim is
    # batch*seq_len and seq_len is variable, so only the model's feature dims matter for
    # tensor cores; memory is the real limit. 16 may fit since most batches are far under
    # 6144, but a worst-case all-6144 batch risks OOM (no recovery in the loop), so go to
    # 16 only with memory monitoring or activation checkpointing.
    batch_size: int = 8         # micro-batch per forward; effective = batch_size * grad_accum
    grad_accum: int = 8         # gradient accumulation steps -> effective batch 64
    max_seq_len: int = 6144     # hard cap on sequence length (image + text); text truncated to fit
    lr: float = 3e-4            # AdamW base learning rate (all params)
    # Base-loaded (pretrained) params are fine-tuned more gently than the fresh
    # vision adapter: their LR is scaled by this factor, while patch_proj/patch_norm/
    # patch_pos_emb always train at the full base LR (multiplier 1).
    lr_mult_pretrained: float = 0.1
    weight_decay: float = 0.05  # light, so fine-tuning doesn't pull the pretrained weights toward 0
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
    num_workers: int = 16           # DataLoader worker processes (image decode+patchify is CPU-heavy;
                                    #   every image is downscaled to the 1536px ceiling, so keep the
                                    #   A100 fed -- set near the box's vCPU count)
    pin_memory: bool = True         # page-locked host buffers for faster H2D copies
    prefetch_factor: int = 4        # batches prefetched per worker (when num_workers>0)
    epochs: int = 2                 # fine-tuning a pretrained base; few passes + keep best-eval ckpt
    log_every: int = 50             # log train loss every N steps
    eval_every: int = 200            # run eval every N steps
    eval_batches: int | None = 50   # cap eval to N batches (None = full eval split)
    ckpt_every: int = 1000          # write a checkpoint every N steps
    out_dir: str = "checkpoints"
    # When set AND the HF_TOKEN env var is present, the best checkpoint is pushed to
    # this HuggingFace Hub repo (created private) at the end of training. None = no upload.
    hf_repo: str | None = None
    use_wandb: bool = True
    wandb_project: str = "nanomark"
    wandb_run_name: str | None = None  # display name for the W&B run (None = W&B auto-generates one)
    device: str | None = None       # None = auto-detect (cuda > mps > cpu)
    seed: int = 0                   # seed for the train/eval split

    @property
    def patch_dim(self) -> int:
        """Flattened input size of one patch (patch_size**2 * channels)."""
        return self.patch_size * self.patch_size * self.img_channels

    @property
    def max_patch_grid(self) -> int:
        """Largest patch-grid side an image can produce (ceil(max_image_px / patch_size)).

        Sizes the learned positional-embedding table (``image_pos_mode="learned"``)
        and bounds the per-patch (row, col) coordinates in :func:`data.build_sample`.
        """
        return math.ceil(self.max_image_px / self.patch_size)

    def __post_init__(self):
        """Validate invariants; raise AssertionError on a bad config.

        The model-shape fields are validated only once they are populated (by
        :meth:`from_base` or by an explicit construction): a bare ``Config()`` for
        reading training-knob defaults leaves them ``None`` and skips those checks.
        """
        # --- always-valid training knobs ---
        assert self.image_pos_mode in ("rope2d", "learned"), \
            "image_pos_mode must be 'rope2d' or 'learned'"
        assert 0.0 < self.eval_frac < 1.0, "eval_frac must be in (0, 1)"
        assert self.epochs >= 1, "epochs must be >= 1"
        assert self.grad_accum >= 1, "grad_accum must be >= 1"

        # --- model shape (resolved from the base model; skip until populated) ---
        shape = (self.d_model, self.n_layers, self.n_heads, self.n_kv_heads,
                 self.head_dim, self.mlp_hidden, self.vocab_size, self.padded_vocab)
        if any(v is None for v in shape):
            return
        # head_dim is decoupled from d_model (Granite-style): n_heads*head_dim may
        # differ from d_model, so the q/o projections are not square. No assert ties them.
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert (self.head_dim // 2) % 2 == 0, "head_dim/2 must be even (row/col RoPE freq interleave)"
        assert self.padded_vocab > max(self.bos_id, self.soc_id, self.eos_id), \
            "padded_vocab must cover all special tokens"
        max_img_tokens = (self.max_image_px // self.patch_size) ** 2
        assert self.max_seq_len >= max_img_tokens + 3, "max_seq_len too small to fit a full image + BOS/SOC/EOS"
        assert self.attn_impl in ("sdpa", "flex"), "attn_impl must be 'sdpa' or 'flex'"

    @classmethod
    def from_base(cls, base_repo: str = BASE_REPO, **overrides) -> "Config":
        """Build a Config whose model shape is read from the base model's HF config.

        The decoder is always initialized from ``base_repo`` (see :mod:`granite`), so
        its shape must match the base exactly. This reads the base model's
        ``transformers`` config once and fills d_model/n_layers/n_heads/etc., rather
        than hardcoding them. Any field can still be pinned via ``overrides`` (and is
        the only way the training CLI sets non-shape knobs).

        Args:
            base_repo: HF repo of the base model (config + tokenizer + weights).
            **overrides: Explicit Config field values; take precedence over both the
                derived shape and ``base_repo``.

        Returns:
            A fully-populated, validated :class:`Config`.
        """
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(base_repo)
        head_dim = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)
        # transformers 5.x nests rope_theta under rope_parameters; fall back to the
        # flat attribute (older configs) then a sane default.
        rope_params = getattr(c, "rope_parameters", None) or {}
        rope_base = rope_params.get("rope_theta") or getattr(c, "rope_theta", None) or 1e6
        # Granite's shared (dense) FFN width; plain transformers expose intermediate_size.
        mlp_hidden = getattr(c, "shared_intermediate_size", None) or c.intermediate_size
        shape = dict(
            d_model=c.hidden_size,
            n_layers=c.num_hidden_layers,
            n_heads=c.num_attention_heads,
            n_kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
            head_dim=head_dim,
            mlp_hidden=mlp_hidden,
            rope_base=float(rope_base),
            rms_norm_eps=float(getattr(c, "rms_norm_eps", 1e-5)),
            # Granite muP multipliers (default to no-ops for non-Granite bases).
            embedding_multiplier=float(getattr(c, "embedding_multiplier", 1.0)),
            attention_multiplier=getattr(c, "attention_multiplier", None),
            residual_multiplier=float(getattr(c, "residual_multiplier", 1.0)),
            logits_scaling=float(getattr(c, "logits_scaling", 1.0)),
            vocab_size=c.vocab_size,
            padded_vocab=c.vocab_size,  # tied tok_emb is copied 1:1 from the base embedding
        )
        return cls(base_repo=base_repo, **{**shape, **overrides})
