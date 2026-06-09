"""The NanoMark model: an encoder-free OCR decoder.

A Qwen3-0.6B-shaped decoder-only transformer with modern components:
    - RMSNorm (``torch.nn.RMSNorm``) instead of LayerNorm
    - 2D RoPE (image patches get (row, col); text gets 1D) instead of learned
      positional embeddings; for text it reduces exactly to Qwen3's 1D RoPE
    - Grouped-query attention (GQA) with QK-Norm, full (non-windowed) attention
      on every layer
    - SwiGLU MLP
    - tied input/output embedding
    - a linear patch projector (no vision encoder) that maps raw image patches
      straight into the model dimension
    - the decoder can be loaded from Qwen3 0.6B Base (see ``qwen.load_qwen3``)

Images and text share one sequence. The model itself does no preprocessing: the
token ids, image patches, 2D-RoPE positions, image-slot mask, and attention mask
are all produced by ``data.py`` and passed into :meth:`NanoMark.forward`. The
attention mask is boolean (bidirectional over image patches, causal over text,
with padding masked) — see :func:`data.build_attn_mask`.

Tensor shape conventions used throughout:
    B = batch, S = sequence length, P = image patches per sample,
    H = query heads, D = head dim, d = d_model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def build_rope_cache(pos_h, pos_w, head_dim, base):
    """Build the cos/sin rotation tables for 2D RoPE (Qwen3-compatible).

    Uses Qwen3's standard 1D-RoPE frequency layout (``head_dim // 2`` frequencies
    over the full head, base ``rope_theta``, duplicated for the rotate-half pair),
    but assigns frequencies to the two image axes by *interleaving*: even-indexed
    frequencies rotate by the row coordinate (``pos_h``), odd-indexed by the column
    (``pos_w``). Each axis therefore spans the full spectrum (no high/low bias).

    The key property: for text tokens the data pipeline sets ``pos_h == pos_w``, so
    every frequency rotates by the same position and this collapses *exactly* to
    Qwen3's ordinary 1D RoPE -- which is what the loaded attention weights expect.

    Args:
        pos_h: Row positions, int tensor of shape [B, S].
        pos_w: Column positions, int tensor of shape [B, S].
        head_dim: Size of each attention head (must be divisible by 4).
        base: RoPE base/theta controlling the frequency range (e.g. 1e6).

    Returns:
        A tuple ``(cos, sin)``, each a float tensor of shape [B, S, head_dim].
    """
    n_freq = head_dim // 2                     # Qwen3 1D-RoPE: one freq per rotate-half pair
    device = pos_h.device
    inv_freq = base ** (-torch.arange(0, n_freq, device=device, dtype=torch.float32) / n_freq)  # [n_freq]

    # interleave the two coordinates across the frequency spectrum
    coord = torch.empty(*pos_h.shape, n_freq, device=device, dtype=torch.float32)  # [B, S, n_freq]
    coord[..., 0::2] = pos_h.float()[..., None]    # even freqs <- row
    coord[..., 1::2] = pos_w.float()[..., None]    # odd  freqs <- col

    theta = coord * inv_freq                       # [B, S, n_freq]
    emb = torch.cat([theta, theta], dim=-1)        # [B, S, head_dim] (standard RoPE duplication)
    return emb.cos(), emb.sin()


def rotate_half(x):
    """Standard RoPE rotate-half: split the last dim in two and map (x1, x2) -> (-x2, x1)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """Apply rotary position embedding to queries or keys.

    The standard GPT-NeoX / Qwen3 ``rotate_half`` convention over the full head dim.

    Args:
        x: Queries or keys, shape [B, H, S, head_dim].
        cos: Cosine table from :func:`build_rope_cache`, shape [B, S, head_dim].
        sin: Sine table from :func:`build_rope_cache`, shape [B, S, head_dim].

    Returns:
        The rotated tensor, same shape as ``x``.
    """
    cos = cos[:, None, :, :]  # [B, 1, S, head_dim]
    sin = sin[:, None, :, :]
    return x * cos + rotate_half(x) * sin


def scaled_dot_product_attention(q, k, v, attn_mask):
    """Run attention via PyTorch's fused kernel, with a naive fallback.

    Prefers ``F.scaled_dot_product_attention`` (which dispatches to FlashAttention
    / memory-efficient kernels) and falls back to an explicit softmax when it is
    unavailable.

    Args:
        q: Queries, shape [B, H, S, D].
        k: Keys, shape [B, H, S, D] (already expanded to H heads for GQA).
        v: Values, shape [B, H, S, D] (already expanded to H heads for GQA).
        attn_mask: Boolean mask of shape [B, 1, S, S] where True = attend and
            False = block. Every query row must have at least one True entry, or
            softmax sees all -inf and returns NaN; :func:`data.collate_fn`
            guarantees this.

    Returns:
        The attention output, shape [B, H, S, D].
    """
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    # naive fallback
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # [B, H, S, S]
    scores = scores.masked_fill(~attn_mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v


class Attention(nn.Module):
    """Grouped-query self-attention with QK-Norm and 2D RoPE.

    Uses ``n_heads`` query heads but only ``n_kv_heads`` key/value heads (GQA);
    the KV heads are repeated to match the query heads before attention. Queries
    and keys are RMS-normalized per head (QK-Norm, Qwen3-style) before RoPE is
    applied. There are no biases on the projections. ``head_dim`` is decoupled
    from ``d_model``, so the q/o projections need not be square.
    """

    def __init__(self, cfg: Config):
        """Create the q/k/v/o projections and the per-head q/k norms from ``cfg``."""
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        # QK-Norm: RMSNorm over head_dim, applied to q and k before RoPE. Stabilizes
        # the softmax (esp. with the high-entropy image tokens) and matches Qwen3.
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=1e-6)

    def forward(self, x, cos, sin, attn_mask):
        """Apply self-attention.

        Args:
            x: Input hidden states, shape [B, S, d_model].
            cos: RoPE cosine table, shape [B, S, head_dim].
            sin: RoPE sine table, shape [B, S, head_dim].
            attn_mask: Boolean attention mask, shape [B, 1, S, S] (True = attend).

        Returns:
            Output hidden states, shape [B, S, d_model].
        """
        B, S, _ = x.shape
        # QK-Norm over the head_dim (last axis) before the head/seq transpose.
        q = self.q_norm(self.q_proj(x).view(B, S, self.n_heads, self.head_dim)).transpose(1, 2)     # [B, Hq, S, D]
        k = self.k_norm(self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)).transpose(1, 2)  # [B, Hkv, S, D]
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: expand kv heads to match query heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        out = scaled_dot_product_attention(q, k, v, attn_mask)         # [B, Hq, S, D]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: ``down(silu(gate(x)) * up(x))``.

    Three bias-free projections (gate, up, down). The SiLU-gated formulation is
    the standard Llama/Qwen MLP; ``cfg.mlp_hidden`` is sized to roughly match a
    plain 4x GELU MLP's parameter count.
    """

    def __init__(self, cfg: Config):
        """Create the gate/up/down projections sized from ``cfg``."""
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        """Apply the MLP. ``x``: [B, S, d_model] -> [B, S, d_model]."""
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """A pre-norm transformer block: attention then MLP, each with a residual."""

    def __init__(self, cfg: Config):
        """Create the two RMSNorms, the attention, and the MLP."""
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask):
        """Run the block.

        Args:
            x: Input hidden states, shape [B, S, d_model].
            cos: RoPE cosine table, shape [B, S, head_dim].
            sin: RoPE sine table, shape [B, S, head_dim].
            attn_mask: Boolean attention mask, shape [B, 1, S, S].

        Returns:
            Output hidden states, shape [B, S, d_model].
        """
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# Parameters with no pretrained counterpart: the linear vision adapter, always
# trained from scratch even when the decoder is loaded from a base model. The
# optimizer (full LR vs. gentler fine-tune LR) and the base loader (which weights
# to fill vs. leave random) both key off this.
VISION_ADAPTER_PREFIXES = ("patch_proj", "patch_norm", "patch_pos_emb")


def is_vision_adapter(name: str) -> bool:
    """True for a parameter that belongs to the from-scratch vision adapter."""
    return name.startswith(VISION_ADAPTER_PREFIXES)


class NanoMark(nn.Module):
    """The full encoder-free OCR model.

    Merges two input streams into one sequence: text token embeddings (via
    ``tok_emb``) and image patch embeddings (via the linear ``patch_proj``), the
    latter spliced into the image-slot positions. The stack of :class:`Block`s
    then runs full attention with 2D RoPE, and the output is projected back to
    vocabulary logits using the (tied) token-embedding weight.
    """

    def __init__(self, cfg: Config):
        """Build the embedding, patch projector + norm, blocks, and final norm.

        After the standard N(0, 0.02) init, the residual output projections are
        downscaled by ``1/sqrt(2*n_layers)`` to keep the residual-stream variance
        stable with depth.

        Args:
            cfg: The model/training configuration.
        """
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.padded_vocab, cfg.d_model)
        self.patch_proj = nn.Linear(cfg.patch_dim, cfg.d_model, bias=False)
        # normalize projected patches so the image-embedding scale is consistent
        # regardless of pixel/pad statistics (cf. Gemma's vision embedder)
        self.patch_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        # "learned" image positions (Gemma 4 Unified-style): a factorized 2D table
        # looked up per-axis (row via [:, 0], col via [:, 1]) and summed onto the
        # projected patch before patch_norm. Zero-init so a fresh learned model is
        # numerically identical to a no-position baseline at step 0. Only created
        # in this mode, so "rope2d" state_dicts are unchanged.
        if cfg.image_pos_mode == "learned":
            self.patch_pos_emb = nn.Parameter(torch.zeros(cfg.max_patch_grid, 2, cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        # the output head is tied to tok_emb.weight (applied in forward)
        self.apply(self._init_weights)
        # GPT-2-style: scale residual output projections by 1/sqrt(2*n_layers)
        # so the residual stream's variance does not grow with depth
        residual_scale = (2 * cfg.n_layers) ** -0.5
        with torch.no_grad():
            for block in self.blocks:
                block.attn.o_proj.weight.mul_(residual_scale)
                block.mlp.down.weight.mul_(residual_scale)

    def _init_weights(self, module):
        """Initialize Linear/Embedding weights to N(0, 0.02); biases to zero.

        ``nn.RMSNorm`` is left at its default (gain initialized to ones).
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def trunk(self, input_ids, patches, image_slots, pos_h, pos_w, attn_mask, patch_pos=None):
        """Run the transformer trunk and return final-normed hidden states.

        Everything in :meth:`forward` except the tied output projection. Split
        out so the loss path can project the head on only the positions that
        contribute to the loss (see :meth:`loss`).

        Args:
            input_ids: Token ids, shape [B, S]. Image-slot positions hold a
                placeholder id (overwritten by ``patches``, so its value is
                ignored).
            patches: Flattened image patches, shape [B, P, patch_dim], right-
                padded per sample (P = max patches in the batch).
            image_slots: Boolean mask, shape [B, S], True at image-patch
                positions. ``image_slots[b].sum()`` equals sample ``b``'s patch
                count, so it aligns with ``patches[b]``.
            pos_h: Row positions for RoPE, shape [B, S].
            pos_w: Column positions for RoPE, shape [B, S].
            attn_mask: Boolean attention mask, shape [B, 1, S, S] (True = attend).
            patch_pos: Per-patch grid coordinates, shape [B, P, 2] (row, col),
                ``-1`` on padding. Required when ``image_pos_mode == "learned"``
                (indexes the learned positional table); ignored under "rope2d".

        Returns:
            Final hidden states, shape [B, S, d_model].
        """
        h = self.tok_emb(input_ids)                          # [B, S, d]
        proj = self.patch_proj(patches)                      # [B, P, d]
        if self.cfg.image_pos_mode == "learned":
            # factorized 2D lookup summed onto the patch, then normed (Gemma order)
            assert patch_pos is not None, "image_pos_mode='learned' requires patch_pos"
            rows = patch_pos[..., 0].clamp(min=0)            # -1 pad -> 0; scatter drops it
            cols = patch_pos[..., 1].clamp(min=0)
            proj = proj + self.patch_pos_emb[rows, 0] + self.patch_pos_emb[cols, 1]
        proj = self.patch_norm(proj)                         # [B, P, d], scale-stabilized

        # splice projected patches into the image slots (per sample; see helper)
        h = _scatter_patches(h, proj, image_slots)

        cos, sin = build_rope_cache(pos_h, pos_w, self.cfg.head_dim, self.cfg.rope_base)
        cos, sin = cos.to(h.dtype), sin.to(h.dtype)
        for block in self.blocks:
            h = block(h, cos, sin, attn_mask)
        return self.final_norm(h)

    def forward(self, input_ids, patches, image_slots, pos_h, pos_w, attn_mask, patch_pos=None):
        """Compute next-token logits over the whole sequence.

        Args:
            input_ids: Token ids, shape [B, S]. Image-slot positions hold a
                placeholder id (overwritten by ``patches``, so its value is
                ignored).
            patches: Flattened image patches, shape [B, P, patch_dim], right-
                padded per sample (P = max patches in the batch).
            image_slots: Boolean mask, shape [B, S], True at image-patch
                positions. ``image_slots[b].sum()`` equals sample ``b``'s patch
                count, so it aligns with ``patches[b]``.
            pos_h: Row positions for RoPE, shape [B, S].
            pos_w: Column positions for RoPE, shape [B, S].
            attn_mask: Boolean attention mask, shape [B, 1, S, S] (True = attend).
            patch_pos: Per-patch grid coordinates [B, P, 2]; required under
                ``image_pos_mode == "learned"`` (see :meth:`trunk`).

        Returns:
            Logits over the padded vocabulary, shape [B, S, padded_vocab].
        """
        h = self.trunk(input_ids, patches, image_slots, pos_h, pos_w, attn_mask, patch_pos)
        return h @ self.tok_emb.weight.T                     # tied head, [B, S, padded_vocab]

    def loss(self, input_ids, patches, image_slots, pos_h, pos_w, attn_mask, labels, patch_pos=None):
        """Label-masked cross-entropy, projecting the head only where it matters.

        Equivalent to ``cross_entropy(forward(...), labels, ignore_index=-100)``
        but the tied output head and the fp32 logits are computed only at the
        positions that carry a label (``labels != -100``). The gather flattens
        across the batch, so image-patch positions and SEQ_PAD padding never
        produce logits — the [B*S, vocab] fp32 tensor that otherwise dominates
        memory shrinks to [num_text_tokens, vocab]. The loss and gradients are
        identical to the full-sequence computation (the dropped rows are exactly
        the ones ``ignore_index`` would have masked out).

        Args:
            input_ids/patches/image_slots/pos_h/pos_w/attn_mask/patch_pos: As in
                :meth:`forward`.
            labels: Next-token targets, shape [B, S], with -100 at positions
                excluded from the loss.

        Returns:
            A scalar cross-entropy loss tensor (computed in fp32).
        """
        h = self.trunk(input_ids, patches, image_slots, pos_h, pos_w, attn_mask, patch_pos)
        keep = labels != -100                                # [B, S], True on text targets
        if not keep.any():                                   # all-masked batch: empty gather -> CE is NaN
            return h.sum() * 0.0                             # finite, zero-gradient, keeps the graph valid
        h_kept = h[keep]                                     # [N, d], gathered across the batch
        logits = (h_kept @ self.tok_emb.weight.T).float()    # [N, padded_vocab]
        return F.cross_entropy(logits, labels[keep])


def _scatter_patches(h, proj, image_slots):
    """Overwrite image-slot embeddings with the projected patch embeddings.

    Done per sample because both ``S`` and ``P`` are padded to the batch maximum:
    ``image_slots[b]`` is True for exactly sample ``b``'s real patch count, which
    matches ``proj[b, :n]`` (the rest of ``proj[b]`` is padding). A single fused
    scatter would misalign these per-sample counts. ``h`` is cloned first so the
    masked write does not touch the embedding table or break autograd.

    Args:
        h: Token embeddings, shape [B, S, d] (text-slot embeddings to keep).
        proj: Projected patches, shape [B, P, d] (right-padded per sample).
        image_slots: Boolean mask, shape [B, S], True at image positions.

    Returns:
        ``h`` with image-slot rows replaced by patch embeddings, shape [B, S, d].
    """
    h = h.clone()
    for b in range(h.shape[0]):
        n = int(image_slots[b].sum())
        h[b, image_slots[b]] = proj[b, :n].to(h.dtype)
    return h
