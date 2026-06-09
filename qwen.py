"""Load Qwen3 0.6B Base weights into a NanoMark model.

NanoMark's decoder is a name-compatible superset of Qwen3: same blocks (GQA +
QK-Norm + SwiGLU, pre-norm, RMSNorm, tied embedding) with a decoupled head_dim.
This module copies every Qwen3 tensor into the matching NanoMark parameter; the
only parameters with no Qwen counterpart are the vision adapter (``patch_proj``,
``patch_norm``), which keep their random init -- that is the one part NanoMark
learns from scratch.

The RoPE base (``rope_base``/``rope_theta``) and the tokenizer must also match
the base model; those live in ``config.py`` / ``data.get_tokenizer``.
"""

from config import BASE_REPO
from model import is_vision_adapter


# Qwen3 per-layer tensor suffix (after ``model.layers.N.``) -> NanoMark name
# template, filled with the layer index.
_LAYER_MAP = {
    "self_attn.q_proj.weight": "blocks.{i}.attn.q_proj.weight",
    "self_attn.k_proj.weight": "blocks.{i}.attn.k_proj.weight",
    "self_attn.v_proj.weight": "blocks.{i}.attn.v_proj.weight",
    "self_attn.o_proj.weight": "blocks.{i}.attn.o_proj.weight",
    "self_attn.q_norm.weight": "blocks.{i}.attn.q_norm.weight",
    "self_attn.k_norm.weight": "blocks.{i}.attn.k_norm.weight",
    "mlp.gate_proj.weight": "blocks.{i}.mlp.gate.weight",
    "mlp.up_proj.weight": "blocks.{i}.mlp.up.weight",
    "mlp.down_proj.weight": "blocks.{i}.mlp.down.weight",
    "input_layernorm.weight": "blocks.{i}.attn_norm.weight",
    "post_attention_layernorm.weight": "blocks.{i}.mlp_norm.weight",
}


def _map_name(qwen_name: str):
    """Map a Qwen3 HF parameter name to its NanoMark name (or None to skip).

    Skips ``lm_head.weight`` (NanoMark ties the head to ``tok_emb``) and anything
    without a NanoMark counterpart.
    """
    if qwen_name == "model.embed_tokens.weight":
        return "tok_emb.weight"
    if qwen_name == "model.norm.weight":
        return "final_norm.weight"
    if qwen_name.startswith("model.layers."):
        i, sub = qwen_name[len("model.layers."):].split(".", 1)
        template = _LAYER_MAP.get(sub)
        return template.format(i=i) if template else None
    return None  # lm_head (tied), rotary buffers, etc.


def load_qwen3(model, repo: str = BASE_REPO):
    """Copy Qwen3 weights from ``repo`` into ``model`` in place.

    Loads the model's safetensors, maps each tensor to its NanoMark parameter, and
    copies it (cast to the model's dtype). Asserts that every NanoMark parameter is
    either filled from Qwen3 or is a known fresh (vision-adapter) parameter, so a
    naming or shape drift fails loudly rather than silently leaving weights random.

    Args:
        model: A :class:`model.NanoMark` whose config matches the base shape.
        repo: HF repo id for the base model.

    Returns:
        The same ``model``, mutated in place.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch

    src = load_file(hf_hub_download(repo, "model.safetensors"))
    sd = model.state_dict()

    loaded = set()
    with torch.no_grad():
        for qname, tensor in src.items():
            name = _map_name(qname)
            if name is None:
                continue
            if name not in sd:
                raise KeyError(f"Qwen tensor {qname!r} mapped to {name!r}, which is not in the model")
            if sd[name].shape != tensor.shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: model {tuple(sd[name].shape)} vs Qwen {tuple(tensor.shape)}"
                )
            sd[name].copy_(tensor.to(sd[name].dtype))
            loaded.add(name)

    missing = {n for n in sd if n not in loaded and not is_vision_adapter(n)}
    if missing:
        raise RuntimeError(f"these params were neither loaded from Qwen nor are fresh: {sorted(missing)}")
    print(f"loaded {len(loaded)} tensors from {repo}; "
          f"{sum(1 for n in sd if is_vision_adapter(n))} vision-adapter tensors kept at random init")
    return model
