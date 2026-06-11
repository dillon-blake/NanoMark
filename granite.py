"""Load Granite-4.0-350M Base weights into a NanoMark model.

NanoMark's decoder is a name-compatible (re)mapping of Granite's dense
``GraniteMoeHybrid`` decoder: same blocks (GQA + SwiGLU, pre-norm, RMSNorm, tied
embedding, muP multipliers) with a decoupled head_dim and *no* QK-Norm. This
module copies every Granite attention/MLP tensor into the matching NanoMark
parameter; the only parameters with no Granite counterpart are the vision adapter
(``patch_proj``, ``patch_norm``), which keep their random init -- that is the one
part NanoMark learns from scratch.

Two name shapes differ from a plain Llama/Qwen decoder and are handled below:
  - Granite fuses the SwiGLU gate+up into a single ``shared_mlp.input_linear``
    (output features [2*intermediate]); the forward is ``act(chunk[0]) * chunk[1]``
    (chunk along the last/output dim), so the first half is the gate and the second
    is the up. We split it row-wise into NanoMark's separate ``mlp.gate``/``mlp.up``.
  - the down projection is ``shared_mlp.output_linear``.

The muP multipliers, RoPE base, RMSNorm eps, and the tokenizer must also match the
base model; those live in ``config.py`` / ``data.get_tokenizer``.
"""

from config import BASE_REPO
from model import is_vision_adapter


# Granite per-layer tensor suffix (after ``model.layers.N.``) -> NanoMark name
# template, filled with the layer index. ``shared_mlp.input_linear`` is handled
# separately (it splits into two NanoMark params), so it is not in this 1:1 map.
_LAYER_MAP = {
    "self_attn.q_proj.weight": "blocks.{i}.attn.q_proj.weight",
    "self_attn.k_proj.weight": "blocks.{i}.attn.k_proj.weight",
    "self_attn.v_proj.weight": "blocks.{i}.attn.v_proj.weight",
    "self_attn.o_proj.weight": "blocks.{i}.attn.o_proj.weight",
    "shared_mlp.output_linear.weight": "blocks.{i}.mlp.down.weight",
    "input_layernorm.weight": "blocks.{i}.attn_norm.weight",
    "post_attention_layernorm.weight": "blocks.{i}.mlp_norm.weight",
}


def _expand(granite_name: str, tensor):
    """Map a Granite HF parameter to NanoMark ``(name, tensor)`` pairs.

    Returns a list because Granite's fused ``shared_mlp.input_linear`` splits into
    two NanoMark params (gate, up); every other mapped tensor yields a single
    pair, and unmapped tensors (``lm_head`` -- tied, rotary buffers, etc.) yield
    an empty list.
    """
    if granite_name == "model.embed_tokens.weight":
        return [("tok_emb.weight", tensor)]
    if granite_name == "model.norm.weight":
        return [("final_norm.weight", tensor)]
    if granite_name.startswith("model.layers."):
        i, sub = granite_name[len("model.layers."):].split(".", 1)
        if sub == "shared_mlp.input_linear.weight":
            # fused gate+up: forward is act(chunk[0]) * chunk[1] -> first half is
            # the gate, second half the up (chunk is over the output/row dim here).
            half = tensor.shape[0] // 2
            return [
                (f"blocks.{i}.mlp.gate.weight", tensor[:half]),
                (f"blocks.{i}.mlp.up.weight", tensor[half:]),
            ]
        template = _LAYER_MAP.get(sub)
        return [(template.format(i=i), tensor)] if template else []
    return []  # lm_head (tied), rotary buffers, etc.


def load_granite(model, repo: str = BASE_REPO):
    """Copy Granite weights from ``repo`` into ``model`` in place.

    Loads the model's safetensors, maps each tensor to its NanoMark parameter(s),
    and copies it (cast to the model's dtype). Asserts that every NanoMark
    parameter is either filled from Granite or is a known fresh (vision-adapter)
    parameter, so a naming or shape drift fails loudly rather than silently
    leaving weights random.

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
        for gname, tensor in src.items():
            for name, t in _expand(gname, tensor):
                if name not in sd:
                    raise KeyError(f"Granite tensor {gname!r} mapped to {name!r}, which is not in the model")
                if sd[name].shape != t.shape:
                    raise ValueError(
                        f"shape mismatch for {name!r}: model {tuple(sd[name].shape)} vs Granite {tuple(t.shape)}"
                    )
                sd[name].copy_(t.to(sd[name].dtype))
                loaded.add(name)

    missing = {n for n in sd if n not in loaded and not is_vision_adapter(n)}
    if missing:
        raise RuntimeError(f"these params were neither loaded from Granite nor are fresh: {sorted(missing)}")
    print(f"loaded {len(loaded)} tensors from {repo}; "
          f"{sum(1 for n in sd if is_vision_adapter(n))} vision-adapter tensors kept at random init")
    return model
