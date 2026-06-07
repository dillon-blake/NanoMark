"""Optimizer setup: the Muon/AdamW hybrid parameter split.

Muon orthogonalizes the momentum update of 2D hidden weight matrices; the
embedding/tied head and all 1D params (norm gains, biases) use AdamW instead.
We use PyTorch's built-in torch.optim.Muon (the algorithm); the value here is
build_optimizers(), the model-specific routing of params to each optimizer.
"""

import torch
from torch.optim import Muon


def build_optimizers(model, cfg):
    """Route the model's parameters into the Muon/AdamW hybrid.

    Routing policy:
        - Muon: 2D weight matrices inside the transformer blocks plus the linear
          patch projector (the "hidden matmuls").
        - AdamW (with weight decay): other 2D params, i.e. the token embedding
          (which is also the tied output head).
        - AdamW (no weight decay): 1D params, i.e. RMSNorm gains and any biases.

    Args:
        model: The :class:`model.NanoMark` (or any module) to optimize.
        cfg: Configuration providing learning rates, betas, and weight decay.

    Returns:
        A tuple ``(muon, adamw)`` of optimizers. The caller is expected to step
        both each iteration. The AdamW optimizer has two param groups (decay /
        no-decay).
    """
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_block_matmul = p.ndim >= 2 and (".blocks." in f".{name}" or name.startswith("patch_proj"))
        if is_block_matmul:
            muon_params.append(p)
        elif p.ndim >= 2:
            adamw_decay.append(p)        # tok_emb (also the tied head)
        else:
            adamw_nodecay.append(p)      # RMSNorm gains, biases

    # adjust_lr_fn="match_rms_adamw" applies the 0.2*sqrt(max(fan_in, fan_out))
    # RMS scaling (Moonshot variant) so Muon and AdamW share a learning rate.
    muon = Muon(muon_params, lr=cfg.lr_muon, momentum=0.95,
                weight_decay=cfg.weight_decay, ns_steps=5,
                adjust_lr_fn="match_rms_adamw")
    adamw = torch.optim.AdamW(
        [
            {"params": adamw_decay, "weight_decay": cfg.weight_decay},
            {"params": adamw_nodecay, "weight_decay": 0.0},
        ],
        lr=cfg.lr_adamw, betas=cfg.adam_betas,
    )
    return muon, adamw
