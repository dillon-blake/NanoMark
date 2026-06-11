"""Run a trained NanoMark on a single image and print the transcription.

    python inference.py --ckpt checkpoints/final.pt --image page.png

Generation is naive (no KV cache): each step re-runs the full forward over the
image prefix + text so far. Correct but O(n^2); a KV cache is the obvious
follow-up speedup.
"""

import argparse

import torch
from PIL import Image

from config import Config
from data import (PAD_IMG, REAL_IMG, TEXT, build_attn_mask, get_tokenizer,
                  patch_grid_coords, preprocess_image)
from model import NanoMark


@torch.no_grad()
def transcribe(model, img, cfg, tokenizer, device, max_new_tokens=256, temperature=0.0):
    """Greedily (or with temperature) decode the transcription of one image.

    Builds the ``[BOS] [image patches] [SOC]`` prefix with the same preprocessing
    as training, then appends one text token at a time until EOS or
    ``max_new_tokens``. Output is restricted to real BPE tokens and EOS (never
    BOS/SOC or vocab padding). No KV cache: each step re-runs the full forward.

    Args:
        model: A trained NanoMark in eval mode.
        img: The document image to transcribe.
        cfg: The model configuration.
        tokenizer: A HuggingFace tokenizer for decoding the output ids.
        device: Device to run on.
        max_new_tokens: Hard cap on generated tokens.
        temperature: 0 for greedy argmax; >0 to sample from the softmax.

    Returns:
        The decoded transcription as a string.
    """
    patches, rr, cc, G = preprocess_image(img, cfg)
    patches = patches.to(device)[None]  # [1, P, patch_dim]

    # prefix: BOS + image patches + SOC
    input_ids = [cfg.bos_id] + [0] * (G * G) + [cfg.soc_id]
    token_type = ([TEXT]
                  + [REAL_IMG if (r < rr and c < cc) else PAD_IMG
                     for r in range(G) for c in range(G)]
                  + [TEXT])
    # grid coords for the learned positional table (analog of build_sample's patch_pos)
    patch_pos = torch.tensor([patch_grid_coords(G)], device=device)  # [1, P, 2]

    # RoPE positions: same scheme as build_sample. "learned" gives image tokens a
    # plain sequential 1D index; "rope2d" places patches on their (r+1, c+1) grid
    # and resumes text past the image extent. The per-step append below (pos[-1]+1)
    # continues either scheme correctly.
    if cfg.image_pos_mode == "learned":
        pos_h = list(range(G * G + 2))  # BOS + P patches + SOC, sequential
        pos_w = list(pos_h)
    else:
        pos_h, pos_w = [0], [0]
        for r in range(G):
            for c in range(G):
                pos_h.append(r + 1)
                pos_w.append(c + 1)
        offset = 1 + max(rr, cc)
        pos_h.append(offset)  # SOC
        pos_w.append(offset)

    # only real tokens and EOS are valid outputs. The Granite structural tokens
    # live *inside* the vocab range, so forbid BOS/SOC explicitly (and the unused
    # vocab-padding rows); EOS stays allowed.
    valid = torch.zeros(cfg.padded_vocab, device=device)
    valid[cfg.vocab_size:] = float("-inf")
    valid[cfg.bos_id] = float("-inf")
    valid[cfg.soc_id] = float("-inf")

    generated = []
    for _ in range(max_new_tokens):
        tt = torch.tensor([token_type], device=device)
        logits = model(
            torch.tensor([input_ids], device=device),
            patches,
            (tt == REAL_IMG) | (tt == PAD_IMG),
            torch.tensor([pos_h], device=device),
            torch.tensor([pos_w], device=device),
            build_attn_mask(tt, cfg.bidirectional_img),
            patch_pos,
        )
        next_logits = logits[0, -1] + valid
        if temperature > 0:
            probs = (next_logits / temperature).softmax(-1)
            nxt = int(torch.multinomial(probs, 1))
        else:
            nxt = int(next_logits.argmax())
        if nxt == cfg.eos_id:
            break
        generated.append(nxt)
        input_ids.append(nxt)
        token_type.append(TEXT)
        pos_h.append(pos_h[-1] + 1)
        pos_w.append(pos_w[-1] + 1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    """Load a checkpoint and print the transcription of ``--image``.

    Rebuilds the model from the config stored in the checkpoint so inference
    matches the trained architecture.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg")
    if not isinstance(cfg, Config):
        raise ValueError("checkpoint has no Config under 'cfg'; cannot rebuild the model shape")
    model = NanoMark(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.image)
    text = transcribe(model, img, cfg, get_tokenizer(cfg.base_repo), device,
                      args.max_new_tokens, args.temperature)
    print(text)


if __name__ == "__main__":
    main()
