"""Focused tests for the parts that fail silently: the attention mask, the
label masking, patchify, 2D RoPE, sequence assembly, optimizer split, plus a
forward/backward smoke test.

Run: pytest -q  (from the repo root)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # tests/ (for synthetic)

from config import Config
from data import (PAD_IMG, REAL_IMG, SEQ_PAD, TEXT, OCRDataset, build_attn_mask,
                  build_sample, collate_fn, get_tokenizer, preprocess_image)
from model import NanoMark, apply_rope, build_rope_cache, is_vision_adapter
from synthetic import render
from train import build_optimizer


def small_cfg(**kw):
    """A tiny but architecturally identical config, for fast tests.

    Passes the model shape explicitly (the real pipeline reads it from the base
    model via ``Config.from_base``; tests build a tiny model with no base). The
    vocab matches the Qwen3 tokenizer the tests tokenize with.
    """
    base = dict(d_model=32, n_heads=4, n_kv_heads=2, head_dim=8, mlp_hidden=64,
                n_layers=2, patch_size=8, max_image_px=64, rope_base=1000000.0,
                vocab_size=151936, padded_vocab=151936)
    base.update(kw)
    return Config(**base)


def make_batch(cfg, text="hello world foo bar"):
    """Build a single-sample batch from rendered synthetic text.

    Returns ``(sample, batch)`` where ``sample`` is the un-collated dict and
    ``batch`` is the collated single-item batch.
    """
    tok = get_tokenizer()
    sample = build_sample(render(text), text, tok, cfg)
    return sample, collate_fn([sample], cfg)


# --------------------------------------------------------------------------- #
# 1. attention mask: hand-computed tiny case
# --------------------------------------------------------------------------- #
def test_attn_mask_matrix():
    # BOS, 2x2 image (rows_real=2, cols_real=1 -> col 0 real, col 1 pad),
    # SOC, two text tokens, EOS
    tt = torch.tensor([[TEXT, REAL_IMG, PAD_IMG, REAL_IMG, PAD_IMG, TEXT, TEXT, TEXT]])
    #   index:            0     1         2        3         4       5(SOC) 6     7(EOS)
    mask = build_attn_mask(tt)[0, 0]  # [S, S] bool
    expected = {
        0: {0},                  # BOS: causal, only itself
        1: {0, 1, 3},            # real img: BOS + real img patches
        2: {0, 1, 3},            # pad img query: same prefix
        3: {0, 1, 3},
        4: {0, 1, 3},
        5: {0, 1, 3, 5},         # SOC text: real prefix + causal text
        6: {0, 1, 3, 5, 6},
        7: {0, 1, 3, 5, 6, 7},
    }
    for i, allowed in expected.items():
        got = set(torch.nonzero(mask[i]).flatten().tolist())
        assert got == allowed, f"row {i}: got {got}, want {allowed}"


def test_seqpad_query_has_self():
    # a fully-pad row must keep at least the diagonal so softmax stays finite
    tt = torch.tensor([[TEXT, REAL_IMG, TEXT, SEQ_PAD, SEQ_PAD]])
    mask = build_attn_mask(tt)[0, 0]
    for i in (3, 4):
        assert mask[i].sum() == 1 and mask[i, i]
    # nothing attends to seq-pad keys
    assert mask[:, 3].sum() == 1 and mask[:, 4].sum() == 1  # only own diagonal


# --------------------------------------------------------------------------- #
# 2-4. behavioral mask tests via a real forward
# --------------------------------------------------------------------------- #
@torch.no_grad()
def test_causality_text():
    cfg = small_cfg()
    torch.manual_seed(0)
    model = NanoMark(cfg).eval()
    _, b = make_batch(cfg)
    tt = b["token_type"][0]
    soc = int((tt == TEXT).nonzero()[1])  # second TEXT block start is SOC; find it robustly
    # SOC is the first TEXT after the image block
    img_end = int((tt == REAL_IMG).nonzero().max() if (tt == REAL_IMG).any() else 0)
    soc = img_end + 1
    while tt[soc] != TEXT:
        soc += 1
    p = soc + 1  # first ocr text token

    out1 = model(b["input_ids"], b["patches"], b["image_slots"], b["pos_h"], b["pos_w"], b["attn_mask"])
    ids2 = b["input_ids"].clone()
    ids2[0, p] = (ids2[0, p] + 7) % cfg.vocab_size  # perturb a future text token
    out2 = model(ids2, b["patches"], b["image_slots"], b["pos_h"], b["pos_w"], b["attn_mask"])

    assert torch.allclose(out1[0, :p], out2[0, :p], atol=1e-5), "earlier positions changed -> not causal"
    assert not torch.allclose(out1[0, p], out2[0, p], atol=1e-4), "perturbation had no effect"


@torch.no_grad()
def test_bidirectional_image():
    cfg = small_cfg()
    torch.manual_seed(0)
    model = NanoMark(cfg).eval()
    _, b = make_batch(cfg)
    tt = b["token_type"][0]
    real_idx = (tt == REAL_IMG).nonzero().flatten()  # sequence positions of real patches
    assert len(real_idx) >= 2
    i_seq, j_seq = int(real_idx[0]), int(real_idx[-1])  # i earlier than j

    out1 = model(b["input_ids"], b["patches"], b["image_slots"], b["pos_h"], b["pos_w"], b["attn_mask"])
    patches2 = b["patches"].clone()
    torch.manual_seed(1)
    # non-uniform perturbation: changes the patch's direction. (A uniform add
    # would be cancelled by patch_norm, which is scale-invariant.) Scaled up so the
    # forward-attention effect clears the tolerance even with QK-Norm damping the
    # attention logits in this tiny random-init net.
    patches2[0, j_seq - 1] += 5.0 * torch.randn(cfg.patch_dim)  # patch array idx = seq pos - 1 (BOS at 0)
    out2 = model(b["input_ids"], patches2, b["image_slots"], b["pos_h"], b["pos_w"], b["attn_mask"])

    # earlier image token's output changes because it attends forward to patch j
    assert not torch.allclose(out1[0, i_seq], out2[0, i_seq], atol=1e-4), "image attention not bidirectional"


@torch.no_grad()
def test_no_attention_to_pad():
    cfg = small_cfg()
    torch.manual_seed(0)
    model = NanoMark(cfg).eval()
    _, b = make_batch(cfg)
    tt = b["token_type"][0]
    pad_idx = (tt == PAD_IMG).nonzero().flatten()
    assert len(pad_idx) >= 1
    j_seq = int(pad_idx[0])

    out1 = model(b["input_ids"], b["patches"], b["image_slots"], b["pos_h"], b["pos_w"], b["attn_mask"])
    patches2 = b["patches"].clone()
    torch.manual_seed(1)
    patches2[0, j_seq - 1] += torch.randn(cfg.patch_dim)  # perturb a full-pad patch (non-uniform)
    out2 = model(b["input_ids"], patches2, b["image_slots"], b["pos_h"], b["pos_w"], b["attn_mask"])

    real_pos = ((tt == TEXT) | (tt == REAL_IMG))
    assert torch.allclose(out1[0, real_pos], out2[0, real_pos], atol=1e-5), "real tokens attended to pad"


# --------------------------------------------------------------------------- #
# 5. label masking
# --------------------------------------------------------------------------- #
def test_label_masking():
    cfg = small_cfg()
    text = "hello world foo"
    tok = get_tokenizer()
    s = build_sample(render(text), text, tok, cfg)
    ids, labels, tt = s["input_ids"], s["labels"], s["token_type"]
    P = int(((tt == REAL_IMG) | (tt == PAD_IMG)).sum())
    soc_index = 1 + P
    eos_index = len(ids) - 1
    n_text = len(tok.encode(text))

    # only ocr-text + EOS positions are targets
    targets = (labels != -100).nonzero().flatten().tolist()
    assert targets == list(range(soc_index, eos_index)), targets
    assert len(targets) == n_text + 1  # text tokens + EOS
    # each label equals the next input token
    for i in targets:
        assert labels[i] == ids[i + 1]
    # nothing over BOS / image / SOC
    assert (labels[: soc_index] == -100).all()


# --------------------------------------------------------------------------- #
# 6. patchify round-trip / padding / scaling
# --------------------------------------------------------------------------- #
def test_patchify_and_padding():
    cfg = small_cfg()  # patch_size=8, max_image_px=64
    img = render("x")  # overwrite with a known constant below
    from PIL import Image
    img = Image.new("L", (16, 8), color=100)  # w=16, h=8, constant gray
    patches, rr, cc, G = preprocess_image(img, cfg)
    assert (rr, cc, G) == (1, 2, 2)
    assert patches.shape == (4, 8 * 8)
    content = 100 / 127.5 - 1.0
    assert torch.allclose(patches[0], torch.full((64,), content), atol=1e-4)  # (0,0) real
    assert torch.allclose(patches[1], torch.full((64,), content), atol=1e-4)  # (0,1) real
    assert torch.allclose(patches[2], torch.ones(64))  # (1,0) full white pad
    assert torch.allclose(patches[3], torch.ones(64))  # (1,1) full white pad


# --------------------------------------------------------------------------- #
# 6b. photometric augmentation (train split only)
# --------------------------------------------------------------------------- #
def _nonuniform_img():
    """A deterministic non-uniform grayscale image so jitter is observable."""
    from PIL import Image
    g = torch.arange(40 * 30, dtype=torch.uint8).reshape(40, 30) % 251  # varied content
    return Image.fromarray(g.numpy())


def test_augment_off_is_deterministic():
    # the default (no-augment) pipeline must be byte-identical across calls
    cfg = small_cfg()
    img = _nonuniform_img()
    a, *ga = preprocess_image(img, cfg, augment=False)
    b, *gb = preprocess_image(img, cfg, augment=False)
    assert torch.equal(a, b), "non-augmented preprocessing is not deterministic"
    assert ga == gb


def test_augment_preserves_geometry_but_changes_pixels():
    # augmentation may only touch pixel values, never the grid / patch shape
    cfg = small_cfg()
    img = _nonuniform_img()
    p0, r0, c0, G0 = preprocess_image(img, cfg, augment=False)
    torch.manual_seed(0)
    p1, r1, c1, G1 = preprocess_image(img, cfg, augment=True)
    assert (r1, c1, G1) == (r0, c0, G0), "augmentation changed the patch geometry"
    assert p1.shape == p0.shape
    assert not torch.allclose(p0, p1), "augment=True did not change any pixels"


def test_augment_keeps_pixels_in_range():
    # clamp to [0,255] before normalization -> patches must stay within [-1, 1]
    cfg = small_cfg()
    torch.manual_seed(0)
    p, *_ = preprocess_image(_nonuniform_img(), cfg, augment=True)
    assert p.min() >= -1.0 - 1e-6 and p.max() <= 1.0 + 1e-6


def test_augment_leaves_pad_white():
    # full-pad patches must remain exactly white (+1); jitter touches content only
    from PIL import Image
    cfg = small_cfg()  # patch_size=8, max_image_px=64
    img = Image.new("L", (16, 8), color=100)  # rr=1, cc=2, G=2 -> patches 2,3 are pad
    torch.manual_seed(0)
    patches, rr, cc, G = preprocess_image(img, cfg, augment=True)
    assert (rr, cc, G) == (1, 2, 2)
    assert torch.allclose(patches[2], torch.ones(64)), "pad patch (1,0) was altered"
    assert torch.allclose(patches[3], torch.ones(64)), "pad patch (1,1) was altered"


def test_augment_draws_differ():
    # successive augmented draws of the same image should differ (per-call RNG)
    cfg = small_cfg()
    img = _nonuniform_img()
    torch.manual_seed(0)
    p1, *_ = preprocess_image(img, cfg, augment=True)
    p2, *_ = preprocess_image(img, cfg, augment=True)
    assert not torch.allclose(p1, p2), "two augmented draws were identical"


def test_augment_zero_magnitudes_is_noop():
    # with all magnitudes zeroed, augment=True must equal augment=False
    cfg = small_cfg(aug_contrast=0.0, aug_brightness=0.0, aug_noise_std=0.0)
    img = _nonuniform_img()
    p_off, *_ = preprocess_image(img, cfg, augment=False)
    torch.manual_seed(0)
    p_on, *_ = preprocess_image(img, cfg, augment=True)
    assert torch.equal(p_off, p_on), "zeroed-magnitude augmentation was not a no-op"


def test_dataset_augment_flag_routing():
    # OCRDataset must augment only when augment=True (the train-split contract)
    cfg = small_cfg()
    rows = [{"image": _nonuniform_img(), "text": "hello"}]
    ds_off = OCRDataset(rows, cfg, "image", "text", augment=False)
    ds_on = OCRDataset(rows, cfg, "image", "text", augment=True)
    # no-augment dataset is stable across repeated access...
    assert torch.equal(ds_off[0]["patches"], ds_off[0]["patches"])
    # ...augmenting dataset varies across repeated access
    torch.manual_seed(0)
    assert not torch.allclose(ds_on[0]["patches"], ds_on[0]["patches"])


# --------------------------------------------------------------------------- #
# 7. 2D RoPE: norm preserved + relative-position property in the 1D (text) regime
# --------------------------------------------------------------------------- #
def test_rope_norm_preserved():
    cfg = small_cfg()
    torch.manual_seed(0)
    S = 6
    x = torch.randn(1, cfg.n_heads, S, cfg.head_dim)
    pos_h = torch.arange(S)[None]
    pos_w = (torch.arange(S) * 2)[None]  # genuine 2D (h != w)
    cos, sin = build_rope_cache(pos_h, pos_w, cfg.head_dim, cfg.rope_base)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_relative_property():
    cfg = small_cfg()
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, cfg.head_dim)
    k = torch.randn(1, 1, 1, cfg.head_dim)

    def roped_dot(m, n):
        # text regime: pos_h == pos_w
        cq = build_rope_cache(torch.tensor([[m]]), torch.tensor([[m]]), cfg.head_dim, cfg.rope_base)
        ck = build_rope_cache(torch.tensor([[n]]), torch.tensor([[n]]), cfg.head_dim, cfg.rope_base)
        rq = apply_rope(q, *cq)
        rk = apply_rope(k, *ck)
        return (rq * rk).sum()

    # same relative offset -> same dot product
    assert torch.allclose(roped_dot(2, 5), roped_dot(4, 7), atol=1e-5)
    assert torch.allclose(roped_dot(0, 3), roped_dot(10, 13), atol=1e-5)


# --------------------------------------------------------------------------- #
# 8. sequence assembly
# --------------------------------------------------------------------------- #
def test_seq_len_truncation():
    from PIL import Image
    cfg = small_cfg(max_seq_len=70)
    tok = get_tokenizer()
    s = build_sample(Image.new("L", (8, 8), 255), "word " * 300, tok, cfg)
    assert s["input_ids"].shape[0] <= cfg.max_seq_len      # capped
    assert s["input_ids"].shape[0] == cfg.max_seq_len      # long text fills it
    assert s["input_ids"][-1] == cfg.eos_id                # still ends with EOS
    for k in ("token_type", "pos_h", "pos_w", "labels"):   # all tensors stay aligned
        assert s[k].shape[0] == s["input_ids"].shape[0]


def test_sequence_assembly():
    cfg = small_cfg()
    text = "data model patch"
    tok = get_tokenizer()
    s = build_sample(render(text), text, tok, cfg)
    ids, tt, ph, pw = s["input_ids"], s["token_type"], s["pos_h"], s["pos_w"]
    P = int(((tt == REAL_IMG) | (tt == PAD_IMG)).sum())

    assert ids[0] == cfg.bos_id
    assert ids[1 + P] == cfg.soc_id
    assert ids[-1] == cfg.eos_id
    assert len(ids) == len(tt) == len(ph) == len(pw)
    # text region positions are 1D (h == w) and strictly increasing by 1
    text_start = 1 + P  # SOC
    th, tw = ph[text_start:], pw[text_start:]
    assert torch.equal(th, tw)
    assert torch.equal(th[1:] - th[:-1], torch.ones(len(th) - 1, dtype=th.dtype))


# --------------------------------------------------------------------------- #
# 9. optimizer split
# --------------------------------------------------------------------------- #
def test_optimizer_split():
    cfg = small_cfg()
    model = NanoMark(cfg)
    opt = build_optimizer(model, cfg)

    # weight-decay groups carry decay; the rest don't
    decay_ids = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    nodecay_ids = {id(p) for g in opt.param_groups if g["weight_decay"] == 0 for p in g["params"]}

    # disjoint and complete
    assert decay_ids.isdisjoint(nodecay_ids)
    all_ids = decay_ids | nodecay_ids
    assert all_ids == {id(p) for p in model.parameters() if p.requires_grad}

    # routing: 2D weights (matmuls, patch_proj, tok_emb) -> decay; 1D norms -> nodecay
    assert id(model.patch_proj.weight) in decay_ids
    assert id(model.blocks[0].attn.q_proj.weight) in decay_ids
    assert id(model.blocks[0].mlp.gate.weight) in decay_ids
    assert id(model.tok_emb.weight) in decay_ids
    assert id(model.final_norm.weight) in nodecay_ids
    # weight-decay groups only ever hold 2D params
    assert all(p.ndim >= 2 for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"])
    assert all(p.ndim == 1 for g in opt.param_groups if g["weight_decay"] == 0 for p in g["params"])


# --------------------------------------------------------------------------- #
# 10. forward / backward smoke
# --------------------------------------------------------------------------- #
def test_forward_backward_smoke():
    cfg = small_cfg()
    torch.manual_seed(0)
    model = NanoMark(cfg)
    _, b = make_batch(cfg)
    logits = model(b["input_ids"], b["patches"], b["image_slots"], b["pos_h"], b["pos_w"], b["attn_mask"])
    assert logits.shape == (1, b["input_ids"].shape[1], cfg.padded_vocab)
    assert torch.isfinite(logits).all()

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), b["labels"].reshape(-1), ignore_index=-100)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------- #
# 11. learned image positional embeddings (Gemma-4-Unified-style)
# --------------------------------------------------------------------------- #
def test_learned_positions_are_sequential():
    # in "learned" mode image tokens take ordinary 1D RoPE positions like text:
    # pos_h == pos_w == sequential index over the WHOLE sequence.
    cfg = small_cfg(image_pos_mode="learned")
    text = "data model patch"
    s = build_sample(render(text), text, get_tokenizer(), cfg)
    ph, pw, S = s["pos_h"], s["pos_w"], s["input_ids"].shape[0]
    assert torch.equal(ph, pw)
    assert torch.equal(ph, torch.arange(S, dtype=ph.dtype))


def test_patch_pos_grid_coords():
    # patch_pos is the raw 0-based (row, col) per patch in raster order
    from PIL import Image
    cfg = small_cfg()  # patch_size=8, max_image_px=64; emitted regardless of mode
    s = build_sample(Image.new("L", (16, 8), 255), "hi", get_tokenizer(), cfg)  # rr=1, cc=2, G=2
    assert s["patch_pos"].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_collate_patch_pos_padding():
    # short sample's patches are padded with the (-1, -1) sentinel up to batch P
    from PIL import Image
    cfg = small_cfg(image_pos_mode="learned")
    tok = get_tokenizer()
    s_small = build_sample(Image.new("L", (8, 8), 255), "hi", tok, cfg)   # G=1, P=1
    s_big = build_sample(Image.new("L", (16, 8), 255), "hi", tok, cfg)    # G=2, P=4
    batch = collate_fn([s_small, s_big], cfg)
    assert batch["patch_pos"].shape == (2, 4, 2)
    assert batch["patch_pos"][0, 0].tolist() == [0, 0]
    assert (batch["patch_pos"][0, 1:] == -1).all()  # padding sentinel
    assert (batch["patch_pos"][1] >= 0).all()        # big sample fully real


def test_learned_table_is_fresh_vision_adapter():
    # the learned table must be recognized as a from-scratch vision-adapter param
    # so the Qwen loader leaves it random and the optimizer gives it the full LR.
    cfg = small_cfg(image_pos_mode="learned")
    model = NanoMark(cfg)
    assert is_vision_adapter("patch_pos_emb")
    assert any(n == "patch_pos_emb" for n, _ in model.named_parameters())
    opt = build_optimizer(model, cfg)
    for g in opt.param_groups:
        if any(p is model.patch_pos_emb for p in g["params"]):
            assert g["lr_mult"] == 1.0                       # fresh -> full LR
            assert g["weight_decay"] == cfg.weight_decay     # ndim>=2 -> decayed
            break
    else:
        raise AssertionError("patch_pos_emb not found in any optimizer group")


def test_rope2d_mode_has_no_table():
    # the default mode creates no positional table -> rope2d state_dicts unchanged
    assert not any(n.startswith("patch_pos_emb") for n, _ in NanoMark(small_cfg()).named_parameters())
    assert hasattr(NanoMark(small_cfg(image_pos_mode="learned")), "patch_pos_emb")


def test_learned_forward_backward_smoke():
    cfg = small_cfg(image_pos_mode="learned")
    torch.manual_seed(0)
    model = NanoMark(cfg)
    _, b = make_batch(cfg)
    logits = model(b["input_ids"], b["patches"], b["image_slots"],
                   b["pos_h"], b["pos_w"], b["attn_mask"], b["patch_pos"])
    assert logits.shape == (1, b["input_ids"].shape[1], cfg.padded_vocab)
    assert torch.isfinite(logits).all()

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), b["labels"].reshape(-1), ignore_index=-100)
    loss.backward()
    # the learned table participates: gradient present, finite, and non-zero
    assert model.patch_pos_emb.grad is not None
    assert torch.isfinite(model.patch_pos_emb.grad).all()
    assert model.patch_pos_emb.grad.abs().sum() > 0
