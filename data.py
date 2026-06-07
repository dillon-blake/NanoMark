"""Data pipeline: image -> patches, text -> tokens, and the attention mask.

This module owns all the awkward sequence/mask/position logic so that the model
(``model.py``) stays a clean forward pass. Per sample we build the sequence

    [BOS] [G*G image patches] [SOC] [ocr text] [EOS]

where the image is resized (longest edge <= ``cfg.max_image_px``, never
upscaled), padded to a square multiple of the patch size with white, and cut
into a G x G grid of patches in raster order. Patches that fall entirely in the
white pad region are marked as pad-image and masked out of attention; the partial
("sub-patch") padding on the content's edge patches is baked in and kept.

``token_type`` codes drive both masking and loss:
    0 = TEXT     - text / special (BOS, SOC, ocr text, EOS)
    1 = REAL_IMG - real image patch
    2 = PAD_IMG  - pad image patch (full white pad, masked)
    3 = SEQ_PAD  - sequence padding (added by ``collate_fn`` to batch, masked)

Tensor shape conventions: B = batch, S = sequence length, P = image patches,
G = patch grid side, patch_dim = patch_size**2 * channels.
"""

import math

import numpy as np
import tiktoken
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import Config

TEXT, REAL_IMG, PAD_IMG, SEQ_PAD = 0, 1, 2, 3


def get_tokenizer():
    """Return the GPT-2 byte-level BPE tokenizer (tiktoken ``gpt2`` encoding).

    Returns:
        A ``tiktoken.Encoding`` with a 50,257-token vocabulary. NanoMark's
        special tokens (BOS/SOC/EOS) live above this range and are added by hand,
        never emitted by this tokenizer.
    """
    return tiktoken.get_encoding("gpt2")


def photometric_augment(content: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply light photometric jitter to a grayscale uint8 content array.

    Randomly perturbs contrast, brightness, and adds Gaussian noise so that
    repeated epochs do not see byte-identical inputs. Geometry is left untouched
    (same height/width), so the caller's patch-grid, position, and mask bookkeeping
    stays exactly valid. Randomness is drawn from ``torch``'s default RNG, which a
    ``DataLoader`` seeds per worker, so workers do not share the same jitter.

    Args:
        content: The resized grayscale content, uint8 array of shape [h, w].
        cfg: Configuration providing the ``aug_*`` magnitudes.

    Returns:
        The jittered content, uint8 array of the same shape, clamped to [0, 255].
    """
    x = torch.from_numpy(np.array(content, dtype=np.float32))  # writable copy
    if cfg.aug_contrast > 0:                                  # scale around mid-gray
        c = 1.0 + (torch.rand(()).item() * 2 - 1) * cfg.aug_contrast
        x = (x - 127.5) * c + 127.5
    if cfg.aug_brightness > 0:                                # shift, fraction of full scale
        b = (torch.rand(()).item() * 2 - 1) * cfg.aug_brightness * 255.0
        x = x + b
    if cfg.aug_noise_std > 0:                                 # additive Gaussian noise
        x = x + torch.randn(x.shape) * cfg.aug_noise_std
    return x.clamp_(0, 255).to(torch.uint8).numpy()


def preprocess_image(img: Image.Image, cfg: Config, augment: bool = False):
    """Convert a PIL image into a grid of flattened patches.

    Steps: convert to grayscale, downscale so the longest edge is at most
    ``cfg.max_image_px`` (never upscaling), optionally apply photometric
    augmentation to the content, pad to a square whose side is a multiple of
    ``cfg.patch_size`` (white fill, content anchored top-left), and cut into a
    G x G grid of patches in raster (row-major) order. Pixels are scaled to
    [-1, 1] with white = +1.

    Args:
        img: The source image (any mode; converted to grayscale).
        cfg: Configuration providing ``patch_size`` and ``max_image_px``.
        augment: If True, apply :func:`photometric_augment` to the content (only
            the real pixels; the white pad is left exactly white so pad-patch
            masking semantics are preserved). Use only for the train split.

    Returns:
        A tuple ``(patches, rows_real, cols_real, G)`` where:
            patches: float tensor [G*G, patch_dim], raster order.
            rows_real: number of patch rows containing real content.
            cols_real: number of patch cols containing real content.
            G: square grid side (in patches); a patch ``(r, c)`` is real iff
                ``r < rows_real and c < cols_real``.
    """
    ps = cfg.patch_size
    img = img.convert("L")
    w, h = img.size  # PIL gives (width, height)
    longest = max(w, h)
    if longest > cfg.max_image_px:  # downscale only, never upscale
        scale = cfg.max_image_px / longest
        w, h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((w, h), Image.BILINEAR)

    rows_real = math.ceil(h / ps)
    cols_real = math.ceil(w / ps)
    G = max(rows_real, cols_real)          # square grid side (in patches)
    side = G * ps                          # square canvas side (in pixels)

    content = np.asarray(img)
    if augment:                            # jitter only the real content, not the pad
        content = photometric_augment(content, cfg)
    canvas = np.full((side, side), 255, dtype=np.uint8)  # white background
    canvas[:h, :w] = content                             # paste content top-left

    t = torch.from_numpy(canvas).float() / 127.5 - 1.0   # [-1, 1], white -> +1
    # [side, side] -> [G, ps, G, ps] -> [G, G, ps, ps] -> [G*G, ps*ps], raster order
    patches = t.reshape(G, ps, G, ps).permute(0, 2, 1, 3).reshape(G * G, ps * ps)
    return patches, rows_real, cols_real, G


def build_sample(img: Image.Image, text: str, tokenizer, cfg: Config, augment: bool = False):
    """Build one training sample from an image and its transcription.

    Assembles the ``[BOS] [patches] [SOC] [text] [EOS]`` sequence along with the
    per-token type codes, 2D-RoPE positions, and next-token labels. Labels are
    -100 everywhere except the ocr-text tokens and EOS, so loss is computed only
    on the transcription. The text is truncated so the full sequence fits within
    ``cfg.max_seq_len``. The 2D-RoPE positions place BOS at (0, 0), patch
    ``(r, c)`` at ``(r+1, c+1)``, and the text region on a 1D diagonal
    (``pos_h == pos_w``) resuming just past the image's extent.

    Args:
        img: The source document image.
        text: The ground-truth transcription.
        tokenizer: A tiktoken encoding (see :func:`get_tokenizer`).
        cfg: Configuration (special-token ids, patch settings).
        augment: If True, photometrically augment the image (train split only);
            passed through to :func:`preprocess_image`.

    Returns:
        A dict of 1D tensors plus the patch tensor, all variable length:
        ``input_ids`` [S], ``token_type`` [S], ``pos_h`` [S], ``pos_w`` [S],
        ``labels`` [S], and ``patches`` [P, patch_dim] (P = G*G).
    """
    patches, rr, cc, G = preprocess_image(img, cfg, augment=augment)
    P = G * G
    text_ids = tokenizer.encode(text)
    # enforce the sequence cap: BOS + P patches + SOC + text + EOS <= max_seq_len
    text_ids = text_ids[: max(0, cfg.max_seq_len - P - 3)]

    # --- token ids ---
    # image slots hold a placeholder (0); the model overwrites them with the
    # projected patch embeddings, so the id value is irrelevant there.
    input_ids = [cfg.bos_id] + [0] * P + [cfg.soc_id] + text_ids + [cfg.eos_id]

    # --- token types ---
    img_types = [REAL_IMG if (r < rr and c < cc) else PAD_IMG
                 for r in range(G) for c in range(G)]
    token_type = [TEXT] + img_types + [TEXT] + [TEXT] * len(text_ids) + [TEXT]

    # --- 2D RoPE positions ---
    pos_h = [0]
    pos_w = [0]
    for r in range(G):
        for c in range(G):
            pos_h.append(r + 1)
            pos_w.append(c + 1)
    offset = 1 + max(rr, cc)               # text resumes after the image extent
    n_text_region = 1 + len(text_ids) + 1  # SOC + text + EOS
    for k in range(n_text_region):
        pos_h.append(offset + k)
        pos_w.append(offset + k)

    # --- labels (next-token; loss only on ocr text + EOS) ---
    S = len(input_ids)
    soc_index = 1 + P
    first_target = soc_index + 1           # first ocr text token (or EOS if empty)
    eos_index = S - 1
    labels = [-100] * S
    for i in range(S - 1):
        if first_target <= i + 1 <= eos_index:
            labels[i] = input_ids[i + 1]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "token_type": torch.tensor(token_type, dtype=torch.long),
        "pos_h": torch.tensor(pos_h, dtype=torch.long),
        "pos_w": torch.tensor(pos_w, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "patches": patches,  # [P, patch_dim]
    }


def build_attn_mask(token_type: torch.Tensor) -> torch.Tensor:
    """Build the boolean self-attention mask from per-token type codes.

    Encodes the prefix-LM masking scheme: the image is a fully-observed prefix
    (bidirectional), while text is generated causally.

    Rules (query i attending to key j):
        - Keys that are pad (PAD_IMG or SEQ_PAD) are never attended to.
        - Image queries (real or pad) attend bidirectionally to the real prefix
          (BOS + real image patches) only -- never to text, so no labels leak
          backward into the image.
        - Text queries attend causally to all real tokens (BOS + real image +
          earlier/equal text).
        - SEQ_PAD queries attend to themselves only, which keeps every query row
          non-empty so softmax stays finite.

    Args:
        token_type: Per-token type codes, shape [B, S] (values TEXT/REAL_IMG/
            PAD_IMG/SEQ_PAD).

    Returns:
        A boolean mask of shape [B, 1, S, S] where True = attend.
    """
    B, S = token_type.shape
    device = token_type.device
    idx = torch.arange(S, device=device)

    is_real = (token_type == TEXT) | (token_type == REAL_IMG)        # [B,S] valid keys
    is_bos = (idx == 0)[None, :]                                     # [1,S]
    prefix_key = (token_type == REAL_IMG) | is_bos                   # [B,S] BOS + real img

    causal = (idx[:, None] >= idx[None, :])[None]                    # [1,S,S] i>=j
    key_real = is_real[:, None, :]                                   # [B,1,S]
    key_prefix = prefix_key[:, None, :]                              # [B,1,S]

    text_allowed = key_real & causal                                 # [B,S,S]
    image_allowed = key_prefix.expand(B, S, S)                       # [B,S,S]
    self_only = (idx[:, None] == idx[None, :])[None].expand(B, S, S) # [B,S,S]

    q_img = ((token_type == REAL_IMG) | (token_type == PAD_IMG))[:, :, None]
    q_text = (token_type == TEXT)[:, :, None]
    q_pad = (token_type == SEQ_PAD)[:, :, None]

    allowed = torch.zeros(B, S, S, dtype=torch.bool, device=device)
    allowed = torch.where(q_text, text_allowed, allowed)
    allowed = torch.where(q_img, image_allowed, allowed)
    allowed = torch.where(q_pad, self_only, allowed)
    return allowed[:, None, :, :]


def collate_fn(batch, cfg: Config):
    """Collate variable-length samples into a padded batch with masks.

    Pads sequences to the batch's max length (``token_type`` padded with SEQ_PAD,
    labels with -100) and patches to the batch's max patch count, then derives the
    image-slot mask and attention mask.

    Args:
        batch: A list of sample dicts from :func:`build_sample`.
        cfg: Configuration (used for ``patch_dim``).

    Returns:
        A dict of batched tensors ready for :meth:`model.NanoMark.forward`:
        ``input_ids`` [B,S], ``patches`` [B,P,patch_dim], ``image_slots`` [B,S]
        (bool), ``pos_h``/``pos_w`` [B,S], ``attn_mask`` [B,1,S,S] (bool),
        ``labels`` [B,S], and ``token_type`` [B,S].
    """
    B = len(batch)
    S = max(s["input_ids"].shape[0] for s in batch)
    P = max(s["patches"].shape[0] for s in batch)

    input_ids = torch.zeros(B, S, dtype=torch.long)
    token_type = torch.full((B, S), SEQ_PAD, dtype=torch.long)
    pos_h = torch.zeros(B, S, dtype=torch.long)
    pos_w = torch.zeros(B, S, dtype=torch.long)
    labels = torch.full((B, S), -100, dtype=torch.long)
    patches = torch.zeros(B, P, cfg.patch_dim, dtype=torch.float)

    for b, s in enumerate(batch):
        n = s["input_ids"].shape[0]
        input_ids[b, :n] = s["input_ids"]
        token_type[b, :n] = s["token_type"]
        pos_h[b, :n] = s["pos_h"]
        pos_w[b, :n] = s["pos_w"]
        labels[b, :n] = s["labels"]
        p = s["patches"].shape[0]
        patches[b, :p] = s["patches"]

    image_slots = (token_type == REAL_IMG) | (token_type == PAD_IMG)
    attn_mask = build_attn_mask(token_type)
    return {
        "input_ids": input_ids,
        "patches": patches,
        "image_slots": image_slots,
        "pos_h": pos_h,
        "pos_w": pos_w,
        "attn_mask": attn_mask,
        "labels": labels,
        "token_type": token_type,
    }


class OCRDataset(Dataset):
    """A ``torch`` dataset wrapping any source with an image and a text column.

    Each item is preprocessed on access via :func:`build_sample`. Use with a
    ``DataLoader`` and :func:`collate_fn` (bound to a ``cfg`` via ``partial``).
    """

    def __init__(self, dataset, cfg: Config, image_col="image", text_col="text", augment=False):
        """Wrap an indexable dataset.

        Args:
            dataset: Any indexable returning dict-like rows (e.g. a HuggingFace
                ``datasets.Dataset`` or a list of dicts).
            cfg: Configuration passed through to :func:`build_sample`.
            image_col: Name of the image column in each row.
            text_col: Name of the text column in each row.
            augment: If True, photometrically augment each image (train split
                only); passed through to :func:`build_sample`.
        """
        self.dataset = dataset
        self.cfg = cfg
        self.image_col = image_col
        self.text_col = text_col
        self.augment = augment
        self.tokenizer = get_tokenizer()

    def __len__(self):
        """Return the number of samples."""
        return len(self.dataset)

    def __getitem__(self, i):
        """Preprocess and return sample ``i`` (see :func:`build_sample`)."""
        row = self.dataset[i]
        img = row[self.image_col]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        return build_sample(img, row[self.text_col], self.tokenizer, self.cfg, augment=self.augment)
