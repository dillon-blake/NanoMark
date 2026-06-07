"""Tiny synthetic OCR data: render short text strings onto white images.

Used by the test suite so the pipeline can run end-to-end without downloading a
real dataset. Not meant for real training quality.
"""

import random

from PIL import Image, ImageDraw

WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
         "invoice", "total", "date", "amount", "name", "address", "page",
         "hello", "world", "data", "model", "patch", "token", "image"]


def render(text: str, size=(320, 96)) -> Image.Image:
    """Render ``text`` as black text on a white image.

    Args:
        text: The string to draw.
        size: (width, height) of the output image in pixels.

    Returns:
        A grayscale (mode "L") PIL image.
    """
    img = Image.new("L", size, color=255)  # white background
    draw = ImageDraw.Draw(img)
    draw.text((6, 6), text, fill=0)        # black text, default bitmap font
    return img


def make_synthetic_dataset(n: int = 16, seed: int = 0):
    """Generate a small deterministic synthetic OCR dataset.

    Args:
        n: Number of samples to generate.
        seed: RNG seed for reproducible word choices.

    Returns:
        A list of ``{'image': PIL.Image, 'text': str}`` dicts.
    """
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        words = [rng.choice(WORDS) for _ in range(rng.randint(2, 5))]
        text = " ".join(words)
        data.append({"image": render(text), "text": text})
    return data
