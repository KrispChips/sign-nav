"""Rendered stand-in signs for dry runs and tests (NOT for calibration: real calibration uses
SiGNgapore2D crops). Each sign comes with its annotation in the dataset's format."""
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORDS = ("ROOM 214", "EXIT", "LIFTS", "TOILETS", "PHARMACY", "RADIOLOGY", "ZONE A", "WARD 42",
         "LEVEL 2", "CAFE", "#04-01 TO #04-05", "MAIN ENTRANCE", "LIBRARY", "CLINIC B",
         "INFORMATION", "STAIRS", "ROOMS 200-250", "TOWER A", "CAR PARK", "LOBBY")
ARROWS = ("left", "right", "straight", "locational")
FONTS = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arial.ttf", "arialbd.ttf")


def _font(px, bold):
    for name in FONTS[1::2] if bold else FONTS[0::2]:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default(px)


def synthetic_signs(n=20, seed=0):
    """-> [{"id", "crop" (BGR), "text": [(text, arrow)], "symbol": []}]"""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        lines = rnd.sample(WORDS, rnd.randint(1, 4))
        cap = rnd.randint(28, 60)
        bold = rnd.random() < 0.5
        dark = rnd.random() < 0.3
        bg, fg = ((30, 60, 120), (245, 245, 245)) if dark else ((240, 240, 235), (25, 25, 25))
        f = _font(cap, bold)
        w = max(int(f.getlength(t)) for t in lines) + 3 * cap
        h = int(len(lines) * cap * 1.6 + cap)
        im = Image.new("RGB", (w, h), bg)
        d = ImageDraw.Draw(im)
        for j, t in enumerate(lines):
            d.text((int(1.5 * cap), int(cap * 0.5 + j * cap * 1.6)), t, font=f, fill=fg)
        crop = np.array(im)[:, :, ::-1].copy()
        crop[:4], crop[-4:], crop[:, :4], crop[:, -4:] = 60, 60, 60, 60
        out.append({"id": f"synth{i:03d}", "crop": crop,
                    "text": [(t, rnd.choice(ARROWS)) for t in lines], "symbol": []})
    return out
