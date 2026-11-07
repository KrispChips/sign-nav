"""Rendered test signs with known text sizes (shared by legibility tests)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf",
                   "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf")
LINES = (("ROOM 214", "EXIT", "LIFTS 3-5"), ("PHARMACY", "B2", "WARD 42"),
         ("ZONE A", "RADIOLOGY", "12"))


def font(px):
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default(px)


def cap_height(f):
    b = f.getbbox("H")
    return b[3] - b[1]


def plate(lines, cap, bg=(245, 245, 245), fg=(20, 20, 20), border=None):
    """A sign plate with `lines` at capital height ~`cap` px. Returns (BGR image, real cap px)."""
    f10 = font(40)
    f = font(max(4, round(cap * 40 / cap_height(f10))))
    real = cap_height(f)
    w = max(int(f.getlength(t)) for t in lines) + 4 * cap
    h = int(len(lines) * cap * 2.2 + 2 * cap)
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        d.text((2 * cap, cap + i * cap * 2.2), t, font=f, fill=fg)
    arr = np.array(im)[:, :, ::-1].copy()
    if border:
        arr[:border], arr[-border:], arr[:, :border], arr[:, -border:] = 40, 40, 40, 40
    return arr, real
