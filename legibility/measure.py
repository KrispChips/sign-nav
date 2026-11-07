"""Part A: how large is the sign's text right now, from one sighting.

  m = measure(crop, info, engine)        # crop, info = rectify(image, box, K)

Heights are text-box heights from the OCR DETECTOR (no recognizer), in SOURCE pixels.
Rectification enlarges small signs to at least 16 px, which adds no detail, so the crop is
first resampled to source scale using info["v_scale"] (source px per crop px at the crop
centre) and only then given to the detector. Scaling the heights afterwards is not enough:
the detector's padding has an additive part, in the pixels it was given. This way the same
sign measured from a crop and from that crop upscaled 2x gives the same number.

Line height: the detector only says WHERE each text line is. Its box edges are padded (by an
amount that depends on the font and has an additive part) and grow with blur, which made
small blurred text read 35-40% too tall in tests. So inside each box the line height is taken
as the full width at half maximum (FWHM) of the row-contrast profile: rows crossing text vary
a lot, background rows don't, and blur widens the profile's tails but hardly moves its
half-maximum points. On rendered signs this tracks the true height within 10% from 8 px up,
blurred or not. The raw box statistic is kept as h_box_p25 for comparison in calibration.
Always get the engine from signfilter.ocr_engine(): calibration and live use must match.

On SMALL signs (under 48 px) boxes taller than 85% of the sign are dropped: when text is too
small to resolve, the detector sometimes returns one box around the whole sign, which would
make a far sign look MORE legible than a near one. Larger signs keep them: SiGNgapore2D has
many boxes drawn tightly around one line of text, whose text really is ~sign-height.

Why detection only, and not the filter's boxes: with recognition on, RapidOCR drops lines it
can't read with confidence, and at a distance those are exactly the lines being measured."""
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Measurement:
    h_p25: Optional[float]      # 25th percentile of text-line heights, source px (main signal)
    h_med: Optional[float]      # median text-line height, source px
    n_lines: int                # number of text lines measured
    sign_px: float              # sign's shorter side in source px, angle-independent (fallback)
    angle: float                # viewing angle, degrees (0 = head-on), from frontality
    sharpness: float            # Laplacian variance of the crop (rectify.py)
    heights: list[float] = field(default_factory=list)   # every line height, source px
    h_box_p25: Optional[float] = None   # same statistic from raw detector box heights
    t: Optional[float] = None   # sighting time, set by the caller

    @property
    def has_text(self) -> bool:
        return self.h_p25 is not None

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()
                if k != "heights"}


def box_height(quad) -> float:
    """Shorter side of a text box quadrilateral (average of opposite sides)."""
    q = np.asarray(quad, float).reshape(4, 2)
    a = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    b = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    return float(min(a, b))


def detect_boxes(crop, engine) -> list:
    """Text boxes from the detector alone, in crop pixels."""
    r = engine(crop, use_det=True, use_cls=False, use_rec=False)
    return [] if r.boxes is None else list(r.boxes)


def to_source_scale(crop, v_scale: float):
    """Resample a rectified crop so one pixel is one source pixel (vertically)."""
    if abs(v_scale - 1.0) < 0.02:
        return crop
    h, w = crop.shape[:2]
    size = (max(1, round(w * v_scale)), max(1, round(h * v_scale)))
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA if v_scale < 1 else cv2.INTER_CUBIC)


def line_height(gray, quad) -> Optional[float]:
    """Full width at half maximum of the per-row contrast inside a text box (sub-pixel)."""
    q = np.asarray(quad, float).reshape(4, 2)
    x1, y1 = np.maximum(np.floor(q.min(0)).astype(int), 0)
    x2, y2 = np.ceil(q.max(0)).astype(int)
    roi = gray[y1:y2, x1:x2].astype(np.float32)
    if roi.shape[0] < 3 or roi.shape[1] < 3:
        return None
    prof = roi.std(axis=1)          # polarity-free: dark-on-light and light-on-dark alike
    lo, hi = float(np.percentile(prof, 5)), float(prof.max())
    if hi - lo < 4.0:               # no contrast: nothing to measure
        return None
    half = lo + 0.5 * (hi - lo)
    rows = np.flatnonzero(prof >= half)
    a, b = int(rows[0]), int(rows[-1])   # first to last: letters can dip mid-height

    def crossing(j0, j1):
        p0, p1 = prof[j0], prof[j1]
        return j0 + (half - p0) / (p1 - p0) if p1 != p0 else float(j0)
    top = crossing(a - 1, a) if a > 0 else a - 0.5
    bottom = crossing(b, b + 1) if b < len(prof) - 1 else b + 0.5
    return float(bottom - top)


SMALL_SIGN_PX = 48


def measure(crop, info: dict, engine=None, min_px: float = 2.0,
            max_frac_of_sign: float = 0.85) -> Measurement:
    """`info` is rectify()'s dict (needs v_scale, frontality, sharpness, height_px, aspect)."""
    src = to_source_scale(crop, float(info.get("v_scale", 1.0)))
    sign_h = float(info.get("height_px", src.shape[0]))
    boxes = [q for q in detect_boxes(src, engine)
             if sign_h >= SMALL_SIGN_PX or box_height(q) <= max_frac_of_sign * sign_h]
    box_hs = [box_height(q) for q in boxes]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src
    heights = sorted(h for h in (line_height(gray, q) for q in boxes) if h and h >= min_px)
    h_p25 = float(np.percentile(heights, 25)) if heights else None
    h_med = float(np.median(heights)) if heights else None
    sign_px = min(sign_h, sign_h * float(info.get("aspect", 1.0)))   # true shorter side
    front = float(np.clip(info.get("frontality", 1.0), 0.0, 1.0))
    return Measurement(h_p25=h_p25, h_med=h_med, n_lines=len(heights), sign_px=sign_px,
                       angle=math.degrees(math.acos(front)),
                       sharpness=float(info.get("sharpness", 0.0)), heights=heights,
                       h_box_p25=float(np.percentile(box_hs, 25)) if box_hs else None)
