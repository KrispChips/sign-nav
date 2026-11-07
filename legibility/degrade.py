"""Part B: simulate a sign seen farther away and at an angle, from a full-resolution crop.

  p = sample_params(angle=30, scale=0.25, seed=stable_seed("sign17", 30, 3))
  scene = degrade(crop, p)                       # camera image + sign box + camera matrix
  crop, info, m = sight(scene.image, scene.box, scene.K, engine)   # the LIVE path

Order: 1) viewing angle (the sign plane rotated about its vertical axis, seen through a
plausible camera), 2) distance (area downscale by `scale`; text height ~ 1/distance), 3) optics
and motion (Gaussian blur, half the time a short mostly-horizontal motion blur), 4) sensor and
stream (Gaussian noise, JPEG at quality 60-85).

The sign is put on a plain background with a margin before step 2, so the result is a small
camera image that goes through rectify() exactly like a live frame, and the camera matrix K is
kept consistent with every transform so rectify's angle estimate is geometrically right."""
import zlib
from dataclasses import asdict, dataclass

import cv2
import numpy as np

ANGLES = (0, 30, 50, 70)
FOCAL_PER_SIGN = 1.8      # camera focal length in units of the sign's larger side (~ a 40% wide sign
                          # in a 70 degree field of view): sets how strong the perspective is
MARGIN = 0.35             # background around the sign, in units of its larger side
BACKGROUND = 118


def log_scales(n=10, smallest=0.08):
    """n log-spaced scales from 1.0 down to `smallest` (halving the scale = doubling distance)."""
    return [float(s) for s in np.geomspace(1.0, smallest, n)]


def stable_seed(*parts) -> int:
    """Same parts -> same seed, across runs and machines (unlike hash())."""
    return zlib.crc32("|".join(map(str, parts)).encode())


@dataclass
class DegradeParams:
    angle: float          # plane rotation about the vertical axis, degrees
    scale: float          # 1 = full resolution
    blur_sigma: float = 0.0
    motion_len: int = 0   # 0 = no motion blur
    motion_angle: float = 0.0
    noise_sigma: float = 0.0
    jpeg_quality: int = 0  # 0 = no JPEG
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def sample_params(angle, scale, seed, motion=None) -> DegradeParams:
    """Random optics/sensor settings for one sample, reproducible from `seed`.
    motion=None: motion blur in half the samples; True/False forces it."""
    rng = np.random.default_rng(seed)
    with_motion = bool(rng.random() < 0.5) if motion is None else motion
    return DegradeParams(
        angle=float(angle), scale=float(scale),
        blur_sigma=float(rng.uniform(0.4, 1.0)),
        motion_len=int(rng.integers(3, 8)) if with_motion else 0,
        motion_angle=float(rng.normal(0.0, 12.0)),     # mostly horizontal head movement
        noise_sigma=float(rng.uniform(1.0, 3.0)),
        jpeg_quality=int(rng.integers(60, 86)),
        seed=int(seed))


@dataclass
class Scene:
    image: np.ndarray     # BGR camera image
    box: tuple            # sign bounding box (x1, y1, x2, y2) in `image`
    K: np.ndarray         # camera matrix of `image`
    corners: np.ndarray   # the sign's 4 corners (tl, tr, br, bl) in `image`
    params: DegradeParams


def plane_homography(w, h, angle_deg, f):
    """Crop pixels -> image of the crop's plane rotated by `angle_deg` about its vertical axis,
    with its centre on the optical axis at depth f (so angle 0 maps 1:1). The centre column
    stays at depth f, so the sign's height there is unchanged by the rotation."""
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]])
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], [0, 0, f]]) @ T


def view(crop, angle):
    """Step 1 plus the background canvas, at full resolution. Returns (canvas, corners, K)."""
    h, w = crop.shape[:2]
    f = FOCAL_PER_SIGN * max(w, h)
    H = plane_homography(w, h, angle, f)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    pts = cv2.perspectiveTransform(src[None], H)[0]
    m = MARGIN * max(w, h)
    off = -pts.min(0) + m
    size = np.ceil(pts.max(0) - pts.min(0) + 2 * m).astype(int)
    shift = np.array([[1, 0, off[0]], [0, 1, off[1]], [0, 0, 1.0]])
    canvas = cv2.warpPerspective(crop, shift @ H, (int(size[0]), int(size[1])), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(BACKGROUND,) * 3)
    K = np.array([[f, 0, off[0]], [0, f, off[1]], [0, 0, 1.0]])   # sign centre = principal point
    return canvas, pts + off, K


def motion_kernel(length, angle_deg):
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    d = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))]) * c
    cv2.line(k, (int(round(c - d[0])), int(round(c - d[1]))), (int(round(c + d[0])), int(round(c + d[1]))),
             1.0, 1, cv2.LINE_AA)
    return k / k.sum()


def degrade(crop, p: DegradeParams) -> Scene:
    canvas, corners, K = view(crop, p.angle)
    # 2) distance
    if p.scale < 1.0:
        size = (max(8, round(canvas.shape[1] * p.scale)), max(8, round(canvas.shape[0] * p.scale)))
        sx, sy = size[0] / canvas.shape[1], size[1] / canvas.shape[0]
        canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
        corners = corners * [sx, sy]
        K = np.diag([sx, sy, 1.0]) @ K
    img = canvas.astype(np.float32)
    # 3) optics and motion
    if p.blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), p.blur_sigma)
    if p.motion_len >= 2:
        img = cv2.filter2D(img, -1, motion_kernel(p.motion_len, p.motion_angle))
    # 4) sensor and stream
    if p.noise_sigma > 0:
        img += np.random.default_rng(p.seed + 1).normal(0, p.noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if p.jpeg_quality:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    x1, y1 = corners.min(0)
    x2, y2 = corners.max(0)
    return Scene(img, (float(x1), float(y1), float(x2), float(y2)), K, corners, p)


def sight(image, box, K, engine):
    """The live path for one sighting: rectify -> measure. Returns (crop, info, Measurement).
    The crop is what would be sent to Gemma."""
    from rectify import rectify
    from legibility.measure import measure
    crop, info = rectify(image, box, K)
    return crop, info, measure(crop, info, engine)
