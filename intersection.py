"""Hallways, entrances and open areas from monocular depth. No training.

Each frame's relative depth map is turned into a top-down scan at eye height, and the scan
is read like a floor plan:

1. Scale. Depth Anything gives inverse depth up to an unknown scale and shift. The shift is
   taken from the farthest part of the frame; the scale from the floor at the bottom of the
   image, which lies `eye_height` below the camera (its inverse depth grows linearly with
   the image row below the horizon). The horizon row comes out of the same fit.
2. Scan. The horizon band of each column (eye height, so through doorways, above most
   furniture) gives one point per column: forward distance Z and lateral offset X, in metres.
3. Walls. A straight line is fitted to each side's points (RANSAC, so pillars, people and
   gaps are outliers). Pillars and cabinets sit inside the wall line and are ignored.
   Recesses (closed doors, alcoves) are less than `beyond_m` behind it and count as wall.
4. Gaps. Columns that see well past the wall line are gaps in the wall. A gap's width
   along the wall comes from the rays at its two edges, so it is a length in metres:
     hallway   width >= hall_min (1.4 m)     two corridors meeting
     entrance  entrance_min .. hall_min       a doorway into a room
     open side width >= open_side_min (5 m)   e.g. a lounge or atrium along one side
5. Open area. No walls near either side and little nearby: announced once on entry
   ("Entering an open area.") and once on leaving. Gaps aren't announced inside it.

Geometry notes. The glasses see about +-35 deg to each side, so a side hallway is only seen
obliquely, from a few metres before it until about 1.5 m before it. The walker announced
"Hallway ahead on your left" therefore gets it before reaching it. Widths scale with the
assumed eye height and the camera's FOV (`--calib` or `--hfov`); a wrong FOV makes everything
wider or narrower in proportion.

Depth through glass looks like an opening: a glass office wall reads as an entrance or an
open side. That needs an image cue, not depth (see README)."""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from direction import where

DEPTH_REPO = "onnx-community/depth-anything-v2-small"
DEPTH_FILE = "onnx/model_int8.onnx"     # CPU: int8 is ~2-3x faster there
DEPTH_FILE_GPU = "onnx/model.onnx"      # GPU: int8 ops don't run on CUDA (each frame would
                                        # bounce between GPU and CPU: ~0.5 s instead of ~20 ms)
GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def _providers():
    """CUDA / DirectML if present, then CPU. Not TensorRT: it needs a separate install and
    otherwise prints a long error before falling back."""
    import onnxruntime as ort
    avail = ort.get_available_providers()
    return [p for p in GPU_PROVIDERS if p in avail] + ["CPUExecutionProvider"]


def resolve_model(path=None):
    """Local .onnx path, or download the default model from Hugging Face: the full-precision
    model when a GPU provider is available, the int8 one for the CPU."""
    if path:
        return path
    from huggingface_hub import hf_hub_download
    if len(_providers()) > 1:
        try:
            return hf_hub_download(repo_id=DEPTH_REPO, filename=DEPTH_FILE_GPU)
        except Exception as e:  # noqa: BLE001
            print(f"[depth] couldn't get {DEPTH_FILE_GPU} ({e}); using {DEPTH_FILE}")
    return hf_hub_download(repo_id=DEPTH_REPO, filename=DEPTH_FILE)


class DepthEstimator:
    """Runs the ONNX depth model on a BGR frame and returns relative inverse depth (larger = closer)."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path, long_side=518):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model_path, providers=_providers())
        used = self.sess.get_providers()
        if used[0] in GPU_PROVIDERS:
            print(f"[depth] on the GPU ({used[0]}), {model_path.replace(chr(92), '/').split('/')[-1]}")
            if "int8" in model_path or "quant" in model_path:
                print("[depth] note: an int8 model on the GPU runs partly on the CPU (slow); "
                      "use the full-precision onnx/model.onnx")
        else:
            print(f"[depth] on the CPU ({', '.join(used)}): for the GPU, "
                  "pip uninstall onnxruntime && pip install onnxruntime-gpu")
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.half = inp.type == "tensor(float16)"
        hw = inp.shape[2:4]
        # Use the model's fixed input size if it has one, otherwise pick our own.
        self.fixed_hw = tuple(hw) if all(isinstance(s, int) for s in hw) else None
        self.long_side = long_side

    def _target_size(self, h, w):
        if self.fixed_hw:
            return self.fixed_hw
        scale = self.long_side / max(h, w)
        # Depth Anything's ViT backbone uses 14-pixel patches, so sides must be multiples of 14.
        to14 = lambda x: max(14, int(round(x * scale / 14)) * 14)
        return to14(h), to14(w)

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        th, tw = self._target_size(h, w)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (x - self.MEAN) / self.STD
        x = x.transpose(2, 0, 1)[None]
        if self.half:
            x = x.astype(np.float16)
        out = self.sess.run(None, {self.input_name: x})[0]
        return np.squeeze(out).astype(np.float32)


# ---------------------------------------------------------------- 1-3: depth map -> scan + walls

@dataclass
class Wall:
    side: str
    m: float            # X = m * Z + c  (m: the corridor's angle to the camera axis)
    c: float            # lateral offset of the wall (m, signed)
    inliers: np.ndarray
    n: int


@dataclass
class Scan:
    ok: bool
    reason: str = ""
    u: Optional[np.ndarray] = None      # tan(bearing) per column, + = right
    X: Optional[np.ndarray] = None      # lateral offset (m, + = right)
    Z: Optional[np.ndarray] = None      # forward distance (m), max_range where farther
    far: Optional[np.ndarray] = None    # True where beyond max_range
    walls: dict = field(default_factory=lambda: {"left": None, "right": None})
    horizon: float = float("nan")       # horizon row as a fraction of the image height
    center_Z: float = float("nan")      # median distance straight ahead
    shift: float = 0.0                  # the depth model's shift, in units of the floor slope
    shift_source: str = ""
    shift_estimate: Optional[float] = None   # this frame's own estimate from the walls
    vp: Optional[tuple] = None          # vanishing point (u, row fraction), if found


def _tol(Z):
    return 0.2 + 0.03 * Z


def _floor_fit(e, rows, c0, c1):
    """e = kf * (row - rh) on the floor rows of the central columns. -> (kf, rh) or None."""
    vals = np.median(e[rows, c0:c1], axis=1)
    keep = np.ones(len(rows), bool)
    kf = q = None
    for _ in range(3):
        if keep.sum() < 4:
            return None
        A = np.stack([rows[keep], np.ones(keep.sum())], 1)
        (kf, q), *_ = np.linalg.lstsq(A, vals[keep], rcond=None)
        if kf <= 0:
            return None
        pred = kf * rows + q
        keep = np.abs(vals - pred) <= 0.2 * np.maximum(np.abs(pred), 1e-6)
    return (float(kf), float(-q / kf)) if keep.sum() >= 4 else None


def _line_fit(u, e, side, a, delta, prior_c=None, min_inliers=5, min_span=0.12,
              max_offset=4.5):
    """Robust line e = s*u + q through one side's columns in inverse-depth space. A wall
    parallel to the walking direction is a straight line there, e + delta = (a / c)(u - m),
    so a candidate line's slope gives the wall's lateral offset c = a / s directly.
    Candidates near `prior_c` (this side's wall in recent frames) count double: walls don't
    jump between frames, while a cabinet face or a gap can win a single frame.
    -> (s, q, inlier mask) or None."""
    sgn = -1 if side == "left" else 1
    # the most distant columns (the end of the corridor) carry no evidence about side walls
    idx = np.where((np.sign(u) == sgn) & (e > 0.08 * e.max()))[0]
    if len(idx) < min_inliers:
        return None
    uu, ee = u[idx], e[idx]
    du = float(np.median(np.diff(np.sort(u)))) if len(u) > 1 else 0.03
    cover = du / np.maximum(np.abs(uu), 0.2) ** 2     # ~ metres of wall per column / its offset
    tol = 0.05 * np.abs(ee) + 0.01 * np.abs(ee).max()
    best = None
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if abs(uu[i] - uu[j]) < min_span / 2:
                continue
            s = (ee[j] - ee[i]) / (uu[j] - uu[i])
            if sgn * s <= 0 or not (0.25 <= a / s * sgn <= max_offset):
                continue
            q = ee[i] - s * uu[i]
            inl = np.abs(ee - (s * uu + q)) <= tol
            # length of wall covered, not column count: near things (a cabinet, a person)
            # fill many columns but only a metre or two of the corridor
            score = float(cover[inl].sum())
            if prior_c is not None and abs(a / s - prior_c) <= 0.25 * abs(prior_c) + 0.15:
                score *= 2
            key = (score, int(inl.sum()))
            if best is None or key > best[0]:
                best = (key, inl)
    if best is None:
        return None
    inl = best[1]
    if inl.sum() < min_inliers or np.ptp(uu[inl]) < min_span:
        return None
    s, q = np.polyfit(uu[inl], ee[inl], 1)
    if sgn * s <= 0:
        return None
    full = np.zeros(len(u), bool)
    full[idx[inl]] = True
    return float(s), float(q), full


def vanishing_point(g, max_u, u):
    """The corridor's vanishing point: centre of the farthest region of the (smoothed) map.
    -> (u, row fraction) or None when there's no single compact far region (open areas,
    facing a wall)."""
    sm = cv2.GaussianBlur(g, (5, 5), 0)
    lo, hi = float(sm.min()), float(np.percentile(sm, 99))
    mask = (sm <= lo + 0.06 * (hi - lo)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    r0, c0 = np.unravel_index(np.argmin(sm), sm.shape)
    k = lab[r0, c0]
    w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
    if w > 0.35 * g.shape[1] or h > 0.45 * g.shape[0]:
        return None
    cx, cy = cent[k]
    ui = float(np.interp(cx, np.arange(len(u)), u))
    if abs(ui) > max_u or not (0.2 <= cy / g.shape[0] <= 0.8):
        return None
    return ui, float(cy / g.shape[0])


def _wall_through(u, e, side, uv, delta, a, prior_c=None, min_inliers=5, max_offset=4.5):
    """A wall parallel to the corridor passes through the vanishing point (uv, -delta) in
    (u, inverse depth): e + delta = s (u - uv). One parameter, so cabinets, door frames and
    gaps can't tilt it. -> (s, inlier mask) or None."""
    sgn = -1 if side == "left" else 1
    idx = np.where((np.sign(u - uv) == sgn) & (np.abs(u - uv) > 0.06) & (e > 0.08 * e.max()))[0]
    if len(idx) < min_inliers:
        return None
    uu, ee = u[idx] - uv, e[idx] + delta
    du = float(np.median(np.diff(np.sort(u))))
    cover = du / np.maximum(np.abs(uu), 0.2) ** 2
    tol = 0.05 * np.abs(ee) + 0.01 * np.abs(ee).max()
    best = None
    for s in ee / uu:
        if sgn * s <= 0 or not (0.25 <= a / s * sgn <= max_offset):
            continue
        inl = np.abs(ee - s * uu) <= tol
        score = float(cover[inl].sum())
        if prior_c is not None and abs(a / s - prior_c) <= 0.25 * abs(prior_c) + 0.15:
            score *= 2
        if best is None or (score, int(inl.sum())) > best[0]:
            best = ((score, int(inl.sum())), inl)
    if best is None or best[1].sum() < min_inliers:
        return None
    inl = best[1]
    s = float(np.sum(ee[inl] * uu[inl]) / np.sum(uu[inl] ** 2))   # LS through the point
    full = np.zeros(len(u), bool)
    full[idx[inl]] = True
    return s, full, float(cover[inl].sum())


def scan_from_depth(inv, K, image_size, eye_height=1.6, n_cols=64, max_range=25.0,
                    floor_rows=(0.72, 1.0), floor_cols=0.3, prior=None,
                    max_angle=30.0, max_offset=4.5, min_shift_span=0.3):
    """Relative inverse depth (any resolution) + the image's K and (width, height) -> Scan.

    The model's shift is unknown. Walls parallel to the corridor are straight lines in
    (u, inverse depth) through the vanishing point, where the true inverse depth is 0. So
    the vanishing point (the centre of the farthest region) and one cleanly seen wall give
    the shift; walls are then fitted through the vanishing point, which cabinets and door
    frames can't tilt. `prior` = {"shift": ..., "c": {"left": .., "right": ..}} from recent
    frames (see IntersectionWatcher) is used when this frame gives no estimate, and its
    shift steadies a frame's own estimate. Without a vanishing point (open areas), the two
    walls' crossing stands in for it; failing that, the farthest point counts as infinitely
    far."""
    prior = prior or {}
    W, H = image_size
    nr = max(8, int(round(n_cols * inv.shape[0] / inv.shape[1])))
    g = cv2.resize(inv.astype(np.float32), (n_cols, nr), interpolation=cv2.INTER_AREA)
    fx, cx = K[0, 0] * n_cols / W, (K[0, 2] + 0.5) * n_cols / W - 0.5
    fy = K[1, 1] * nr / H
    u = (np.arange(n_cols) - cx) / fx
    p1 = float(np.percentile(g, 1))
    e = g - p1
    c0 = max(0, int(round(cx - floor_cols / 2 * n_cols)))
    c1 = min(n_cols, int(round(cx + floor_cols / 2 * n_cols)) + 1)
    rows = np.arange(int(floor_rows[0] * nr), int(floor_rows[1] * nr))
    fit = _floor_fit(e, rows, c0, c1)
    if fit is None:
        return Scan(False, "no floor at the bottom of the image")
    mmax = math.tan(math.radians(max_angle))
    vp = vanishing_point(g, mmax, u)
    delta, source, estimate = 0.0, "farthest point", None
    if prior.get("shift") is not None:
        delta, source = p1 - prior["shift"] * fit[0], "recent frames"
    lines = {}
    for it in range(2):      # horizon -> walls -> shift -> horizon
        fit = _floor_fit(e + delta, rows, c0, c1)
        if fit is None:
            return Scan(False, "no floor at the bottom of the image")
        kf, rh = fit
        if not (-0.1 * nr <= rh <= 0.7 * nr):
            if vp is None:
                return Scan(False, f"implausible horizon row {rh / nr:.2f}")
            rh = vp[1] * nr
            kf = float(np.median((e[rows, c0:c1].mean(axis=1) + delta) / np.maximum(rows - rh, 1)))
            if kf <= 0:
                return Scan(False, "no floor at the bottom of the image")
        a = kf * eye_height * fy           # e + delta = a / Z on any surface
        r0, r1 = int(math.floor(rh - 0.035 * nr)), int(math.ceil(rh + 0.01 * nr)) + 1
        band = e[max(0, r0):min(nr, r1)]
        if band.shape[0] == 0:
            return Scan(False, "horizon outside the image")
        ec = np.median(band, axis=0)
        lines = {sd: _line_fit(u, ec, sd, a, delta, prior.get("c", {}).get(sd),
                               max_offset=max_offset) for sd in ("left", "right")}
        long_ = {sd: ln for sd, ln in lines.items() if ln and np.ptp(u[ln[2]]) >= min_shift_span}
        # candidate vanishing points (u, shift): each long wall extended to the farthest
        # region's centre, and the two walls' crossing. Keep the one through which both
        # walls fit best: a cluttered wall (cabinets, door frames) gives a bad line, and
        # off-centre the near wall is so steep that a small error in u moves the shift a lot
        cands = []
        if vp is not None:
            cands += [(vp[0], -(ln[0] * vp[0] + ln[1])) for ln in long_.values()]
        if len(long_) == 2:
            (sl, ql, _), (sr, qr, _) = lines["left"], lines["right"]
            um = (qr - ql) / (sl - sr)
            if abs(um) <= mmax:
                cands.append((float(um), -(sl * um + ql)))
        d = None
        if cands:
            def support(c):
                fits = [_wall_through(u, ec, sd, c[0], c[1], a, max_offset=max_offset)
                        for sd in ("left", "right")]
                return sum(f[2] for f in fits if f)
            uv, d = max(cands, key=support)
            if vp is not None or len(cands) == 1:
                vp = (float(uv), vp[1] if vp else rh / nr)
        if d is not None and abs(d) <= 0.5 * ec.max():
            estimate = float(d)
            if prior.get("shift") is None:
                delta, source = float(d), "walls"
        if source == "recent frames" or it == 1:
            break
    et = ec + delta
    far = et <= a / max_range
    Z = np.where(far, max_range, a / np.maximum(et, 1e-9))
    X = Z * u
    walls = {}
    if vp is not None:
        for sd in ("left", "right"):
            fitc = _wall_through(u, ec, sd, vp[0], delta, a, prior.get("c", {}).get(sd),
                                 max_offset=max_offset)
            walls[sd] = None
            if fitc:
                sgn = -1 if sd == "left" else 1
                inl = fitc[1] & ~far
                c = a / fitc[0]
                if inl.sum() >= 4 and 0.25 <= sgn * c <= max_offset:
                    walls[sd] = Wall(sd, float(vp[0]), float(c), inl, int(inl.sum()))
    else:
        for sd, ln in lines.items():
            walls[sd] = None
            if not ln:
                continue
            sgn = -1 if sd == "left" else 1
            inl = ln[2] & ~far
            if inl.sum() < 4:
                continue
            m, c = np.polyfit(Z[inl], X[inl], 1)
            if abs(m) > mmax or not (0.25 <= sgn * c <= max_offset):
                continue
            walls[sd] = Wall(sd, float(m), float(c), inl, int(inl.sum()))
        if walls["left"] and walls["right"]:       # parallel walls: share the angle
            wl, wr = walls["left"], walls["right"]
            m = (wl.m * wl.n + wr.m * wr.n) / (wl.n + wr.n)
            for w in (wl, wr):
                w.c = float(np.median(X[w.inliers] - m * Z[w.inliers]))
                w.m = float(m)
    mid = np.abs(u - (vp[0] if vp else 0.0)) < 0.1
    return Scan(True, u=u, X=X, Z=Z, far=far, walls=walls, horizon=float(rh / nr),
                center_Z=float(np.median(Z[mid])) if mid.any() else float("nan"),
                shift=float((p1 - delta) / kf), shift_source=source,
                shift_estimate=None if estimate is None else float((p1 - estimate) / kf),
                vp=vp)


# ---------------------------------------------------------------- 4: gaps

WALL, RECESS, BEYOND, INSIDE = "wall", "recess", "beyond", "inside"


@dataclass
class Gap:
    side: str
    z_near: float           # where the gap starts along the wall (forward m)
    z_far: float
    near_seen: bool         # False: the gap runs off the edge of the image
    far_seen: bool          # False: it runs into the distance (end not visible yet)
    depth: float            # how far past the wall line it sees (m)
    cols: list = field(default_factory=list)
    wall_c: float = 1.0     # lateral distance of this side's wall (m)

    @property
    def width(self):
        return self.z_far - self.z_near

    @property
    def measured(self):
        return self.near_seen and self.far_seen


def classify_columns(scan, wall, beyond_m=0.45):
    """Per column of this wall's side: wall / recess / beyond (a gap) / inside (an obstacle)."""
    sgn = -1 if wall.side == "left" else 1
    off = sgn * (scan.X - (wall.m * scan.Z + wall.c))
    tol = _tol(scan.Z)
    cls = np.where(off > beyond_m, BEYOND,
                   np.where(off > tol, RECESS, np.where(off < -tol, INSIDE, WALL))).astype(object)
    cls[scan.far] = BEYOND
    return cls, off


def find_gaps(scan, wall, beyond_m=0.45, gap_range=12.0, min_depth=1.0):
    sgn = -1 if wall.side == "left" else 1
    cls, off = classify_columns(scan, wall, beyond_m)
    # columns of this side, from the image edge (nearest wall) toward the vanishing point
    cols = [i for i in np.argsort(-sgn * scan.u) if np.sign(scan.u[i]) == sgn]

    def z_line(uu):          # where a ray of slope uu meets the wall line
        den = uu - wall.m
        return wall.c / den if sgn * den > 1e-6 else math.inf

    usable = [i for i in cols if z_line(scan.u[i]) <= gap_range]
    gaps, k = [], 0
    while k < len(usable):
        if cls[usable[k]] != BEYOND:
            k += 1
            continue
        start = k
        while k < len(usable) and (cls[usable[k]] == BEYOND or (
                cls[usable[k]] == INSIDE and k + 1 < len(usable) and cls[usable[k + 1]] == BEYOND)):
            k += 1
        run = usable[start:k]
        near_seen = start > 0
        if near_seen:
            z_near = z_line((scan.u[usable[start - 1]] + scan.u[run[0]]) / 2)
        else:
            z_near = z_line(scan.u[run[0]])
        deep = float(np.max(np.where(scan.far[run], 99.0, off[run]))) >= min_depth
        zr, xr = scan.Z[run], np.abs(scan.X[run])
        # a side hallway seen obliquely shows its far wall, which faces us (constant Z) and
        # may be only ~0.5 m past our wall line; the wall behind a cabinet runs parallel to
        # ours (constant X): only the first is an opening
        facing = (len(run) >= 2 and not scan.far[run].any()
                  and np.ptp(zr) / np.median(zr) < 0.5 * np.ptp(xr) / max(np.median(xr), 1e-6))
        last = run[-1]
        nxt = usable[k] if k < len(usable) else None
        far_seen = True
        if nxt is not None and cls[nxt] in (WALL, RECESS):
            if scan.Z[last] - scan.Z[nxt] > 0.1 * scan.Z[nxt] + 0.1:
                # a jump back to our wall: a door jamb or a wall end, at the edge ray
                z_far = z_line((scan.u[last] + scan.u[nxt]) / 2)
            else:
                # no jump: a crease where the far wall of a side hallway meets ours; that far
                # wall faces us, so its distance is the corner's
                z_far = float(scan.Z[last])
        elif facing and nxt is not None:
            z_far = float(scan.Z[last])     # the far wall carries on across (a T junction)
        else:
            far_seen = False
            z_far = z_line(scan.u[last]) if nxt is not None else gap_range
        if not (deep or facing):
            continue
        gaps.append(Gap(wall.side, float(z_near), float(max(z_far, z_near)), near_seen,
                        far_seen, float(np.median(off[run])), [int(i) for i in run],
                        abs(wall.c)))
    return gaps


# ---------------------------------------------------------------- 5: over time

HALLWAY, ENTRANCE, OPEN_SIDE = "hallway", "entrance", "open_side"
OPEN_AREA, LEAVE_OPEN, OPEN_AHEAD = "open_area", "leave_open_area", "open_ahead"


@dataclass
class GapTrack:
    id: int
    side: str
    t_first: float
    t_last: float
    z_near: float
    z_far: float
    widths: list = field(default_factory=list)
    span: float = 0.0           # longest stretch seen, for gaps never seen whole
    kind: Optional[str] = None
    t_kind: float = 0.0
    left_view: bool = False     # measured, and now its near edge is past the image edge
    depth: float = 0.0          # deepest median view past the wall line (m)
    announced: Optional[str] = None
    bearing: float = 0.0        # camera bearing of its middle at the last sighting (deg)

    def width(self):
        """Median of the 3 latest (= closest, finest) measurements. If they disagree by ~2x,
        the smallest: from a distance a room's back wall can line up with our wall and make
        a doorway look like a hallway corner, an error that only ever over-measures."""
        if not self.widths:
            return None
        w = self.widths[-3:]
        return float(min(w)) if self.unsettled() else float(np.median(w))

    def unsettled(self):
        w = self.widths[-3:]
        return len(w) >= 2 and max(w) > 1.8 * min(w)


class PlaceTracker:
    """Gap tracks per side, the open-area state, and events to announce."""

    def __init__(self, hall_min=1.4, entrance_min=0.6, open_side_min=5.0, min_measures=3,
                 open_enter_s=2.0, open_leave_s=4.0, open_free=2.5, open_near_frac=0.15,
                 forget_s=1.5, wall_ahead_max=10.0, measure_range=3.0,
                 open_side_hold=1.5, cross_hold=0.8, open_side_depth=2.0):
        self.hall_min, self.entrance_min, self.open_side_min = hall_min, entrance_min, open_side_min
        self.min_measures = min_measures
        self.open_enter_s, self.open_leave_s = open_enter_s, open_leave_s
        self.open_free, self.open_near_frac = open_free, open_near_frac
        self.forget_s, self.wall_ahead_max = forget_s, wall_ahead_max
        self.measure_range, self.open_side_hold = measure_range, open_side_hold
        self.cross_hold, self.open_side_depth = cross_hold, open_side_depth
        self.reset()

    def reset(self):
        self.tracks, self.next_id = [], 1
        self.in_open, self.open_hist = False, []      # (t, open-looking frame?)
        self.open_ahead_t = -math.inf
        self.quiet_until = {"left": -math.inf, "right": -math.inf}

    # ---- open area
    def _open_frame(self, scan, walls):
        near = float(np.mean(scan.Z < 2.5))
        free = [abs(w.c) if w else math.inf for w in walls.values()]
        return near <= self.open_near_frac and min(free) >= self.open_free

    def _update_open(self, t, looks_open):
        self.open_hist = [(tt, v) for tt, v in self.open_hist
                          if t - tt <= max(self.open_enter_s, self.open_leave_s)] + [(t, looks_open)]
        span = t - self.open_hist[0][0]
        if not self.in_open:
            recent = [v for tt, v in self.open_hist if t - tt <= self.open_enter_s]
            if span >= self.open_enter_s * 0.9 and np.mean(recent) >= 0.8:
                self.in_open = True
                return OPEN_AREA
        else:
            recent = [v for tt, v in self.open_hist if t - tt <= self.open_leave_s]
            if span >= self.open_leave_s * 0.9 and np.mean(recent) <= 0.2:
                self.in_open = False
                return LEAVE_OPEN
        return None

    # ---- gaps
    def _associate(self, g, t):
        best = None
        for tr in self.tracks:
            if tr.side != g.side or t - tr.t_last > self.forget_s:
                continue
            # the walker moves forward, so a gap comes closer: allow for that
            lo, hi = tr.z_near - 1.0 - 2.0 * (t - tr.t_last), tr.z_far + 0.5
            if g.z_near <= hi and g.z_far >= lo:
                d = abs(g.z_near - tr.z_near)
                if best is None or d < best[0]:
                    best = (d, tr)
        return best[1] if best else None

    def _long_open(self, tr):
        return tr.span >= 0.6 * self.open_side_min and tr.depth >= self.open_side_depth

    def _classify(self, tr):
        w = tr.width()
        if w is not None and w >= self.open_side_min and len(tr.widths) >= 2:
            return OPEN_SIDE        # measured end to end
        if self._long_open(tr) and any(self._long_open(o) for o in self.tracks
                                       if o.side != tr.side):
            # far end not seen, but both sides open up deep at once: an open area ahead. On
            # one side alone that's as often a wall behind clutter, or a door faced head-on
            return OPEN_SIDE
        n = len(tr.widths)
        agree = n >= 2 and max(tr.widths[-2:]) <= 1.5 * min(tr.widths[-2:])
        # a side hallway is only measurable from ~3 m until it leaves the view at ~1.4 m,
        # so: 3 measurements, or 2 that agree, or whatever we have once it has left the view
        if w is None or not ((n >= self.min_measures and not tr.unsettled()) or agree
                             or (n >= 1 and tr.left_view)):
            return None
        if w >= self.hall_min:
            return HALLWAY
        if w >= self.entrance_min:
            return ENTRANCE
        return None

    def update(self, scan, walls, gaps, t):
        """Returns (events, gaps_with_track_ids). Events: dicts with kind, side(s), bearing."""
        events = []
        ev = self._update_open(t, self._open_frame(scan, walls))
        if ev:
            # "Open area ahead" was just said: entering it needs no second announcement
            events.append({"kind": ev, "quiet": ev == OPEN_AREA and t - self.open_ahead_t <= 10.0})
        seen = []
        for g in gaps:
            tr = self._associate(g, t)
            if tr is None:
                tr = GapTrack(self.next_id, g.side, t, t, g.z_near, g.z_far)
                self.next_id += 1
                self.tracks.append(tr)
            tr.t_last, tr.z_near, tr.z_far = t, g.z_near, g.z_far
            tr.left_view = bool(tr.widths) and not g.near_seen
            tr.depth = max(tr.depth, g.depth)
            # precision depends on the viewing angle: measure once the gap's near edge is
            # within measure_range x the wall's lateral distance (~18 deg off straight ahead)
            close = g.z_near <= self.measure_range * max(g.wall_c, 0.8)
            if g.measured and g.width >= 0.3 and close:
                tr.widths.append(g.width)
            if g.near_seen and close:
                tr.span = max(tr.span, g.width)
            zm = (g.z_near + g.z_far) / 2
            w = walls[g.side]
            tr.bearing = math.degrees(math.atan2(w.m * zm + w.c, zm))
            g.track = tr.id
            seen.append(tr)
        self.tracks = [tr for tr in self.tracks if t - tr.t_last <= self.forget_s]
        for tr in seen:
            kind = self._classify(tr)
            if kind and _rank(kind) > _rank(tr.kind):
                tr.kind, tr.t_kind = kind, t
        if self.in_open or t - self.open_ahead_t <= 6.0:
            # inside an open area (or walking into one just announced) gaps aren't hallways
            for tr in self.tracks:
                tr.announced = tr.kind or tr.announced
            return events, gaps

        def waiting(tr):
            # an open side waits a moment: if the other side opens too, it's "open area ahead";
            # a hallway waits while a gap opposite it hasn't been measured yet (a crossing)
            if tr.kind == OPEN_SIDE:
                return t - tr.t_kind < self.open_side_hold
            if tr.kind == HALLWAY and t - tr.t_kind < self.cross_hold:
                return any(o.side != tr.side and o.kind is None and not o.widths
                           and abs(o.z_near - tr.z_near) < 2.5 for o in self.tracks)
            return False
        for tr in self.tracks:      # alongside an open side, its pieces aren't new openings
            if tr.announced == OPEN_SIDE:   # quiet until we're past its far end (~1.2 m/s)
                self.quiet_until[tr.side] = max(self.quiet_until[tr.side],
                                                t + min(tr.z_far, 15.0) / 1.2 + 1.0)
        due = []
        for tr in self.tracks:
            if not tr.kind or _rank(tr.kind) <= _rank(tr.announced) or waiting(tr):
                continue
            if t <= self.quiet_until[tr.side] and tr.kind != OPEN_SIDE:
                tr.announced = tr.kind
                continue
            due.append(tr)
        for tr in sorted(due, key=lambda x: -_rank(x.kind)):
            if _rank(tr.kind) <= _rank(tr.announced):      # merged into an earlier event
                continue
            if t <= self.quiet_until[tr.side] and tr.kind != OPEN_SIDE:
                tr.announced = tr.kind
                continue
            if tr.kind == OPEN_SIDE:
                self.quiet_until[tr.side] = t + min(tr.z_far, 15.0) / 1.2 + 1.0
            e = {"kind": tr.kind, "side": tr.side, "track": tr.id, "width": tr.width(),
                 "bearing": tr.bearing, "z": round((tr.z_near + tr.z_far) / 2, 2)}
            tr.announced = tr.kind
            if tr.kind == OPEN_SIDE:    # both sides open at once: open area ahead
                opp = [o for o in self.tracks if o.side != tr.side and (
                    o.kind == OPEN_SIDE or o.span >= 0.6 * self.open_side_min)]
            else:
                opp = [o for o in self.tracks if o.side != tr.side and abs(o.z_near - tr.z_near) < 2.5
                       and (o.kind == tr.kind or (tr.kind == HALLWAY and o.widths and max(o.widths) >= 1.0))]
            if opp and any(o.announced == HALLWAY for o in opp) and tr.kind == HALLWAY:
                e["also"] = True        # the opposite side was announced a moment ago
            elif opp:
                e["side"] = "both"
                for o in opp:
                    o.kind = o.announced = tr.kind if _rank(tr.kind) > _rank(o.kind) else o.kind
                if tr.kind == OPEN_SIDE:
                    e["kind"] = OPEN_AHEAD
                    self.open_ahead_t = t
            if e["kind"] == HALLWAY:
                e["wall_ahead"] = bool(not math.isnan(scan.center_Z)
                                       and scan.center_Z <= self.wall_ahead_max
                                       and scan.center_Z <= e["z"] + 3.5)
            events.append(e)
        return events, gaps


def _rank(kind):
    return {ENTRANCE: 1, HALLWAY: 2, OPEN_SIDE: 3}.get(kind, 0)


# ---------------------------------------------------------------- phrases

def phrase(event, rel=None):
    """Sentence for an event. `rel`: bearing of the gap relative to the walking direction."""
    k = event["kind"]
    if k == OPEN_AREA:
        return "Entering an open area."
    if k == LEAVE_OPEN:
        return "Leaving the open area."
    if k == OPEN_AHEAD:
        return "Open area ahead."
    side = event.get("side")
    if k == HALLWAY and side == "both":
        return ("Hallway ends ahead; it goes left and right." if event.get("wall_ahead")
                else "Hallways cross ahead, on both sides.")
    if k == HALLWAY and event.get("also"):
        return f"Hallway on your {side} too."
    if k == HALLWAY and event.get("wall_ahead"):
        return f"Hallway turns {side} ahead."
    spot = where(rel) if rel is not None else f"on your {side}"
    if spot == "ahead":                         # it's a side opening: keep the side
        spot = f"ahead on your {side}"
    return {HALLWAY: f"Hallway {spot}.", ENTRANCE: f"Entrance {spot}.",
            OPEN_SIDE: f"Open area {spot}."}[k]


SPOKEN = {HALLWAY, OPEN_SIDE, OPEN_AHEAD, OPEN_AREA, LEAVE_OPEN}


def _median(values):
    return float(np.median(values)) if len(values) >= 3 else None


class IntersectionWatcher:
    """Depth model + scan + walls + gaps + tracker for the main loop. Call once per Frame."""

    def __init__(self, estimator, tracker=None, eye_height=1.6, n_cols=64, max_head_turn=25.0,
                 speak_entrances=False):
        self.estimator = estimator
        self.tracker = tracker or PlaceTracker()
        self.eye_height, self.n_cols = eye_height, n_cols
        self.max_head_turn = max_head_turn
        self.spoken = SPOKEN | ({ENTRANCE} if speak_entrances else set())
        self.last = None   # latest result, for drawing
        # recent per-frame estimates: the depth model's shift and each wall's offset
        self.shifts = deque(maxlen=30)
        self.cs = {"left": deque(maxlen=8), "right": deque(maxlen=8)}

    def __call__(self, frame, heading, inv=None):
        """Returns (result, [(sentence or None, event)]). `result` is None while paused."""
        if frame.yaw_deg is not None and abs(heading.relative(0.0, frame.yaw_deg)) > self.max_head_turn:
            self.last = None
            return None, []
        if inv is None:
            inv = self.estimator(frame.image)
        h, w = frame.image.shape[:2]
        res, events = self.process(inv, frame.K, (w, h), frame.t)
        said = []
        for e in events:
            rel = None
            if "bearing" in e:
                rel = heading.relative(e["bearing"], frame.yaw_deg)
                e["rel"] = round(rel, 1)
            sentence = phrase(e, rel)
            e["sentence"] = sentence
            said.append((sentence if e["kind"] in self.spoken and not e.get("quiet") else None, e))
        self.last = res
        return res, said

    def process(self, inv, K, size, t):
        prior = {"shift": _median(self.shifts), "c": {sd: _median(h) for sd, h in self.cs.items()}}
        scan = scan_from_depth(inv, K, size, self.eye_height, self.n_cols, prior=prior)
        if not scan.ok:
            return {"ok": False, "reason": scan.reason, "in_open": self.tracker.in_open}, []
        cur = _median(self.shifts)
        if scan.shift_estimate is not None and (
                cur is None or len(self.shifts) < 5
                or abs(scan.shift_estimate - cur) <= 0.25 * abs(cur) + 0.2):
            # once settled, outliers (a cabinet face taken for a wall) don't move it
            self.shifts.append(scan.shift_estimate)
        for sd, w in scan.walls.items():
            if w is not None:
                self.cs[sd].append(w.c)
        walls = scan.walls
        gaps = [g for w in walls.values() if w for g in find_gaps(scan, w)]
        events, gaps = self.tracker.update(scan, walls, gaps, t)
        return result_dict(scan, walls, gaps, self.tracker, events), events


def result_dict(scan, walls, gaps, tracker, events):
    r2 = lambda v: round(float(v), 2)   # noqa: E731
    return {
        "ok": True, "in_open": tracker.in_open, "horizon": r2(scan.horizon),
        "shift": r2(scan.shift), "shift_from": scan.shift_source,
        "center_z": r2(scan.center_Z),
        "walls": {s: (None if w is None else {"m": r2(w.m), "c": r2(w.c), "n": w.n})
                  for s, w in walls.items()},
        "width": (r2(walls["right"].c - walls["left"].c)
                  if walls["left"] and walls["right"] else None),
        "gaps": [{"side": g.side, "z": [r2(g.z_near), r2(g.z_far)], "w": r2(g.width),
                  "seen": [g.near_seen, g.far_seen], "depth": r2(g.depth),
                  "track": getattr(g, "track", None)} for g in gaps],
        "scan": {"x": [r2(v) for v in scan.X], "z": [r2(v) for v in scan.Z]},
        "events": [{k: v for k, v in e.items()} for e in events],
        "tracks": {tr.id: tr.kind for tr in tracker.tracks if tr.kind},
    }
