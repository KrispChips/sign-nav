"""Hallway opening (intersection) detection from monocular depth. No training.

Ported from geometric_intersection_detection.ipynb (sections 2-3). Each frame goes through
Depth Anything V2 (ONNX), the depth map is collapsed into a left-to-right "farness" profile
across the middle of the image, and a side counts as OPEN when its edge of the profile stays
far for long enough. One event fires per new opening.

Detects openings on the left, the right, or both. It does NOT tell a T from a four-way or
detect a wall ahead: with relative depth the farthest thing in view always looks "far".

Tied into the sign pipeline:
- runs on the same undistorted frames, so `--calib` applies here too
- camera left/right is turned into walker left/right with the IMU heading (see `phrase`)
- updates pause while the head is turned away from the walking direction, because then the
  edges of the image are not the sides of the hallway"""
import math

import cv2
import numpy as np

from direction import where

DEPTH_REPO = "onnx-community/depth-anything-v2-small"
DEPTH_FILE = "onnx/model_int8.onnx"


def resolve_model(path=None):
    """Local .onnx path, or download the default model from Hugging Face."""
    if path:
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=DEPTH_REPO, filename=DEPTH_FILE)


class DepthEstimator:
    """Runs the ONNX depth model on a BGR frame and returns relative inverse depth (larger = closer)."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path, long_side=518):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
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
        out = self.sess.run(None, {self.input_name: x})[0]
        return np.squeeze(out).astype(np.float32)


def depth_profile(inv_depth, band=(0.35, 0.65), n_cols=24):
    """Collapse a depth map into a left-to-right 'farness' profile (0 = nearest, 1 = farthest).

    The model's output has an arbitrary scale and shift, and inverse depth squashes far
    distances together, so each pixel is replaced by its percentile rank within the frame
    (what fraction of the frame is closer than it). Ranks are unaffected by scale, shift,
    or any other order-preserving distortion. Then we take the median of each vertical
    column inside a horizontal band, skipping floor and ceiling.
    """
    small = cv2.resize(inv_depth, (inv_depth.shape[1] // 2, inv_depth.shape[0] // 2),
                       interpolation=cv2.INTER_AREA)
    flat = small.ravel()
    ranks = np.empty(flat.size, dtype=np.float32)
    ranks[np.argsort(-flat, kind="stable")] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    far = ranks.reshape(small.shape)  # larger inverse depth = closer = lower rank
    h = far.shape[0]
    strip = far[int(band[0] * h):int(band[1] * h)]
    cols = np.array_split(strip, n_cols, axis=1)
    return np.array([np.median(c) for c in cols], dtype=np.float32)


def region_scores(profile, side_frac=0.3):
    """Average farness over the left edge, center, and right edge of the profile."""
    n = len(profile)
    k = max(1, int(round(n * side_frac)))
    return {
        "left": float(profile[:k].mean()),
        "center": float(profile[k:n - k].mean()),
        "right": float(profile[n - k:].mean()),
    }


class IntersectionDetector:
    """Smooths side-openness scores over time and fires one event per opening.

    A side counts as open only after its smoothed score stays above `on_thresh` for
    `min_on_s` seconds, and closes when it drops below `off_thresh` (hysteresis).
    """

    def __init__(self, on_thresh=0.55, off_thresh=0.40, min_on_s=0.8, alpha=0.4, cooldown_s=4.0):
        self.on_thresh, self.off_thresh = on_thresh, off_thresh
        self.min_on_s, self.alpha, self.cooldown_s = min_on_s, alpha, cooldown_s
        self.reset()

    def reset(self):
        self.ema = None
        self.open = {"left": False, "right": False}
        self.above_since = {"left": None, "right": None}
        self.last_event_t = -1e9
        self.announced = "corridor"  # last state we fired an event for

    def update(self, scores, t):
        if self.ema is None:
            self.ema = dict(scores)
        else:
            for k, v in scores.items():
                self.ema[k] = self.alpha * v + (1 - self.alpha) * self.ema[k]

        for side in ("left", "right"):
            s = self.ema[side]
            if not self.open[side]:
                if s >= self.on_thresh:
                    if self.above_since[side] is None:
                        self.above_since[side] = t
                    if t - self.above_since[side] >= self.min_on_s:
                        self.open[side] = True
                else:
                    self.above_since[side] = None
            elif s < self.off_thresh:
                self.open[side] = False
                self.above_since[side] = None

        if self.open["left"] and self.open["right"]:
            label = "opening_both"
        elif self.open["left"]:
            label = "opening_left"
        elif self.open["right"]:
            label = "opening_right"
        else:
            label = "corridor"

        # Fire when a new opening state appears. If it appears during the cooldown,
        # it fires as soon as the cooldown ends (as long as it's still open).
        event = None
        if label == "corridor":
            self.announced = "corridor"
        elif label != self.announced and t - self.last_event_t >= self.cooldown_s - 1e-6:
            event = label
            self.last_event_t = t
            self.announced = label
        return {"label": label, "event": event, "ema": dict(self.ema)}


def side_bearings(K, width, side_frac=0.3):
    """Camera bearing (deg, + = right) of the middle of the left and right profile regions."""
    fx, cx = K[0, 0], K[0, 2]
    x_left, x_right = 0.5 * side_frac * width, (1 - 0.5 * side_frac) * width
    return {"left": math.degrees(math.atan2(x_left - cx, fx)),
            "right": math.degrees(math.atan2(x_right - cx, fx))}


def phrase(event, rels):
    """Sentence for an event. `rels` maps side -> bearing relative to the walking direction."""
    sides = {"opening_left": ["left"], "opening_right": ["right"],
             "opening_both": ["left", "right"]}[event]
    spots = [where(rels[s]) for s in sides]
    if len(spots) == 2:
        if spots[0] == "ahead on your left" and spots[1] == "ahead on your right":
            return "Openings ahead on both sides."
        return f"Openings {spots[0]} and {spots[1]}."
    return f"Opening {spots[0]}."


class IntersectionWatcher:
    """Depth model + profile + detector for the main loop. Call once per Frame."""

    def __init__(self, estimator, detector, band=(0.35, 0.65), n_cols=24, side_frac=0.3,
                 max_head_turn=25.0):
        self.estimator, self.detector = estimator, detector
        self.band, self.n_cols, self.side_frac = band, n_cols, side_frac
        self.max_head_turn = max_head_turn
        self.last = None   # latest result, for drawing

    def __call__(self, frame, heading):
        """Returns (result, sentence or None). `result` is None while the head is turned."""
        if frame.yaw_deg is not None and abs(heading.relative(0.0, frame.yaw_deg)) > self.max_head_turn:
            self.last = None
            return None, None
        inv = self.estimator(frame.image)
        prof = depth_profile(inv, self.band, self.n_cols)
        scores = region_scores(prof, self.side_frac)
        res = self.detector.update(scores, frame.t)
        res.update(scores=scores, profile=[round(float(v), 3) for v in prof])
        sentence = None
        if res["event"]:
            cam = side_bearings(frame.K, frame.image.shape[1], self.side_frac)
            rels = {s: heading.relative(b, frame.yaw_deg) for s, b in cam.items()}
            res["rels"] = rels
            sentence = phrase(res["event"], rels)
        self.last = res
        return res, sentence
