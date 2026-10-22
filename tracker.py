"""Tracks signs across frames and keeps the BEST sighting of each, so the VLM reads every
physical sign once, from its most legible view.

A track is handed back for reading (once) when:
  - its best quality is excellent (>= great_q), or
  - its best quality is good enough (>= ready_q) and hasn't improved for `patience` seconds
    (you've stopped getting closer, or you're looking away), or
  - it is lost, and its best quality is at least min_q.
Tracks need `min_hits` sightings first, to drop one-frame false positives."""
import math

import numpy as np

from rectify import rectify


def quality(info, score):
    """0..1 legibility estimate. Its terms are stored in info["q_terms"] for tuning."""
    terms = dict(
        size=min(1.0, info["height_px"] / 100.0),            # ~100px tall reads reliably
        sharp=min(1.0, info["sharpness"] / 200.0) ** 0.5,     # motion blur kills OCR
        front=0.4 + 0.6 * info["frontality"],
        whole=0.3 if info["truncated"] else 1.0,
        quad=1.0 if info["quad_found"] else 0.85,
        det=min(1.0, score / 0.5) ** 0.5,
    )
    info["q_terms"] = {k: round(v, 3) for k, v in terms.items()}
    return float(np.prod(list(terms.values())))


class Track:
    def __init__(self, tid, box, t):
        self.id, self.box = tid, np.asarray(box, float)
        self.first_t = self.last_t = self.best_t = t
        self.hits, self.submitted = 0, False
        self.best_q, self.best_crop, self.best_info = 0.0, None, None
        self.last_theta, self.last_yaw = 0.0, None

    def add(self, frame, box, score):
        crop, info = rectify(frame.image, box, frame.K)
        q = quality(info, score)
        self.box, self.last_t, self.hits = np.asarray(box, float), frame.t, self.hits + 1
        self.last_theta, self.last_yaw = info["theta"], frame.yaw_deg
        if q > self.best_q:
            self.best_q, self.best_crop, self.best_info, self.best_t = q, crop, info, frame.t


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    def __init__(self, ready_q=0.5, great_q=0.85, min_q=0.2, min_hits=2, patience=0.6,
                 max_age=1.5, iou_thr=0.2):
        self.ready_q, self.great_q, self.min_q = ready_q, great_q, min_q
        self.min_hits, self.patience, self.max_age, self.iou_thr = min_hits, patience, max_age, iou_thr
        self.tracks, self._next_id = [], 1

    def _predict(self, tr, frame):
        """Shift the box by the head rotation since it was last seen. At 5 fps a quick head
        turn moves a sign further than its own width, which plain IoU matching can't follow."""
        if frame.yaw_deg is None or tr.last_yaw is None:
            return tr.box
        fx, cx = frame.K[0, 0], frame.K[0, 2]
        bx = 0.5 * (tr.box[0] + tr.box[2])
        th = math.atan2(bx - cx, fx) - math.radians(frame.yaw_deg - tr.last_yaw)
        if abs(th) > math.radians(85):
            return tr.box
        dx = cx + fx * math.tan(th) - bx
        return tr.box + np.array([dx, 0, dx, 0])

    def _match_score(self, pred, box):
        iou = _iou(pred, box)
        if iou >= self.iou_thr:
            return iou
        pc, bc = (pred[:2] + pred[2:]) / 2, (box[:2] + box[2:]) / 2
        size = max(pred[2] - pred[0], pred[3] - pred[1])
        ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(1e-6, (pred[2] - pred[0]) * (pred[3] - pred[1]))
        dist = np.linalg.norm(pc - bc)
        if dist < 0.6 * size and 1 / 3 < ratio < 3:   # close and similar size: probably the same sign
            return 0.1 * (1 - dist / (0.6 * size))
        return -1.0

    def update(self, frame, dets):
        """dets: [(box xyxy, score)]. Returns tracks that should be read now."""
        preds = [self._predict(tr, frame) for tr in self.tracks]
        pairs = sorted(((self._match_score(p, np.asarray(b, float)), i, j)
                        for i, p in enumerate(preds) for j, (b, _) in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for s, i, j in pairs:
            if s < 0:
                break
            if i in used_t or j in used_d:
                continue
            used_t.add(i); used_d.add(j)
            self.tracks[i].add(frame, dets[j][0], dets[j][1])
        for j, (box, score) in enumerate(dets):
            if j not in used_d:
                tr = Track(self._next_id, box, frame.t)
                self._next_id += 1
                tr.add(frame, box, score)
                self.tracks.append(tr)

        ready = []
        for tr in self.tracks:
            if tr.submitted or tr.hits < self.min_hits:
                continue
            lost = frame.t - tr.last_t > self.max_age
            settled = tr.best_q >= self.ready_q and frame.t - tr.best_t >= self.patience
            if tr.best_q >= self.great_q or settled or (lost and tr.best_q >= self.min_q):
                tr.submitted = True
                ready.append(tr)
        self.tracks = [tr for tr in self.tracks if frame.t - tr.last_t <= self.max_age]
        return ready

    def flush(self):
        """At end of stream: everything worth reading that hasn't been read yet."""
        out = [tr for tr in self.tracks
               if not tr.submitted and tr.hits >= self.min_hits and tr.best_q >= self.min_q]
        for tr in out:
            tr.submitted = True
        return out
