"""Part G: per tracked sign, decide WHEN to spend a Gemma call.

  trig = ReadTrigger(model)
  d = trig.update(measurement)        # every sighting
  if d.action == READ:  send the crop to Gemma, then  trig.read_result(ok)
  elif d.action == APPROACH:  d.scale_needed  (get to 1/scale_needed of the current distance)
  elif d.action == FACE_SIGN: too oblique, no size is enough: move toward the front of the sign

Rules:
- guidance uses the median of the last 3 sightings (single frames are noisy);
- READ fires when the sign is legible in 2 of the last 3 sightings, the latest sighting is
  sharp enough, and the (median) angle isn't blocked;
- and the number of text lines hasn't grown for `settle` sightings: while new lines are still
  appearing, smaller text is still emerging and the height comes from the larger lines that
  survive the distance (on SiGNgapore2D, text accuracy at the same measured height was
  0.40-0.46 with lines still missing vs 0.56-0.65 with all lines visible);
- after a failed read (the step 1 OCR cross-check failed, or nothing came back), this sign's
  threshold is raised x1.3 and the history is cleared, so the next read needs new, closer
  sightings: this adapts to odd fonts or poor contrast without touching the calibration;
- after 2 failed reads of a sign that measured as legible, it's UNREADABLE: not a distance
  problem, walking closer won't help;
- at most 3 reads per sign."""
import statistics
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

from legibility.measure import Measurement
from legibility.model import LegibilityModel

WAIT, READ, APPROACH, FACE_SIGN, DONE, UNREADABLE = (
    "wait", "read", "approach", "face_sign", "done", "unreadable")


@dataclass
class Decision:
    action: str
    scale_needed: Optional[float] = None   # from the smoothed (median) measurement
    reason: str = ""


def _median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def smoothed(history) -> Measurement:
    """Median of the recent sightings, field by field."""
    last = history[-1]
    return replace(last, h_p25=_median(m.h_p25 for m in history),
                   h_med=_median(m.h_med for m in history),
                   sign_px=_median(m.sign_px for m in history),
                   angle=_median(m.angle for m in history),
                   sharpness=_median(m.sharpness for m in history))


class ReadTrigger:
    def __init__(self, model: LegibilityModel, window: int = 3, need: int = 2,
                 min_sharpness: Optional[float] = None, max_reads: int = 3,
                 raise_factor: float = 1.3, max_failed: int = 2, settle: Optional[int] = None):
        self.model, self.need = model, need
        self.min_sharpness = model.min_sharpness if min_sharpness is None else min_sharpness
        self.max_reads, self.raise_factor, self.max_failed = max_reads, raise_factor, max_failed
        self.settle = (model.meta.get("trigger", {}).get("settle_sightings", 2) if settle is None
                       else settle)
        self.max_lines, self.since_growth = 0, 0
        self.history: deque = deque(maxlen=window)
        self.factor = 1.0          # this sign's threshold multiplier
        self.reads = self.failed = 0
        self.pending = False       # a read was fired and its result hasn't come back
        self.state: Optional[str] = None   # DONE or UNREADABLE once finished

    def update(self, m: Measurement) -> Decision:
        if self.state:
            return Decision(self.state, reason="finished")
        self.history.append(m)
        if m.n_lines > self.max_lines:
            self.max_lines, self.since_growth = m.n_lines, 0
        else:
            self.since_growth += 1
        if self.pending:
            return Decision(WAIT, reason="read in progress")
        sm = smoothed(self.history)
        if self.model.blocked_by_angle(sm):
            return Decision(FACE_SIGN, reason=f"angle {sm.angle:.0f} deg: no text size is enough")
        k = self.model.scale_needed(sm, self.factor)
        legible = sum(self.model.is_legible(x, self.factor) for x in self.history)
        if legible < self.need:
            return Decision(APPROACH, k, f"legible in {legible}/{len(self.history)} sightings")
        if self.since_growth < self.settle:
            return Decision(WAIT, k, f"text lines still appearing ({m.n_lines})")
        if m.sharpness < self.min_sharpness:
            return Decision(WAIT, k, f"too blurred ({m.sharpness:.0f} < {self.min_sharpness:.0f})")
        if self.reads >= self.max_reads:
            self.state = UNREADABLE
            return Decision(UNREADABLE, k, f"read cap ({self.max_reads}) reached")
        self.pending = True
        return Decision(READ, k, f"legible in {legible}/{len(self.history)} sightings")

    def read_result(self, ok: bool) -> Decision:
        """Report how the read went: ok = the step 1 match passed its OCR cross-check."""
        self.pending = False
        self.reads += 1
        if ok:
            self.state = DONE
            return Decision(DONE, reason=f"read {self.reads} succeeded")
        self.failed += 1
        if self.failed >= self.max_failed:
            self.state = UNREADABLE
            return Decision(UNREADABLE, reason=f"{self.failed} failed reads while legible: "
                                               "not a distance problem")
        if self.reads >= self.max_reads:
            self.state = UNREADABLE
            return Decision(UNREADABLE, reason=f"read cap ({self.max_reads}) reached")
        self.factor *= self.raise_factor
        self.history.clear()
        return Decision(APPROACH, reason=f"read failed: threshold x{self.factor:.2f}, keep approaching")
