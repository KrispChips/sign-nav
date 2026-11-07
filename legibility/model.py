"""Part F: turn a Measurement into decisions, using the fitted calibration file.

  model = LegibilityModel.load("legibility_calib.json")
  model.is_legible(m)              # read now?
  model.scale_needed(m)            # text must get this many times larger (1.0 = fine now)
  model.steps_closer(m, d_now)     # ...in steps, once a distance estimate exists (step 4)
  model.blocked_by_angle(m)        # no size is enough from this angle: move to face the sign

Thresholds in the file are in LIVE measurement units (what measure() reports), converted by
calibrate.py from the true-height accuracy curve through the fitted live-vs-true relation.

Text height scales as 1/distance, so text that must get k times larger means getting to 1/k
of the current distance."""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from legibility.measure import Measurement

BLOCKED = math.inf


@dataclass
class AngleBin:
    angle: float
    h_star: float            # live-measurement threshold (inf = blocked)
    sign_px_star: Optional[float] = None   # pictogram/fallback threshold at this angle, if fitted
    floor: Optional[float] = None          # accuracy floor at this angle, if fitted
    no_text_sign_px: Optional[float] = None   # sign size at which text signs show their text


@dataclass
class LegibilityModel:
    bins: list[AngleBin]
    floor: float                       # below this the text measurement is unreliable
    sign_px_star: float                # pictogram / no-text fallback threshold
    min_sharpness: float = 100.0
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "LegibilityModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, c: dict) -> "LegibilityModel":
        bins = [AngleBin(float(b["angle"]),
                         BLOCKED if b.get("blocked") or b.get("h_star") is None else float(b["h_star"]),
                         b.get("sign_px_star"), b.get("floor"), b.get("no_text_sign_px"))
                for b in c["angles"]]
        return cls(sorted(bins, key=lambda b: b.angle), float(c["estimator"]["floor"]),
                   float(c["pictogram"]["sign_px_star"]),
                   float(c.get("trigger", {}).get("min_sharpness", 100.0)), c)

    # ---------------------------------------------------------------- thresholds

    def h_star(self, angle: float) -> float:
        """Threshold for this viewing angle, log-interpolated between bins. Between a readable
        bin and a blocked one, it is blocked from the midpoint on."""
        return self._interp(angle, [b.h_star for b in self.bins])

    def sign_px_threshold(self, angle: float) -> float:
        per_bin = [b.sign_px_star for b in self.bins]
        if any(v is None for v in per_bin):
            return BLOCKED if self.h_star(angle) == BLOCKED else self.sign_px_star
        return self._interp(angle, [BLOCKED if b.h_star == BLOCKED else v
                                    for b, v in zip(self.bins, per_bin)])

    def floor_at(self, angle: float) -> float:
        """Accuracy floor for this angle (linear between bins), else the global one."""
        return self._linear(angle, "floor", self.floor)

    @property
    def pictograms_calibrated(self) -> bool:
        """Whether the no-text (pictogram) fallback was fitted on symbol-only signs. If not,
        a sign with no detected text is never read: on text signs that fallback fired on far
        signs whose text simply wasn't detectable yet."""
        pic = self.meta.get("pictogram", {})
        return bool(pic) and "note" not in pic

    def no_text_threshold(self, angle: float) -> float:
        """Sign size needed before a sign WITHOUT detected text is read (as a pictogram):
        the pictogram threshold, but never before text signs would have shown their text."""
        return max(self.sign_px_threshold(angle), self._linear(angle, "no_text_sign_px", 0.0))

    def _linear(self, angle: float, attr: str, default: float) -> float:
        known = [(b.angle, getattr(b, attr)) for b in self.bins if getattr(b, attr) is not None]
        if not known:
            return default
        a = abs(angle)
        if a <= known[0][0]:
            return known[0][1]
        for (a0, f0), (a1, f1) in zip(known, known[1:]):
            if a <= a1:
                return f0 + (a - a0) / (a1 - a0) * (f1 - f0)
        return known[-1][1]

    def _interp(self, angle: float, values: list) -> float:
        angles = [b.angle for b in self.bins]
        a = abs(angle)
        if a <= angles[0]:
            return values[0]
        if a >= angles[-1]:
            return values[-1]
        i = max(j for j in range(len(angles)) if angles[j] <= a)
        lo, hi = values[i], values[i + 1]
        t = (a - angles[i]) / (angles[i + 1] - angles[i])
        if math.isinf(lo):
            return BLOCKED
        if math.isinf(hi):                 # readable bin -> blocked bin
            return BLOCKED if t >= 0.5 else lo
        return math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))

    # ---------------------------------------------------------------- decisions

    def blocked_by_angle(self, m: Measurement) -> bool:
        return math.isinf(self.h_star(m.angle))

    def uses_text(self, m: Measurement) -> bool:
        """Whether the text measurement drives the decision (else the sign-size fallback)."""
        return m.h_p25 is not None

    def scale_needed(self, m: Measurement, factor: float = 1.0) -> float:
        """How many times larger the text must get (1.0 = legible now, inf = blocked).
        `factor` raises the threshold for one sign (read trigger after a failed read)."""
        if self.blocked_by_angle(m):
            return BLOCKED
        if self.uses_text(m):
            h = m.h_p25   # the floors are reported by calibration, not used here
            ratio = self.h_star(m.angle) * factor / max(h, 1e-6)
        else:
            ratio = self.no_text_threshold(m.angle) * factor / max(m.sign_px, 1e-6)
        return max(1.0, ratio)

    def is_legible(self, m: Measurement, factor: float = 1.0) -> bool:
        if self.blocked_by_angle(m):
            return False
        if self.uses_text(m):
            # The floor never vetoes a read: on real signs the live estimate is noisy at every
            # height, which pushed the fitted floor far above the threshold and blocked all reads.
            return m.h_p25 >= self.h_star(m.angle) * factor
        if not self.pictograms_calibrated:
            return False      # no text seen and no pictogram calibration: keep approaching
        return m.sign_px >= self.no_text_threshold(m.angle) * factor

    def steps_closer(self, m: Measurement, d_now: float, step_len: float = 0.7,
                     factor: float = 1.0) -> Optional[int]:
        """Steps to take toward the sign, given the current distance (metres, from step 4).
        None when blocked by angle (walking closer won't help)."""
        k = self.scale_needed(m, factor)
        if math.isinf(k):
            return None
        if k <= 1.0:
            return 0
        d_target = d_now / k
        return math.ceil((d_now - d_target) / step_len - 1e-9)
