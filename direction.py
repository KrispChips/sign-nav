"""Where a sign is, and where its arrows point, relative to the user's walking direction.

Angles are degrees relative to the heading, + = right, in [-180, 180).

Walking direction ("heading") is approximated by smoothed head yaw: a quick glance to the
side doesn't move it, so a sign you glance at is reported as "on your right" etc.

Arrows are drawn in the frame of someone standing in front of the sign looking at it.
  facing sign (across the corridor, end of hall, overhead): reader looks along the heading,
      so left = left, right = right, up = straight ahead.
  wall sign (on a side wall, parallel to the corridor): reader looks at the wall, so on the
      RIGHT wall the sign's left is ahead and its right is behind; mirrored on the left wall.
      Up/down on a wall sign has no agreed meaning, so it is announced literally."""
import math

WALL_MAX_FRONTALITY = 0.6   # below this the sign is seen obliquely: wall-mounted


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class Heading:
    """Exponentially smoothed head yaw (time constant `tau` seconds), unwrapped internally."""

    def __init__(self, tau=1.5):
        self.tau, self._v, self._t, self._raw, self._u = tau, None, None, None, None

    def update(self, yaw, t):
        if yaw is None:
            return
        self._u = yaw if self._raw is None else self._u + wrap(yaw - self._raw)
        self._raw = yaw
        if self._v is None:
            self._v = self._u
        else:
            a = 1.0 - math.exp(-max(0.0, t - self._t) / self.tau)
            self._v += a * (self._u - self._v)
        self._t = t

    @property
    def value(self):
        return 0.0 if self._v is None else self._v

    def relative(self, theta, yaw):
        """Bearing of something seen at camera angle `theta` while the head was at `yaw`."""
        if yaw is None or self._v is None:
            return wrap(theta)
        return wrap(yaw + theta - self._v)


def mounting(theta, frontality):
    """'wall' if the sign is seen obliquely, else 'facing'. A wall sign looked at squarely
    is handled in describe(): a 'facing' sign far to the side must be on a side wall."""
    if frontality is not None and frontality < WALL_MAX_FRONTALITY:
        return "wall"
    return "facing"


def _reader_dir(rel, mount):
    """Direction the reader looks when facing the sign, snapped to 0 / +-90 / 180."""
    if mount == "wall":
        return 90.0 if rel >= 0 else -90.0
    for d in (0.0, 90.0, -90.0):
        if abs(wrap(rel - d)) <= 45:
            return d
    return 180.0


def where(rel):
    """Where the sign itself is."""
    a, side = abs(rel), "right" if rel >= 0 else "left"
    if a <= 20:
        return "ahead"
    if a <= 65:
        return f"ahead on your {side}"
    if a <= 115:
        return f"on your {side}"
    if a <= 160:
        return f"behind you on your {side}"
    return "behind you"


def go(d):
    """Where to go, for a direction relative to the heading."""
    a, side = abs(wrap(d)), "right" if wrap(d) >= 0 else "left"
    if a <= 22.5:
        return "straight ahead"
    if a <= 67.5:
        return f"ahead to the {side}"
    if a <= 112.5:
        return f"to the {side}"
    if a <= 157.5:
        return f"back to the {side}"
    return "behind you"


_HORIZ = {"left": -90, "right": 90, "up-left": -90, "down-left": -90, "up-right": 90, "down-right": 90}
_FACING = {"up": 0, "left": -90, "right": 90, "up-left": -45, "up-right": 45,
           "down-left": -135, "down-right": 135}


def arrow_phrase(arrow, rel, mount):
    if arrow == "none":
        return None
    look = _reader_dir(rel, mount)
    if look == 0.0:                                  # sign faces us: full 2D mapping
        if arrow in _FACING:
            return go(_FACING[arrow])
    elif arrow in _HORIZ:                            # side-wall / behind: only left-right is meaningful
        return go(look + _HORIZ[arrow])
    return f"{arrow.replace('-', ' ')} arrow"        # ambiguous: say what is drawn


def _clean(text):
    return " ".join(str(text).split()).rstrip(".,;:")


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def describe(entries, rel, mount):
    """One sentence for TTS, or '' if nothing was read."""
    if not entries:
        return ""
    labels, groups = [], {}
    for e in entries:
        text = _clean(e.get("text", ""))
        if not text:
            continue
        p = arrow_phrase(e.get("arrow", "none"), rel, mount)
        if p is None:
            labels.append(text)
        else:
            groups.setdefault(p, []).append(text)
    if not labels and not groups:
        return ""
    loc = where(rel)
    parts = []
    if labels:
        parts.append(f"{_join(labels)}, {loc}.")
    if groups:
        head = "" if labels else f"Sign {loc}: "
        parts.append(head + " ".join(f"{_join(t)}, {p}." for p, t in groups.items()))
    s = " ".join(parts)
    return s[0].upper() + s[1:]
