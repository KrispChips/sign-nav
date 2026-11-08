"""Synthetic indoor scenes for the intersection detector: a ray caster that renders what a
relative depth model would output (affine inverse depth) for a walker in a floor plan.

World coordinates: x to the right, y forward along the main corridor, z up (metres).
Walls are vertical segments in plan with a height range, so a door is a gap in a full-height
wall plus a lintel segment above it."""
import math
from dataclasses import dataclass, field

import numpy as np

CEILING = 2.7


@dataclass
class World:
    walls: list = field(default_factory=list)     # (x0, y0, x1, y1, z0, z1)
    ceiling: float = CEILING

    def wall(self, x0, y0, x1, y1, z0=0.0, z1=None):
        self.walls.append((x0, y0, x1, y1, z0, self.ceiling if z1 is None else z1))
        return self

    def box(self, x0, y0, x1, y1):
        """A pillar / cabinet: four walls."""
        for s in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
            self.wall(*s)
        return self

    def side_wall(self, x, y0, y1, gaps=(), lintel=None):
        """Wall along y at lateral position x, with gaps [(ya, yb)]. `lintel` (height) closes
        each gap above that height (a doorway)."""
        y = y0
        for ga, gb in sorted(gaps):
            if ga > y:
                self.wall(x, y, x, ga)
            if lintel is not None:
                self.wall(x, ga, x, gb, lintel, self.ceiling)
            y = gb
        if y < y1:
            self.wall(x, y, x, y1)
        return self


def camera_K(width=72, height=96, hfov=70.0):
    f = width / 2 / math.tan(math.radians(hfov) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def render(world, pos, yaw_deg=0.0, pitch_deg=10.0, eye=1.6, K=None, size=(72, 96),
           scale=3.0, shift=0.05, noise=0.01, rng=None):
    """Inverse depth (larger = closer) seen from `pos` (x, y), as a depth model outputs it:
    scale * 1/Z + shift with Z the optical-axis depth, plus multiplicative noise."""
    w, h = size
    K = camera_K(w, h) if K is None else K
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    f0 = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    r = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    fwd = math.cos(pitch) * f0 - math.sin(pitch) * up
    down = -(math.cos(pitch) * up + math.sin(pitch) * f0)
    xs = (np.arange(w) + 0.5 - K[0, 2]) / K[0, 0]
    ys = (np.arange(h) + 0.5 - K[1, 2]) / K[1, 1]
    U, V = np.meshgrid(xs, ys)
    d = U[..., None] * r + V[..., None] * down + fwd        # forward component 1 -> t = Z
    o = np.array([pos[0], pos[1], eye])
    t = np.full((h, w), np.inf)
    dz = d[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(dz < 0, -eye / dz, np.inf)
        tc = np.where(dz > 0, (world.ceiling - eye) / dz, np.inf)
    t = np.minimum(t, np.minimum(tf, tc))
    dx, dy = d[..., 0], d[..., 1]
    for x0, y0, x1, y1, z0, z1 in world.walls:
        ex, ey = x1 - x0, y1 - y0
        den = dx * (-ey) - dy * (-ex)
        with np.errstate(divide="ignore", invalid="ignore"):
            px, py = x0 - o[0], y0 - o[1]
            tt = (px * (-ey) - py * (-ex)) / den
            s = (dx * py - dy * px) / den
        z = eye + tt * dz
        hit = (np.abs(den) > 1e-9) & (tt > 1e-3) & (s >= 0) & (s <= 1) & (z >= z0) & (z <= z1)
        t = np.where(hit & (tt < t), tt, t)
    t = np.minimum(t, 80.0)
    inv = scale / t + shift
    if noise:
        rng = rng or np.random.default_rng(0)
        inv = inv * (1 + noise * rng.standard_normal(inv.shape))
    return inv.astype(np.float32), K


# ---- scenes: a 2 m wide corridor (x in [-1, 1]) running along +y ----

def corridor(width=2.0, length=60.0):
    wd = World()
    return wd, width / 2, length


def scene_doors():
    """Straight corridor with three open doorways (0.9 m, lintel at 2.05 m) into rooms."""
    wd, hw, L = corridor()
    doors_r, doors_l = [(10, 10.9), (24, 24.9)], [(17, 17.9)]
    wd.side_wall(hw, -5, L, doors_r, lintel=2.05)
    wd.side_wall(-hw, -5, L, doors_l, lintel=2.05)
    for ya, _ in doors_r:          # 4 x 5 m room behind each door
        wd.wall(hw, ya - 2, hw + 4, ya - 2).wall(hw + 4, ya - 2, hw + 4, ya + 3).wall(hw + 4, ya + 3, hw, ya + 3)
    for ya, _ in doors_l:
        wd.wall(-hw, ya - 2, -hw - 4, ya - 2).wall(-hw - 4, ya - 2, -hw - 4, ya + 3).wall(-hw - 4, ya + 3, -hw, ya + 3)
    wd.wall(-hw, L, hw, L)
    return wd, {"entrance": [("right", 10.45), ("left", 17.45), ("right", 24.45)]}


def scene_side_hall(side="left", at=20.0, w=2.0, length=15.0):
    wd, hw, L = corridor()
    s = -1 if side == "left" else 1
    wd.side_wall(s * hw, -5, L, [(at, at + w)])
    wd.side_wall(-s * hw, -5, L)
    wd.wall(s * hw, at, s * (hw + length), at).wall(s * hw, at + w, s * (hw + length), at + w)
    wd.wall(s * (hw + length), at, s * (hw + length), at + w)
    wd.wall(-hw, L, hw, L)
    return wd, {"hallway": [(side, at + w / 2)]}


def scene_cross(at=20.0, w=2.0):
    wd, hw, L = corridor()
    for s in (-1, 1):
        wd.side_wall(s * hw, -5, L, [(at, at + w)])
        wd.wall(s * hw, at, s * 15, at).wall(s * hw, at + w, s * 15, at + w).wall(s * 15, at, s * 15, at + w)
    wd.wall(-hw, L, hw, L)
    return wd, {"hallway": [("both", at + w / 2)]}


def scene_t(at=20.0, w=2.0):
    """Corridor ends in a wall; a crossing corridor goes left and right."""
    wd, hw, _ = corridor()
    for s in (-1, 1):
        wd.side_wall(s * hw, -5, at)
        wd.wall(s * hw, at, s * 15, at).wall(s * 15, at, s * 15, at + w)
    wd.wall(-15, at + w, 15, at + w)
    return wd, {"hallway": [("both", at + w / 2)]}


def scene_pillars_recesses():
    """No openings: pillars along both walls, door recesses (0.3 m), cabinets."""
    wd, hw, L = corridor(width=2.4)
    hw = 1.2
    recesses = [(9, 10), (21, 22)]
    wd.side_wall(hw, -5, L, recesses)
    for ya, yb in recesses:        # closed door 0.3 m back
        wd.wall(hw, ya, hw + 0.3, ya).wall(hw + 0.3, ya, hw + 0.3, yb).wall(hw + 0.3, yb, hw, yb)
    wd.side_wall(-hw, -5, L)
    for y in range(6, 40, 6):
        wd.box(-hw, y, -hw + 0.4, y + 0.4)          # pillar against the left wall
    wd.box(hw - 0.5, 14, hw, 16)                    # cabinet on the right
    wd.wall(-hw, L, hw, L)
    return wd, {}


def scene_lobby(enter=15.0, size=20.0):
    """Corridor that opens into a 20 x 20 m lobby with a few columns."""
    wd, hw, _ = corridor()
    wd.side_wall(-hw, -5, enter).side_wall(hw, -5, enter)
    h = size / 2
    wd.wall(-hw, enter, -h, enter).wall(hw, enter, h, enter)
    wd.wall(-h, enter, -h, enter + size).wall(h, enter, h, enter + size).wall(-h, enter + size, h, enter + size)
    for x, y in ((-4, enter + 6), (4, enter + 6), (-4, enter + 13), (4, enter + 13)):
        wd.box(x, y, x + 0.5, y + 0.5)
    return wd, {"open_area": [enter + 1.0]}


def walk(world, y0=0.0, y1=30.0, speed=1.2, fps=5.0, x=0.0, seed=0, yaw_jitter=3.0,
         pitch=10.0, **kw):
    """Frames (t, y, inverse depth, K) walking from y0 to y1 with head sway."""
    rng = np.random.default_rng(seed)
    n = int((y1 - y0) / speed * fps)
    for i in range(n):
        t = i / fps
        y = y0 + speed * t
        yaw = yaw_jitter * math.sin(2 * math.pi * t / 2.3) + rng.normal(0, 0.5)
        p = pitch + 2 * math.sin(2 * math.pi * t / 1.1)
        inv, K = render(world, (x + 0.05 * math.sin(2 * math.pi * t / 1.1), y), yaw, p, rng=rng, **kw)
        yield t, y, inv, K


def scene_lounge(at=15.0, length=8.0, depth=7.0):
    """The right wall gives way to an 8 m long, 7 m deep lounge along the corridor."""
    wd, hw, L = corridor()
    wd.side_wall(-hw, -5, L)
    wd.side_wall(hw, -5, L, [(at, at + length)])
    wd.wall(hw, at, hw + depth, at).wall(hw + depth, at, hw + depth, at + length)
    wd.wall(hw + depth, at + length, hw, at + length)
    wd.box(hw + 2, at + 3, hw + 3, at + 4)     # a sofa / table in it
    wd.wall(-hw, L, hw, L)
    return wd, {"open_side": [("right", at)]}
