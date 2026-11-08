"""Hallway / entrance / open-area detection on rendered walks (tests/corridor_world.py).

The renderer outputs what a relative depth model gives (affine inverse depth, unknown scale
and shift, noise), so these cover the geometry: scale from the floor, shift from the walls,
wall fitting, gap widths, and the announcement logic. They don't cover Depth Anything's own
errors on real footage (glass, reflections): that's what intersection_replay.py is for."""
import math

import numpy as np
import pytest

import corridor_world as cw
from direction import Heading
from intersection import (IntersectionWatcher, find_gaps, phrase, scan_from_depth)
from intersection_replay import score
from sources import Frame


def walk_events(scene, y0, y1, size=(72, 96), hfov=70.0, **kw):
    wd, _ = scene()
    w = IntersectionWatcher(None)
    K = cw.camera_K(*size, hfov)
    out = []
    for t, y, inv, K in cw.walk(wd, y0, y1, size=size, K=K, **kw):
        _, ev = w.process(inv, K, size, t)
        out += [(e["kind"], e.get("side"), round(y, 1)) for e in ev if not e.get("quiet")]
    return out


def kinds(events):
    return [(k, s) for k, s, _ in events]


# ---- scan geometry

def test_scan_recovers_corridor_walls_and_shift():
    wd, _ = cw.scene_doors()
    inv, K = cw.render(wd, (0.0, 3.0), 0.0, 10.0, noise=0.0)
    s = scan_from_depth(inv, K, (72, 96))
    assert s.ok and s.shift_source == "walls"
    assert s.walls["left"].c == pytest.approx(-1.0, abs=0.05)
    assert s.walls["right"].c == pytest.approx(1.0, abs=0.05)
    # the model's shift (its output at infinity, 0.05 here) in units of the floor slope
    kf = 3.0 * math.cos(math.radians(10)) / (1.6 * K[1, 1] * round(64 * 96 / 72) / 96)
    assert s.shift * kf == pytest.approx(0.05, abs=0.01)


def test_scan_unaffected_by_model_scale_and_shift():
    wd, _ = cw.scene_doors()
    walls = []
    for scale, shift in ((3.0, 0.05), (0.7, 2.0), (12.0, -0.1)):
        inv, K = cw.render(wd, (0.0, 3.0), 0.0, 10.0, noise=0.0, scale=scale, shift=shift)
        s = scan_from_depth(inv, K, (72, 96))
        walls.append((s.walls["left"].c, s.walls["right"].c))
    assert np.ptp([w[0] for w in walls]) < 0.05 and np.ptp([w[1] for w in walls]) < 0.05


def test_scan_rejects_frames_without_floor():
    inv = np.ones((96, 72), np.float32)          # a flat wall in front of the camera
    assert not scan_from_depth(inv, cw.camera_K(), (72, 96)).ok


def test_doorway_width_measured_close_up():
    wd, _ = cw.scene_doors()                     # right door at y 10..10.9
    inv, K = cw.render(wd, (0.0, 8.0), 0.0, 10.0, noise=0.0)
    s = scan_from_depth(inv, K, (72, 96))
    gaps = find_gaps(s, s.walls["right"])
    assert len(gaps) == 1 and gaps[0].measured
    assert gaps[0].z_near == pytest.approx(2.0, abs=0.25)
    assert gaps[0].width == pytest.approx(0.9, abs=0.3)


def test_side_hallway_width():
    wd, _ = cw.scene_side_hall("left", at=20.0, w=2.0)
    inv, K = cw.render(wd, (0.0, 17.5), 0.0, 10.0, noise=0.0)
    s = scan_from_depth(inv, K, (72, 96))
    g = [g for g in find_gaps(s, s.walls["left"]) if g.measured]
    assert g and g[0].width == pytest.approx(2.0, abs=0.35)


def test_recesses_are_not_gaps():
    """(The cabinet needs the walls of earlier frames: covered by the pillars walk.)"""
    wd, _ = cw.scene_pillars_recesses()
    for y in (5.0, 8.0, 17.0):
        inv, K = cw.render(wd, (0.0, y), 0.0, 10.0, noise=0.0)
        s = scan_from_depth(inv, K, (72, 96))
        for w in s.walls.values():
            assert not [g for g in find_gaps(s, w) if g.z_near < 5]


# ---- whole walks

@pytest.mark.parametrize("scene,y0,y1,want", [
    (cw.scene_doors, 0, 28, [("entrance", "right"), ("entrance", "left"), ("entrance", "right")]),
    (lambda: cw.scene_side_hall("left"), 5, 26, [("hallway", "left")]),
    (lambda: cw.scene_side_hall("right", w=3.0), 5, 26, [("hallway", "right")]),
    (cw.scene_cross, 5, 26, [("hallway", "both")]),
    (cw.scene_t, 5, 19, [("hallway", "both")]),
    (cw.scene_pillars_recesses, 0, 40, []),
    (cw.scene_lounge, 0, 26, [("open_side", "right")]),
    (cw.scene_lobby, 0, 28, [("open_ahead", "both")]),
])
def test_walks(scene, y0, y1, want):
    assert kinds(walk_events(scene, y0, y1)) == want


def test_announced_before_reaching_it():
    ev = walk_events(lambda: cw.scene_side_hall("left", at=20.0), 5, 26)
    assert ev and ev[0][2] < 20.0          # said while the hallway (y 20-22) is still ahead
    ev = walk_events(cw.scene_doors, 0, 28)
    assert all(y < door for (_, _, y), door in zip(ev, (10.0, 17.0, 24.0)))


def test_walking_off_centre_and_noisy():
    assert kinds(walk_events(cw.scene_doors, 0, 28, x=0.45, seed=1)) == [
        ("entrance", "right"), ("entrance", "left"), ("entrance", "right")]
    assert kinds(walk_events(cw.scene_pillars_recesses, 0, 40, noise=0.03)) == []


def test_wide_landscape_lens():
    ev = walk_events(cw.scene_cross, 5, 26, size=(96, 72), hfov=100.0)
    assert kinds(ev) == [("hallway", "both")]


def test_t_junction_says_wall_ahead():
    wd, _ = cw.scene_t()
    w = IntersectionWatcher(None)
    said = [phrase(e) for t, y, inv, K in cw.walk(wd, 5, 19)
            for e in w.process(inv, K, (72, 96), t)[1]]
    assert said == ["Hallway ends ahead; it goes left and right."]


# ---- announcements

def test_phrases():
    assert phrase({"kind": "hallway", "side": "left"}, -40.0) == "Hallway ahead on your left."
    assert phrase({"kind": "hallway", "side": "right"}, 80.0) == "Hallway on your right."
    assert phrase({"kind": "hallway", "side": "left"}, -5.0) == "Hallway ahead on your left."
    assert phrase({"kind": "hallway", "side": "both"}) == "Hallways cross ahead, on both sides."
    assert phrase({"kind": "hallway", "side": "left", "wall_ahead": True}) == "Hallway turns left ahead."
    assert phrase({"kind": "hallway", "side": "right", "also": True}) == "Hallway on your right too."
    assert phrase({"kind": "entrance", "side": "right"}, 50.0) == "Entrance ahead on your right."
    assert phrase({"kind": "open_ahead", "side": "both"}) == "Open area ahead."
    assert phrase({"kind": "open_area"}) == "Entering an open area."


@pytest.mark.parametrize("speak_entrances", [False, True])
def test_watcher_speaks_entrances_only_when_asked(speak_entrances):
    wd, _ = cw.scene_doors()
    w = IntersectionWatcher(None, speak_entrances=speak_entrances)
    heading, said = Heading(), []
    for t, y, inv, K in cw.walk(wd, 0, 12):
        frame = Frame(np.zeros((96, 72, 3), np.uint8), t, K, None)
        res, out = w(frame, heading, inv=inv)
        assert res["ok"]
        said += out
    assert [e["kind"] for _, e in said] == ["entrance"]
    assert said[0][1]["sentence"] == "Entrance ahead on your right."
    assert (said[0][0] is not None) == speak_entrances


def test_watcher_pauses_while_head_turned():
    w = IntersectionWatcher(None)
    heading = Heading()
    for t in np.arange(0, 3, 0.2):
        heading.update(0.0, t)
    inv, K = cw.render(cw.scene_doors()[0], (0.0, 3.0))
    res, said = w(Frame(np.zeros((96, 72, 3), np.uint8), 3.0, K, 60.0), heading, inv=inv)
    assert res is None and said == []


def test_replay_scoring():
    events = [{"kind": "hallway", "side": "left", "t": 31.0}, {"kind": "entrance", "side": "right", "t": 40.0},
              {"kind": "open_ahead", "side": "both", "t": 63.0}, {"kind": "hallway", "side": "right", "t": 79.0}]
    labels = {"hallway": [[32.0, "left"], [83.0, "both"]], "entrance": [[41.0, "right"]],
              "open_area": [[64.0, 70.0]]}
    s = score(events, labels)
    assert s["hallway"] == {"tp": 1, "fp": [79.0], "missed": [[83.0, "both"]]}
    assert s["entrance"]["tp"] == 1 and s["open_area"]["tp"] == 1
