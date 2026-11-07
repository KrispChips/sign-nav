"""Tests 4-5: model math (hand-computed examples) and read-trigger behaviour on synthetic
measurement sequences. No OCR, no files: the calibration is given inline."""
import math
import random

import pytest

from legibility.measure import Measurement
from legibility.model import LegibilityModel
from legibility.trigger import APPROACH, DONE, FACE_SIGN, READ, UNREADABLE, WAIT, ReadTrigger

CALIB = {
    "angles": [
        {"angle": 0, "h_star": 15.0},
        {"angle": 30, "h_star": 18.0},
        {"angle": 50, "h_star": 24.0},
        {"angle": 70, "h_star": None, "blocked": True},
    ],
    "estimator": {"floor": 6.0},
    "pictogram": {"sign_px_star": 40.0},
    "trigger": {"min_sharpness": 100.0},
}


@pytest.fixture
def model():
    return LegibilityModel.from_dict(CALIB)


def M(h=None, angle=0.0, sign_px=80.0, sharp=500.0):
    return Measurement(h_p25=h, h_med=h, n_lines=0 if h is None else 3, sign_px=sign_px,
                       angle=angle, sharpness=sharp)


# ---------------------------------------------------------------- test 4: model math

def test_plan_example(model):
    """Text at 6 px, threshold 15 px: 2.5x larger, i.e. get to 40% of the current distance."""
    m = M(h=6.0)
    assert model.scale_needed(m) == pytest.approx(2.5)
    assert 1 / model.scale_needed(m) == pytest.approx(0.40)
    assert not model.is_legible(m)


def test_steps_closer(model):
    m = M(h=7.5)                                   # needs 2x: from 10 m to 5 m
    assert model.steps_closer(m, 10.0) == math.ceil(5.0 / 0.7)    # 8 steps
    assert model.steps_closer(m, 10.0, step_len=0.5) == 10
    assert model.steps_closer(M(h=20.0), 10.0) == 0
    assert model.steps_closer(M(h=10.0, angle=70), 10.0) is None


def test_legible_at_threshold(model):
    assert model.is_legible(M(h=15.0)) and not model.is_legible(M(h=14.9))
    assert model.scale_needed(M(h=30.0)) == 1.0


def test_interpolation_between_angle_bins(model):
    # log-linear between 15 (0 deg) and 18 (30 deg): at 15 deg, sqrt(15 * 18)
    assert model.h_star(15) == pytest.approx(math.sqrt(15 * 18))
    assert model.h_star(40) == pytest.approx(math.sqrt(18 * 24))
    assert model.h_star(-30) == pytest.approx(18.0)          # sign doesn't matter
    m = M(h=16.0, angle=15)
    assert model.scale_needed(m) == pytest.approx(math.sqrt(15 * 18) / 16)


def test_blocked_by_angle(model):
    assert not model.blocked_by_angle(M(h=10, angle=55))     # below the 50-70 midpoint
    assert model.h_star(55) == pytest.approx(24.0)
    assert model.blocked_by_angle(M(h=10, angle=60))         # from the midpoint on
    assert model.blocked_by_angle(M(h=100, angle=75))
    assert not model.is_legible(M(h=100, angle=75))
    assert math.isinf(model.scale_needed(M(h=100, angle=75)))


def test_floor_and_fallback(model):
    assert not model.is_legible(M(h=4.0))                    # detected but under the floor
    assert model.scale_needed(M(h=4.0)) == pytest.approx(15.0 / 6.0)   # uses the floor
    assert model.is_legible(M(h=None, sign_px=45))           # no text: pictogram fallback
    assert not model.is_legible(M(h=None, sign_px=20))
    assert model.scale_needed(M(h=None, sign_px=20)) == pytest.approx(2.0)


def test_factor_raises_threshold(model):
    assert model.is_legible(M(h=16.0)) and not model.is_legible(M(h=16.0), factor=1.3)


def test_load_roundtrip(tmp_path, model):
    import json
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(CALIB))
    assert LegibilityModel.load(p).h_star(30) == model.h_star(30)


# ---------------------------------------------------------------- test 5: trigger

def approach(start_m=12.0, end_m=1.0, step=0.35, h_at_1m=60.0, noise=0.08, seed=0, angle=0.0):
    """Heights ~ 1/distance with multiplicative noise, one sighting per step."""
    rnd = random.Random(seed)
    d = start_m
    while d >= end_m:
        yield d, M(h=h_at_1m / d * (1 + rnd.gauss(0, noise)), angle=angle)
        d -= step


def run(trigger, seq, outcomes):
    """Feed sightings; answer each READ with the next outcome. Returns the READ distances."""
    outcomes = list(outcomes)
    reads = []
    for d, m in seq:
        dec = trigger.update(m)
        if dec.action == READ:
            reads.append(d)
            trigger.read_result(outcomes.pop(0) if outcomes else True)
    return reads


def test_fires_once_at_the_right_point(model):
    t = ReadTrigger(model)
    reads = run(t, approach(), [True])
    assert len(reads) == 1 and t.state == DONE
    # threshold 15 px at 60 px/m: legible from 4 m; with 2-of-3 and noise, read near there
    assert 3.0 <= reads[0] <= 4.6


def test_single_noisy_frame_does_not_fire(model):
    t = ReadTrigger(model)
    for h in (5, 5, 40, 5, 5):     # one spike
        assert t.update(M(h=h)).action != READ


def test_failed_cross_check_raises_threshold_and_reads_again_closer(model):
    t = ReadTrigger(model)
    reads = run(t, approach(), [False, True])
    assert len(reads) == 2 and t.state == DONE
    assert t.factor == pytest.approx(1.3)
    assert reads[1] < reads[0] / 1.15          # ~1.3x closer, minus noise


def test_two_failures_mean_unreadable(model):
    t = ReadTrigger(model)
    reads = run(t, approach(), [False, False, True])
    assert len(reads) == 2 and t.state == UNREADABLE
    assert t.update(M(h=99)).action == UNREADABLE   # stays finished


def test_read_cap(model):
    t = ReadTrigger(model, max_failed=99)           # only the cap can stop it
    reads = run(t, approach(end_m=0.3, step=0.15), [False] * 10)
    assert len(reads) == 3 and t.state == UNREADABLE


def test_blurred_frames_wait(model):
    t = ReadTrigger(model)
    for _ in range(3):
        d = t.update(M(h=30, sharp=20))
    assert d.action == WAIT
    assert t.update(M(h=30, sharp=500)).action == READ


def test_blocked_angle_asks_to_face_the_sign(model):
    t = ReadTrigger(model)
    for _ in range(3):
        d = t.update(M(h=80, angle=72))
    assert d.action == FACE_SIGN


def test_approach_guidance_uses_median(model):
    t = ReadTrigger(model)
    t.update(M(h=5.0)); t.update(M(h=7.5)); d = t.update(M(h=50.0))
    assert d.action == APPROACH and d.scale_needed == pytest.approx(2.0)   # median 7.5


def test_waits_while_a_read_is_pending(model):
    t = ReadTrigger(model)
    for _ in range(2):
        t.update(M(h=30))
    assert t.update(M(h=30)).action in (READ, WAIT)
    assert t.update(M(h=30)).action == WAIT        # no second Gemma call before the result


def test_per_angle_floor():
    c = {**CALIB, "angles": [dict(b) for b in CALIB["angles"]]}
    c["angles"][0]["floor"], c["angles"][1]["floor"] = 6.0, 12.0
    m = LegibilityModel.from_dict(c)
    assert m.floor_at(0) == 6.0 and m.floor_at(15) == pytest.approx(9.0) and m.floor_at(30) == 12.0
    assert m.is_legible(M(h=19.0, angle=30)) and not m.is_legible(M(h=11.0, angle=30))


def test_no_text_fallback_waits_until_text_would_show():
    c = {**CALIB, "angles": [dict(b) for b in CALIB["angles"]]}
    for b in c["angles"]:
        b["no_text_sign_px"] = 150.0
    m = LegibilityModel.from_dict(c)
    assert not m.is_legible(M(h=None, sign_px=100))     # pictogram threshold (40) alone would say yes
    assert m.is_legible(M(h=None, sign_px=160))
    assert m.scale_needed(M(h=None, sign_px=75)) == pytest.approx(2.0)


def test_waits_while_text_lines_are_still_appearing(model):
    """Same height, but new lines keep showing up: smaller text is still emerging."""
    t = ReadTrigger(model)
    for n in (1, 2, 3, 4):
        m = M(h=30.0); m.n_lines = n
        assert t.update(m).action != READ
    m = M(h=30.0); m.n_lines = 4
    assert t.update(m).action != READ          # 1 sighting without growth
    m = M(h=30.0); m.n_lines = 4
    assert t.update(m).action == READ          # 2 without growth: settled
