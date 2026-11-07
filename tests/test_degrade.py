"""Part B tests: the synthetic degradation behaves like distance and viewing angle."""
import cv2
import numpy as np
import pytest

from legibility.degrade import (DegradeParams, degrade, log_scales, plane_homography,
                                sample_params, sight, stable_seed)
from render import LINES, plate

pytest.importorskip("rapidocr")


@pytest.fixture(scope="module")
def engine():
    from signfilter import ocr_engine
    return ocr_engine()


@pytest.fixture(scope="module")
def sign():
    return plate(LINES[0], 32, border=6)[0]


def test_scales_are_log_spaced():
    s = log_scales(10, 0.08)
    assert s[0] == 1.0 and s[-1] == pytest.approx(0.08) and len(s) == 10
    ratios = np.array(s[1:]) / np.array(s[:-1])
    assert np.allclose(ratios, ratios[0])


def test_reproducible():
    assert stable_seed("sign7", 30, 2) == stable_seed("sign7", 30, 2)
    assert stable_seed("sign7", 30.0, 2) == stable_seed("sign7", 30, 2)    # CLI floats = defaults
    assert stable_seed("sign7", 30.5, 2) != stable_seed("sign7", 30, 2)
    assert sample_params(30, 0.5, 42) == sample_params(30, 0.5, 42)
    crop = plate(LINES[1], 20)[0]
    a = degrade(crop, sample_params(30, 0.5, 42)).image
    b = degrade(crop, sample_params(30, 0.5, 42)).image
    assert np.array_equal(a, b)


def test_motion_blur_in_about_half_the_samples():
    with_motion = sum(sample_params(0, 0.5, seed).motion_len > 0 for seed in range(400))
    assert 150 < with_motion < 250
    p = sample_params(0, 0.5, 1)
    assert 0.4 <= p.blur_sigma <= 1.0 and 60 <= p.jpeg_quality <= 85


@pytest.mark.parametrize("s", [1.0, 0.5, 0.2])
def test_sign_height_scales_with_s(sign, s):
    sc = degrade(sign, DegradeParams(angle=0, scale=s))
    tl, tr, br, bl = sc.corners
    height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    assert height == pytest.approx(sign.shape[0] * s, rel=0.02, abs=0.6)


@pytest.mark.parametrize("angle", [0, 30, 50, 70])
def test_rotation_keeps_height_at_the_centre(sign, angle):
    h, w = sign.shape[:2]
    H = plane_homography(w, h, angle, 1.8 * max(w, h))
    top, bottom = cv2.perspectiveTransform(np.array([[[w / 2, 0], [w / 2, h]]], np.float64), H)[0]
    assert np.linalg.norm(bottom - top) == pytest.approx(h, rel=1e-6)


@pytest.mark.parametrize("angle", [30, 50, 70])
def test_rotation_squeezes_horizontally(sign, angle):
    sc = degrade(sign, DegradeParams(angle=angle, scale=1.0))
    width = sc.corners[:, 0].max() - sc.corners[:, 0].min()
    assert width < sign.shape[1] * (np.cos(np.radians(angle)) + 0.15)


@pytest.mark.parametrize("s", [1.0, 0.6, 0.35])
def test_measured_height_follows_scale(engine, sign, s):
    """True height at scale s = full-resolution height * s; the live path should agree."""
    full = sight(*_scene(sign, 0, 1.0), engine)[2]
    m = sight(*_scene(sign, 0, s), engine)[2]
    assert m.h_p25 == pytest.approx(full.h_p25 * s, rel=0.15)


@pytest.mark.parametrize("angle", [30, 50, 70])
def test_live_path_recovers_the_angle(engine, sign, angle):
    m = sight(*_scene(sign, angle, 1.0), engine)[2]
    assert m.angle == pytest.approx(angle, abs=8)


@pytest.mark.parametrize("angle", [30, 50])
def test_angle_does_not_change_measured_height_much(engine, sign, angle):
    flat = sight(*_scene(sign, 0, 1.0), engine)[2]
    turned = sight(*_scene(sign, angle, 1.0), engine)[2]
    assert turned.h_p25 == pytest.approx(flat.h_p25, rel=0.15)


def _scene(sign, angle, s):
    sc = degrade(sign, DegradeParams(angle=angle, scale=s))
    return sc.image, sc.box, sc.K
