"""Part A tests: text-box heights on rendered text, and source-pixel scale invariance."""
import cv2
import numpy as np
import pytest

from legibility.measure import box_height, measure
from rectify import rectify
from render import LINES, plate
from signfilter import ocr_engine

pytest.importorskip("rapidocr")


@pytest.fixture(scope="module")
def engine():
    return ocr_engine()


def frontal_info(crop):
    return dict(v_scale=1.0, frontality=1.0, sharpness=100.0, height_px=crop.shape[0], aspect=1.0)


@pytest.fixture(scope="module")
def offset(engine):
    """The detector's padding as box = a * cap + b, fitted ONCE on sizes not used in the test."""
    xs, ys = [], []
    for lines in LINES:
        for cap in (10, 14, 20, 28, 40):
            img, real = plate(lines, cap)
            m = measure(img, frontal_info(img), engine)
            xs += [real] * len(m.heights)
            ys += m.heights
    x, y = np.array(xs, float), np.array(ys, float)
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))
              if x[j] != x[i]]
    a = float(np.median(slopes))
    return a, float(np.median(y - a * x))


def test_box_height_is_shorter_side():
    assert box_height([[0, 0], [100, 0], [100, 20], [0, 20]]) == pytest.approx(20)
    assert box_height([[0, 0], [10, 0], [10, 50], [0, 50]]) == pytest.approx(10)


@pytest.mark.parametrize("cap", [8, 12, 16, 24, 32])
@pytest.mark.parametrize("lines", LINES, ids=["set1", "set2", "set3"])
def test_h_p25_matches_rendered_text(engine, offset, lines, cap):
    a, b = offset
    img, real = plate(lines, cap)
    m = measure(img, frontal_info(img), engine)
    assert m.n_lines >= 2, m
    expected = a * real + b
    assert m.h_p25 == pytest.approx(expected, rel=0.20), (m.heights, expected)


def place_on_canvas(sign, canvas=(720, 1280), at=(500, 300)):
    img = np.full((*canvas, 3), 110, np.uint8)
    x, y = at
    img[y:y + sign.shape[0], x:x + sign.shape[1]] = sign
    K = np.array([[800.0, 0, canvas[1] / 2], [0, 800.0, canvas[0] / 2], [0, 0, 1]])
    return img, (x, y, x + sign.shape[1], y + sign.shape[0]), K


@pytest.mark.parametrize("cap", [10, 20])
def test_source_pixel_height_is_scale_invariant(engine, cap):
    sign, _ = plate(LINES[0], cap, border=3)
    img, box, K = place_on_canvas(sign)
    crop, info = rectify(img, box, K)
    m1 = measure(crop, info, engine)
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    m2 = measure(big, dict(info, v_scale=info["v_scale"] / 2), engine)
    assert m1.h_p25 and m2.h_p25
    assert m2.h_p25 == pytest.approx(m1.h_p25, rel=0.05)
    assert m2.sign_px == pytest.approx(m1.sign_px, rel=0.05)


def test_small_sign_enlarged_by_rectify_reports_source_pixels(engine):
    """A sign under 16 px tall is enlarged by rectify; the measurement must not grow with it."""
    sign, _ = plate(("EXIT",), 4, border=1)
    img, box, K = place_on_canvas(sign)
    crop, info = rectify(img, box, K)
    assert crop.shape[0] > sign.shape[0] and info["v_scale"] < 1.0   # it was enlarged
    m = measure(crop, info, engine)
    assert m.sign_px == pytest.approx(sign.shape[0], rel=0.15)
    if m.h_p25:   # tiny text may not be detected at all; if it is, it's in source pixels
        assert m.h_p25 < sign.shape[0]


class FakeEngine:
    """Returns fixed boxes (and what the recognizer 'read' in each), to test the filtering."""
    def __init__(self, boxes, txts=None, scores=None):
        self.boxes = boxes
        self.txts = txts or ["EXIT"] * len(boxes)
        self.scores = scores or [0.95] * len(boxes)

    def __call__(self, img, **kw):
        return type("R", (), {"boxes": self.boxes, "txts": self.txts, "scores": self.scores})()


def test_whole_sign_box_is_not_a_text_line_on_a_small_sign():
    img = np.full((40, 120, 3), 240, np.uint8)
    cv2.putText(img, "EXIT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    whole = [[0, 0], [120, 0], [120, 40], [0, 40]]
    line = [[8, 12], [60, 12], [60, 32], [8, 32]]
    m = measure(img, dict(frontal_info(img), height_px=40), FakeEngine([whole, line]))
    assert m.n_lines == 1 and m.h_p25 < 20


def test_sign_height_text_kept_on_a_large_sign():
    """SiGNgapore2D often boxes a single line tightly: its text really is ~sign-height."""
    img = np.full((60, 260, 3), 240, np.uint8)
    cv2.putText(img, "EXIT", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 0), 5)
    tall = [[2, 2], [180, 2], [180, 58], [2, 58]]
    m = measure(img, dict(frontal_info(img), height_px=60), FakeEngine([tall]))
    assert m.n_lines == 1 and m.h_p25 > 30


def test_line_height_ignores_blur(engine):
    """Half-maximum width barely moves with blur; the detector box grows."""
    img, real = plate(("ROOM 214",), 16)
    sharp = measure(img, frontal_info(img), engine)
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    soft = measure(blurred, frontal_info(blurred), engine)
    assert soft.h_p25 == pytest.approx(sharp.h_p25, rel=0.10)


def test_shared_engine_still_recognizes_after_measuring(engine):
    """RapidOCR keeps call options between calls; measure() must not turn the filter's
    recognition off when the two share an engine."""
    from signfilter import OcrCheck
    img, _ = plate(("ROOM 214",), 20)
    measure(img, frontal_info(img), engine)                  # detection only
    keep, n, text = OcrCheck(engine=engine)(img)              # must still recognize
    assert keep and n >= 4, text


def test_unreadable_detections_are_not_text_lines():
    """Far away, almost every detection is not text: only boxes that read as text count."""
    img = np.full((80, 200, 3), 240, np.uint8)
    cv2.putText(img, "EXIT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    blob = [[10, 5], [190, 5], [190, 75], [10, 75]]
    line = [[15, 25], [100, 25], [100, 58], [15, 58]]
    eng = FakeEngine([blob, line], txts=["~", "EXIT"], scores=[0.3, 0.97])
    m = measure(img, dict(frontal_info(img), height_px=80), eng)
    assert m.n_lines == 1 and m.h_p25 < 40
    assert measure(img, dict(frontal_info(img), height_px=80),
                   FakeEngine([blob], txts=["I"], scores=[0.9])).n_lines == 0   # 1 char: not text
