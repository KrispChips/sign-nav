"""Calibration code: read scoring, fitting on rows with a known answer, and a tiny end-to-end
run (rendered signs, OCR stand-in reader) including the cache."""
import json
import math
import random

import numpy as np
import pytest

from legibility.calibrate import (fit, fit_logistic, load_samples, run_experiment, score_read,
                                  _detection_floor)
from legibility.model import LegibilityModel


def test_score_read():
    gt = [("ROOM 214", "left"), ("EXIT", "right"), ("LIFTS", "up")]
    out = [{"text": "Room 214", "arrow": "left"},       # correct
           {"text": "EXIT", "arrow": "left"},           # right text, wrong arrow
           {"text": "CAFETERIA", "arrow": "up"}]        # invented
    correct, halluc, wrong = score_read(gt, out)
    assert correct == [True, False, False] and halluc == 1 and wrong == 1


def test_score_read_one_output_per_entry():
    gt = [("EXIT", "left"), ("EXIT", "right")]           # list-valued direction in the dataset
    correct, _, _ = score_read(gt, [{"text": "EXIT", "arrow": "left"}])
    assert correct == [True, False]


def test_symbol_synonyms():
    correct, halluc, _ = score_read([("TOILET", "left")], [{"text": "Restrooms", "arrow": "left"}],
                                    symbols=True)
    assert correct == [True] and halluc == 0


def test_fit_logistic_recovers_parameters():
    rnd = np.random.default_rng(0)
    x = rnd.uniform(0, 4, 4000)
    y = rnd.random(4000) < 1 / (1 + np.exp(-(-6 + 3 * x)))
    a, b = fit_logistic(x, y)
    assert a == pytest.approx(-6, abs=0.5) and b == pytest.approx(3, abs=0.3)


def test_detection_floor():
    pairs = [(h, h if h >= 10 or random.Random(int(h * 10)).random() < 0.3 else None)
             for h in np.geomspace(2, 50, 300)]
    floor, bins = _detection_floor(pairs)
    assert 8 <= floor <= 11 and bins


def synthetic_rows(h_star_by_angle, n=1500, seed=0):
    """Rows whose entry correctness follows a known logistic in log height."""
    rnd = np.random.default_rng(seed)
    rows = []
    for angle, h_star in h_star_by_angle.items():
        for _ in range(n):
            h = float(np.exp(rnd.uniform(np.log(2), np.log(60))))
            if h_star is None:
                p = 0.3
            else:   # 90% at h_star, slope 4 per log unit
                p = 1 / (1 + np.exp(-(math.log(9) + 4 * (math.log(h) - math.log(h_star)))))
            live = h * rnd.normal(1.05, 0.04) if h > 5 else (None if rnd.random() < 0.6 else h)
            rows.append({"sign": f"s{rnd.integers(50)}", "kind": "text", "angle": angle, "scale": 1,
                         "live": {"h_p25": live, "sign_px": 4 * h, "sharpness": 300.0},
                         "truth": {"h_p25": h, "sign_px": 4 * h},
                         "entries": [{"h": h, "correct": bool(rnd.random() < p)}],
                         "n_out": 1, "hallucinated": int(rnd.random() < 0.01), "wrong_arrow": 0,
                         "reader": "test", "dry_run": True})
    return rows


def test_fit_recovers_known_thresholds_and_blocking():
    calib, _ = fit(synthetic_rows({0: 12.0, 30: 18.0, 70: None}))
    by = {a["angle"]: a for a in calib["angles"]}
    assert by[0]["h_star_true"] == pytest.approx(12.0, rel=0.12)
    assert by[30]["h_star_true"] == pytest.approx(18.0, rel=0.12)
    assert by[0]["h_star"] == pytest.approx(12.0 * 1.05, rel=0.15)     # converted to live units
    assert by[70]["blocked"]
    assert calib["estimator"]["median_ratio"] == pytest.approx(1.05, abs=0.03)
    m = LegibilityModel.from_dict(calib)                                # the model can load it
    assert m.blocked_by_angle(type("M", (), {"angle": 75})())


def test_high_hallucination_raises_threshold():
    rows = synthetic_rows({0: 10.0})
    for r in rows:   # lots of invented entries below 20 px
        r["hallucinated"] = int(r["truth"]["h_p25"] < 20 and random.Random(id(r)).random() < 0.5)
    calib, _ = fit(rows)
    a0 = calib["angles"][0]
    assert a0["raised_for_hallucination"] and a0["h_star_true"] > 10.0 * 1.2


def test_end_to_end_tiny_run_and_cache(tmp_path):
    pytest.importorskip("rapidocr")
    from legibility.calibrate import OcrProxyReader
    from legibility.synth import synthetic_signs
    signs = synthetic_signs(2, seed=3)
    reader = OcrProxyReader()
    calls = []
    real = reader.__call__

    class Counting:
        tag, ignore_arrows, dry_run = reader.tag, True, True

        def __call__(self, crop):
            calls.append(1)
            return real(crop)
    path = run_experiment(signs, Counting(), tmp_path, angles=[0, 50], scales=[1.0, 0.5, 0.25, 0.1],
                          log=lambda *a: None)
    first = len(calls)
    rows = load_samples([path])
    assert rows and first == len(rows)
    assert {"truth", "live", "entries", "params"} <= set(rows[0])
    assert all(r["dry_run"] for r in rows)
    run_experiment(signs, Counting(), tmp_path, angles=[0, 50], scales=[1.0, 0.5, 0.25, 0.1],
                   log=lambda *a: None)
    assert len(calls) == first          # second run is served entirely from the cache
    json.dumps(rows)                    # rows are plain JSON


def test_dataset_boxes_are_centre_based():
    """SiGNgapore2D boundingBox = [centre x, centre y, w, h] (IMG_0056_frame_0011: the board
    spans x 608-1197, y 229-483 for [903, 356, 589, 254])."""
    from legibility.calibrate import crop_box
    im = np.zeros((1080, 1920, 3), np.uint8)
    im[229:483, 608:1197] = 255
    crop = crop_box(im, [903, 356, 589, 254])
    assert abs(crop.shape[0] - 254) <= 1 and abs(crop.shape[1] - 589) <= 1 and crop.mean() > 245
    assert crop_box(im, [10, 10, 40, 40]).shape[:2] == (30, 30)     # clipped at the edge


def test_threshold_is_relative_to_the_plateau():
    """A reader that tops out at 80%: the threshold is where it reaches 90% of that (72%),
    not where a plain curve would reach 90% (never)."""
    rnd = np.random.default_rng(1)
    rows = synthetic_rows({0: 12.0})
    for r in rows:
        r["entries"][0]["correct"] = bool(r["entries"][0]["correct"] and rnd.random() < 0.8)
    calib, _ = fit(rows)
    a0 = calib["angles"][0]
    assert not a0["blocked"]
    assert a0["curve"]["plateau"] == pytest.approx(0.8, abs=0.06)
    assert a0["h_star_true"] == pytest.approx(12.0, rel=0.2)


def test_named_pictograms_are_not_hallucinations():
    from legibility.calibrate import hallucinated
    r = {"kind": "text", "entries": [{"text": "EXIT", "arrow": "left", "correct": True}],
         "read": [{"text": "Exit", "arrow": "left"}, {"text": "Toilets", "arrow": "right"},
                  {"text": "Wheelchair accessibility", "arrow": "right"},
                  {"text": "Steven Medlock Street", "arrow": "up"}],
         "symbols": ["WHEELCHAIR ACCESSIBILITY"]}
    assert hallucinated(r) == 1                    # only the street name is invented
