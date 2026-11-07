"""Test 6 helpers. The full simulated approach (with Gemma) runs on the PC:
python -m legibility.simulate --calib legibility_calib.json --gt ... --images ... --reader gemma"""
import numpy as np
import pytest

from legibility.simulate import approach_scales, cross_check, summarize


def test_approach_is_constant_speed():
    s = approach_scales(0.1, 10, dwell=3)
    assert s[-3:] == [1.0, 1.0, 1.0]
    s = s[:-3]
    d = 1 / np.array(s)
    assert s[0] == pytest.approx(0.1) and s[-1] == pytest.approx(1.0)
    assert np.allclose(np.diff(d), np.diff(d)[0])          # equal distance steps


def test_cross_check():
    assert cross_check([{"text": "Room 214"}, {"text": "Exit"}], "ROOM 214 | EXIT")
    assert not cross_check([{"text": "Cafeteria"}, {"text": "Library"}], "ROOM 214")
    assert not cross_check([], "ROOM 214")


def test_summary_counts_only_evaluated_signs():
    rows = [{"category": "evaluated", "reads": 1, "final_recall": 1.0, "ceiling_recall": 1.0, "state": "done"},
            {"category": "evaluated", "reads": 2, "final_recall": 0.5, "ceiling_recall": 1.0, "state": "done"},
            {"category": "blocked", "reads": 0, "final_recall": 0.0, "ceiling_recall": 1.0, "state": "never fired"},
            {"category": "out_of_range", "reads": 0, "final_recall": 0.0, "ceiling_recall": 1.0, "state": "never fired"}]
    s = summarize(rows)
    assert s["mean_reads"] == pytest.approx(0.75) and s["evaluated"] == 2
    assert s["mean_final_recall_evaluated"] == pytest.approx(0.75)
    assert s["blocked"] == 1 and s["out_of_range"] == 1
