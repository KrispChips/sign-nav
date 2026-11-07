"""Dataset-driven checks on SiGNgapore2D ground truth (see tests/sg_dataset.py for the adapter
and `python tests/sg_dataset.py` for the full report). Skipped when the GT file isn't found."""
import pytest

import sg_dataset as sg
from targets import load_rule

GT = sg.find_gt()
pytestmark = pytest.mark.skipif(GT is None, reason="SiGNgapore2D gt_annotation.json not found: "
                                "set SIGN_UNDERSTANDING_GT or clone AdaCompNUS/sign-understanding "
                                "next to this repo")
RULE = load_rule("singapore_units")


@pytest.fixture(scope="module")
def signs():
    return sg.load_signs(GT)


def test_unknown_rate_is_reported_and_small(signs):
    types, unknown = sg.coverage(signs, RULE)
    rate = types["UNKNOWN"] / sum(types.values())
    print(f"\nentry types: {dict(types)}  UNKNOWN {100 * rate:.1f}%: {sorted(set(unknown))}")
    assert rate < 0.05


def test_hand_labelled_types():
    rows = sg.label_accuracy(RULE)
    assert len(rows) == 50
    misses = [r for r in rows if not r["ok"]]
    # Each known miss carries a note saying why; a new miss without one fails the test.
    unexplained = [r for r in misses if "no config" not in r.get("note", "")]
    assert not unexplained, unexplained
    assert len(misses) / len(rows) <= 0.05


def test_self_consistency(signs):
    rows = sg.self_consistency(signs, RULE)
    ok = sum(r["ok"] for r in rows) / len(rows)
    fails = [r["explain"] for r in rows if not r["ok"]][:10]
    assert ok >= 0.95, fails


def test_no_false_exact_or_range_on_negatives(signs):
    rows, _ = sg.negatives(signs, RULE)
    bad = [r["explain"] for r in rows if r["bad"]]
    assert len(rows) > 300
    assert not bad, bad[:10]
