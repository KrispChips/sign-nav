"""Hand-written sign set (tests/data/synthetic_signs.json): room numbers, ranges,
building-qualified IDs, floors and facilities, each with expected results for a few targets.
SiGNgapore2D has few room numbers, so this set is what covers them."""
import json
from pathlib import Path

import pytest

from matcher import Context, MatchLevel, match_sign
from targets import load_rule, parse_target

DATA = json.loads((Path(__file__).parent / "data" / "synthetic_signs.json").read_text())
RULES = {}


def rule(name):
    if name not in RULES:
        RULES[name] = load_rule(name)
    return RULES[name]


CASES = [(s, e) for s in DATA["signs"] for e in s["expect"]]


@pytest.mark.parametrize("sign,exp", CASES,
                         ids=[f"{s['id']}:{e['target']}" + (f"@F{e['user_floor']}" if "user_floor" in e else "")
                              for s, e in CASES])
def test_synthetic(sign, exp):
    r = rule(sign["rule"])
    ctx = Context(user_floor=exp.get("user_floor"))
    m = match_sign(parse_target(exp["target"], r), sign["entries"], r, context=ctx)
    why = m.explain()
    assert m.level == MatchLevel[exp["level"]], why
    if exp.get("ambiguous"):
        assert m.ambiguous, why
    else:
        assert not m.ambiguous, why
        assert m.arrow == exp["arrow"], why
    assert sorted(c.contradiction for c in m.contradictions) == sorted(exp.get("contradictions", [])), why
    assert m.arrived == exp.get("arrived", False), why
    assert m.inside_building == exp.get("inside_building", False), why


def test_set_size():
    assert len(DATA["signs"]) >= 40
