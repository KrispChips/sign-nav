"""Matcher unit tests for targets 214 (default rule) and 014-100 (multi-building rule)."""
import pytest

from matcher import Context, MatchLevel as L, match_sign
from targets import load_rule, parse_target

D = load_rule("default")
M = load_rule("multi_building")


def sign(*entries):
    return [{"text": t, "arrow": a} for t, a in entries]


def run(target, entries, rule, **kw):
    m = match_sign(target, entries, rule, **kw)
    assert m.explain()   # every result explains itself
    return m


T214 = parse_target("214", D, aliases=("Radiology",), area=("WING", "EAST"))
T014 = parse_target("014-100", M, aliases=("Radiology",), area=("WING", "EAST"))

# (target, rule, sign entries, context, expected level, expected arrow)
LEVELS = [
    (T214, D, [("Room 214", "right")], None, L.EXACT, "right"),
    (T214, D, [("Rooms 200-250", "left")], None, L.RANGE, "left"),
    (T214, D, [("East Wing", "left")], None, L.AREA, "left"),
    (T214, D, [("Level 2", "up")], None, L.FLOOR, "up"),
    (T214, D, [("Radiology", "right")], None, L.NAMED, "right"),
    (T214, D, [("Elevators", "up")], Context(user_floor=1), L.SUBGOAL, "up"),
    (T214, D, [("Cafeteria", "right")], None, L.NONE, None),
    (T014, M, [("014-100", "right")], None, L.EXACT, "right"),
    (T014, M, [("014-100 to 014-150", "right")], None, L.RANGE, "right"),
    (T014, M, [("East Wing", "left")], None, L.AREA, "left"),
    (T014, M, [("Building 14", "left")], None, L.BUILDING, "left"),
    (T014, M, [("Bldg 014", "left")], None, L.BUILDING, "left"),
    (T014, M, [("Level 1", "up")], None, L.FLOOR, "up"),
    (T014, M, [("Radiology Dept", "right")], None, L.NAMED, "right"),
    (T014, M, [("Stairs", "down")], Context(user_floor=3), L.SUBGOAL, "down"),
    (T014, M, [("Cafeteria", "right")], None, L.NONE, None),
]


@pytest.mark.parametrize("target,rule,entries,ctx,level,arrow", LEVELS,
                         ids=[f"{c[0].raw}:{c[2][0][0]}" for c in LEVELS])
def test_levels(target, rule, entries, ctx, level, arrow):
    m = run(target, sign(*entries), rule, context=ctx)
    assert m.level == level, m.explain()
    assert m.arrow == arrow


def test_best_level_wins():
    m = run(T214, sign(("Level 2", "up"), ("Rooms 200-250", "left"), ("Room 214", "right")), D)
    assert m.level == L.EXACT and m.arrow == "right"


@pytest.mark.parametrize("target,rule,entry,kind", [
    (T014, M, ("Building 15", "left"), "building"),
    (T014, M, ("015-100", "left"), "building"),
    (T014, M, ("015-100 to 015-150", "left"), "building"),
    (T014, M, ("014-120 to 014-150", "left"), "range"),     # floor 1 range without 100
    (T214, D, ("Rooms 220-250", "left"), "range"),
    (T214, D, ("Level 3", "up"), "floor"),
    (T014, M, ("Level 2", "up"), "floor"),
    (T214, D, ("West Wing", "right"), "area"),
], ids=lambda c: c if isinstance(c, str) else None)
def test_contradictions(target, rule, entry, kind):
    m = run(target, sign(entry), rule)
    assert m.level == L.NONE
    assert [c.contradiction for c in m.contradictions] == [kind], m.explain()


def test_range_on_other_floor_is_not_a_contradiction():
    # 300-350 says nothing about where 214 is.
    m = run(T214, sign(("Rooms 300-350", "left")), D)
    assert m.level == L.NONE and not m.contradictions


def test_contradictions_kept_alongside_a_match():
    m = run(T014, sign(("014-100 to 014-150", "right"), ("Building 15", "left")), M)
    assert m.level == L.RANGE and m.arrow == "right"
    assert [(c.contradiction, c.entry.arrow) for c in m.contradictions] == [("building", "left")]


def test_locational_exact_means_arrived():
    m = run(T214, sign(("Room 214", "none")), D)
    assert m.level == L.EXACT and m.locational and m.arrived


def test_locational_building_means_inside():
    m = run(T014, sign(("Building 14", "locational")), M)
    assert m.level == L.BUILDING and m.inside_building and not m.arrived


def test_ambiguous_tie():
    m = run(T214, sign(("Rooms 200-250", "left"), ("Rooms 210-220", "right")), D)
    assert m.level == L.RANGE and m.ambiguous
    assert {t.entry.arrow for t in m.ties} == {"left", "right"}


def test_same_arrow_twice_is_not_ambiguous():
    m = run(T214, sign(("Rooms 200-250", "left"), ("Room 214", "left"), ("214", "left")), D)
    assert not m.ambiguous


def test_ocr_cross_check_halves_confidence():
    good = run(T214, sign(("Rooms 200-250", "right")), D, ocr_text="ROOMS 200 - 250")
    bad = run(T214, sign(("Rooms 200-250", "right")), D, ocr_text="ROOMS 300 - 350 LIFTS")
    assert good.ocr_ok and not bad.ocr_ok
    assert bad.confidence == pytest.approx(good.confidence / 2)


def test_ocr_cross_check_tolerates_ocr_noise():
    m = run(T214, sign(("Room 214", "right")), D, ocr_text="R00M 2I4")
    assert m.ocr_ok


@pytest.mark.parametrize("entry,level", [
    ("214A", L.EXACT),
    ("Room 214", L.RANGE),        # same number, other suffix: close but not exact
    ("Rooms 200-214", L.RANGE),   # suffix-less bound: 214A counts as inside
    ("Rooms 214B-214D", L.NONE),
])
def test_suffix_rules(entry, level):
    m = run(parse_target("214A", D), sign((entry, "left")), D)
    assert m.level == level, m.explain()


def test_parity():
    t = parse_target("214", D)
    assert run(t, sign(("Even numbers 200-250", "left")), D).level == L.RANGE
    assert run(t, sign(("Odd numbers 201-249", "left")), D).level == L.NONE


def test_leading_zeros_compare_numerically():
    t = parse_target("14-100".replace("14", "014"), M)
    assert run(t, sign(("014-090 to 014-110", "left")), M).level == L.RANGE


def test_subgoal_needs_a_different_floor():
    assert run(T214, sign(("Elevators", "up")), D, context=Context(user_floor=2)).level == L.NONE
    assert run(T214, sign(("Elevators", "up")), D).level == L.NONE   # user floor unknown


def test_facility_target_and_labels():
    exit_a = parse_target("Exit A", D)
    m = run(exit_a, sign(("Exit", "left"), ("Exit A", "right")), D)
    assert m.level == L.EXACT and m.arrow == "right" and not m.ambiguous   # labelled beats generic
    m = run(exit_a, sign(("Exit B", "right")), D)
    assert m.level == L.NONE and m.contradictions[0].contradiction == "facility"
    assert run(parse_target("restroom", D), sign(("Toilets", "left")), D).level == L.EXACT


def test_named_numbers_must_agree():
    t = parse_target("Platform 1", D)
    assert run(t, sign(("Platform 2", "left")), D).level == L.NONE
    assert run(t, sign(("Platform 1", "left")), D).level == L.NAMED


def test_area_target():
    t = parse_target("Zone B", D)
    assert run(t, sign(("Zone B", "left")), D).level == L.EXACT
    m = run(t, sign(("Zone C", "left")), D)
    assert m.level == L.NONE and m.contradictions[0].contradiction == "area"


def test_compound_entry_matches_its_part():
    m = run(T214, sign(("Rooms 200–250 & Restrooms", "right")), D)
    assert m.level == L.RANGE and m.arrow == "right"


def test_explain_reads_well():
    m = run(T014, sign(("014-100 to 014-150", "right"), ("Building 15", "left")), M,
            ocr_text="014-100 TO 014-150 BUILDING 15")
    text = m.explain()
    assert "RANGE" in text and "right" in text and "building" in text and "OCR agrees" in text


def test_up_and_down_escalators_differ():
    up = parse_target("Up Escalator", D)
    m = run(up, sign(("Up Escalator", "right"), ("Down Escalator", "up")), D)
    assert m.level == L.EXACT and m.arrow == "right" and not m.ambiguous
    m = run(parse_target("escalator", D), sign(("Up Escalator", "right"), ("Down Escalator", "up")), D)
    assert m.ambiguous
