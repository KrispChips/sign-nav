"""Edge cases from the plan. Handled ones are pinned here; deferred ones are skipped with a
note so the gap stays visible."""
import pytest

from entries import parse_entry
from matcher import MatchLevel as L, match_sign
from targets import load_rule, parse_target

D = load_rule("default")
M = load_rule("multi_building")


@pytest.mark.parametrize("text,floor", [("1st Floor", 1), ("2nd Floor", 2), ("3rd floor", 3),
                                        ("4th Level", 4), ("11th Floor", 11)])
def test_ordinal_floors(text, floor):
    e, = parse_entry(text, "up", D)
    assert e.type == "FLOOR" and e.floors == [floor]


def test_floor_above_9_in_4_digit_rooms():
    assert parse_target("1204", D).floor == 12
    m = match_sign(parse_target("1204", D), [{"text": "Level 12", "arrow": "up"}], D)
    assert m.level == L.FLOOR


@pytest.mark.parametrize("text,lo,hi", [("100s", "100", "199"), ("Rooms 200s", "200", "299"),
                                        ("300 series", "300", "399")])
def test_hundreds_series(text, lo, hi):
    e, = parse_entry(text, "left", D)
    assert e.type == "RANGE" and (e.ranges[0][0].text, e.ranges[0][1].text) == (lo, hi)


def test_merged_lines_room_range_and_facility():
    parts = parse_entry("Rooms 200-250 Toilets", "left", D)
    assert [p.type for p in parts] == ["RANGE", "FACILITY"]


def test_building_without_leading_zeros():
    t = parse_target("014-100", M)
    for text in ("Building 14", "Bldg 014", "014"):
        assert match_sign(t, [{"text": text, "arrow": "left"}], M).level == L.BUILDING, text


@pytest.mark.skip(reason="DEFERRED to Phase 2: the same room ID on several signs (one says left, a "
                         "later one right) needs memory across signs; match_sign judges one sign.")
def test_same_room_on_several_signs():
    pass


@pytest.mark.skip(reason="DEFERRED: two destination NAMES merged into one VLM line "
                         "('PHARMACY RADIOLOGY') can't be split without knowing the building's "
                         "names. Only room/range + facility merges are split today.")
def test_merged_names():
    parts = parse_entry("Pharmacy Radiology", "left", D)
    assert len(parts) == 2
