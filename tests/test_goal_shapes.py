"""Building-agnostic room numbering (docs/step1-goal.md, sections 1-2 and 6): the goal's
written form is the only hint, and every behaviour is checked in more than one convention.
The goals here are examples of conventions, not of any one building."""
import pytest

from entries import parse_entry
from matcher import PARTIAL_WEIGHT, Context, MatchLevel as L, match_sign
from roomid import goal_shape, rule_for_goal
from targets import parse_target


def goal(text):
    rule = rule_for_goal(text)
    return parse_target(text, rule), rule


def sign(target_text, *entries, ctx=None, ocr=None):
    t, rule = goal(target_text)
    return match_sign(t, [{"text": a, "arrow": b} for a, b in entries], rule,
                      ocr_text=ocr, context=ctx or Context())


# ---------------------------------------------------------------- 1. goal parsing

@pytest.mark.parametrize("text,kind,prefix,groups,suffix", [
    ("014-190", "building-room", "", ("014", "190"), ""),
    ("14-100", "building-room", "", ("14", "100"), ""),
    ("B14-100", "building-room", "B", ("14", "100"), ""),
    ("A140", "lettered", "A", ("140",), ""),
    ("C2.15", "sectioned", "C", ("2", "15"), ""),
    ("120", "plain", "", ("120",), ""),
    ("1200", "plain", "", ("1200",), ""),
    ("120B", "plain", "", ("120",), "B"),
    ("B2-105", "building-room", "B", ("2", "105"), ""),
    ("2.105", "sectioned", "", ("2", "105"), ""),
    ("L3-07", "sectioned", "L", ("3", "07"), ""),
    ("2 105", "sectioned", "", ("2", "105"), ""),
    ("Room 1200", "plain", "", ("1200",), ""),
    ("rm. A140", "lettered", "A", ("140",), ""),
    ("#04-03B", "sectioned", "", ("04", "03"), "B"),
    ("014 – 190", "building-room", "", ("014", "190"), ""),
])
def test_goal_shape(text, kind, prefix, groups, suffix):
    s = goal_shape(text)
    assert (s.kind, s.prefix, s.groups, s.suffix) == (kind, prefix, groups, suffix)


@pytest.mark.parametrize("text", ["Chemistry Lab", "Registrar", "restroom", "Exit A"])
def test_names_are_not_room_shapes(text):
    assert goal_shape(text) is None
    assert parse_target(text, rule_for_goal(text)).kind in ("named", "facility")


@pytest.mark.parametrize("text,building,section,prefix,key,floor", [
    ("014-190", 14, None, None, (0, 190), 1),     # leading zeros ignored
    ("14-100", 14, None, None, (0, 100), 1),
    ("B14-100", 14, None, "B", (0, 100), 1),
    ("A140", None, None, "A", (0, 140), 1),
    ("C2.15", None, 2, "C", (0, 15), 2),
    ("2.105", None, 2, None, (0, 105), 2),
    ("L3-07", None, 3, "L", (0, 7), 3),
    ("1200", None, None, None, (0, 1200), 12),
    ("#04-03B", None, 4, None, (0, 3), 4),
])
def test_goal_room_parts(text, building, section, prefix, key, floor):
    t, _ = goal(text)
    r = t.room
    assert t.kind == "room"
    assert (r.building, r.section, r.prefix, r.key, r.floor) == (building, section, prefix, key, floor)
    assert t.floor_stated is False   # from the number: a hint only


def test_suffix_letter_kept():
    t, _ = goal("120B")
    assert (t.room.key, t.room.suffix) == ((0, 120), "B")


def test_prefix_letter_survives_ocr_repair():
    # Without care, OCR digit repair reads a leading L as 1 ("L3-07" -> "13-07").
    t, rule = goal("L3-07")
    e, = parse_entry("L3-07", "none", rule)
    assert e.type == "ROOM" and e.rooms[0].prefix == "L" and e.rooms[0].section == 3


def test_floor_in_words_counts_fully():
    t, _ = goal("Level 2")
    assert t.kind == "floor" and t.floor == 2 and t.floor_stated


# ---------------------------------------------------------------- 2. same shape, ranges

@pytest.mark.parametrize("target,text", [
    ("A140", "A121 - A159"),
    ("014-190", "014-150 to 014-199"),
    ("014-190", "14-100 – 14-200"),
    ("C2.15", "C2.10 - C2.30"),
    ("1200", "1100 thru 1299"),
    ("L3-07", "L3-01 to L3-20"),
    ("2.105", "2.100-2.120"),
])
def test_same_shape_range(target, text):
    m = sign(target, (text, "right"))
    assert m.level == L.RANGE and m.arrow == "right" and not m.partial


@pytest.mark.parametrize("target,text", [
    ("A140", "A140"), ("014-190", "014-190"), ("014-190", "14-190"), ("C2.15", "C2.15"),
    ("120B", "120B"), ("L3-07", "L3-07"), ("1200", "Room 1200"),
])
def test_exact_same_shape(target, text):
    m = sign(target, (text, "left"))
    assert m.level == L.EXACT and not m.partial and m.confidence == pytest.approx(0.95)


@pytest.mark.parametrize("target,text", [("A140", "A141 - A159"), ("1200", "1201 - 1299"),
                                         ("014-190", "014-191 to 014-199")])
def test_range_without_the_goal(target, text):
    assert sign(target, (text, "left")).level == L.NONE


def test_odd_even_ranges():
    assert sign("A141", ("A121 - A159 odd", "left")).level == L.RANGE
    assert sign("A140", ("A121 - A159 odd", "left")).level == L.NONE


# ---------------------------------------------------------------- the 14-100 ambiguity

def test_14_100_is_a_room_for_a_building_room_goal():
    t, rule = goal("014-190")
    e, = parse_entry("14-100", "left", rule)
    assert e.type == "ROOM" and e.rooms[0].building == 14 and e.rooms[0].key == (0, 100)


@pytest.mark.parametrize("target", ["50", "99"])
def test_14_100_is_a_range_for_a_plain_goal(target):
    t, rule = goal(target)
    e, = parse_entry("14-100", "left", rule)
    assert e.type == "RANGE"
    assert sign(target, ("14-100", "left")).level == L.RANGE


@pytest.mark.parametrize("text,want", [
    ("14-100", "ROOM"),        # different digit counts: building-room
    ("014-100", "ROOM"),       # a leading zero: building-room
    ("100-150", "RANGE"),      # equal counts: a range
    ("214", "ROOM"), ("B12", "ROOM"),
])
def test_no_hint_reading(text, want):
    rule = rule_for_goal("Registrar")
    assert [p.type for p in parse_entry(text, "left", rule)] == [want]


# ---------------------------------------------------------------- partial IDs

@pytest.mark.parametrize("target,plate", [("014-190", "190"), ("A140", "140"),
                                          ("C2.15", "15"), ("B14-100", "14-100")])
def test_partial_plate_is_a_weaker_match(target, plate):
    m = sign(target, (plate, "none"))
    assert m.level == L.EXACT and m.partial and m.arrived
    assert m.confidence == pytest.approx(0.95 * PARTIAL_WEIGHT)


@pytest.mark.parametrize("target,plate,ctx", [
    ("014-190", "190", Context(user_building=14)),
    ("A140", "140", Context(user_prefix="A")),
    ("C2.15", "C15", Context(user_section=2)),
])
def test_partial_plate_full_once_the_missing_part_is_known(target, plate, ctx):
    m = sign(target, (plate, "none"), ctx=ctx)
    assert m.level == L.EXACT and not m.partial and m.confidence == pytest.approx(0.95)


def test_full_id_beats_partial_on_the_same_sign():
    m = sign("A140", ("140", "left"), ("A140", "right"))
    assert m.arrow == "right" and not m.partial and not m.ambiguous


def test_partial_range():
    m = sign("A140", ("121 - 159", "left"))
    assert m.level == L.RANGE and m.partial


# ---------------------------------------------------------------- different prefix

@pytest.mark.parametrize("target,text,kind", [
    ("A140", "B140", "prefix"),
    ("A140", "B121 - B159", "prefix"),
    ("014-190", "015-190", "building"),
    ("014-190", "15-100 to 15-200", "building"),
    ("C2.15", "C3.15", "section"),
    ("L3-07", "L4-07", "section"),
])
def test_different_prefix_is_not_a_match(target, text, kind):
    m = sign(target, (text, "left"))
    assert m.level == L.NONE
    assert [c.contradiction for c in m.contradictions] == [kind]
    assert not m.contradictions[0].weak


# ---------------------------------------------------------------- building / wing level

@pytest.mark.parametrize("target,text,level", [
    ("014-190", "Building 14", L.BUILDING),
    ("14-100", "Bldg 014", L.BUILDING),
    ("A140", "Wing A", L.AREA),
    ("A140", "Block A", L.AREA),
    ("C2.15", "Zone C", L.AREA),
])
def test_building_and_wing_level(target, text, level):
    m = sign(target, (text, "up"))
    assert m.level == level and m.arrow == "up"


@pytest.mark.parametrize("target,text,kind", [("014-190", "Building 15", "building"),
                                              ("A140", "Wing B", "area")])
def test_other_building_or_wing_contradicts(target, text, kind):
    m = sign(target, (text, "left"))
    assert m.level == L.NONE and m.contradictions[0].contradiction == kind


def test_more_specific_entry_wins():
    m = sign("A140", ("Wing A", "left"), ("A121 - A159", "right"))
    assert m.level == L.RANGE and m.arrow == "right"


# ---------------------------------------------------------------- floors are hints

@pytest.mark.parametrize("target,text", [("1200", "Level 3"), ("214", "Level 5"),
                                         ("014-190", "2nd Floor")])
def test_guessed_floor_never_rules_out(target, text):
    m = sign(target, (text, "up"))
    assert all(c.weak for c in m.contradictions)


@pytest.mark.parametrize("target,text", [("1200", "1201 - 1250"), ("A140", "A101 - A120")])
def test_range_on_guessed_floor_is_a_hint(target, text):
    m = sign(target, (text, "left"))
    assert m.level == L.NONE
    assert m.contradictions and all(c.weak for c in m.contradictions)


def test_guessed_floor_does_not_block_other_entries():
    m = sign("1200", ("Level 3", "up"), ("1100 - 1299", "left"))
    assert m.level == L.RANGE and m.arrow == "left"


@pytest.mark.parametrize("target", ["1200", "L3-07"])
def test_guessed_floor_suggests_the_lift(target):
    m = sign(target, ("Lifts", "left"), ctx=Context(user_floor=1))
    assert m.level == L.SUBGOAL and m.arrow == "left"


def test_stated_floor_contradiction_is_not_weak():
    m = sign("Level 2", ("Level 3", "up"))
    assert m.contradictions and not m.contradictions[0].weak


# ---------------------------------------------------------------- OCR cross-check still applies

def test_ocr_disagreeing_halves_confidence():
    assert sign("C2.15", ("C2.15", "left"), ocr="C2.15").ocr_ok
    m = sign("C2.15", ("C2.15", "left"), ocr="EXIT")
    assert m.ocr_ok is False and m.confidence == pytest.approx(0.95 * 0.5)


# ---------------------------------------------------------------- YAML override still works

def test_yaml_override():
    rule = rule_for_goal("014-100", override="multi_building")
    assert rule.name == "multi_building"
    t = parse_target("014-100", rule)
    assert t.room.building == 14
