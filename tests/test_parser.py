"""Parser unit tests: entry text -> expected parse, under the default and multi-building rules."""
import pytest

from entries import canonical_arrow, normalize, parse_entry
from targets import load_rule, parse_target

D = load_rule("default")
M = load_rule("multi_building")
S = load_rule("singapore_units")


def one(text, rule=D, arrow="right"):
    parts = parse_entry(text, arrow, rule)
    assert len(parts) == 1, [p.summary() for p in parts]
    return parts[0]


def rooms(e):
    return [r.text for r in e.rooms]


def rng(e):
    return [(lo.text, hi.text) for lo, hi in e.ranges]


# (text, rule, type, check) -- check receives the single ParsedEntry
CASES = [
    ("214", D, "ROOM", lambda e: rooms(e) == ["214"] and e.rooms[0].floor == 2),
    ("Rm 214A", D, "ROOM", lambda e: rooms(e) == ["214A"] and e.rooms[0].suffix == "A" and "RM" in e.fillers),
    ("Rooms 200–250", D, "RANGE", lambda e: rng(e) == [("200", "250")] and "ROOMS" in e.fillers),
    ("200 TO 250", D, "RANGE", lambda e: rng(e) == [("200", "250")]),
    ("250 thru 200", D, "RANGE", lambda e: rng(e) == [("200", "250")]),
    ("101, 103, 105", D, "ROOM", lambda e: rooms(e) == ["101", "103", "105"]),
    ("014-100", M, "ROOM", lambda e: rooms(e) == ["014-100"] and e.rooms[0].building == 14
     and e.rooms[0].floor == 1),
    ("014-100 – 014-150", M, "RANGE", lambda e: rng(e) == [("014-100", "014-150")]),
    ("014-100 to 014-150", M, "RANGE", lambda e: rng(e) == [("014-100", "014-150")]),
    ("Bldg 14", D, "BUILDING", lambda e: e.building == 14),
    ("Building 014", D, "BUILDING", lambda e: e.building == 14),
    ("014", M, "BUILDING", lambda e: e.building == 14),
    ("2nd Floor", D, "FLOOR", lambda e: e.floors == [2]),
    ("Level B1", D, "FLOOR", lambda e: e.floors == [-1]),
    ("Level 12", D, "FLOOR", lambda e: e.floors == [12]),
    ("L2", D, "FLOOR", lambda e: e.floors == [2]),
    ("B1", D, "FLOOR", lambda e: e.floors == [-1]),
    ("Basement", D, "FLOOR", lambda e: e.floors == [-1]),
    ("East Wing", D, "AREA", lambda e: e.area == ("WING", "EAST")),
    ("Zone B", D, "AREA", lambda e: e.area == ("ZONE", "B")),
    ("Tower A (Clinics)", D, "AREA", lambda e: e.area == ("TOWER", "A")),
    ("North", D, "AREA", lambda e: e.area == ("DIRECTION", "NORTH")),
    ("Passenger Lifts", D, "FACILITY", lambda e: e.facility == "elevator"),
    ("Toilets", D, "FACILITY", lambda e: e.facility == "restroom"),
    ("WC", D, "FACILITY", lambda e: e.facility == "restroom"),
    ("Staircase", D, "FACILITY", lambda e: e.facility == "stairs"),
    ("Help Desk", D, "FACILITY", lambda e: e.facility == "information"),
    ("Main Entrance", D, "FACILITY", lambda e: e.facility == "entrance"),
    ("Exit A", D, "FACILITY", lambda e: e.facility == "exit" and e.facility_label == "A"),
    ("Lift to Platform", D, "FACILITY", lambda e: e.facility == "elevator"),
    ("R0OM 2I4", D, "ROOM", lambda e: rooms(e) == ["214"] and e.repaired and e.confidence < 0.95),
    ("Cafeteria", D, "NAMED", lambda e: e.name == "CAFETERIA"),
    ("Emergency", D, "NAMED", lambda e: e.name == "EMERGENCY"),              # a department, not an exit
    ("Breastfeeding Room", D, "NAMED", lambda e: e.name == "BREASTFEEDING ROOM"),   # ROOM kept: no number
    ("Car Pickup/Dropoff via Lift", D, "NAMED", lambda e: "PICKUP" in e.name),     # names aren't split
    ("MD7", D, "NAMED", lambda e: e.name == "MD7"),                          # short alphanumeric code
    ("Orchard MRT@B1/B2", D, "NAMED", lambda e: e.name == "ORCHARD MRT" and e.floors == [-1, -2]),
    ("@B1/B3/B4", D, "FLOOR", lambda e: e.floors == [-1, -3, -4]),
    ("1204", D, "ROOM", lambda e: e.rooms[0].floor == 12),                  # floor above 9
    ("B12", D, "ROOM", lambda e: e.rooms[0].floor == -1),
    ("odd numbers 101–149", D, "RANGE", lambda e: e.parity == "odd"),
    ("300s", D, "RANGE", lambda e: rng(e) == [("300", "399")]),
    ("100 series", D, "RANGE", lambda e: rng(e) == [("100", "199")]),
    ("#04-03A TO #04-03D", S, "RANGE", lambda e: rng(e) == [("04-03A", "04-03D")]),
    ("#04-01", S, "ROOM", lambda e: rooms(e) == ["04-01"] and e.rooms[0].floor == 4),
    ("??", D, "UNKNOWN", lambda e: True),
]


@pytest.mark.parametrize("text,rule,type_,check", CASES, ids=[c[0] for c in CASES])
def test_parse(text, rule, type_, check):
    e = one(text, rule)
    assert e.type == type_, e.summary()
    assert check(e), e


def test_compound_splits_and_shares_arrow():
    parts = parse_entry("Rooms 200–250 & Restrooms", "right", D)
    assert [p.type for p in parts] == ["RANGE", "FACILITY"]
    assert {p.arrow for p in parts} == {"right"}
    assert parts[0].ranges[0][0].text == "200" and parts[1].facility == "restroom"


@pytest.mark.parametrize("text", ["Toilets / Lifts", "Toilets and Lifts", "Toilets, Lifts", "Toilets + Lifts"])
def test_split_connectors(text):
    assert [p.facility for p in parse_entry(text, "left", D)] == ["restroom", "elevator"]


def test_merged_lines_room_and_facility():
    # The VLM merged two sign lines into one entry without a separator.
    parts = parse_entry("Rooms 200-250 Restrooms", "up", D)
    assert [p.type for p in parts] == ["RANGE", "FACILITY"]


def test_multi_word_name_with_comma_not_split():
    parts = parse_entry("National University Centre for Oral Health, Singapore", "up", D)
    assert len(parts) == 1 and parts[0].type == "NAMED"


def test_building_qualified_room_under_default_rule():
    e = one("Bldg 14 Room 100")
    assert e.type == "ROOM" and e.rooms[0].building == 14


def test_multi_building_id_is_not_a_range():
    # The whole point of matching room IDs before connectors.
    e = one("014-100", M)
    assert e.type == "ROOM" and not e.ranges


def test_ocr_repair_never_touches_words_or_codes():
    assert normalize("LOBBY").text == "LOBBY"
    assert normalize("TE14").text == "TE14"
    assert normalize("L12").text == "L12"     # level 12, not room 112
    assert normalize("2I4").text == "214"
    assert normalize("1O1").text == "101"


def test_dash_variants():
    for d in "–—−~":
        assert rng(one(f"200{d}250")) == [("200", "250")]


@pytest.mark.parametrize("given,expected", [
    ("straight", "up"), ("straight-left", "up-left"), ("straight-right", "up-right"),
    ("locational", "none"), ("down-right", "down-right"), ("Left", "left"), (None, "none"),
    ("sideways", "none"),
])
def test_arrow_adapter(given, expected):
    assert canonical_arrow(given) == expected


@pytest.mark.parametrize("text,rule,kind,check", [
    ("room 214", D, "room", lambda t: t.room_id == "214" and t.floor == 2),
    ("214A", D, "room", lambda t: t.room.suffix == "A"),
    ("014-100", M, "room", lambda t: t.building == 14 and t.room_value == (0, 100) and t.floor == 1),
    ("radiology", D, "named", lambda t: t.aliases == ["RADIOLOGY"]),
    ("restroom", D, "facility", lambda t: t.facility == "restroom"),
    ("zone B", D, "area", lambda t: t.area == ("ZONE", "B")),
    ("Level 2", D, "floor", lambda t: t.floor == 2),
    ("Building 14", D, "building", lambda t: t.building == 14),
])
def test_parse_target(text, rule, kind, check):
    t = parse_target(text, rule)
    assert t.kind == kind and check(t), t
