"""Room numbering for any building, read from the goal itself (docs/step1-goal.md, sections
1 and 2). No per-building setup: the way the user writes the goal ("014-190", "A140",
"C2.15", "1200", "L3-07") is the best clue to how this building numbers its rooms, so it is
parsed into a Shape and signs are then read through that shape.

  rule = rule_for_goal("A140")           # a BuildingRule built from the goal's shape
  target = parse_target("A140", rule)    # room 140, prefix A (wing/block A)

Signs read through the shape:
- the goal's prefix (wing/block letters) and leading number group (building or section) may
  be missing on a sign: "140" for A140, "190" for 014-190 is a partial ID (a weaker match,
  see matcher.py); a different prefix or group is read so it can contradict ("B140");
- "14-100" means building 14, room 100 when the goal looks like 014-190, but rooms 14 to 100
  when the goal is a plain number; with no goal shape (a named goal), different digit counts
  read as building-room and equal counts as a range;
- floors guessed from numbers are hints only (RoomId.floor).

Per-building YAML rules (targets.load_rule) stay available as overrides for odd conventions.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from targets import BuildingRule, RoomId, load_rule

_BUILDING_REGEXES = (re.compile(r"\b(?:BUILDING|BLDG|BLD)\s*(?P<building>\d{1,3})\b"),)
_GOAL_FILLER = re.compile(r"^(?:(?:ROOMS?|RM|NO|SUITE|STE|UNIT)\b\.?|#)\s*")
_GOAL = re.compile(r"^(?P<prefix>[A-Z]{1,2})?[- ]?(?P<body>\d+(?:[-. ]\d+){0,2})(?P<suffix>[A-Z])?$")


@dataclass(frozen=True)
class Shape:
    """The written form of a room goal: prefix letters, digit groups and the separators
    between them (after the last one: the room number), suffix letter."""
    raw: str
    prefix: str
    groups: tuple[str, ...]
    seps: tuple[str, ...]
    suffix: str
    hash_style: bool = False   # written with '#' (#04-03): the first group is a level

    @property
    def kind(self) -> str:
        if len(self.groups) == 1:
            return "lettered" if self.prefix else "plain"
        return "building-room" if self.section_is_building else "sectioned"

    @property
    def section_is_building(self) -> bool:
        """Is the leading group a building number (014-190, B14-100) rather than a level or
        section (C2.15, 2.105, L3-07, #04-03)?"""
        return (len(self.groups) >= 2 and self.seps[0] == "-" and self.prefix != "L"
                and not self.hash_style)

    @property
    def floor_from_section(self) -> bool:
        """Is the leading group the floor (L3-07, 2.105, #04-03)?"""
        return len(self.groups) >= 2 and not self.section_is_building and (
            self.prefix == "L" or self.seps[0] in ". " or self.hash_style)


def _clean_goal(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).upper().strip()
    t = re.sub(r"[‐-―−﹘﹣－]", "-", t)
    hash_style = "#" in t
    t = _GOAL_FILLER.sub("", t).replace("#", "")
    t = re.sub(r"\s*-\s*", "-", t)
    t = re.sub(r"\s*\.\s*", ".", t)
    return re.sub(r"\s+", " ", t).strip(), hash_style


def goal_shape(text: str) -> Optional[Shape]:
    """'014-190', 'A140', 'C2.15', '120B', 'Room 1200', 'L3-07', '#04-03B' -> Shape.
    None for anything that isn't a room ID (a name, a facility)."""
    t, hash_style = _clean_goal(text)
    m = _GOAL.match(t)
    if not m:
        return None
    body = m.group("body")
    return Shape(str(text), m.group("prefix") or "", tuple(re.split(r"[-. ]", body)),
                 tuple(re.findall(r"[-. ]", body)), m.group("suffix") or "", hash_style)


# ---------------------------------------------------------------- rules

def _sep_regex(sep: str) -> str:
    # entries.normalize turns "." into a space, so "C2.15" on a sign reads "C2 15".
    return "-" if sep == "-" else " "


def _floor_from_room(digits: str) -> Optional[int]:
    """The common convention, as a hint only: 214 -> 2, 1204 -> 12."""
    n = {3: 1, 4: 2}.get(len(digits))
    return int(digits[:n]) if n else None


@dataclass(frozen=True)
class GoalRule(BuildingRule):
    """A BuildingRule built from a goal's Shape. Room IDs carry prefix/section/building as far
    as the sign shows them; entries.py and matcher.py use it like any other rule."""
    shape: Optional[Shape] = None

    def find_rooms(self, text: str) -> list[tuple[RoomId, int, int]]:
        out = super().find_rooms(text)
        if self.shape.section_is_building and len(out) == 1:
            r, s, e = out[0]
            # A bare "014" on its own is a building here, not room 14 (leading zero).
            if (s, e) == (0, len(text.strip())) and r.building is None and not r.prefix \
                    and text.strip().startswith("0"):
                return []
        return out

    def _room(self, m: re.Match) -> RoomId:
        g = m.groupdict()
        s = self.shape
        prefix = g.get("prefix") or None
        sec = g.get("section")
        room = g["room"]
        mid = g.get("mid")
        key = (0,) + ((int(mid),) if mid else ()) + (int(room),)
        building = int(sec) if sec and s.section_is_building else None
        section = int(sec) if sec and not s.section_is_building else None
        if s.floor_from_section:
            floor = int(sec) if sec else None
        else:
            floor = _floor_from_room(room)
        suffix = g.get("suffix") or ""
        # Written the goal's way (signs lose the "." in C2.15 to normalization).
        text = (prefix or "") + (sec + s.seps[0] if sec else "") \
            + (mid + s.seps[1] if mid else "") + room + suffix
        return RoomId(text, key, suffix, building, floor, prefix=prefix, section=section)


def _goal_regex(s: Shape) -> str:
    rx = r"(?<![A-Z0-9])"
    if s.prefix:
        rx += rf"(?:(?P<prefix>[A-Z]{{{len(s.prefix)}}})-?)?"
    if len(s.groups) >= 2:
        lead = s.groups[0]
        width = max(3, len(lead)) if s.section_is_building else max(2, len(lead))
        rx += rf"(?:(?P<section>\d{{1,{width}}}){_sep_regex(s.seps[0])})?"
        if len(s.groups) == 3:
            rx += rf"(?P<mid>\d{{{len(s.groups[1])}}}){_sep_regex(s.seps[1])}"
        rx += rf"(?P<room>\d{{{len(s.groups[-1])}}})"
    elif s.prefix:
        rx += rf"(?P<room>\d{{{len(s.groups[0])}}})"
    else:
        # Plain numbers: allow other lengths, so "14-100" reads as two rooms (a range).
        rx += rf"(?P<room>\d{{2,{len(s.groups[0]) + 1}}})"
    return rx + r"(?P<suffix>[A-Z])?(?![A-Z0-9])"


@dataclass(frozen=True)
class NoHintRule(BuildingRule):
    """For goals that aren't room IDs: the default numbering (214, 1204, B12) plus
    building-room IDs, where "14-100" (different digit counts) or "014-100" (a leading zero)
    is building 14, room 100 and "100-150" (equal counts) is a range."""

    def find_rooms(self, text: str) -> list[tuple[RoomId, int, int]]:
        out = []
        for m in self.room_regex.finditer(text):
            sec, num = m.group("section"), m.group("number")
            if sec and len(sec) == len(num) and not sec.startswith("0") and not m.group("basement"):
                s0, e0 = m.span("section")
                out.append((self._plain(sec, ""), s0, e0))
                out.append((self._plain(num, m.group("suffix") or ""), m.start("number"), m.end()))
            else:
                out.append((self._room(m), m.start(), m.end()))
        return out

    def parse_room(self, text: str) -> Optional[RoomId]:
        found = self.find_rooms(text.strip())
        if len(found) == 1 and found[0][1] == 0 and found[0][2] == len(text.strip()):
            return found[0][0]
        return None

    @staticmethod
    def _plain(digits: str, suffix: str) -> RoomId:
        return RoomId(digits + suffix, (0, int(digits)), suffix, None, _floor_from_room(digits))

    def _room(self, m: re.Match) -> RoomId:
        sec, num, basement = m.group("section"), m.group("number"), bool(m.group("basement"))
        suffix = m.group("suffix") or ""
        if basement:
            floor = -max(1, int(num[0]))
        else:
            floor = _floor_from_room(num)
        return RoomId(m.group(0), (int(basement), int(num)), suffix,
                      int(sec) if sec else None, floor)


_NO_HINT_REGEX = re.compile(
    r"(?<![A-Z0-9])(?:(?P<section>\d{1,3})-(?=B?\d))?(?P<basement>B)?"
    r"(?P<number>(?(basement)\d{2,4}|\d{3,4}))(?P<suffix>[A-Z])?(?![A-Z0-9])")


def _base(name: str, regex: re.Pattern, keep: tuple[str, ...]) -> dict:
    return dict(name=name, room_regex=regex, key_groups=(), id_format="", floor_group=None,
                leading_digits={}, basement_group=None, ground_floor=1,
                building_regexes=_BUILDING_REGEXES, keep_prefixes=keep)


def rule_for_goal(goal: str, override: Optional[str] = None) -> BuildingRule:
    """The rule to read signs with for this goal: `override` (a YAML rule name or path) if
    given, else one built from the goal's shape, else the no-hint rule."""
    if override:
        return load_rule(override)
    s = goal_shape(goal)
    if s is None:
        return NoHintRule(**_base("no-hint", _NO_HINT_REGEX, ()))
    keep = tuple(dict.fromkeys(s.prefix)) if s.prefix else ()
    base = _base(f"goal:{s.kind}", re.compile(_goal_regex(s)), keep)
    if s.section_is_building:
        # Building-room numbering: a bare number on a sign ("14 ->", "16 <-") is a building
        # (campus direction signs list them); a bare room-length number stays a room plate.
        base["building_regexes"] = _BUILDING_REGEXES + (re.compile(r"^(?P<building>\d{1,3})$"),)
    return GoalRule(**base, shape=s)
