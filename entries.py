"""Normalize and parse what the VLM read off a sign: one {text, arrow} entry in, one or
more ParsedEntry out (compound entries like "Rooms 200-250 & Restrooms" become two
entries sharing the arrow).

Room IDs are found with the building's regex FIRST, and only then do we look at what joins
them, so under the multi-building rule "014-100" is one room, not a range from 14 to 100.

Type order: RANGE, ROOM, BUILDING, FLOOR, AREA, FACILITY, NAMED, UNKNOWN."""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import facilities
from targets import BuildingRule, RoomId

TYPES = ("RANGE", "ROOM", "BUILDING", "FLOOR", "AREA", "FACILITY", "NAMED", "UNKNOWN")
TYPED = frozenset(TYPES[:6])   # anything but NAMED/UNKNOWN

CONFIDENCE = {"RANGE": 0.95, "ROOM": 0.95, "BUILDING": 0.9, "FLOOR": 0.9, "AREA": 0.9,
              "FACILITY": 0.9, "NAMED": 0.7, "UNKNOWN": 0.2}
OCR_REPAIR_PENALTY = 0.8

# ---------------------------------------------------------------- arrows

ARROWS = ("left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right", "none")
_ARROW_ALIASES = {
    "straight": "up", "ahead": "up", "forward": "up", "straight-left": "up-left",
    "straight-right": "up-right", "locational": "none", "here": "none", "": "none",
    "left-up": "up-left", "right-up": "up-right", "left-down": "down-left",
    "right-down": "down-right",
}


def canonical_arrow(value: Optional[str]) -> str:
    """VLM and dataset arrow words -> the VLM vocabulary (straight -> up, locational -> none).
    Unknown words become "none" rather than raising."""
    v = (value or "").strip().lower().replace("_", "-").replace(" ", "-")
    v = _ARROW_ALIASES.get(v, v)
    return v if v in ARROWS else "none"


# ---------------------------------------------------------------- normalization

_DASHES = re.compile(r"[‐-―−⁃﹘﹣－~]")
_FILLER = re.compile(r"(?:\bR[O0]{2}MS?\b|\bRMS?\b|\bNO\.|\bSUITES?\b|\bSTE\b|#)\s*(?=[#\s]*\S*\d)")
_PUNCT = re.compile(r"[^A-Z0-9 \-/,&+@()]")
_CHUNK = re.compile(r"[A-Z0-9-]+")


@dataclass(frozen=True)
class Normalized:
    text: str
    fillers: tuple[str, ...] = ()   # filler words seen ("ROOMS" hints a range or list follows)
    repaired: bool = False          # an OCR digit repair was applied


def normalize(text: str) -> Normalized:
    """NFKC + upper case, plain hyphens, fillers stripped (and recorded), punctuation dropped
    except - / , & (plus @ and parentheses, which parse_entry uses), OCR digit repair."""
    t = unicodedata.normalize("NFKC", str(text)).upper()
    t = _DASHES.sub("-", t)
    fillers = []

    def drop(m):
        word = re.sub(r"[^A-Z#.]", "", m.group(0).replace("0", "O")).rstrip(".")
        fillers.append(word or "#")
        return " "
    t = _FILLER.sub(drop, t)
    t = t.replace("'", "").replace("’", "")
    t = _PUNCT.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    repaired = False

    def fix(m):
        nonlocal repaired
        tok = m.group(0)
        digits = sum(c.isdigit() for c in tok)
        letters = sum(c.isalpha() for c in tok)
        if not digits or digits <= letters:
            return tok   # alphabetic words (and codes like TE14, L2, B1) are never touched
        out = []
        for i, c in enumerate(tok):
            if c == "O" or (c in "IL" and (i > 0 or len(tok) >= 4)):
                out.append("0" if c == "O" else "1")   # a leading L/I is a prefix (L12 = level)
            else:
                out.append(c)
        new = "".join(out)
        repaired |= new != tok
        return new
    t = _CHUNK.sub(fix, t)
    return Normalized(t, tuple(fillers), repaired)


# ---------------------------------------------------------------- parsed entry

@dataclass
class ParsedEntry:
    raw: str                                  # this entry's normalized text
    arrow: str                                # canonical arrow
    type: str
    source: str = ""                          # the original VLM text (before splitting)
    rooms: list[RoomId] = field(default_factory=list)
    ranges: list[tuple[RoomId, RoomId]] = field(default_factory=list)
    parity: Optional[str] = None              # "odd" / "even" for "odd numbers 101-149"
    building: Optional[int] = None
    floors: list[int] = field(default_factory=list)
    area: Optional[tuple[str, str]] = None    # ("WING", "EAST"), ("ZONE", "B")
    facility: Optional[str] = None
    facility_label: Optional[str] = None
    name: Optional[str] = None                # destination name for NAMED (and others' leftovers)
    confidence: float = 0.0
    fillers: tuple[str, ...] = ()
    repaired: bool = False

    @property
    def floor(self) -> Optional[int]:
        return self.floors[0] if self.floors else None

    @property
    def locational(self) -> bool:
        return self.arrow == "none"

    def summary(self) -> str:
        if self.type == "RANGE":
            what = ", ".join(f"{lo}..{hi}" for lo, hi in self.ranges)
            what += f" ({self.parity})" if self.parity else ""
        elif self.type == "ROOM":
            what = ", ".join(map(str, self.rooms))
        elif self.type == "BUILDING":
            what = f"building {self.building}"
        elif self.type == "FLOOR":
            what = "floor " + "/".join(map(str, self.floors))
        elif self.type == "AREA":
            what = " ".join(self.area)
        elif self.type == "FACILITY":
            what = self.facility + (f" {self.facility_label}" if self.facility_label else "")
        else:
            what = self.name or self.raw
        return f"{self.type} {what}"


# ---------------------------------------------------------------- parsing

_SPLIT = re.compile(r"\s*(?:,|&|/|\+|\bAND\b)\s*")
_TAIL = re.compile(r"[@(]")
_PAREN = re.compile(r"\([^)]*\)?")
_CONNECTORS = {"-", "TO", "THRU", "THROUGH", "TILL", "UNTIL"}
_PARITY = re.compile(r"\b(?P<p>ODD|EVEN)\b(?:\s+(?:NUMBERS|NUMBERED|NOS|NUMBER|SIDE))?")
_SERIES = re.compile(r"\b(?P<n>\d{1,2})00\s*(?:S|SERIES)\b")

_FLOOR_PATTERNS = (
    re.compile(r"\b(?:LEVEL|LVL|LV|FLOOR|FLR|FL)\s*(?P<b>B)?(?P<n>\d{1,2})\b"),
    re.compile(r"\b(?P<n>\d{1,2})(?:ST|ND|RD|TH)\s*(?:FLOOR|FLR|FL|LEVEL|LVL)\b"),
    re.compile(r"\bL(?P<n>\d{1,2})\b"),
    re.compile(r"\b(?P<b>B)(?P<n>\d)\b"),
    re.compile(r"\bBASEMENT(?:\s*(?P<n>\d))?\b"),
    re.compile(r"\b(?P<g>GROUND)(?:\s+(?:FLOOR|LEVEL))?\b"),
    re.compile(r"\b(?:LEVEL|FLOOR)\s+(?P<g>G)\b"),
)
_AREA_PATTERNS = (
    re.compile(r"^(?P<kind>WING|ZONE|TOWER|BLOCK|BLK|SECTOR|POD|AREA|SECTION)\s+(?P<v>[A-Z0-9]{1,3})$"),
    re.compile(r"^(?P<v>[A-Z][A-Z ]*?)\s+(?P<kind>WING|TOWER|ZONE)$"),
    re.compile(r"^(?P<v>NORTH|SOUTH|EAST|WEST|NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST)$"),
)
_CODE = re.compile(r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9-]{2,8}$")   # MD7, TE14, E2I


def parse_entry(text: str, arrow: Optional[str], rule: BuildingRule) -> list[ParsedEntry]:
    """One VLM entry -> one or more ParsedEntry (compound entries are split)."""
    arrow = canonical_arrow(arrow)
    norm = normalize(text)
    m = _TAIL.search(norm.text)
    head, tail = (norm.text[:m.start()], norm.text[m.start():]) if m else (norm.text, "")
    pieces = [p for p in _SPLIT.split(head) if p.strip()]
    if len(pieces) > 1:
        parts = [e for p in pieces for e in _parse_one((p + tail).strip(), arrow, rule, text, norm)]
        if any(e.type in TYPED for e in parts):
            return _merge_rooms(parts)
    return _parse_one(norm.text, arrow, rule, text, norm)


def _parse_one(text: str, arrow: str, rule: BuildingRule, source: str,
               norm: Normalized) -> list[ParsedEntry]:
    def entry(type_, **kw):
        conf = CONFIDENCE[type_] * (OCR_REPAIR_PENALTY if norm.repaired else 1.0)
        return ParsedEntry(raw=text, arrow=arrow, type=type_, source=str(source),
                           confidence=conf, fillers=norm.fillers, repaired=norm.repaired, **kw)

    at = text.find("@")
    core = _PAREN.sub(" ", text[:at] if at >= 0 else text)
    core = re.sub(r"\s+", " ", core).strip(" -")
    tail_floors = _floors(text[at:]) if at >= 0 else []

    parity = None
    pm = _PARITY.search(core)
    if pm:
        parity = pm.group("p").lower()
        core = (core[:pm.start()] + core[pm.end():]).strip()

    sm = _SERIES.search(core)
    if sm:
        n = int(sm.group("n"))
        lo, hi = rule.parse_room(f"{n}00"), rule.parse_room(f"{n}99")
        if lo and hi:
            return [entry("RANGE", ranges=[(lo, hi)], parity=parity, floors=tail_floors)]

    # "BUILDING 014" must not be read as room 014: take written-out building phrases out first.
    bm = rule.find_building(core)
    building = bm[0] if bm else None
    if bm and re.search(r"[A-Z]", core[bm[1]:bm[2]]):
        room_text = core[:bm[1]] + " " * (bm[2] - bm[1]) + core[bm[2]:]
    else:
        room_text = core
    found = rule.find_rooms(room_text)
    if found:
        if building is not None and room_text is not core:   # "BLDG 14 ROOM 100"
            found = [(RoomId(r.text, r.key, r.suffix, r.building or building, r.floor), s, e)
                     for r, s, e in found]
        out = []
        if len(found) >= 2 and core[found[0][2]:found[1][1]].strip() in _CONNECTORS:
            lo, hi = found[0][0], found[1][0]
            if (hi.key, hi.suffix) < (lo.key, lo.suffix):
                lo, hi = hi, lo
            out.append(entry("RANGE", ranges=[(lo, hi)], parity=parity, floors=tail_floors))
            rest = found[2:]
            spans = [(found[0][1], found[1][2])] + [(s, e) for _, s, e in rest]
            if rest:   # "200-250 260" (rare): keep the extra rooms on the same entry
                out[0].rooms = [r for r, _, _ in rest]
        else:
            out.append(entry("ROOM", rooms=[r for r, _, _ in found], parity=parity,
                             floors=tail_floors))
            spans = [(s, e) for _, s, e in found]
        leftover = _remove_spans(room_text, spans)
        leftover = " ".join(w for w in leftover.split() if w not in _CONNECTORS)
        fac = facilities.lookup(leftover) if leftover else None
        if fac:   # two sign lines merged by the VLM: "ROOMS 200-250 RESTROOMS"
            out.append(entry("FACILITY", facility=fac[0], facility_label=fac[1]))
        elif leftover and len(re.findall(r"[A-Z]", leftover)) >= 3:
            out[0].name = leftover
        return out

    if building is not None:
        return [entry("BUILDING", building=building, floors=tail_floors)]

    floors = _floors(core)
    rest = _strip_floors(core)
    if floors and not rest:
        return [entry("FLOOR", floors=floors + [f for f in tail_floors if f not in floors])]
    if not core and tail_floors:   # "@B1/B3/B4"
        return [entry("FLOOR", floors=tail_floors)]
    floors = floors + [f for f in tail_floors if f not in floors]

    for rx in _AREA_PATTERNS:
        am = rx.match(core)
        if am:
            kind = am.groupdict().get("kind") or "DIRECTION"
            kind = {"BLK": "BLOCK"}.get(kind, kind)
            return [entry("AREA", area=(kind, am.group("v").strip()), floors=floors)]

    fac = facilities.lookup(core)
    if fac:
        return [entry("FACILITY", facility=fac[0], facility_label=fac[1], floors=floors)]

    if len(re.findall(r"[A-Z]", core)) >= 3 or _CODE.match(core.replace(" ", "")):
        return [entry("NAMED", name=core, floors=floors)]
    return [entry("UNKNOWN", name=core or None, floors=floors)]


def _floors(text: str) -> list[int]:
    out = []
    for rx in _FLOOR_PATTERNS:
        for m in rx.finditer(text):
            g = m.groupdict()
            if g.get("g"):
                value = 1   # ground; rule.ground_floor is applied by the matcher if needed
            elif g.get("n") is None:
                value = -1  # bare BASEMENT
            else:
                value = -max(1, int(g["n"])) if g.get("b") or "BASEMENT" in m.group(0) else int(g["n"])
            if value not in out:
                out.append(value)
    return out


def _strip_floors(text: str) -> str:
    for rx in _FLOOR_PATTERNS:
        text = rx.sub(" ", text)
    return re.sub(r"[\s,/&-]+", " ", text).strip()


def _remove_spans(text: str, spans) -> str:
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + " " + text[e:]
    return re.sub(r"\s+", " ", text).strip()


def _merge_rooms(parts: list[ParsedEntry]) -> list[ParsedEntry]:
    """Adjacent single-room parts ("101, 103, 105") become one ROOM entry with a list."""
    out: list[ParsedEntry] = []
    for e in parts:
        prev = out[-1] if out else None
        if (prev is not None and e.type == "ROOM" and prev.type == "ROOM" and not e.name
                and not prev.name and e.arrow == prev.arrow):
            prev.rooms.extend(e.rooms)
            prev.raw = f"{prev.raw}, {e.raw}"
        else:
            out.append(e)
    return out
