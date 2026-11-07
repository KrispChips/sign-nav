"""Does this sign help reach the target, and how?

  m = match_sign(target, [{"text": "Rooms 200-250", "arrow": "right"}], rule,
                 ocr_text="ROOMS 200-250", context=Context(user_floor=1))
  m.level, m.arrow, m.confidence, m.contradictions, m.ambiguous, m.explain()

Pure functions: no I/O, no ML. Arrows are already in the VLM vocabulary (entries.canonical_arrow)."""
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Optional, Union

from rapidfuzz import fuzz

import facilities
from entries import ParsedEntry, normalize, parse_entry
from targets import BuildingRule, RoomId, Target

NAMED_MIN_SCORE = 90      # fuzzy alias match
OCR_MIN_SCORE = 80        # below this the VLM's text isn't backed by OCR: halve confidence


class MatchLevel(IntEnum):
    """Higher is better."""
    NONE = 0
    SUBGOAL = 1    # elevator/stairs/escalator, and the target is on another floor
    NAMED = 2      # fuzzy match against the target's aliases
    FLOOR = 3      # the room target's floor
    BUILDING = 4   # the room target's building
    AREA = 5       # the room target's wing/zone
    RANGE = 6      # a range containing the target room
    EXACT = 7      # the entry names the target


@dataclass
class Context:
    """What the navigation stack knows about where the user is."""
    user_floor: Optional[int] = None
    user_building: Optional[int] = None


@dataclass
class EntryMatch:
    entry: ParsedEntry
    level: MatchLevel = MatchLevel.NONE
    contradiction: Optional[str] = None   # "building", "floor", "range", "area", "facility"
    detail: str = ""
    key_text: str = ""                    # what to look for in the OCR text
    score: float = 0.0                    # within a level: decides ties (and so ambiguity)
    specificity: float = 0.0              # among ties: which one to report (narrower range)

    def line(self) -> str:
        s = f"{self.entry.raw!r} -> {self.entry.arrow}: {self.level.name}"
        if self.contradiction:
            s += f" (contradicts: {self.contradiction})"
        return s + (f" [{self.detail}]" if self.detail else "")


@dataclass
class SignMatch:
    target: Target
    level: MatchLevel
    best: Optional[EntryMatch]
    arrow: Optional[str]
    locational: bool                 # the best entry labels THIS place (arrow none)
    arrived: bool                    # EXACT + locational: the user is there
    inside_building: bool            # BUILDING + locational
    confidence: float
    contradictions: list[EntryMatch]
    ambiguous: bool                  # best level tied between entries pointing different ways
    ties: list[EntryMatch]
    results: list[EntryMatch]
    ocr_ok: Optional[bool] = None    # None when no OCR text was given

    @property
    def entry(self) -> Optional[ParsedEntry]:
        return self.best.entry if self.best else None

    def explain(self) -> str:
        parts = [f"target {self.target.describe()}"]
        if self.best is None:
            parts.append(f"no helpful entry among {len(self.results)}")
        else:
            b = self.best
            ocr = {None: "", True: ", OCR agrees", False: ", OCR disagrees (halved)"}[self.ocr_ok]
            parts.append(f"{self.level.name} via {b.entry.raw!r} -> {b.entry.arrow}"
                         f" (conf {self.confidence:.2f}{ocr})" + (f" [{b.detail}]" if b.detail else ""))
            if self.arrived:
                parts.append("ARRIVED: the sign labels the target itself")
            elif self.inside_building:
                parts.append("inside the target building")
            if self.ambiguous:
                parts.append("AMBIGUOUS: " + ", ".join(f"{t.entry.raw!r} -> {t.entry.arrow}"
                                                       for t in self.ties))
        if self.contradictions:
            parts.append("contradicts: " + ", ".join(
                f"{c.entry.raw!r} -> {c.entry.arrow} ({c.contradiction}: {c.detail})"
                for c in self.contradictions))
        return "; ".join(parts)


# ---------------------------------------------------------------- per entry

def match_entry(target: Target, entry: ParsedEntry, context: Optional[Context] = None) -> EntryMatch:
    ctx = context or Context()
    m = EntryMatch(entry)
    if target.kind == "room":
        _match_room(target, entry, ctx, m)
    elif target.kind == "facility":
        _match_facility(target, entry, m)
    elif target.kind == "area":
        _match_area(target, entry, m, exact=True)
    elif target.kind == "building":
        if entry.type == "BUILDING" and entry.building == target.building:
            _set(m, MatchLevel.EXACT, f"building {entry.building}", entry.raw)
        elif entry.type == "BUILDING":
            _contradict(m, "building", f"building {entry.building}, target is {target.building}")
    elif target.kind == "floor":
        _match_floor(target, entry, ctx, m, exact=True)
    if m.level < MatchLevel.NAMED and not m.contradiction and target.aliases:
        _match_named(target, entry, m)
    return m


def _set(m: EntryMatch, level: MatchLevel, detail: str, key_text: str, score: float = 1.0):
    if level > m.level or (level == m.level and score > m.score):
        m.level, m.detail, m.key_text, m.score = level, detail, key_text, score
        m.contradiction = None   # an entry that helps isn't a contradiction


def _contradict(m: EntryMatch, kind: str, detail: str):
    if m.level == MatchLevel.NONE and not m.contradiction:
        m.contradiction, m.detail = kind, detail


def _match_room(target: Target, e: ParsedEntry, ctx: Context, m: EntryMatch):
    t = target.room
    if e.type == "ROOM" or e.rooms:
        for r in e.rooms:
            if t.building is not None and r.building is not None and r.building != t.building:
                _contradict(m, "building", f"{r} is in building {r.building}, not {t.building}")
            elif r.key == t.key and r.suffix == t.suffix:
                _set(m, MatchLevel.EXACT, f"names {r}", r.text)
            elif r.key == t.key:   # 214 vs 214A: same number, different suffix
                _set(m, MatchLevel.RANGE, f"{r} is the same number as {t}", r.text, 0.5)
    for lo, hi in e.ranges:
        if not (lo.same_building(t) and hi.same_building(t)):
            _contradict(m, "building", f"{lo}..{hi} is in building {lo.building}, not {t.building}")
        elif _in_range(t, lo, hi, e.parity):
            _set(m, MatchLevel.RANGE, f"{t} is in {lo}..{hi}" + (f" ({e.parity})" if e.parity else ""),
                 lo.text)
            m.specificity = max(m.specificity, _range_specificity(lo, hi))
        elif t.floor is not None and lo.floor == hi.floor == t.floor:
            _contradict(m, "range", f"{lo}..{hi} is on floor {t.floor} but doesn't contain {t}")
    if e.type == "BUILDING" and target.building is not None:
        if e.building == target.building:
            _set(m, MatchLevel.BUILDING, f"building {e.building}", e.raw)
        else:
            _contradict(m, "building", f"building {e.building}, target is in {target.building}")
    if e.type == "AREA" and target.area:
        _match_area(target, e, m, exact=False)
    _match_floor(target, e, ctx, m, exact=False)


def _range_specificity(lo: RoomId, hi: RoomId) -> float:
    """Narrower ranges are more specific: 04-03A..04-03D beats 04-02..04-04."""
    width = hi.key[-1] - lo.key[-1] if hi.key[:-1] == lo.key[:-1] else 10_000
    return 1.0 / (1.0 + width / 100.0)


def _in_range(t: RoomId, lo: RoomId, hi: RoomId, parity: Optional[str]) -> bool:
    if t.key[0] != lo.key[0] or t.key[0] != hi.key[0]:   # basement flag must agree
        return False
    # A bound without a suffix compares numbers only, so 214A is inside 200..214.
    above = (t.key, t.suffix) >= (lo.key, lo.suffix) if lo.suffix else t.key >= lo.key
    below = (t.key, t.suffix) <= (hi.key, hi.suffix) if hi.suffix else t.key <= hi.key
    if not (above and below):
        return False
    if parity:
        return (t.key[-1] % 2 == 1) == (parity == "odd")
    return True


def _match_floor(target: Target, e: ParsedEntry, ctx: Context, m: EntryMatch, exact: bool):
    if e.type == "FLOOR" and target.floor is not None:
        if target.floor in e.floors:
            _set(m, MatchLevel.EXACT if exact else MatchLevel.FLOOR, f"floor {target.floor}", e.raw)
        else:
            _contradict(m, "floor", f"floor {'/'.join(map(str, e.floors))}, target is on {target.floor}")
    if (e.type == "FACILITY" and e.facility in facilities.FLOOR_CHANGERS and target.floor is not None
            and ctx.user_floor is not None and ctx.user_floor != target.floor):
        _set(m, MatchLevel.SUBGOAL, f"{e.facility}: target is on floor {target.floor}, "
                                    f"you are on {ctx.user_floor}", e.raw)


def _match_area(target: Target, e: ParsedEntry, m: EntryMatch, exact: bool):
    if e.type != "AREA" or not target.area:
        return
    (tk, tv), (ek, ev) = target.area, e.area
    same_kind = tk == ek or "DIRECTION" in (tk, ek)
    if same_kind and tv == ev:
        _set(m, MatchLevel.EXACT if exact else MatchLevel.AREA, " ".join(e.area), e.raw)
    elif tk == ek:
        _contradict(m, "area", f"{ek} {ev}, target is {tk} {tv}")


def _match_facility(target: Target, e: ParsedEntry, m: EntryMatch):
    if e.type != "FACILITY" or e.facility != target.facility:
        return
    tl, el = target.facility_label, e.facility_label
    if tl and el and tl != el:
        _contradict(m, "facility", f"{e.facility} {el}, target is {e.facility} {tl}")
    else:   # a labelled match ("EXIT A" for EXIT A) beats a generic one ("EXIT")
        _set(m, MatchLevel.EXACT, e.facility + (f" {el}" if el else ""), e.raw,
             1.0 if tl == el else 0.5)


_DISTINCT = re.compile(r"\b(?:[A-Z]*\d[A-Z0-9]*|[A-Z])\b")


def _match_named(target: Target, e: ParsedEntry, m: EntryMatch):
    candidates = {c for c in (e.name, e.raw, normalize(e.source).text if e.source else None) if c}
    best, best_c = 0.0, ""
    for alias in target.aliases:
        a = normalize(alias).text
        for c in candidates:
            # Numbers and single letters must agree: "PLATFORM 1" is not "PLATFORM 2"
            # (their plain fuzzy score is exactly 90).
            da, dc = set(_DISTINCT.findall(a)), set(_DISTINCT.findall(c))
            if da and dc and da != dc:
                continue
            s = fuzz.token_set_ratio(a, c)
            if s > best:
                best, best_c = s, c
    if best >= NAMED_MIN_SCORE:
        _set(m, MatchLevel.NAMED, f"name {best_c!r} ~ alias ({best:.0f})", best_c, best / 100)


# ---------------------------------------------------------------- whole sign

EntryLike = Union[ParsedEntry, dict]


def parse_sign(entries: Iterable[EntryLike], rule: BuildingRule) -> list[ParsedEntry]:
    """VLM entries ({text, arrow}) -> ParsedEntry list. Already-parsed entries pass through."""
    out = []
    for e in entries:
        if isinstance(e, ParsedEntry):
            out.append(e)
        else:
            out.extend(parse_entry(e.get("text", ""), e.get("arrow"), rule))
    return out


def match_sign(target: Target, entries: Iterable[EntryLike], rule: BuildingRule,
               ocr_text: Optional[str] = None, context: Optional[Context] = None) -> SignMatch:
    parsed = parse_sign(entries, rule)
    results = [match_entry(target, e, context) for e in parsed]
    contradictions = [r for r in results if r.contradiction]
    top = max((r.level for r in results), default=MatchLevel.NONE)
    if top == MatchLevel.NONE:
        return SignMatch(target, top, None, None, False, False, False, 0.0, contradictions,
                         False, [], results)
    at_top = [r for r in results if r.level == top]
    best_score = max(r.score for r in at_top)
    ties = [r for r in at_top if abs(r.score - best_score) < 1e-9]
    # Among equally good entries, report the most specific one (the narrower range), then the
    # one whose text is closest to the request ("MRT" over "ORCHARD MRT", "PASSENGER LIFTS"
    # over "LIFT TO STREET"). Whether the result is ambiguous still depends on all of them.
    asked = normalize(target.raw).text
    best = max(ties, key=lambda r: (r.specificity, fuzz.ratio(asked, r.entry.raw)))
    ambiguous = len({r.entry.arrow for r in ties}) > 1

    confidence, ocr_ok = best.entry.confidence, None
    if ocr_text is not None:
        key = normalize(best.key_text or best.entry.raw).text
        ocr_ok = fuzz.partial_ratio(key, normalize(ocr_text).text) >= OCR_MIN_SCORE
        if not ocr_ok:   # the VLM may have invented it: let the caller get closer and re-read
            confidence *= 0.5
    locational = best.entry.locational
    return SignMatch(target, top, best, best.entry.arrow, locational,
                     arrived=locational and top == MatchLevel.EXACT,
                     inside_building=locational and top == MatchLevel.BUILDING,
                     confidence=confidence, contradictions=contradictions,
                     ambiguous=ambiguous, ties=[best] + [r for r in ties if r is not best] if ambiguous
                     else [best], results=results,
                     ocr_ok=ocr_ok)
