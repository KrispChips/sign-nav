"""Facility synonyms: maps what signs say ("PASSENGER LIFTS", "WC", "GENTS") to one canonical
facility ("elevator", "restroom", ...).

An entry counts as a facility only when, after dropping a few modifier words (PASSENGER,
PUBLIC, MAIN, ...) and any trailing "VIA ...", "@ ..." or "(...)" part, what's left is exactly
a synonym, optionally followed by a short label ("EXIT A", "LIFT 3"). Substrings don't
count: "DROPOFF VIA LIFT" is not an elevator, and "EMERGENCY" alone is a hospital
department, not an exit."""
import re
from typing import Optional

FACILITIES: dict[str, tuple[str, ...]] = {
    "elevator": ("ELEVATOR", "ELEVATORS", "LIFT", "LIFTS", "LIFT LOBBY"),
    "stairs": ("STAIRS", "STAIR", "STAIRWAY", "STAIRWAYS", "STAIRCASE", "STAIRWELL"),
    "escalator": ("ESCALATOR", "ESCALATORS"),
    "restroom": ("RESTROOM", "RESTROOMS", "TOILET", "TOILETS", "WC", "WASHROOM", "WASHROOMS",
                 "BATHROOM", "BATHROOMS", "MEN", "WOMEN", "MENS", "WOMENS", "MALE", "FEMALE",
                 "GENTS", "LADIES", "MALE TOILET", "FEMALE TOILET", "ACCESSIBLE TOILET"),
    "exit": ("EXIT", "EXITS", "WAY OUT", "EMERGENCY EXIT", "FIRE EXIT"),
    "information": ("INFORMATION", "INFO", "INFORMATION DESK", "INFORMATION COUNTER",
                    "RECEPTION", "HELP DESK", "HELPDESK", "CONCIERGE", "ENQUIRY", "ENQUIRIES"),
    "entrance": ("ENTRANCE", "ENTRANCES", "MAIN ENTRANCE", "ENTRY"),
}

# Facilities that take you to another floor: relevant as a SUBGOAL when the target is elsewhere.
FLOOR_CHANGERS = frozenset({"elevator", "stairs", "escalator"})

# For these, UP/DOWN is kept as the label instead of being dropped.
DIRECTED = frozenset({"escalator", "stairs"})

# Dropped before lookup, but only if something is left ("EMERGENCY" alone stays a name).
MODIFIERS = frozenset({"PASSENGER", "PASSENGERS", "PUBLIC", "MAIN", "UP", "DOWN", "ALL",
                       "ACCESSIBLE", "DISABLED", "VISITOR", "VISITORS", "STAFF", "THE", "S"})

_LOOKUP = {syn: name for name, syns in FACILITIES.items() for syn in syns}
_TAIL = re.compile(r"\s*(?:\(.*|@.*|\bVIA\b.*|\bTO\b.*|\bFOR\b.*)$")
_LABEL = re.compile(r"^(?P<head>.+?)\s+(?P<label>[A-Z]|\d{1,2}|[A-Z]\d{1,2})$")


def lookup(text: str) -> Optional[tuple[str, Optional[str]]]:
    """Normalized (upper-case) entry text -> (facility, label) or None.

    "PASSENGER LIFTS" -> ("elevator", None); "EXIT A" -> ("exit", "A");
    "LIFT TO PLATFORM" -> ("elevator", None); "CAR PICKUP" -> None."""
    core = _TAIL.sub("", text).strip()
    if not core:
        return None
    for candidate, label in _with_label(core):
        words = candidate.split()
        kept = [w for w in words if w not in MODIFIERS] or words
        for phrase in (" ".join(kept), candidate):
            if phrase in _LOOKUP:
                name = _LOOKUP[phrase]
                if label is None and name in DIRECTED:   # UP vs DOWN escalator are different things
                    label = next((w for w in words if w in ("UP", "DOWN")), None)
                return name, label
    return None


def _with_label(core: str):
    yield core, None
    m = _LABEL.match(core)
    if m:
        yield m.group("head"), m.group("label")
