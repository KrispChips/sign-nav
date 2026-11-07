"""What the user asked for (the Target), and the building numbering rules used to read it.

  rule = load_rule("multi_building")            # config/buildings/multi_building.yaml
  target = parse_target("014-100", rule)        # building 14, room 100, floor 1

The only I/O in the parser/matcher modules is load_rule() reading a YAML file."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config" / "buildings"


@dataclass(frozen=True)
class RoomId:
    """One room ID. `key` holds the numeric groups (compared numerically, so 014 == 14),
    `suffix` the letter after the number (214A). `building` is None when the ID doesn't
    say (then it means "the current building")."""
    text: str
    key: tuple[int, ...]
    suffix: str = ""
    building: Optional[int] = None
    floor: Optional[int] = None

    def same_building(self, other: "RoomId") -> bool:
        return self.building is None or other.building is None or self.building == other.building

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class BuildingRule:
    name: str
    room_regex: re.Pattern
    key_groups: tuple[str, ...]
    id_format: str
    floor_group: Optional[str]
    leading_digits: dict[int, int]
    basement_group: Optional[str]
    ground_floor: int
    building_regexes: tuple[re.Pattern, ...]

    def find_rooms(self, text: str) -> list[tuple[RoomId, int, int]]:
        """All room IDs in normalized text, with their (start, end) spans."""
        return [(self._room(m), m.start(), m.end()) for m in self.room_regex.finditer(text)]

    def parse_room(self, text: str) -> Optional[RoomId]:
        m = self.room_regex.fullmatch(text.strip())
        return self._room(m) if m else None

    def _room(self, m: re.Match) -> RoomId:
        g = {k: v for k, v in m.groupdict().items()}
        basement = bool(self.basement_group and g.get(self.basement_group))
        key = tuple(int(g[k]) for k in self.key_groups)
        floor = self._floor(g, basement)
        building = int(g["building"]) if g.get("building") else None
        fmt = {k: (int(v) if v and v.isdigit() else (v or "")) for k, v in g.items()}
        return RoomId(self.id_format.format(**fmt), (int(basement),) + key,
                      g.get("suffix") or "", building, floor)

    def _floor(self, g: dict, basement: bool) -> Optional[int]:
        digits = g.get(self.floor_group) if self.floor_group else None
        if not digits:
            return None
        n = self.leading_digits.get(len(digits)) if self.leading_digits else len(digits)
        if n is None:
            return None
        value = int(digits[:n])
        return -max(1, value) if basement else value

    def find_building(self, text: str) -> Optional[tuple[int, int, int]]:
        """(building number, start, end) of the first building phrase in the text, or None."""
        for rx in self.building_regexes:
            m = rx.search(text)
            if m:
                return int(m.group("building")), m.start(), m.end()
        return None


def load_rule(name_or_path: str = "default") -> BuildingRule:
    """A building numbering config by name (config/buildings/<name>.yaml) or path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    floor = c.get("floor") or {}
    return BuildingRule(
        name=c.get("name", path.stem),
        room_regex=re.compile(c["room_regex"]),
        key_groups=tuple(c["key_groups"]),
        id_format=c["id_format"],
        floor_group=floor.get("group"),
        leading_digits={int(k): int(v) for k, v in (floor.get("leading_digits") or {}).items()},
        basement_group=floor.get("basement_group"),
        ground_floor=int(c.get("ground_floor", 1)),
        building_regexes=tuple(re.compile(r) for r in c.get("building_regexes", [])),
    )


@dataclass
class Target:
    """What the user is looking for. `kind` says which fields matter:
    room (room/room_id/floor/building), facility, area, building, floor, or named (aliases)."""
    raw: str
    kind: str
    room: Optional[RoomId] = None
    floor: Optional[int] = None
    building: Optional[int] = None
    area: Optional[tuple[str, str]] = None        # ("WING", "EAST"), ("ZONE", "B"), ...
    facility: Optional[str] = None
    facility_label: Optional[str] = None          # "A" in "EXIT A"
    aliases: list[str] = field(default_factory=list)   # names from the map, e.g. "RADIOLOGY"

    @property
    def room_id(self) -> Optional[str]:
        return self.room.text if self.room else None

    @property
    def room_value(self) -> Optional[tuple[int, ...]]:
        return self.room.key if self.room else None

    def describe(self) -> str:
        bits = [f"{self.kind} {self.raw!r}"]
        if self.room:
            bits.append(f"id {self.room.text}")
        if self.building is not None:
            bits.append(f"bldg {self.building}")
        if self.floor is not None:
            bits.append(f"floor {self.floor}")
        if self.area:
            bits.append(" ".join(self.area))
        return ", ".join(bits)


def parse_target(text: str, rule: BuildingRule, aliases: tuple[str, ...] = (),
                 area: Optional[tuple[str, str]] = None) -> Target:
    """'room 214', '014-100', 'restroom', 'zone B', 'radiology' -> Target.
    `aliases` and `area` come from the building map when it knows them."""
    from entries import parse_entry   # entries imports this module
    parts = parse_entry(text, "none", rule)
    e = parts[0] if len(parts) == 1 else None
    names = list(aliases)
    if e is not None and e.type == "ROOM" and len(e.rooms) == 1:
        r = e.rooms[0]
        return Target(text, "room", room=r, floor=r.floor, building=r.building, area=area,
                      aliases=names)
    if e is not None and e.type == "FACILITY":
        return Target(text, "facility", facility=e.facility, facility_label=e.facility_label,
                      aliases=names)
    if e is not None and e.type == "AREA":
        return Target(text, "area", area=e.area, aliases=names)
    if e is not None and e.type == "BUILDING":
        return Target(text, "building", building=e.building, aliases=names)
    if e is not None and e.type == "FLOOR" and len(e.floors) == 1:
        return Target(text, "floor", floor=e.floors[0], aliases=names)
    from entries import normalize
    name = normalize(text).text
    return Target(text, "named", aliases=[name] + [a for a in names if a != name])
