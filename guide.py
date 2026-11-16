"""Goal-directed sign guidance (docs/step1-goal.md, section 3): for each OCR-confirmed sign
read, decide what to say. The user stays in charge: we say what signs say and where they are,
ask what they already know, and never override what they told us.

  g = Guide("014-190")
  g.start(t)                          # "Guiding to 014-190. Do you know anything about the way..."
  d = g.on_read(Read(sign_id=7, entries=[{"text": "014-150 to 014-199", "arrow": "right"}],
                     t=12.0, rel=-40.0, mount="wall", ocr_text="014-150 TO 014-199"))
  d.kind, d.text                      # "spoken", "Sign on your left: rooms 014-150 to 014-199, ..."
  g.on_command(commands.parse("read it"), t)

Decisions: spoken / arrival / asked / noted / read (on request) / repeat / note / question /
unclear / stop / nothing. Pure Python, the clock is passed in; no I/O (main.py logs
Decision.log()).
"""
import re
from dataclasses import dataclass, field
from typing import Optional

import commands
import direction
from matcher import Context, MatchLevel, SignMatch, match_sign
from roomid import rule_for_goal
from targets import BuildingRule, parse_target

ASK_TTL = 10.0          # a "Want to hear it?" question expires after this (s)
READ_GRACE = 30.0       # ...but "read it" still reads that sign this long after it was asked
NOTE_WINDOW = 30.0      # after the opening question, any reply counts as route knowledge
SAFETY_FACILITIES = frozenset({"exit", "stairs"})
COARSE_REPEAT_S = 20.0  # building / wing / floor: a new direction is re-said at most this often
# (successive signs are seen from different angles and arrows are read noisily)
_POSTED_PLAN = re.compile(r"\b(?:PLAN|EVACUATION|ASSEMBLY|FIRE ESCAPE)\b")


@dataclass
class Read:
    """One OCR-confirmed sign read. `rel`: where the sign is, degrees from the user's heading
    (+ = right); `mount`: "wall" / "facing" (direction.mounting)."""
    sign_id: int
    entries: list[dict]
    t: float
    rel: float = 0.0
    mount: str = "facing"
    ocr_text: Optional[str] = None
    heading: Optional[float] = None


@dataclass
class Decision:
    kind: str
    text: Optional[str]
    t: float
    sign_id: Optional[int] = None
    reason: str = ""
    match: Optional[SignMatch] = None

    def log(self) -> dict:
        m = self.match
        out = dict(decision=self.kind, said=self.text, t=self.t, sign=self.sign_id,
                   reason=self.reason)
        if m is not None:
            out.update(level=m.level.name, arrow=m.arrow, confidence=round(m.confidence, 3),
                       entry=m.entry.raw if m.entry else None, partial=m.partial,
                       ambiguous=m.ambiguous, arrived=m.arrived,
                       contradictions=[dict(entry=c.entry.raw, arrow=c.entry.arrow,
                                            kind=c.contradiction, weak=c.weak)
                                       for c in m.contradictions])
        return out


@dataclass
class Instruction:
    """What we last told the user about the goal: re-said only when it changes or gets more
    specific."""
    level: MatchLevel
    way: Optional[str]            # direction phrase in the user's frame ("to the left")
    sign_id: int
    t: float
    text: str


@dataclass
class _Asked:
    read: Read
    t: float


def _side(phrase: Optional[str]) -> Optional[str]:
    """'ahead to the left' -> 'left'; for comparing with what the user said."""
    if not phrase:
        return None
    p = phrase.lower()
    if "left" in p:
        return "left"
    if "right" in p:
        return "right"
    if "straight" in p or p.startswith("ahead"):
        return "ahead"
    if "behind" in p or "back" in p:
        return "back"
    return None


def _worded(raw: str) -> bool:
    """'BUILDING 14', 'LEVEL 2', 'BASEMENT' rather than a bare code like '14' or 'B1'."""
    return bool(re.search(r"[A-Z]{3,}", raw))


def _note_side(text: str) -> Optional[str]:
    """A clear direction in what the user said ("it's on the right", "I usually go left")."""
    words = set(commands._clean(text).split())
    sides = [s for s in ("left", "right") if s in words]
    if len(sides) == 1:
        return sides[0]
    if not sides and words & {"straight", "ahead"}:
        return "ahead"
    return None


@dataclass
class Guide:
    goal: str
    rule: Optional[BuildingRule] = None
    rule_override: Optional[str] = None
    context: Context = field(default_factory=Context)

    def __post_init__(self):
        if self.rule is None:
            self.rule = rule_for_goal(self.goal, self.rule_override)
        self.target = parse_target(self.goal, self.rule)
        self.current: Optional[Instruction] = None
        self.asked: Optional[_Asked] = None
        self.asked_signs: set[int] = set()
        self.noted: list[dict] = []          # for step 2 (topological memory)
        self.user_notes: list[dict] = []
        self.user_side: Optional[str] = None
        self.disagreement_said = False
        self.done = False
        self.arrived_text: Optional[str] = None
        self.question_t: Optional[float] = None

    # ------------------------------------------------------------ conversation

    def start(self, t: float) -> Decision:
        self.question_t = t
        return Decision("question", f"Guiding to {self._goal_words()}. Do you know anything about "
                        "the way, like which side it's on, or a route you usually take?", t)

    def expecting_note(self, t: float) -> bool:
        return self.question_t is not None and t - self.question_t <= NOTE_WINDOW

    def hear(self, text: str, t: float) -> Decision:
        """Recognised speech -> decision (parses with commands.parse)."""
        return self.on_command(commands.parse(text, self.expecting_note(t)), t)

    def on_command(self, cmd: commands.Command, t: float) -> Decision:
        if cmd.kind == "note":
            return self.add_user_note(cmd.text, t)
        if cmd.kind == "no":
            if self.expecting_note(t):
                self.question_t = None
                return Decision("note", "Okay.", t, reason="no route knowledge")
            if self.asked is not None:
                self.asked = None
                return Decision("nothing", "Okay.", t, reason="declined the sign")
            return Decision("nothing", None, t, reason="no")
        if cmd.kind == "stop":
            self.done = True
            self.asked = None
            return Decision("stop", "Stopped guiding.", t)
        if cmd.kind in ("read", "yes"):
            a = self.asked
            if a is not None and t - a.t <= READ_GRACE:
                self.asked = None
                text = direction.describe(a.read.entries, a.read.rel, a.read.mount)
                return Decision("read", text or "I couldn't make out that sign.", t,
                                a.read.sign_id, reason=f"{cmd.source} on a pending question")
            if cmd.kind == "yes":
                return Decision("nothing", None, t, reason="yes with nothing pending")
            return self._repeat(t, cmd.source)
        if cmd.kind == "repeat":
            return self._repeat(t, cmd.source)
        return Decision("unclear", "Sorry, I didn't catch that.", t, reason=f"heard {cmd.text!r}")

    def add_user_note(self, text: str, t: float) -> Decision:
        side = _note_side(text)
        self.user_notes.append(dict(t=t, text=text, side=side))
        self.question_t = None
        if side:
            self.user_side, self.disagreement_said = side, False
        return Decision("note", "Got it.", t, reason=f"user note (side: {side})")

    def _repeat(self, t: float, source: str) -> Decision:
        if self.done and self.arrived_text:
            return Decision("repeat", self.arrived_text, t, reason=source)
        if self.current is not None:
            return Decision("repeat", self.current.text, t, self.current.sign_id, reason=source)
        return Decision("nothing", "Nothing to read yet.", t, reason=source)

    # ------------------------------------------------------------ sign reads

    def on_read(self, read: Read) -> Decision:
        self._establish(read)
        m = match_sign(self.target, read.entries, self.rule, read.ocr_text, self.context)
        where = direction.where(read.rel)
        way = direction.arrow_phrase(m.arrow, read.rel, read.mount) if m.arrow else None

        if self.done:
            return self._note(read, m, "guidance finished")

        if m.arrived:
            text = self._arrival_text(m, where)
            if not m.partial:
                self.done, self.asked, self.arrived_text = True, None, text
                return Decision("arrival", text, read.t, read.sign_id, "goal's own plate", m)
            return self._speak(read, m, text, MatchLevel.RANGE, "here", "partial door plate")

        if m.level >= MatchLevel.SUBGOAL and m.ambiguous:
            ways = []
            for r in m.ties:
                p = direction.arrow_phrase(r.entry.arrow, read.rel, read.mount) or "no arrow"
                if p not in ways:
                    ways.append(p)
            text = (f"Sign {where} is unclear for {self._goal_words()}: one arrow says "
                    f"{ways[0]}, another says {ways[1] if len(ways) > 1 else 'something else'}.")
            return self._speak(read, m, text, m.level, "unclear", "matching arrows disagree")

        if m.level >= MatchLevel.SUBGOAL and way:
            text = self._guidance_text(m, where, way)
            return self._speak(read, m, text, m.level, way, "helps reach the goal")

        if m.level >= MatchLevel.SUBGOAL and m.locational and self._labels_here(m):
            text = self._here_text(m)
            if text:
                return self._speak(read, m, text, m.level, "here", "says where the user is")

        if self._has_content(read, m) and not self._is_safety(m) and read.sign_id not in self.asked_signs:
            if self.asked is not None and read.t - self.asked.t <= ASK_TTL:
                return self._note(read, m, "another question is pending")
            self.asked = _Asked(read, read.t)
            self.asked_signs.add(read.sign_id)
            return Decision("asked", f"Sign {where}. Want to hear it?", read.t, read.sign_id,
                            "content, nothing about the goal", m)
        reason = ("safety sign" if self._is_safety(m) else "door plate" if self._is_plate(m)
                  else "building directory" if self._is_building_directory(m)
                  else "posted plan" if self._is_posted_plan(m)
                  else "already asked" if read.sign_id in self.asked_signs else "little content")
        return self._note(read, m, reason)

    def _speak(self, read: Read, m: SignMatch, text: str, level: MatchLevel, way: Optional[str],
               reason: str) -> Decision:
        cur = self.current
        if cur is not None and level < cur.level and cur.way != "here":
            return self._note(read, m, f"less specific than the current instruction ({reason})")
        side = _side(way)
        if cur is not None and level == cur.level and (
                way == cur.way or read.sign_id == cur.sign_id
                or (side is not None and side == _side(cur.way))):
            return self._note(read, m, "same as the current instruction")
        if (cur is not None and level == cur.level and level <= MatchLevel.AREA
                and read.t - cur.t < COARSE_REPEAT_S):
            return self._note(read, m, "coarse direction changed again too soon (arrow noise?)")
        if self.user_side and side and side != self.user_side and not self.disagreement_said:
            text += f" You said {self._side_words(self.user_side)}; it's your call."
            self.disagreement_said = True
            reason += "; disagrees with the user's note"
        self.current = Instruction(level, way, read.sign_id, read.t, text)
        self.asked = None     # a new instruction takes priority over a pending question
        return Decision("spoken", text, read.t, read.sign_id, reason, m)

    def _note(self, read: Read, m: SignMatch, reason: str) -> Decision:
        self.noted.append(dict(t=read.t, sign=read.sign_id, rel=read.rel, heading=read.heading,
                               mount=read.mount, entries=read.entries, level=m.level.name,
                               reason=reason,
                               parsed=[r.entry.summary() for r in m.results]))
        return Decision("noted", None, read.t, read.sign_id, reason, m)

    # ------------------------------------------------------------ what's established

    def _establish(self, read: Read):
        """Building / wing / level / floor known from plates and signs that label this place,
        so partial plates ("190" for 014-190) can count fully."""
        from matcher import parse_sign
        parsed = parse_sign(read.entries, self.rule)
        if any(_POSTED_PLAN.search(e.raw) for e in parsed):
            return   # a posted map / evacuation plan names places it shows, not where you stand
        n_buildings = sum(e.type == "BUILDING" for e in parsed)
        for e in parsed:
            if not e.locational:
                continue
            if e.type == "BUILDING" and e.building is not None and n_buildings == 1 \
                    and _worded(e.raw):
                self.context.user_building = e.building
            if e.type == "AREA" and e.area and len(e.area[1]) <= 2:
                self.context.user_prefix = e.area[1]
            if e.type == "FLOOR" and len(e.floors) == 1 and _worded(e.raw):
                self.context.user_floor = e.floors[0]
            for r in e.rooms:
                if r.building is not None:
                    self.context.user_building = r.building
                if r.prefix:
                    self.context.user_prefix = r.prefix
                if r.section is not None:
                    self.context.user_section = r.section

    # ------------------------------------------------------------ classification

    @staticmethod
    def _labels_here(m: SignMatch) -> bool:
        """A sign labels this place only if it names one building / wing / floor in words
        ("Building 14", "Level 2"): a list of numbers without arrows ("14, 16, 18 thru 60")
        is a directory, a bare "14" or "B1" too little to go on, a posted plan names what
        it shows."""
        es = [r.entry for r in m.results]
        if any(_POSTED_PLAN.search(e.raw) for e in es):
            return False
        kinds = [e for e in es if e.type == m.entry.type]
        return len(kinds) == 1 and (m.entry.type == "AREA" or _worded(m.entry.raw))

    @staticmethod
    def _is_plate(m: SignMatch) -> bool:
        es = [r.entry for r in m.results]
        return bool(es) and all(e.locational and e.type in ("ROOM", "RANGE") for e in es)

    @staticmethod
    def _is_safety(m: SignMatch) -> bool:
        es = [r.entry for r in m.results]
        return bool(es) and all(
            (e.type == "FACILITY" and e.facility in SAFETY_FACILITIES)
            or "FIRE" in (e.name or e.raw) for e in es)

    @staticmethod
    def _is_posted_plan(m: SignMatch) -> bool:
        return any(_POSTED_PLAN.search(r.entry.raw) for r in m.results)

    @staticmethod
    def _is_building_directory(m: SignMatch) -> bool:
        """Mostly bare building numbers ("14, 16, 20, 22 ->"): when the goal's building isn't
        on it, the rest isn't worth a question (revisit after user tests)."""
        es = [r.entry for r in m.results]
        bare = sum(e.type == "BUILDING" and not _worded(e.raw) for e in es)
        return bool(es) and bare * 2 >= len(es)

    def _has_content(self, read: Read, m: SignMatch) -> bool:
        es = [r.entry for r in m.results if r.entry.type != "UNKNOWN"]
        if self._is_plate(m) or self._is_building_directory(m):
            return False
        if self._is_posted_plan(m):
            return False   # maps / evacuation plans: noted until we can read maps
        # An arrow, or a real list of labels (two bare words like "BIOL / UNIV" aren't).
        return any(not e.locational for e in es) or len(es) >= 3

    # ------------------------------------------------------------ wording (information, not orders)

    def _goal_words(self) -> str:
        return self.target.room_id if self.target.room is not None else self.goal

    @staticmethod
    def _side_words(side: str) -> str:
        return {"ahead": "it's straight ahead"}.get(side, f"it's to the {side}")

    def _arrival_text(self, m: SignMatch, where: str) -> str:
        if m.partial:
            return (f"Door plate {m.entry.rooms[0] if m.entry.rooms else m.entry.raw}, {where}. "
                    f"It may be {self._goal_words()}.")
        return f"You've arrived: {self._goal_words()}, {where}."

    def _guidance_text(self, m: SignMatch, where: str, way: str) -> str:
        e = m.entry
        goal = self._goal_words()
        if m.level == MatchLevel.EXACT:
            what = goal if not m.partial else (e.rooms[0].text if e.rooms else e.raw)
            tail = f" That's probably {goal}." if m.partial else ""
            return f"Sign {where}: {what}, {way}.{tail}"
        if m.level == MatchLevel.RANGE and e.ranges:
            lo, hi = e.ranges[0]
            return f"Sign {where}: rooms {lo} to {hi}, {way}. That includes {goal}."
        if m.level == MatchLevel.RANGE:
            return f"Sign {where}: {e.raw}, {way}. Close to {goal}."
        if m.level == MatchLevel.BUILDING:
            return f"Sign {where}: building {e.building}, {way}. {goal} is in building {e.building}."
        if m.level == MatchLevel.AREA:
            kind, value = e.area
            name = value if kind == "DIRECTION" else f"{kind.lower()} {value}"
            return f"Sign {where}: {name}, {way}."
        if m.level == MatchLevel.FLOOR:
            return f"Sign {where}: {e.source or e.raw}, {way}."
        if m.level == MatchLevel.SUBGOAL:
            floor = self.target.floor
            hint = "probably " if not self.target.floor_stated else ""
            return (f"Sign {where}: {e.facility if e.facility != 'elevator' else 'lift'}, {way}. "
                    f"{goal} is {hint}on floor {floor}.")
        return f"Sign {where}: {e.source or e.raw}, {way}."

    def _here_text(self, m: SignMatch) -> Optional[str]:
        e = m.entry
        if m.level == MatchLevel.BUILDING:
            return f"You're in building {e.building}."
        if m.level == MatchLevel.AREA:
            kind, value = e.area
            return f"You're in {value if kind == 'DIRECTION' else kind.lower() + ' ' + value}."
        if m.level == MatchLevel.FLOOR:
            return f"You're on {e.source or e.raw}."
        return None
