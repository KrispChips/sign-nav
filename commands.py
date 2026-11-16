"""What the user said (recognised speech), a glasses gesture or the phone's button -> one
Command for guide.Guide (docs/step1-goal.md, section 4a). Simple word rules, no guessing:
anything that isn't a command or a plausible piece of route knowledge is "unknown"."""
import re
from dataclasses import dataclass
from typing import Optional

KINDS = ("read", "repeat", "stop", "yes", "no", "note", "unknown")


@dataclass(frozen=True)
class Command:
    kind: str
    text: str = ""          # the words, for notes and logs
    source: str = "voice"   # voice / gesture / button


def _clean(text: str) -> str:
    t = re.sub(r"[^a-z0-9' ]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


_READ = re.compile(r"^(?:please )?(?:read(?: it| that| the sign| me the sign)?|what does it say|"
                   r"what does the sign say|tell me more|go ahead)(?: please)?$")
_REPEAT = re.compile(r"^(?:what was that|say (?:it|that) again|repeat(?: that| it)?|"
                     r"come again|pardon|sorry)(?: please)?$")
_STOP = re.compile(r"^(?:stop|stop guiding|stop guidance|cancel|end guidance|i'?m done)(?: please)?$")
_YES = re.compile(r"^(?:yes|yeah|yep|sure|ok|okay)(?: please)?$")
_NO = re.compile(r"^(?:no|nope|no thanks|nothing|not really|i don'?t know|no idea|skip)$")
# Route knowledge: directions, layout and habit words. Only then is free speech kept as a note.
_NOTE_WORDS = re.compile(
    r"\b(?:left|right|straight|ahead|back|behind|north|south|east|west|upstairs|downstairs|"
    r"floor|level|lift|elevator|stairs|corridor|hallway|hall|door|entrance|usually|always|"
    r"normally|shortcut|shorter|route|way|side|end|next to|near|past|opposite|turn|go)\b")


def parse(text: str, expecting_note: bool = False) -> Command:
    """Recognised speech -> Command. `expecting_note`: the user was just asked what they
    know about the way, so any reply that isn't a command or a "no" is their answer."""
    t = _clean(text)
    if not t:
        return Command("unknown", "")
    for kind, rx in (("read", _READ), ("repeat", _REPEAT), ("stop", _STOP), ("yes", _YES),
                     ("no", _NO)):
        if rx.match(t):
            return Command(kind, t)
    if expecting_note or _NOTE_WORDS.search(t):
        return Command("note", str(text).strip())
    return Command("unknown", t)


def from_gesture(event: str, direction: Optional[str] = None) -> Command:
    """Glasses Inputs / phone button events (section 4): select or the button reads what is
    waiting (or repeats); a capture press repeats."""
    if event in ("select", "button", "read"):
        return Command("read", event, "gesture" if event != "button" else "button")
    if event == "capture":
        return Command("repeat", event, "gesture")
    return Command("unknown", f"{event} {direction or ''}".strip(), "gesture")
