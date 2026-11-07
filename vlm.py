"""Reads a rectified sign crop with a local VLM (Gemma 4 E4B via Ollama on the PC).

Prompt versions (compare them with `python vlm_eval.py`):
  v1      the original prompt.
  v2      arrow rules: arrows shown as glyphs, diagonals kept, one arrow can serve several
          lines, "none" for labels of this place (incl. floor markers like L2 / B1).
  v3      layout first: list the sign's arrows (direction + where they are), then give each
          entry the id of its arrow. Aimed at pairing text with the wrong arrow, the main
          arrow error on signs whose entries point different ways.
All versions return the same entries: [{"text", "arrow"}] in the step 1 arrow vocabulary."""
import json

import cv2

from entries import canonical_arrow

PROMPT = """You are reading a sign for a blind pedestrian. The image is a cropped, perspective-corrected sign.
Return ONLY JSON in this form:
{"entries": [{"text": "...", "arrow": "left|right|up|down|up-left|up-right|down-left|down-right|none"}]}
Rules:
- One entry per destination, room number, or label on the sign.
- "arrow" is the direction of the arrow drawn next to that entry, exactly as it appears on the sign.
- Use "none" when the text labels THIS place (e.g. a room number or name on a door).
- Copy text exactly. Never guess text you cannot read. If nothing is legible, return {"entries": []}."""

_ARROW_KEY = """Arrow directions, as drawn:
  ← left    → right    ↑ up    ↓ down
  ↖ up-left    ↗ up-right    ↙ down-left    ↘ down-right
A diagonal arrow is a diagonal: report up-left/up-right/down-left/down-right, not left or right."""

PROMPT_V2 = f"""You are reading a sign for a blind pedestrian. The image is a cropped, perspective-corrected sign.
Return ONLY JSON in this form:
{{"entries": [{{"text": "...", "arrow": "left|right|up|down|up-left|up-right|down-left|down-right|none"}}]}}
{_ARROW_KEY}
Rules:
- One entry per destination, room number, or label on the sign.
- "arrow" is the arrow that points the way to THAT entry. One arrow often serves several lines
  (a block of destinations next to one arrow, or a row with the arrow at its end): give each of
  those lines that arrow. Look at which side of the text the arrow is on and which row it is in.
- Use "none" only when no arrow belongs to the entry: room numbers or names on a door, "you are
  here", and floor markers such as "L2" or "B1" that say which level THIS is.
- Never invent an arrow, and never drop one: if an arrow belongs to the entry, report it.
- Copy text exactly. Never guess text you cannot read. If nothing is legible, return {{"entries": []}}."""

PROMPT_V3 = f"""You are reading a sign for a blind pedestrian. The image is a cropped, perspective-corrected sign.
First list every arrow drawn on the sign, then every entry with the arrow that belongs to it.
Return ONLY JSON in this form:
{{"arrows": [{{"id": "A", "direction": "left|right|up|down|up-left|up-right|down-left|down-right", "where": "e.g. left end of the top row"}}],
  "entries": [{{"text": "...", "arrow": "A"}}]}}
{_ARROW_KEY}
Rules:
- "arrows": one item per arrow symbol actually drawn, with its direction exactly as drawn.
- "entries": one item per destination, room number, or label. "arrow" is the id of the arrow that
  points the way to it. One arrow often serves several lines (a block of destinations next to it,
  or the row it ends): give each of those lines that arrow's id.
- Use "none" when no arrow belongs to the entry: room numbers or names on a door, "you are here",
  and floor markers such as "L2" or "B1" that say which level THIS is.
- Copy text exactly. Never guess text you cannot read. If nothing is legible, return
  {{"arrows": [], "entries": []}}."""

PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2, "v3": PROMPT_V3}
DEFAULT_PROMPT = "v1"
ARROWS = {"left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right", "none"}


def _prep(crop, long_side=640):
    h, w = crop.shape[:2]
    s = long_side / max(h, w)
    if s > 1:  # upscale small crops; the vision encoder resizes anyway
        crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return crop


def parse(raw, version="v1"):
    """Model output -> [{"text", "arrow"}]. Arrows may be ids (v3), words or glyphs."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    ids = {}
    for a in data.get("arrows", []) if isinstance(data.get("arrows"), list) else []:
        if isinstance(a, dict) and a.get("id") is not None:
            ids[str(a["id"]).strip().upper()] = canonical_arrow(str(a.get("direction", "")))
    entries = []
    for e in data.get("entries", []) if isinstance(data.get("entries"), list) else []:
        if not isinstance(e, dict):
            continue
        text = str(e.get("text", "")).strip()
        a = str(e.get("arrow", "none")).strip()
        arrow = ids.get(a.upper()) if a.upper() in ids else canonical_arrow(a)
        if text:
            entries.append({"text": text, "arrow": arrow if arrow in ARROWS else "none"})
    return entries


class SignReader:
    def __init__(self, model="gemma4:e4b", host=None, prompt=DEFAULT_PROMPT):
        import ollama
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model, self.version = model, prompt
        self.prompt = PROMPTS[prompt]

    def __call__(self, crop):
        ok, buf = cv2.imencode(".jpg", _prep(crop), [cv2.IMWRITE_JPEG_QUALITY, 92])
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": self.prompt, "images": [buf.tobytes()]}],
            format="json", options={"temperature": 0})
        raw = resp["message"]["content"]
        return parse(raw, self.version), raw
