"""Reads a rectified sign crop with a local VLM (Gemma 4 E4B via Ollama on the PC)."""
import json

import cv2

PROMPT = """You are reading a sign for a blind pedestrian. The image is a cropped, perspective-corrected sign.
Return ONLY JSON in this form:
{"entries": [{"text": "...", "arrow": "left|right|up|down|up-left|up-right|down-left|down-right|none"}]}
Rules:
- One entry per destination, room number, or label on the sign.
- "arrow" is the direction of the arrow drawn next to that entry, exactly as it appears on the sign.
- Use "none" when the text labels THIS place (e.g. a room number or name on a door).
- Copy text exactly. Never guess text you cannot read. If nothing is legible, return {"entries": []}."""

ARROWS = {"left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right", "none"}


def _prep(crop, long_side=640):
    h, w = crop.shape[:2]
    s = long_side / max(h, w)
    if s > 1:  # upscale small crops; the vision encoder resizes anyway
        crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return crop


class SignReader:
    def __init__(self, model="gemma4:e4b", host=None):
        import ollama
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model

    def __call__(self, crop):
        ok, buf = cv2.imencode(".jpg", _prep(crop), [cv2.IMWRITE_JPEG_QUALITY, 92])
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": PROMPT, "images": [buf.tobytes()]}],
            format="json", options={"temperature": 0})
        raw = resp["message"]["content"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [], raw
        entries = []
        for e in data.get("entries", []) if isinstance(data, dict) else []:
            text = str(e.get("text", "")).strip()
            arrow = str(e.get("arrow", "none")).lower().replace(" ", "-")
            if text:
                entries.append({"text": text, "arrow": arrow if arrow in ARROWS else "none"})
        return entries, raw
