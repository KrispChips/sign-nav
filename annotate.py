"""Annotated output video: the scene with detections on the left, a chat log of what the
system read and said on the right. One output frame per processed frame, so the video
plays at real speed when written at the processing fps.

Text in the panel is drawn with PIL, since cv2.putText can't draw non-ASCII sign text."""
import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ORANGE, GREEN, RED, WHITE = (0, 165, 255), (0, 200, 0), (0, 0, 255), (255, 255, 255)
# Panel colors are RGB (PIL).
BG, CARD, MUTED, TEXT = (24, 24, 28), (40, 42, 48), (150, 150, 160), (235, 235, 240)
SIGN_COL, OPEN_COL, WAIT_COL, ERR_COL = (110, 200, 120), (90, 170, 255), (240, 170, 60), (240, 90, 90)


def _font(size, bold=False):
    names = (["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold else
             ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"])
    for n in names + ["/usr/share/fonts/truetype/dejavu/" + names[-1]]:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:   # Pillow < 10.1
        return ImageFont.load_default()


class ChatLog:
    """Messages for the right-hand panel. Sign cards are updated in place when Gemma answers."""

    def __init__(self):
        self.items, self._by_track = [], {}

    def sign_pending(self, tr, t, check=None):
        title = f"Sign #{tr.id}" + (f" (retry {tr.attempts})" if tr.attempts > 1 else "")
        item = dict(t=t, color=WAIT_COL, title=title, status="reading...",
                    lines=[], thumb=tr.best_crop.copy(), check=check)
        self._by_track[tr.id] = item
        self.items.append(item)

    def sign_filtered(self, tr, t, reason, ocr_text):
        """Not sent to Gemma. Kept small: there can be many of these."""
        self.items.append(dict(t=t, color=MUTED, title=f"Filtered #{tr.id}", status=reason,
                               lines=[f"OCR: {ocr_text}"] if ocr_text else [],
                               thumb=tr.best_crop.copy(), thumb_h=40))

    def sign_result(self, tr, t, sentence, entries, mount, error=None, confirmed=True):
        item = self._by_track.get(tr.id)
        if item is None:
            return
        item["t_done"] = t
        if error:
            item.update(color=ERR_COL, status="read failed", lines=[str(error)[:200]])
            return
        item.update(color=SIGN_COL if entries and confirmed else MUTED,
                    status=(f"read, {mount}" if confirmed else "not confirmed by OCR: not spoken")
                    if entries else "nothing legible")
        item["lines"] = ([f"“{sentence}”"] if sentence else []) + (
            [f"filter: {item['check']}"] if item.get("check") else []) + [
            f"• {e['text']}" + ("" if e["arrow"] == "none" else f"  (arrow: {e['arrow']})")
            for e in entries]

    def opening(self, t, sentence, res):
        e = res["ema"]
        self.items.append(dict(t=t, color=OPEN_COL, title="Intersection",
                               status=res["event"].replace("_", " "), thumb=None,
                               lines=[f"“{sentence}”",
                                      f"L {e['left']:.2f}   C {e['center']:.2f}   R {e['right']:.2f}"]))


def _wrap(draw, text, font, width):
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            cand = word if not line else line + " " + word
            if draw.textlength(cand, font=font) <= width or not line:
                line = cand
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def render_panel(log, width, height):
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    f_head, f_body, f_small = _font(20, bold=True), _font(17), _font(14)
    d.text((16, 12), "What the system read and said", font=f_head, fill=TEXT)
    top, pad, inner = 48, 12, width - 2 * 16 - 2 * 12
    y = height - 10
    for item in reversed(log.items):   # newest at the bottom, older ones scroll off the top
        lines = [ln for s in item["lines"] for ln in _wrap(d, s, f_body, inner)]
        thumb = None
        if item.get("thumb") is not None:
            th = item["thumb"]
            s = min(inner / th.shape[1], item.get("thumb_h", 90) / th.shape[0], 2.0)
            thumb = Image.fromarray(cv2.cvtColor(
                cv2.resize(th, (max(1, int(th.shape[1] * s)), max(1, int(th.shape[0] * s)))),
                cv2.COLOR_BGR2RGB))
        h = pad + 22 + (thumb.height + 8 if thumb else 0) + 22 * len(lines) + pad
        if y - h < top:
            break
        y0 = y - h
        d.rounded_rectangle((16, y0, width - 16, y), radius=10, fill=CARD)
        d.rectangle((16, y0 + 8, 20, y - 8), fill=item["color"])
        cy = y0 + pad
        d.text((16 + pad, cy), f"{item['t']:6.1f}s  {item['title']}", font=f_head, fill=item["color"])
        d.text((width - 16 - pad - d.textlength(item["status"], font=f_small), cy + 4),
               item["status"], font=f_small, fill=MUTED)
        cy += 22
        if thumb:
            img.paste(thumb, (16 + pad, cy + 4))
            cy += thumb.height + 8
        for ln in lines:
            d.text((16 + pad, cy), ln, font=f_body, fill=TEXT)
            cy += 22
        y = y0 - 8
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_scene(image, tracks, inter, t, info, flash=None, height=720):
    """Frame scaled to `height` with track boxes, intersection scores and a status line."""
    s = height / image.shape[0]
    vis = cv2.resize(image, (int(round(image.shape[1] * s)), height), interpolation=cv2.INTER_AREA)
    for tr in tracks:
        x1, y1, x2, y2 = [int(v * s) for v in tr.box]
        rejected = not tr.submitted and tr.rejected_q is not None
        col = GREEN if tr.submitted else RED if rejected else ORANGE
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        status = getattr(tr, "legibility", None) or f"q={tr.best_q:.2f}"
        cv2.putText(vis, f"#{tr.id} " + ("filtered" if rejected else status),
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    w = vis.shape[1]
    cv2.rectangle(vis, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(vis, info, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    if inter is not None:   # bottom strip: smoothed farness per side, like the notebook's plot
        cv2.rectangle(vis, (0, height - 34), (w, height), (0, 0, 0), -1)
        bw = (w - 40) // 3
        for i, side in enumerate(("left", "center", "right")):
            x0, v = 10 + i * (bw + 10), inter["ema"][side]
            cv2.rectangle(vis, (x0, height - 26), (x0 + bw, height - 8), (70, 70, 70), -1)
            cv2.rectangle(vis, (x0, height - 26), (x0 + int(bw * v), height - 8),
                          (0, 140, 255) if side != "center" else (170, 170, 170), -1)
            cv2.putText(vis, f"{side[0].upper()} {v:.2f}", (x0 + 4, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    if flash:
        cv2.rectangle(vis, (0, 0), (w - 1, height - 1), RED, 8)
        cv2.putText(vis, flash, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2, cv2.LINE_AA)
    return vis


class VideoOut:
    """Writes scene|panel frames. Re-encodes to H.264 with ffmpeg if it is on PATH."""

    def __init__(self, path, fps, height=720, panel_w=560):
        self.path, self.fps, self.height, self.panel_w = path, fps, height, panel_w
        self.raw = os.path.splitext(path)[0] + "_raw.mp4"
        self.writer, self.size, self.last = None, None, None

    def compose(self, scene, log):
        panel = render_panel(log, self.panel_w, self.height)
        out = np.hstack([scene, panel])
        if self.size is None:
            self.size = (out.shape[1] - out.shape[1] % 2, self.height - self.height % 2)
        return cv2.resize(out, self.size) if (out.shape[1], out.shape[0]) != self.size else out

    def write(self, frame):
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.raw, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        self.writer.write(frame)
        self.last = frame

    def close(self):
        if self.writer is None:
            return
        self.writer.release()
        if shutil.which("ffmpeg"):
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.raw, "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "20", self.path])
            if r.returncode == 0:
                os.remove(self.raw)
                print(f"[video] wrote {self.path}")
                return
        os.replace(self.raw, self.path)
        print(f"[video] wrote {self.path} (mp4v; install ffmpeg for a more widely playable H.264 file)")
