"""A stand-in for the phone app (android/), to test the PC side without the phone or glasses.

Serves the same three things the app does (docs/LIVE.md):
  http://<host>:8080/frames.mjpg   a video file played in real time as MJPEG (looped)
  tcp :9002                        prints every sentence main.py sends, as the phone would
                                   speak it (dropping ones that waited past their ttl_s)
  tcp :9003                        yaw lines {"yaw_deg": ...}: 0, or a slow sweep (--yaw-sweep)

Type to play the user (sent back on :9002 like the app does): a line is recognised speech
("read it", "I usually go left"); "!button", "!select", "!capture", "!nav left" are the
app's Read it button and glasses gestures.

  python tools/fake_phone.py walk.mp4
  python main.py --source http://localhost:8080/frames.mjpg --speak-to localhost:9002 \\
      --imu-from localhost:9003 --hfov 70 --no-tts --show
"""
import argparse
import json
import math
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class Latest:
    """The newest JPEG, with a counter so each client sends each frame once."""

    def __init__(self):
        self.jpeg, self.n, self.done = None, 0, False
        self.cond = threading.Condition()

    def finish(self):
        with self.cond:
            self.done = True
            self.cond.notify_all()

    def put(self, jpeg):
        with self.cond:
            self.jpeg, self.n = jpeg, self.n + 1
            self.cond.notify_all()

    def wait(self, seen, timeout=5.0):
        with self.cond:
            self.cond.wait_for(lambda: self.n != seen or self.done, timeout)
            return self.jpeg, self.n


def play(path, latest, fps, quality, loop=True, stop=None):
    """Reads `path` at real speed and publishes `fps` JPEGs per second."""
    while stop is None or not stop.is_set():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit(f"cannot open {path}")
        t_start, next_t = time.monotonic(), 0.0
        while stop is None or not stop.is_set():
            ok = cap.grab()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / fps
            ok, img = cap.retrieve()
            if not ok:
                break
            delay = t - (time.monotonic() - t_start)
            if delay > 0:
                time.sleep(delay)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            latest.put(buf.tobytes())
        cap.release()
        if not loop:
            latest.finish()       # streams end: like the phone app stopping
            return


def mjpeg_handler(latest):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/snapshot.jpg"):
                jpeg, _ = latest.wait(-1)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if not self.path.startswith("/frames.mjpg"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            print(f"[mjpeg] client {self.client_address[0]} connected", flush=True)
            seen = -1
            try:
                while True:
                    jpeg, seen2 = latest.wait(seen)
                    if seen2 == seen and latest.done:
                        break
                    if jpeg is None or seen2 == seen:
                        continue
                    seen = seen2
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                print(f"[mjpeg] client {self.client_address[0]} left", flush=True)
    return Handler


def event_line(typed: str) -> dict:
    """What the user typed -> the app's input event (docs/LIVE.md)."""
    typed = typed.strip()
    if typed.startswith("!"):
        name, _, arg = typed[1:].partition(" ")
        ev = {"event": name.strip().lower(), "t": round(time.time(), 3)}
        if arg.strip():
            ev["direction"] = arg.strip().lower()
        return ev
    return {"event": "speech", "text": typed, "t": round(time.time(), 3)}


class SpeechHandler(socketserver.StreamRequestHandler):
    """Prints sentences the way the phone would speak them: in order, ~0.08 s per character,
    dropping ones that waited in the queue past their ttl_s. `send()` writes input events back
    to every connected PC, as the app does."""
    spoken = []      # (kind, text) for tests
    writers = []

    @classmethod
    def send(cls, ev: dict):
        line = (json.dumps(ev) + "\n").encode("utf-8")
        for w in list(cls.writers):
            try:
                w.write(line)
                w.flush()
            except OSError:
                cls.writers.remove(w)

    def handle(self):
        SpeechHandler.writers.append(self.wfile)
        print(f"[speech] PC connected from {self.client_address[0]}", flush=True)
        try:
            self._read()
        except (ConnectionResetError, ConnectionAbortedError):
            pass   # the PC went away (main.py ended)
        finally:
            if self.wfile in SpeechHandler.writers:
                SpeechHandler.writers.remove(self.wfile)
        print("[speech] PC disconnected", flush=True)

    def _read(self):
        for raw in self.rfile:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            arrived = time.time()
            print(f"[speech] {msg.get('kind', '?'):5s} ttl {msg.get('ttl_s')}s: {msg.get('text')}",
                  flush=True)
            SpeechHandler.spoken.append((msg.get("kind"), msg.get("text"), arrived))


class YawHandler(socketserver.StreamRequestHandler):
    sweep = 0.0      # degrees amplitude of a slow head sweep (0 = straight ahead)

    def handle(self):
        print(f"[imu] PC connected from {self.client_address[0]}", flush=True)
        t0 = time.time()
        try:
            while True:
                t = time.time() - t0
                yaw = self.sweep * math.sin(2 * math.pi * t / 8.0)
                self.wfile.write((json.dumps({"yaw_deg": round(yaw, 2), "t": round(time.time(), 3)})
                                  + "\n").encode())
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            print("[imu] PC disconnected", flush=True)


class _TCP(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(video, host="0.0.0.0", http_port=8080, speech_port=9002, imu_port=9003, fps=10.0,
          quality=70, yaw_sweep=0.0, loop=True, stop=None):
    """Starts everything in background threads; returns the servers (for tests)."""
    latest = Latest()
    threading.Thread(target=play, args=(video, latest, fps, quality, loop, stop), daemon=True).start()
    httpd = ThreadingHTTPServer((host, http_port), mjpeg_handler(latest))
    httpd.daemon_threads = True
    YawHandler.sweep = yaw_sweep
    speech = _TCP((host, speech_port), SpeechHandler)
    imu = _TCP((host, imu_port), YawHandler)
    for srv in (httpd, speech, imu):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    return httpd, speech, imu


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--yaw-sweep", type=float, default=0.0, help="degrees of simulated head sweep")
    ap.add_argument("--once", action="store_true", help="don't loop the video")
    a = ap.parse_args()
    serve(a.video, fps=a.fps, quality=a.quality, yaw_sweep=a.yaw_sweep, loop=not a.once)
    print("fake phone: http://localhost:8080/frames.mjpg, speech :9002, yaw :9003 (Ctrl+C to stop)")
    print("type what the user says (or !button / !select / !capture / !nav left)")
    try:
        for line in sys.stdin:
            if line.strip():
                ev = event_line(line)
                SpeechHandler.send(ev)
                print(f"[you] {ev}", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
