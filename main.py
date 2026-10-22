"""Sign reading demo: frames -> detect -> track/best-sighting -> rectify -> Gemma -> speech.

  python main.py --source walk.mov --imu walk_imu.csv --show
  python main.py --source rtsp://<phone-ip>:8554/live --udp-imu 9001 --show
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from direction import Heading, describe, mounting
from sources import Camera, ImuTrack, UdpImu, video_frames
from tracker import Tracker


class Announcer:
    """Speaks in a background thread; skips repeats within `cooldown` seconds."""

    def __init__(self, enabled=True, cooldown=20.0):
        self.q, self.recent, self.cooldown = queue.Queue(), {}, cooldown
        self.enabled = enabled
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        now = time.time()
        if now - self.recent.get(text, -1e9) < self.cooldown:
            return
        self.recent[text] = now
        print(f"[SPEAK] {text}", flush=True)
        if self.enabled:
            self.q.put(text)

    def _run(self):
        engine = None
        if sys.platform != "darwin":
            try:
                import pyttsx3
                engine = pyttsx3.init()
            except Exception as e:  # noqa: BLE001
                print(f"[tts] disabled: {e}")
        while True:
            text = self.q.get()
            if sys.platform == "darwin":
                subprocess.run(["say", text])
            elif engine:
                engine.say(text); engine.runAndWait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, rtsp/http URL, or webcam index")
    ap.add_argument("--calib", help="calibration .npz from calibrate.py")
    ap.add_argument("--hfov", type=float, default=100.0, help="horizontal FOV if no calib")
    ap.add_argument("--imu", help="IMU CSV (t,yaw_deg or t,gyro_z)")
    ap.add_argument("--imu-offset", type=float, default=0.0, help="seconds added to IMU timestamps")
    ap.add_argument("--gyro-sign", type=float, default=1.0)
    ap.add_argument("--udp-imu", type=int, help="port for live IMU JSON packets")
    ap.add_argument("--fps", type=float, default=5.0, help="processing rate")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--ollama-host", default=None)
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--ready-q", type=float, default=0.5)
    ap.add_argument("--debug-dir", default="debug_out")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-tts", action="store_true")
    a = ap.parse_args()

    from detector import SignDetector   # heavy imports after arg parsing
    from vlm import SignReader

    imu = (ImuTrack(a.imu, a.imu_offset, a.gyro_sign) if a.imu else
           UdpImu(a.udp_imu) if a.udp_imu else None)
    camera = Camera(a.calib, a.hfov)
    detector = SignDetector(box_thr=a.box_thr)
    reader = SignReader(a.model, a.ollama_host)
    tracker = Tracker(ready_q=a.ready_q)
    heading = Heading()
    speaker = Announcer(enabled=not a.no_tts)
    pool = ThreadPoolExecutor(max_workers=1)   # VLM never blocks the camera loop
    pending = {}
    os.makedirs(a.debug_dir, exist_ok=True)
    log = open(os.path.join(a.debug_dir, "reads.jsonl"), "a")

    for frame in video_frames(a.source, camera, imu, a.fps):
        heading.update(frame.yaw_deg, frame.t)
        t0 = time.time()
        dets = detector(frame.image)
        for tr in tracker.update(frame, dets):
            pending[pool.submit(reader, tr.best_crop)] = tr
            cv2.imwrite(os.path.join(a.debug_dir, f"track{tr.id:04d}.jpg"), tr.best_crop)

        for fut in [f for f in pending if f.done()]:
            tr = pending.pop(fut)
            try:
                entries, raw = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[vlm] error on track {tr.id}: {e}")
                continue
            mount = mounting(tr.best_info["theta"], tr.best_info["frontality"])
            rel = heading.relative(tr.last_theta, tr.last_yaw)   # where it is NOW-ish
            sentence = describe(entries, rel, mount)
            log.write(json.dumps(dict(track=tr.id, t=frame.t, entries=entries, raw=raw,
                                      mount=mount, rel=rel, q=tr.best_q, info=tr.best_info,
                                      said=sentence)) + "\n"); log.flush()
            if sentence:
                speaker.say(sentence)

        if a.show:
            vis = frame.image.copy()
            for t in tracker.tracks:
                x1, y1, x2, y2 = map(int, t.box)
                col = (0, 200, 0) if t.submitted else (0, 165, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                cv2.putText(vis, f"#{t.id} q={t.best_q:.2f} th={t.last_theta:+.0f}",
                            (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            info = f"t={frame.t:6.1f}s det={1000*(time.time()-t0):.0f}ms pending={len(pending)}"
            if frame.yaw_deg is not None:
                info += f" yaw={frame.yaw_deg:+.0f} heading={heading.value:+.0f}"
            cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            h, w = vis.shape[:2]
            s = 960 / max(h, w)
            cv2.imshow("signs", cv2.resize(vis, (int(w * s), int(h * s))))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    for tr in tracker.flush():   # signs still in view when the stream ended
        pending[pool.submit(reader, tr.best_crop)] = tr
        cv2.imwrite(os.path.join(a.debug_dir, f"track{tr.id:04d}.jpg"), tr.best_crop)
    for fut, tr in list(pending.items()):   # flush outstanding reads at end of file
        entries, raw = fut.result()
        s = describe(entries, heading.relative(tr.last_theta, tr.last_yaw),
                     mounting(tr.best_info["theta"], tr.best_info["frontality"]))
        if s:
            speaker.say(s)
    while not speaker.q.empty():
        time.sleep(0.2)
    time.sleep(3.0)  # let the last sentence finish


if __name__ == "__main__":
    main()
