"""Sign reading demo: frames -> detect -> track/best-sighting -> rectify -> Gemma -> speech,
plus hallway opening detection from monocular depth on the same frames.

  python main.py --source walk.mov --imu walk_imu.csv --show
  python main.py --source rtsp://<phone-ip>:8554/live --udp-imu 9001 --show
"""
import argparse
import json
import os
import queue
import re
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

    def say(self, text, dedupe=True):
        now = time.time()
        if dedupe and now - self.recent.get(text, -1e9) < self.cooldown:
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


def _legibility_label(m, d, model):
    """Short status for the video overlay: measured text height vs this angle's threshold."""
    h = "no text" if m.h_p25 is None else f"{m.h_p25:.0f}/{model.h_star(m.angle):.0f}px"
    if d.action == "approach" and d.scale_needed and d.scale_needed != float("inf"):
        if d.scale_needed <= 1.01:   # big enough, waiting for 2 of 3 sightings / lines to settle
            return f"{h} legible, confirming"
        return f"{h} closer x{d.scale_needed:.1f}"
    return f"{h} {d.action.replace('_', ' ')}"


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
    ap.add_argument("--no-filter", action="store_true", help="send every track to the VLM (no OCR/CLIP check)")
    ap.add_argument("--no-clip", action="store_true", help="OCR only: no CLIP rescue of symbol-only signs")
    ap.add_argument("--no-distractor-prompts", action="store_true",
                    help="detector prompt without distractor phrases (to compare recall)")
    ap.add_argument("--ocr-min-chars", type=int, default=2, help="letters/digits needed to keep a crop")
    ap.add_argument("--ocr-min-conf", type=float, default=0.6, help="OCR line confidence that counts")
    ap.add_argument("--clip-margin", type=float, default=0.0,
                    help="CLIP P(sign) - P(distractor) needed to rescue an OCR reject")
    ap.add_argument("--no-intersections", action="store_true", help="skip opening detection")
    ap.add_argument("--depth-model", help="ONNX depth model (default: download Depth Anything V2 small)")
    ap.add_argument("--depth-long-side", type=int, default=518, help="depth input size, multiple of 14")
    ap.add_argument("--eye-height", type=float, default=1.6,
                    help="camera height above the floor (m): sets the scale of all distances")
    ap.add_argument("--hall-min", type=float, default=1.4,
                    help="gap width (m) from which a side opening is a hallway, not an entrance")
    ap.add_argument("--speak-entrances", action="store_true",
                    help="also speak doorways (always shown in the video and logged)")
    ap.add_argument("--legibility", default="legibility_calib.json",
                    help="calibration from legibility.calibrate: read a sign when its text is legible")
    ap.add_argument("--no-legibility", action="store_true",
                    help="old behaviour: read each sign at its best-quality view")
    ap.add_argument("--debug-dir", default="debug_out")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out-video", help="write an annotated video (scene + chat log), e.g. annotated.mp4")
    ap.add_argument("--no-tts", action="store_true")
    a = ap.parse_args()

    from detector import DISTRACTORS, SignDetector   # heavy imports after arg parsing
    from vlm import SignReader
    from intersection import DepthEstimator, IntersectionWatcher, PlaceTracker, resolve_model
    from annotate import ChatLog, VideoOut, draw_scene

    imu = (ImuTrack(a.imu, a.imu_offset, a.gyro_sign) if a.imu else
           UdpImu(a.udp_imu) if a.udp_imu else None)
    camera = Camera(a.calib, a.hfov)
    detector = SignDetector(box_thr=a.box_thr,
                            distractors=() if a.no_distractor_prompts else DISTRACTORS)
    sign_filter = None
    if not a.no_filter:
        from signfilter import ClipCheck, OcrCheck, SignFilter
        sign_filter = SignFilter(OcrCheck(a.ocr_min_conf, a.ocr_min_chars),
                                 None if a.no_clip else ClipCheck(margin=a.clip_margin,
                                                                  device=detector.device))
    reader = SignReader(a.model, a.ollama_host)
    legibility = None
    if not a.no_legibility:
        if os.path.exists(a.legibility):
            from legibility.measure import measure
            from legibility.model import LegibilityModel
            from legibility.trigger import READ, ReadTrigger, cross_check
            from signfilter import ocr_engine
            legibility = LegibilityModel.load(a.legibility)
            legibility_engine = ocr_engine()
            triggers = {}
            print(f"[legibility] {a.legibility}: thresholds "
                  + ", ".join(f"{b.angle:.0f} deg {b.h_star:.0f} px" for b in legibility.bins))
        else:
            print(f"[legibility] {a.legibility} not found: reading each sign at its best view instead")
    tracker = Tracker(ready_q=a.ready_q, auto_ready=legibility is None)
    heading = Heading()
    speaker = Announcer(enabled=not a.no_tts)
    watcher = None if a.no_intersections else IntersectionWatcher(
        DepthEstimator(resolve_model(a.depth_model), long_side=a.depth_long_side),
        PlaceTracker(hall_min=a.hall_min), eye_height=a.eye_height,
        speak_entrances=a.speak_entrances)
    pool = ThreadPoolExecutor(max_workers=1)   # VLM never blocks the camera loop
    pending = {}
    os.makedirs(a.debug_dir, exist_ok=True)
    log = open(os.path.join(a.debug_dir, "reads.jsonl"), "a")
    ilog = open(os.path.join(a.debug_dir, "intersections.jsonl"), "a")
    flog = open(os.path.join(a.debug_dir, "filter.jsonl"), "a")
    llog = open(os.path.join(a.debug_dir, "legibility.jsonl"), "a")
    rejected_dir = os.path.join(a.debug_dir, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)
    sent_to_vlm = 0

    chat = ChatLog()
    video = VideoOut(a.out_video, a.fps) if a.out_video else None
    flash_until, flash = -1.0, None

    def handle_read(fut, job, t):
        tr, info, ocr_text = job["track"], job["info"], job["ocr"]
        try:
            entries, raw = fut.result()
        except Exception as e:  # noqa: BLE001
            print(f"[vlm] error on track {tr.id}: {e}")
            chat.sign_result(tr, t, None, [], None, error=e)
            if legibility is not None:
                triggers[tr.id].read_result(False)
            return
        mount = mounting(info["theta"], info["frontality"])
        rel = heading.relative(tr.last_theta, tr.last_yaw)   # where it is NOW-ish
        sentence = describe(entries, rel, mount)
        confirmed, outcome = True, None
        if legibility is not None:
            # Same check as the simulation: the returned printed text must be backed by OCR on
            # the same view. Unconfirmed reads aren't spoken; the trigger raises this sign's
            # threshold and reads again closer (or gives up after two).
            confirmed = cross_check(entries, ocr_text)
            outcome = triggers[tr.id].read_result(confirmed).action
        log.write(json.dumps(dict(track=tr.id, t=t, entries=entries, raw=raw,
                                  mount=mount, rel=rel, q=tr.best_q, info=info,
                                  attempt=tr.attempts, confirmed=confirmed, outcome=outcome,
                                  said=sentence if confirmed else None)) + "\n"); log.flush()
        chat.sign_result(tr, t, sentence if confirmed else None, entries, mount, confirmed=confirmed)
        if sentence and confirmed:
            speaker.say(sentence)

    def submit(tr, t, crop=None, info=None, ocr_text=""):
        """Filter, then read. `crop` defaults to the track's best view (no legibility)."""
        nonlocal sent_to_vlm
        crop = tr.best_crop if crop is None else crop
        info = tr.best_info if info is None else info
        check = None
        if sign_filter:
            keep, reason, ocr_text_f = sign_filter(crop)
            check = dict(keep=keep, reason=reason, ocr=ocr_text_f)
            flog.write(json.dumps(dict(track=tr.id, t=t, attempt=tr.attempts, q=tr.best_q,
                                       ms=round(sign_filter.last_ms, 1), **check)) + "\n")
            flog.flush()
            if not keep:
                name = re.sub(r"[^0-9A-Za-z.+-]+", "_", reason)   # also safe on Windows
                cv2.imwrite(os.path.join(rejected_dir, f"track{tr.id:04d}_a{tr.attempts}_{name}.jpg"),
                            crop)
                tracker.reject(tr, reason)
                chat.sign_filtered(tr, t, reason, ocr_text_f)
                if legibility is not None:
                    triggers[tr.id].read_result(False)   # not a sign (yet): counts as a failed read
                return
        sent_to_vlm += 1
        pending[pool.submit(reader, crop)] = {"track": tr, "info": info, "ocr": ocr_text}
        cv2.imwrite(os.path.join(a.debug_dir, f"track{tr.id:04d}.jpg"), crop)
        chat.sign_pending(tr, t, check["reason"] if check else None)

    frame = None
    for frame in video_frames(a.source, camera, imu, a.fps):
        heading.update(frame.yaw_deg, frame.t)
        t0 = time.time()
        if watcher:
            res, said = watcher(frame, heading)
            if res is not None:
                ilog.write(json.dumps(dict(t=frame.t, yaw=frame.yaw_deg, heading=heading.value,
                                           said=[x for x, _ in said], **res)) + "\n")
                ilog.flush()
            for sentence, event in said:
                kind = event["kind"]
                cv2.imwrite(os.path.join(a.debug_dir, f"{kind}_{frame.t:07.1f}s.jpg"), frame.image)
                if sentence:
                    speaker.say(sentence, dedupe=False)   # the tracker announces each place once
                chat.place(frame.t, event, spoken=bool(sentence))
                flash_until, flash = frame.t + 1.0, event["sentence"].rstrip(".").upper()
        dets = detector(frame.image)
        for tr in tracker.update(frame, dets):
            submit(tr, frame.t)
        if legibility is not None:
            for tr in tracker.tracks:
                if tr.last_t != frame.t:          # not seen in this frame
                    continue
                m = measure(tr.last_crop, tr.last_info, legibility_engine)
                trig = triggers.setdefault(tr.id, ReadTrigger(legibility))
                d = trig.update(m)
                tr.legibility = _legibility_label(m, d, legibility)
                llog.write(json.dumps(dict(m.to_dict(), t=frame.t, track=tr.id, action=d.action,
                                           scale_needed=d.scale_needed, reason=d.reason)) + "\n")
                if d.action == READ:
                    tr.submitted, tr.attempts = True, tr.attempts + 1
                    submit(tr, frame.t, tr.last_crop, tr.last_info, m.ocr_text)
            llog.flush()

        for fut in [f for f in pending if f.done()]:
            handle_read(fut, pending.pop(fut), frame.t)

        if a.show or video:
            info = f"t={frame.t:6.1f}s  {1000*(time.time()-t0):.0f}ms/frame  reading={len(pending)}"
            if frame.yaw_deg is not None:
                info += f"  yaw={frame.yaw_deg:+.0f} heading={heading.value:+.0f}"
            if watcher and watcher.last is None:
                info += "  depth paused: head turned"
            elif watcher and not watcher.last.get("ok", True):
                info += f"  depth: {watcher.last['reason']}"
            scene = draw_scene(frame.image, tracker.tracks, watcher.last if watcher else None,
                               frame.t, info, flash if frame.t <= flash_until else None)
            vis = video.compose(scene, chat) if video else scene
            if video:
                video.write(vis)
            if a.show:
                s = min(1.0, 1600 / vis.shape[1])
                cv2.imshow("signs", cv2.resize(vis, None, fx=s, fy=s) if s < 1 else vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    for tr in tracker.flush():   # signs still in view when the stream ended (no legibility)
        submit(tr, frame.t if frame else 0.0)
    for fut, job in list(pending.items()):   # wait for outstanding reads
        handle_read(fut, job, frame.t if frame else 0.0)
    if video and frame is not None:   # hold the last frame so late reads are visible
        info = f"t={frame.t:6.1f}s  end of video"
        scene = draw_scene(frame.image, [], None, frame.t, info)
        for _ in range(int(3 * a.fps)):
            video.write(video.compose(scene, chat))
    if video:
        video.close()
    summary = dict(boxes_dropped_by_distractor_label=detector.dropped, sent_to_vlm=sent_to_vlm,
                   **({f"filter_{k}": v for k, v in sign_filter.stats.items()} if sign_filter else {}))
    with open(os.path.join(a.debug_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("[summary] " + "  ".join(f"{k}={v}" for k, v in summary.items()))
    while not speaker.q.empty():
        time.sleep(0.2)
    time.sleep(3.0)  # let the last sentence finish

if __name__ == "__main__":
    main()
