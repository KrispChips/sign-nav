"""Tune the hallway / entrance / open-area detector on a recorded walk without the video.

1. On the PC (runs the depth model once, ~1 min for a 90 s walk):
     python intersection_replay.py dump --source walk.mp4 --out walk_depth.npz
   walk_depth.npz holds small depth maps and thumbnails, ~15 MB for 90 s at 5 fps:
   small enough to share.

2. Anywhere, in seconds, no GPU or depth model needed:
     python intersection_replay.py replay walk_depth.npz --sheet events.jpg
     python intersection_replay.py replay walk_depth.npz --labels walk_labels.json
     python intersection_replay.py replay walk_depth.npz --video replay.mp4

Labels (times in seconds of the video, sides as the walker sees them):
  {"hallway":   [[31.5, "right"], [83.0, "both"]],
   "entrance":  [[40.0, "left"]],
   "open_area": [[0, 12], [64, 70]],        <- spans [start, end]: being in/next to one
   "open_side": [[65.0, "right"]]}
An event counts as correct when its kind and side match a label within --tol seconds
(open-area events: anywhere in, or up to --tol before, a labelled span)."""
import argparse
import json
import sys

import cv2
import numpy as np

from intersection import IntersectionWatcher, PlaceTracker, phrase


def dump(a):
    from intersection import DepthEstimator, resolve_model
    from sources import Camera, video_frames
    est = DepthEstimator(resolve_model(a.depth_model), long_side=a.depth_long_side)
    cam = Camera(a.calib, a.hfov)
    invs, ts, thumbs, K, size = [], [], [], None, None
    for f in video_frames(a.source, cam, None, a.fps):
        inv = est(f.image)
        h, w = inv.shape
        s = a.depth_width / w
        invs.append(cv2.resize(inv, (a.depth_width, max(8, int(round(h * s)))),
                               interpolation=cv2.INTER_AREA).astype(np.float16))
        ts.append(f.t)
        th = cv2.resize(f.image, (int(round(f.image.shape[1] * a.thumb / f.image.shape[0])), a.thumb),
                        interpolation=cv2.INTER_AREA)
        thumbs.append(cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 70])[1])
        K, size = f.K, (f.image.shape[1], f.image.shape[0])
        if len(ts) % 50 == 0:
            print(f"  {f.t:6.1f} s")
    offs = np.cumsum([0] + [len(b) for b in thumbs])
    np.savez_compressed(a.out, inv=np.stack(invs), t=np.array(ts), K=K, size=np.array(size),
                        thumbs=np.concatenate(thumbs).astype(np.uint8), thumb_offsets=offs,
                        meta=json.dumps({"source": str(a.source), "fps": a.fps, "hfov": a.hfov,
                                         "calib": a.calib}))
    print(f"wrote {a.out}: {len(ts)} frames")


class Dump:
    def __init__(self, path):
        d = np.load(path)
        self.inv, self.t, self.K = d["inv"], d["t"], d["K"]
        self.size = tuple(int(v) for v in d["size"])
        self._thumbs, self._offs = d["thumbs"], d["thumb_offsets"]
        self.meta = json.loads(str(d["meta"]))

    def __len__(self):
        return len(self.t)

    def thumb(self, i):
        return cv2.imdecode(self._thumbs[self._offs[i]:self._offs[i + 1]], cv2.IMREAD_COLOR)


def run(dump_, eye_height=1.6, hall_min=1.4, quiet=False):
    """-> (per-frame results, events). Events carry their frame index and time."""
    w = IntersectionWatcher(None, PlaceTracker(hall_min=hall_min), eye_height=eye_height)
    results, events = [], []
    for i in range(len(dump_)):
        t = float(dump_.t[i])
        res, ev = w.process(dump_.inv[i].astype(np.float32), dump_.K, dump_.size, t)
        results.append(res)
        for e in ev:
            e.update(t=t, frame=i, sentence=phrase(e))
            events.append(e)
            if not quiet:
                print(f"{t:7.1f}s  {e['kind']:16s} {e.get('side') or '':6s} "
                      + (f"w={e['width']:.1f}m " if e.get("width") else "")
                      + ("(not spoken) " if e.get("quiet") else "") + e["sentence"])
    return results, events


def _kind(e):
    return {"open_ahead": "open_area", "leave_open_area": None}.get(e["kind"], e["kind"])


def score(events, labels, tol=3.0):
    """Per kind: true positives, false positives, missed labels."""
    out, used = {}, set()
    for e in events:
        k = _kind(e)
        if k is None:
            continue
        hit = None
        for j, lab in enumerate(labels.get(k, [])):
            if (k, j) in used:
                continue
            if k == "open_area":
                ok = lab[0] - tol <= e["t"] <= lab[1]
            else:
                ok = abs(e["t"] - lab[0]) <= tol and (
                    lab[1] == e.get("side") or "both" in (lab[1], e.get("side")))
            if ok:
                hit = j
                break
        s = out.setdefault(k, {"tp": 0, "fp": [], "missed": []})
        if hit is None:
            s["fp"].append(round(e["t"], 1))
        else:
            s["tp"] += 1
            used.add((k, hit))
    for k, labs in labels.items():
        s = out.setdefault(k, {"tp": 0, "fp": [], "missed": []})
        s["missed"] = [lab for j, lab in enumerate(labs) if (k, j) not in used]
    return out


def sheet(dump_, results, events, path, per_row=5):
    from annotate import draw_topdown
    tiles = []
    for e in events:
        img = dump_.thumb(e["frame"])
        img = cv2.resize(img, (int(img.shape[1] * 320 / img.shape[0]), 320))
        if results[e["frame"]].get("ok"):
            draw_topdown(img, results[e["frame"]], size=(110, 140))
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, f"{e['t']:.1f}s {e['kind']} {e.get('side') or ''}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        if e.get("width"):
            cv2.putText(img, f"gap {e['width']:.1f} m", (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        print("no events")
        return
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 2, 2, 2, 2 + w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack((tiles[i:i + per_row] + [blank] * per_row)[:per_row])
            for i in range(0, len(tiles), per_row)]
    cv2.imwrite(path, np.vstack(rows))
    print(f"wrote {path}")


def video(dump_, results, events, path):
    from annotate import draw_topdown
    by_frame = {}
    for e in events:
        by_frame.setdefault(e["frame"], []).append(e)
    fps = float(dump_.meta.get("fps", 5.0))
    writer, flash, until = None, "", -1.0
    for i in range(len(dump_)):
        img = dump_.thumb(i)
        img = cv2.resize(img, (int(img.shape[1] * 480 / img.shape[0]) // 2 * 2, 480))
        if results[i].get("ok"):
            draw_topdown(img, results[i])
        for e in by_frame.get(i, []):
            flash, until = e["sentence"], dump_.t[i] + 2.0
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(img, f"{dump_.t[i]:.1f}s  " + (flash if dump_.t[i] <= until else ""), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer:
        writer.release()
        print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="run the depth model over a video and save the maps")
    d.add_argument("--source", required=True)
    d.add_argument("--out", default="walk_depth.npz")
    d.add_argument("--calib"); d.add_argument("--hfov", type=float, default=100.0)
    d.add_argument("--fps", type=float, default=5.0)
    d.add_argument("--depth-model"); d.add_argument("--depth-long-side", type=int, default=518)
    d.add_argument("--depth-width", type=int, default=96, help="stored depth map width")
    d.add_argument("--thumb", type=int, default=240, help="thumbnail height (px)")
    r = sub.add_parser("replay", help="run the detector over a dump")
    r.add_argument("dump")
    r.add_argument("--eye-height", type=float, default=1.6)
    r.add_argument("--hall-min", type=float, default=1.4)
    r.add_argument("--labels"); r.add_argument("--tol", type=float, default=3.0)
    r.add_argument("--sheet", help="contact sheet of the event frames (jpg)")
    r.add_argument("--video", help="thumbnails + bird's-eye inset + events (mp4)")
    r.add_argument("--json", help="write per-frame results and events")
    a = ap.parse_args(argv)
    if a.cmd == "dump":
        return dump(a)
    dm = Dump(a.dump)
    results, events = run(dm, a.eye_height, a.hall_min)
    ok = sum(r.get("ok", False) for r in results)
    print(f"{len(dm)} frames, scan ok in {ok} ({100 * ok / max(1, len(dm)):.0f}%)")
    if a.labels:
        with open(a.labels) as f:
            sc = score(events, json.load(f), a.tol)
        for k, s in sc.items():
            print(f"  {k:10s} correct {s['tp']}, false {s['fp']}, missed {s['missed']}")
    if a.sheet:
        sheet(dm, results, events, a.sheet)
    if a.video:
        video(dm, results, events, a.video)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"events": events, "frames": results}, f)


if __name__ == "__main__":
    sys.exit(main())
