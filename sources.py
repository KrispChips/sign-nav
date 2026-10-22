"""Frame sources: video/stream frames, camera undistortion, head yaw.

This is the ONLY file that changes for the glasses: write another generator that yields
Frame(image, t, K, yaw_deg) with `image` already undistorted and `K` its pinhole matrix."""
import csv
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray          # undistorted BGR
    t: float                   # seconds: video time for files, time since start for live
    K: np.ndarray              # 3x3 pinhole intrinsics of `image`
    yaw_deg: Optional[float]   # head yaw, increases when turning right; None without IMU


class Camera:
    """Undistorts frames and gives the matching pinhole K.

    With a calibration file, the calibrated K is rescaled if frames come at a different
    resolution of the same aspect. Without one, K comes from `hfov` (no distortion model).
    `alpha` is passed to getOptimalNewCameraMatrix: 0 crops to valid pixels only, 1 keeps
    every source pixel. 0.5 keeps most of the wide-angle edges, where side signs appear."""

    def __init__(self, calib=None, hfov=100.0, alpha=0.5):
        self.hfov, self.alpha = hfov, alpha
        self.calib = None
        if calib:
            d = np.load(calib)
            self.calib = (d["K"].astype(float), d["dist"], tuple(int(v) for v in d["size"]))
        self.K, self._size, self._maps = None, None, None

    def _setup(self, w, h):
        self._maps = None
        if self.calib is None:
            f = (w / 2) / np.tan(np.radians(self.hfov) / 2)
            self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
        else:
            K, dist, (cw, ch) = self.calib
            if (cw, ch) != (w, h):
                if abs(cw / ch - w / h) > 0.01:
                    raise SystemExit(f"calibrated at {cw}x{ch} but frames are {w}x{h}: different "
                                     "aspect/orientation. Recalibrate at the recording settings.")
                K = K.copy()
                K[:2] *= w / cw
            self.K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), self.alpha, (w, h))
            self._maps = cv2.initUndistortRectifyMap(K, dist, None, self.K, (w, h), cv2.CV_16SC2)
        self._size = (w, h)

    def __call__(self, img):
        h, w = img.shape[:2]
        if self._size != (w, h):
            self._setup(w, h)
        if self._maps is not None:
            img = cv2.remap(img, *self._maps, cv2.INTER_LINEAR)
        return img, self.K


class ImuTrack:
    """Recorded yaw from a CSV with columns t,yaw_deg or t,gyro_z (rad/s).

    `offset` is added to the CSV timestamps to line them up with video time.
    `gyro_sign` flips either column so that yaw increases when turning right."""

    def __init__(self, path, offset=0.0, gyro_sign=1.0):
        with open(path, newline="") as f:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        if not rows:
            raise SystemExit(f"{path} is empty")
        t = np.array([float(r["t"]) for r in rows])
        order = np.argsort(t)
        t = t[order]
        if "yaw_deg" in rows[0]:
            yaw = np.array([float(r["yaw_deg"]) for r in rows])[order]
            yaw = np.degrees(np.unwrap(np.radians(yaw)))   # no jumps at +-180
        elif "gyro_z" in rows[0]:
            g = np.array([float(r["gyro_z"]) for r in rows])[order]
            yaw = np.degrees(np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(t))]))
        else:
            raise SystemExit(f"{path} needs columns t,yaw_deg or t,gyro_z")
        self.t, self.yaw = t + offset, yaw * gyro_sign

    def yaw_at(self, t):
        return float(np.interp(t, self.t, self.yaw))


class UdpImu:
    """Live yaw from UDP JSON packets like {"yaw_deg": 12.3}. Returns the latest value."""

    def __init__(self, port, gyro_sign=1.0):
        self.yaw, self.sign = None, gyro_sign
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            data, _ = self.sock.recvfrom(4096)
            try:
                self.yaw = self.sign * float(json.loads(data)["yaw_deg"])
            except (ValueError, KeyError, TypeError):
                continue

    def yaw_at(self, t):  # live: always the latest packet
        return self.yaw


class _LatestFrame:
    """Reads a live stream in the background so we always process the newest frame
    instead of falling further behind the OpenCV buffer."""

    def __init__(self, cap):
        self.cap, self.img, self.n, self.done = cap, None, 0, False
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ok, img = self.cap.read()
            with self.cond:
                if not ok:
                    self.done = True
                else:
                    self.img, self.n = img, self.n + 1
                self.cond.notify_all()
            if not ok:
                return

    def next(self, seen):
        with self.cond:
            self.cond.wait_for(lambda: self.n != seen or self.done)
            return (None, seen) if self.n == seen else (self.img, self.n)


def video_frames(source, camera, imu=None, fps=5.0):
    """Yields undistorted Frames at about `fps` from a file, URL or webcam index."""
    src = int(source) if str(source).isdigit() else source
    live = not os.path.isfile(str(source))
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    step = 1.0 / fps

    def make(img, t):
        img, K = camera(img)
        yaw = imu.yaw_at(t) if imu is not None else None
        return Frame(img, t, K, yaw)

    if live:
        grab, seen, t0, last = _LatestFrame(cap), 0, time.monotonic(), -1e9
        while True:
            img, seen = grab.next(seen)
            if img is None:
                break
            t = time.monotonic() - t0
            if t - last < step:
                continue
            last = t
            yield make(img, t)
    else:
        next_t = 0.0
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t + 1e-6 < next_t:
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            next_t = max(next_t + step, t)
            yield make(img, t)
    cap.release()
