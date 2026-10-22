"""Calibrate a camera from a video of a printed checkerboard.

  python calibrate.py --video board.mov --cols 9 --rows 6 --square 0.025 --out iphone_uw.npz

Move the board all over the frame, ESPECIALLY the edges and corners: that is where
side signs appear and where wide-angle distortion is worst. Re-run this for the glasses."""
import argparse

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
ap.add_argument("--square", type=float, default=0.025, help="square size in meters")
ap.add_argument("--every", type=int, default=10, help="use every Nth frame")
ap.add_argument("--out", default="calib.npz")
a = ap.parse_args()

objp = np.zeros((a.rows * a.cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
obj_pts, img_pts, size = [], [], None
cap, i = cv2.VideoCapture(a.video), 0
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
while True:
    ok, img = cap.read()
    if not ok:
        break
    i += 1
    if i % a.every:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    found, corners = cv2.findChessboardCorners(gray, (a.cols, a.rows))
    if found:
        obj_pts.append(objp)
        img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit))
print(f"boards found in {len(img_pts)} frames")
if len(img_pts) < 15:
    raise SystemExit("Need at least ~15 good views. Record a longer, slower video.")
# Rational model handles wide-angle lenses better than the default 5-coefficient model.
rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None,
                                         flags=cv2.CALIB_RATIONAL_MODEL)
print(f"RMS reprojection error: {rms:.3f}px (under ~0.8 is good)\nK=\n{K}")
np.savez(a.out, K=K, dist=dist, size=np.array(size))
print(f"saved {a.out}")
