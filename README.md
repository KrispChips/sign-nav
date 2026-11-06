# Sign reading + intersection demo (iPhone now, Meta glasses later)

Pipeline, all on the PC:

```
frames + IMU ──► undistort ──► Grounding DINO ──► tracker (keeps BEST sighting per sign)
                                                       │  when good enough, or when sign is lost
                                                       ▼
            speech ◄── describe() ◄── arrow → user frame ◄── Gemma 4 E4B ◄── perspective-rectified crop

undistorted frame ──► Depth Anything V2 (ONNX) ──► farness profile ──► opening detector ──► camera side → user side ──► speech
```

| File | Role | Changes for glasses? |
|---|---|---|
| `sources.py` | frames, camera undistortion, IMU | **Yes, only this** (new frame generator + calibration file) |
| `detector.py` | open-vocab sign detection | No |
| `rectify.py` | quad fit, homography warp, true aspect from intrinsics, frontality | No |
| `tracker.py` | tracking + best-sighting selection | Maybe tune thresholds |
| `direction.py` | bearing, heading, wall-sign arrow mapping | No |
| `vlm.py` | Gemma via Ollama, JSON output | No |
| `signfilter.py` | OCR + CLIP check of each track's best crop before Gemma | No |
| `intersection.py` | depth model, farness profile, opening detector (from `geometric_intersection_detection.ipynb`) | No |
| `main.py` | the loop, TTS, debug logging | No |

## Setup (PC)

```bash
pip install -r requirements.txt
ollama pull gemma4:e4b          # ollama must be running
```
Install torch for your GPU first (https://pytorch.org/get-started/locally). RTX 50-series
(Blackwell) cards need a CUDA 12.8+ build, e.g. `--index-url https://download.pytorch.org/whl/cu130`;
older builds like cu126 load but fail on the first GPU op. The detector then falls back to CPU.

Grounding DINO tiny, CLIP ViT-B/32 and Depth Anything V2 small (int8 ONNX) download from
Hugging Face on first run. RapidOCR's models ship inside its wheel. A GPU helps a lot (CPU works, slowly).

## 1. Record test walks with the iPhone

- Settings > Camera > Formats > **Most Compatible** (H.264, so OpenCV can read it).
- Use the **0.5x ultra-wide** lens (closest to the glasses' field of view).
- Strap the phone to your head. Don't aim it at signs.
- Record IMU at the same time with any sensor-logging app, then convert to CSV with
  columns `t,yaw_deg` (or `t,gyro_z` in rad/s). Yaw must **increase when you turn right**;
  if not, pass `--gyro-sign -1`.
- To sync video and IMU, do a sharp head shake at the start. Find the time in each
  and pass the difference as `--imu-offset`.

## 2. Calibrate the ultra-wide (once)

Film a printed checkerboard, covering the frame edges and corners, at the same resolution you record at:
```bash
python calibrate.py --video board.mov --cols 9 --rows 6 --square 0.025 --out iphone_uw.npz
```
Without calibration the demo assumes `--hfov 100` with no distortion, which is fine for a first run
but makes bearings and rectification worse at the image edges.

## 3. Run

```bash
python main.py --source walk.mov --imu walk_imu.csv --calib iphone_uw.npz --show
```
Live, any stream OpenCV can open works, e.g. `--source rtsp://<phone-ip>:8554/live`, or the
webcam index if the iPhone shows up as a camera on your PC. Live IMU: send UDP JSON
`{"yaw_deg": 12.3}` to `--udp-imu 9001`.

## Distractor filter (lights, outlets, paintings, ...)

Three stages, cheapest first. Stages 2-3 run once per track, on its best rectified crop,
right before the Gemma call:

1. **Distractor prompts** (`detector.py`): phrases like "ceiling light", "electrical outlet",
   "painting" are appended to the Grounding DINO prompt, and boxes labelled with them are
   dropped before NMS. A merged label such as "sign picture frame" is kept: only labels with
   a distractor word and none of sign/plate/number are dropped. Add phrases to `DISTRACTORS`
   one at a time and check recall after each (`--no-distractor-prompts` to compare); phrases
   close to the targets, like "poster" or "door", can steal real signs.
2. **OCR** (`signfilter.OcrCheck`, RapidOCR = PP-OCR det+rec on ONNX Runtime): keep if lines
   with confidence >= 0.6 hold >= 2 letters/digits ("B2" passes, an outlet read as "O" doesn't).
   The detector resize limit is set to `max` 320 (default `min` 736 was ~10-90x slower on
   crops) and the angle classifier is off, since crops are already rectified.
3. **CLIP ViT-B/32** (`signfilter.ClipCheck`), only for OCR rejects: rescues symbol-only
   signs (restroom, exit, elevator, stairs pictograms). Keep if P(sign prompts) - P(distractor
   prompts) > `--clip-margin` (0). Prompt lists are at the top of `signfilter.py`.

A rejected track can be retried once its best quality beats the rejected one by 30%, at most
two attempts in total (a far, unreadable real sign gets a second look up close).

Tuning: run a few corridor walks, then go through `debug_out/rejected/` by hand.
Real signs there mean the filter is too strict: lower `--ocr-min-chars` or `--clip-margin`.
Distractors reaching Gemma (`track*.jpg`) mean add them to the prompt lists.
Recall matters more than precision: when in doubt, set the thresholds to keep.
`--no-clip` runs OCR only, `--no-filter` turns stages 2-3 off.

## Annotated video

```bash
python main.py --source walk.mp4 --hfov 69 --no-tts --out-video annotated.mp4
```
Left: the video with sign boxes (orange = still collecting sightings, green = sent to Gemma,
red = rejected by the filter),
the live left/center/right farness bars, and a red flash on each opening event.
Right: a chat log, newest at the bottom. Each sign card shows the crop Gemma saw, "reading..."
until it answers, then the spoken sentence and every entry with its arrow. Opening events get
their own cards. One output frame per processed frame, so it plays at real speed.
If ffmpeg is on PATH the file is re-encoded to H.264; otherwise it stays mp4v (VLC plays it).
Works together with `--show`, which then shows the same side-by-side view live.

## Debugging

Everything lands in `debug_out/`:
- `trackNNNN.jpg`: the exact crop sent to Gemma (check rectification here first)
- `reads.jsonl`: raw model output, parsed entries, quality, bearing, wall/facing decision, and what was spoken

The `--show` window draws tracks: orange = still collecting sightings, green = sent to the VLM.

- `filter.jsonl`: every filter decision: track, attempt, quality, keep, reason
  (`ocr:0 chars, clip:-0.41`), raw OCR lines with confidences, time taken
- `rejected/trackNNNN_aK_<reason>.jpg`: every crop the filter rejected
- `summary.json` (also printed at the end): boxes dropped by distractor labels, crops checked,
  kept by OCR, rejected by OCR, rescued by CLIP, rejected, sent to the VLM
- `intersections.jsonl`: per frame, raw and smoothed left/center/right farness, the 24-column
  profile, open/closed label, and for events the user-frame bearings and what was spoken
- `opening_NNNNN.Ns.jpg`: the frame at each opening event

Knobs: `--ready-q` (lower = read earlier but from worse views), `--box-thr` (detector
confidence), `--fps`. Wall vs facing thresholds are in `direction.mounting()`.

## Intersections

Openings on the left, right, or both are announced as e.g. *"Opening ahead on your left."*
Tune `--open-thr` / `--close-thr` (and the other `IntersectionDetector` settings) on your clips
the same way as in the notebook: the logged `ema` values are the notebook's smoothed L/C/R curves.
`--depth-long-side` trades depth quality for speed; `--no-intersections` turns it off.
Depth updates pause while the head is turned more than 25° from the walking direction,
since the image edges are then not the sides of the hallway.

## Moving to the glasses

Write a generator like `video_frames()` that yields `Frame(image, t, K, yaw_deg)` from the
glasses stream and the glasses IMU, recalibrate with `calibrate.py` on glasses footage,
and keep everything else. Hook `Announcer.say()` into your navigation stack's speech output.

## Known limits

- Heading = smoothed head yaw. If your nav stack has a better body heading, use it in `Heading`.
- Intersections: left/right/both openings only. No T vs four-way, no wall-ahead detection
  (relative depth always calls the farthest thing "far"). Open doors can look like openings.
- The head-turn pause only covers glances. A turn held longer than about a second becomes the
  new walking direction, so detection resumes.
- Up/down arrows on wall-mounted signs are ambiguous by convention; they're announced as such.
- Grounding DINO is slow per frame. Once it works, auto-label your recordings with it and
  train a small YOLO for the real-time detector.
