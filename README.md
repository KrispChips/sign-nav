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
| `targets.py`, `entries.py`, `matcher.py`, `facilities.py` | target parser + sign matcher (pure Python) | No |
| `config/buildings/*.yaml` | room numbering per building | Per building, not per device |
| `legibility/` | when is a sign legible enough to read, and how much closer to get | Recalibrate on glasses footage |
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

## Target parser and sign matcher

Given what the user asked for and what the VLM read off a sign: does this sign help, and how?
Pure Python (no I/O except loading a building config, no ML; `rapidfuzz` for fuzzy text).

```python
from targets import load_rule, parse_target
from matcher import Context, match_sign

rule = load_rule("multi_building")                 # config/buildings/multi_building.yaml
target = parse_target("014-100", rule)             # building 14, room 100, floor 1
m = match_sign(target, [{"text": "014-100 to 014-150", "arrow": "right"},
                        {"text": "Building 15", "arrow": "left"}],
               rule, ocr_text="014-100 TO 014-150 BUILDING 15", context=Context(user_floor=1))
m.level, m.arrow, m.confidence      # MatchLevel.RANGE, "right", 0.95
m.contradictions                    # "Building 15" <- left: rules that direction out
print(m.explain())
```

- **Targets**: rooms ("room 214", "014-100", "#04-03B"), facilities ("restroom", "exit A"),
  areas ("zone B"), buildings, floors, or names ("radiology", matched fuzzily against `aliases`).
- **Entries**: normalized (NFKC, dashes, filler words like ROOM/RM/# stripped when a number
  follows, OCR digit repair inside mostly-digit tokens only), split on `, & AND / +` (but not
  when every part is just a name, e.g. "CAR PICKUP/DROPOFF"), then typed in this order:
  RANGE, ROOM, BUILDING, FLOOR, AREA, FACILITY, NAMED, UNKNOWN. Room IDs are found with the
  building's regex before looking at connectors, so "014-100" is a room, not 14 to 100.
- **Levels** (best first): EXACT, RANGE, AREA, BUILDING, FLOOR, NAMED, SUBGOAL (elevator /
  stairs / escalator when the target is on another floor), NONE. **Contradictions** (a
  different building, a range on the target's floor without it, a different floor, another
  zone/wing, EXIT B for EXIT A) are listed separately for routing.
- **Combining**: best level wins; among ties the narrowest range / closest text is reported,
  and ties pointing different ways set `ambiguous`. A locational EXACT entry sets `arrived`,
  a locational BUILDING sets `inside_building`. If the matched text isn't in the OCR text
  (partial ratio < 80) the confidence is halved: get closer and re-read.
- **Arrows**: `entries.canonical_arrow()` maps dataset words to the VLM vocabulary
  (straight -> up, straight-left -> up-left, locational -> none).
- **Building configs**: `default` (214, 1204 = floor 12, B12 = floor -1), `multi_building`
  (014-100), `singapore_units` (#04-01, #B1-05). Add one YAML per building.

Tests: `python -m pytest`. Parser and matcher tables, 40 hand-written signs with expected
results (`tests/data/synthetic_signs.json`), pinned edge cases with the deferred ones skipped
and explained, and a dataset test on SiGNgapore2D ground truth. The GT isn't copied here (its
repo has no license file): clone https://github.com/AdaCompNUS/sign-understanding next to this
repo or set `SIGN_UNDERSTANDING_GT`; without it that test is skipped. Full report:
`python tests/sg_dataset.py`. Current numbers (205 signs, 1262 entries): UNKNOWN 1.3% (bare
"1"-"4", "A", "B"), 49/50 hand-labelled types right (the miss: "F01-11 TO F01-33", a unit
format no config covers yet), self-consistency 100% (1241 targets), 0 false EXACT/RANGE on 553
negatives. Self-consistency only shows the matcher finds an entry when asked for it; it is
not a measure of parsing accuracy (that's what the hand labels are for).

## Legibility estimator (`legibility/`)

Decides, per sighting, whether a sign's text is large enough to read now and, if not, how
much closer to get. Not wired into `main.py` yet: it needs a real `legibility_calib.json` first.

- `measure.py` (Part A): from the rectified crop, the OCR **detector** (no recognizer, same
  320 px settings as the filter) finds text lines. Line height = width at half maximum of the
  row-contrast profile inside each box, in **source pixels** (the crop is resampled to source
  scale first, so an enlarged crop gives the same number). `h_p25`, `h_med`, `n_lines`,
  `sign_px` (true shorter side), `angle` (from frontality), `sharpness`.
- `degrade.py` (Part B): plane rotation (0/30/50/70 deg), log-spaced area downscale, Gaussian +
  (half the time) motion blur, noise, JPEG 60-85; seeded per sample; output is a camera image +
  box + K that goes through the live `rectify -> measure` path.
- `calibrate.py` (Parts C-E): `run` degrades signs, reads each sample (cached by image + prompt
  hash, early stop after two zero-recall scales), writes `samples.jsonl`; `fit` writes
  `legibility_calib.json` + plots. Real-footage rows with the same fields can replace the
  synthetic ones later without touching the model.
- `model.py` (Part F): `is_legible`, `scale_needed`, `steps_closer(m, d_now, step_len=0.7)`,
  `blocked_by_angle`, with log interpolation between angle bins.
- `trigger.py` (Part G): median of 3, read when legible in 2 of 3 + sharp enough + not blocked;
  failed OCR cross-check -> threshold x1.3 and approach; 2 failures -> unreadable; max 3 reads.
- `simulate.py` (test 6): constant-speed approaches through the live path + trigger + reader.

Differences from the plan, found while building it:
- Line height is the half-maximum width inside each detector box, not the box itself. The
  box is padded and grows with blur: small blurred text read 35-40% too tall; half-maximum
  width tracks the truth within 10% from 8 px up on rendered signs.
- Boxes taller than 85% of the sign are dropped: with tiny text the detector sometimes
  returns one box around the whole sign, which made far signs look *more* legible.
- Thresholds are stored in live-measurement units (the accuracy curve is fitted on true
  height, then converted through the fitted live-vs-true relation), per angle.
- Two floors per angle: the detection floor (below it text often isn't found) and the
  accuracy floor (below it detected measurements stop being within 20%). The model trusts
  measurements above the accuracy floor; detected text below it counts as not legible.
- No-text fallback: a sign with no detected text is read as a pictogram only once it is as
  large as text signs are when their text gets detected (also fitted), so far-away text signs
  don't fire a premature read.
- Hallucination check uses the rate above each sign's full-resolution baseline: SiGNgapore2D
  only annotates navigational cues, so other real text on a sign isn't an invention.
- RapidOCR keeps `use_det/use_cls/use_rec` between calls on the same engine; every call in
  this repo now passes all three (regression test in `tests/test_measure.py`).

### Running the calibration (on the PC, needs Gemma via Ollama)

Download SiGNgapore2D (link in https://github.com/AdaCompNUS/sign-understanding, the
`dataset.zip` holds `images/` and the ground-truth JSON) and unzip it to `data\SiGNgapore2D`.

```powershell
python -m pip install -r requirements.txt
# ~80 text signs + all symbol-only signs, 8 scales, 4 angles; early stopping keeps it to
# roughly 1500 Gemma calls (about an hour on an RTX 5060 Ti). Reruns are served from the cache.
python -m legibility.calibrate run --gt data\SiGNgapore2D\gt_annotation.json --images data\SiGNgapore2D\images --reader gemma --out calib_runs\v1
python -m legibility.calibrate fit --samples calib_runs\v1\samples.jsonl --out legibility_calib.json
# test 6: 20 simulated approaches with Gemma
python -m legibility.simulate --calib legibility_calib.json --gt data\SiGNgapore2D\gt_annotation.json --images data\SiGNgapore2D\images --reader gemma --signs 20 --out sim_runs\v1
```
`fit` prints, per angle, Gemma's plateau (the share of text it reads even when big), the
threshold in true and live pixels, text accuracy above it, the hallucination rate there (and
its baseline), and arrow accuracy; plus both floors and the estimator error. Plots are saved
next to the JSON. `--target` is a fraction of the plateau (default 0.9: the threshold is where
getting closer stops paying off). The simulation reports final recall against the reader's
own ceiling at the closest view, for text and for text + arrow.

What the first real run (80 signs, 1615 Gemma reads) showed, and what changed because of it:
- SiGNgapore2D boxes are [centre x, centre y, w, h]; crops were offset until fixed.
- Gemma 4 E4B reads ~80-85% of annotated text even when it is large, so a fixed 90% target
  blocked every angle. Thresholds are now relative to the fitted plateau.
- Arrow errors (left/right swaps, "up" read as "right") don't shrink with size: ~60% of arrows
  right when the text is read. Legibility is fitted on text; arrows are reported separately
  and need their own fix (prompt / arrow handling), not a closer view.
- Most "hallucinations" were formatting ("TE 14"), partial reads or real unannotated text. An
  entry is now invented only if it is unrelated to every annotation.
- Live `h_p25` overstates legibility while small lines haven't been detected yet (the larger
  lines survive the distance): at the same measured height, text accuracy was 0.40-0.46 with
  lines still missing vs 0.56-0.65 with all lines visible. The trigger now waits until the
  line count stops growing, and the live-unit conversion uses only all-lines-visible samples.
  The estimator is still the weakest part (~39% median error with all lines visible).
- The "box taller than 85% of the sign" filter only applies to signs under 48 px: many
  dataset boxes are drawn tightly around one line, whose text really is sign-height.
- Thresholds from that run (text): 0 deg 23 px, 30 deg 29 px, 50 deg 33 px, 70 deg 58 px (true
  height); plateau 75-86%; hallucination at threshold 3-4% vs baseline 0.2-1.3%.
- Far away, almost every detection was not text (24 of 25 at scale <= 0.16, true text < 4 px)
  and measured as large text, firing failed reads: only lines the recognizer reads count now.
  The estimator error with all lines visible dropped from 38% to 28% median.

Simulated approaches with Gemma (20 reachable signs, 4 angles, walk + 4 close sightings):
1.05 reads per sign; text recall 67% vs the reader's 78% ceiling at the closest view (text +
arrow 50% vs 64%). The remaining text gap came from three signs, all read at full size:
two were Gemma's run-to-run variation at the same scale as the ceiling view, one failed the
OCR cross-check during the final close views and the walk ended before a second read. None
were premature. Not met: the plan's absolute targets (90% recall, estimator within 20%),
which this reader and this single-sighting estimator can't reach.

Dry run without Gemma or the dataset (OCR stand-in reader, rendered signs; output is marked
`dry_run` and is NOT a calibration): `--synthetic 12 --reader ocr` on `run`, then `fit` and
`simulate --synthetic 20 --reader ocr`. Here: 0.85 reads/sign, 79% final recall vs a 94%
reader ceiling, which says nothing about Gemma: the stand-in's cross-check can't fail.

## Reader prompts and arrows (`vlm.py`, `vlm_eval.py`)

From 3479 entries whose text Gemma read in the calibration run: arrows were right 62% of the
time. Diagonals are almost never reported (up-left right 2%, read as left 46%); labels of this
place such as floor markers "L2" get an invented arrow (20% read as up); left/right confusions
(18% / 9%). Signs whose entries share one arrow: 76%; signs whose entries point different
ways: 59%. Angle and distance barely matter, so this is the reader, not the pipeline. Answers
are unstable (only 18% of entries get the same arrow in every view) but correlated within a
sign, so majority votes help little (3 reads: 64.7% -> 68.4%).

`vlm.py` now has prompt versions: `v1` (original), `v2` (arrow glyph key, diagonals kept, one
arrow can serve several lines, "none" for labels of this place incl. floor markers), `v3`
(list the arrows first with where they are, then entries reference an arrow id). Compare:

```powershell
python vlm_eval.py --gt data\SiGNgapore2D\gt\gt_annotation.json --images data\SiGNgapore2D\images --prompts v1 v2 v3 --cache calib_runs\v1\cache
```
It reads each calibration sign at its closest view from two angles (the same views as the
calibration run, so v1 comes from its cache), and prints text recall, arrow accuracy (exact and
within 45 degrees), split by one-arrow vs mixed-arrow signs, straight / diagonal / "none", the
share of entries given the same arrow from both views, and invented entries per read. Add
`--model <ollama tag>` to try a larger model.

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
