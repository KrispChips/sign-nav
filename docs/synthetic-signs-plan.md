# Synthetic sign simulator: plan and hand-off brief

This document is self-contained: it is meant to start a new Claude Code session (or a new
repo) on the synthetic sign simulator without any prior conversation. Section 1 gives the
context, section 2 the goal, sections 3-6 the interfaces, sections 7-9 the plan.

Status: planned, not started. Nothing here may be specialised to one building or one dataset:
the system must work in any building; datasets and venues named below are evaluation data.

---

## 1. Context: the system this feeds

**sign-nav** (github.com/KrispChips/sign-nav, Python) helps a blind pedestrian find a room in
an unfamiliar building using its signs. Live setup: Ray-Ban Meta glasses stream their camera
(through an Android app built on Meta's Wearables DAT SDK) to a PC with a GPU (RTX 5060 Ti,
16 GB), which detects signs, reads them, matches them against the user's goal and speaks
directions back through the glasses. It also detects hallways/entrances/open areas from depth.

Pipeline per frame (sign part):
1. **Detection**: Grounding DINO tiny (open-vocabulary prompts for signs + distractor prompts).
2. **Tracking + rectification**: tracks signs over frames, fits the sign quad, warps it to a
   front view (`rectify.py`).
3. **Filter**: OCR (RapidOCR) + CLIP check that the crop is a sign (`signfilter.py`).
4. **Legibility trigger**: reads a sign only once its text is big enough in pixels
   (`legibility/`: thresholds calibrated per viewing angle on real signs).
5. **Reader**: Gemma 4 E4B via Ollama (`vlm.py`, prompt versions v1-v5, default v4), thinking
   off (~0.2-0.5 s per read), JSON output, then an OCR cross-check.
6. **Matcher + guide**: goal parsing for any room-numbering convention, sign matching, what to
   say (`targets.py`, `matcher.py`, `guide.py`; spec in `docs/step1-goal.md`).

Camera facts that the simulator must reproduce:
- Glasses live stream: portrait, compressed HEVC, **720x1280 (HIGH) or 504x896 (MEDIUM)**,
  decoded and re-encoded as JPEG (quality ~70) for the PC. Recordings are 1440x1920.
- Horizontal field of view about **70 degrees** in portrait (not yet calibrated with a
  checkerboard; do that in Phase 4).
- Head-mounted: walking bob, glances sideways, signs often at the frame edge or cut off.

## 2. Why: where the reader fails today

Measured on real data:

| measurement | value |
|---|---|
| SiGNgapore2D (205 signs), Gemma v4: arrows exact / within 45 deg | 74.7% / 83.2% |
| same: text recall | 65.8% |
| same: annotated pictograms reported | ~26% |
| same: **diagonal arrows** | **~0-2%** (read as left/right/up), under every prompt tried |
| for comparison, Gemini 2.0 Flash with the NUS paper's prompt (their published outputs) | arrows ~75%, diagonal straight-right 36%, pictograms >=43% |
| SiGNgapore venue E4 sign parsing, goal-sign pairs, Gemma v4 | 52/89 |
| same, SignScene (Gemini 2.5 Pro parsing) per the SiGNgapore paper | 79/90 |

Error groups on E4: destinations not found (mostly pictogram-only destinations such as a taxi
symbol), escalator icons (a qualifier "via escalator", not a direction), forward read as
right, "locational" (no arrow) given when an arrow exists, and diagonals. Prompting fixed some
groups and broke others; **diagonals and pictograms did not respond to prompting at all**. That
is the motivation: perfectly labelled synthetic data for what the model cannot see.

## 3. Goal

Generate synthetic sign images (and walking sequences) realistic enough that training on them
improves sign detection and reading on **real photos from venues never seen**, with a
complete, exact label for every image. Secondary: release the generator and dataset (no public
training set for navigational sign reading exists).

Success is measured only on real, held-out data (section 6).

## 4. Repository

Build it in a **new repo** (e.g. `sign-sim`), not inside sign-nav:
- different dependencies (Blender, BlenderProc, training stacks) and size (assets, renders);
- released separately with its own licenses;
- its own writer, so sessions don't collide on sign-nav's `main`.

Connections to sign-nav:
1. **Label format** (section 5) matches what sign-nav's reader outputs.
2. **Evaluation on real photos lives in sign-nav** (it runs the real reader); the simulator
   repo calls it to score trained models.
3. **Trained weights** go out as releases (GitHub Releases / Hugging Face), loaded by path or
   version in sign-nav. Never commit weights, renders or datasets to git.

Suggested layout:
```
sign-sim/
  README.md            phases, how to run
  LICENSES.md          source + license of every asset (required for release)
  docs/label-format.md section 5 of this file, kept current
  assets/              icons (svg/png), fonts, materials, hdri  (large ones via Git LFS or downloaded by script)
  config/              sign_config.yaml, scene configs, style families
  signsim/             generator code (2D signs, layouts, degradation, labels)
  blender/             scene modules (Phase 3+)
  train/               arrow/pictogram classifiers, detector, LoRA
  eval/                glue to sign-nav's scorer
  renders/ datasets/ checkpoints/   git-ignored
```

## 5. Label format

One JSON per image (and per sign within it). Two layers: what is **drawn** (what the reader
must output) and what is **true in the scene** (for navigation research, Phase 3+).

Reader-facing (must match sign-nav's `vlm.parse` output exactly):
```json
{"entries": [
  {"text": "Lecture Theatres 1-4", "from": "text",   "arrow": "up-left"},
  {"text": "Toilets",              "from": "symbol", "arrow": "down-right", "via": null},
  {"text": "Food Court @ L3",      "from": "text",   "arrow": "up",         "via": "escalator"},
  {"text": "L1",                   "from": "text",   "arrow": "none"}
]}
```
- `arrow` values (sign-nav `vlm.ARROWS`): `left, right, up, down, up-left, up-right,
  down-left, down-right, none`. As drawn on the sign, in the sign's frame ("up" = straight
  ahead). `none` = no arrow belongs to the entry (door names, "you are here", a level panel
  naming this floor).
- `from`: `text` or `symbol` (symbols are named by the place they stand for, e.g. "Taxi").
- `via`: optional way of travel shown next to the entry (`escalator`, `stairs`, `lift`); it is
  a qualifier, never the direction. (On real signs: destination, escalator icon, then the
  arrow; the arrow is the direction.)
- A level written beside a destination stays in its text and the entry keeps its arrow.

Extra per-sign fields (generator side):
- `sign_id`, `sequence_id`, `frame`; `sign_type` (section 8, Phase 2); `style_family`.
- per element: kind (text/symbol/arrow/panel), polygon in the image, for arrows the exact angle
  in degrees, for symbols the **hierarchical class** (e.g. `restroom/all_gender`).
- geometry: distance (m), viewing angle (deg), **character height in pixels at the glasses'
  stream resolution**, occlusion fraction, cut off at the frame edge (bool).
- `readable`: from sign-nav's legibility thresholds; unreadable signs get the target
  `"unreadable"` instead of their text, so models aren't trained to guess.
- Phase 3+: `true_direction` in the building frame for each entry, user-relative position
  (left/right/ahead/overhead).

## 6. Evaluation (Phase 0, before the simulator)

One command scores any model on real photos, broken down by distance/angle (character-height
bins) and by sign type:
- text accuracy (exact match for room numbers), arrow accuracy (exact, within 45 deg, and
  diagonals separately), pictogram accuracy, false signs, readable-or-not accuracy;
- sign-nav's goal-pair metric (as in its SiGNgapore evaluation), so reading gains show up as
  navigation gains.

Data:
- **Held-out test**: all of SiGNgapore (+ SiGNgapore2D; same group and city, likely shared
  venues, so both are test-only) and **one US building never used for style inspiration**.
- **Development**: your own photos and glasses walks from other buildings.
- Real photos stay private and outside both repos (they're test data and may show people).
- Photos used to set style weights must not come from held-out venues.

Baselines to record first: Gemma v4 and the v4+v5 mix ("v6"), on every split.

## 7. Two tracks

Get an answer to "does synthetic data transfer at all?" in weeks, not months:

| | Track A: 2D glyph classifiers (first) | Track B: scenes (the full plan) |
|---|---|---|
| when | weeks 2-3 | weeks 3-10 |
| renderer | Python/Pillow renders + perspective, blur, noise, JPEG; pasted on real backgrounds | Blender + BlenderProc scenes |
| targets | **arrow direction** (8 classes) and **pictogram** classifiers on small crops | detector + reader fine-tune on full frames and crops |
| sim-to-real gap | small (an arrow glyph is an arrow glyph) | larger (lighting, materials, scenes) |
| plugs into sign-nav as | arrows measured from the crop and paired with rows (Gemma reads text and says which row) | better detector, better reader |

sign-nav already has reusable pieces for Track A: `legibility/degrade.py` (plane rotation at
set angles, scale, blur, motion blur, noise, JPEG; seeded) and `legibility/synth.py`
(text-only rendered signs with labels, for dry runs).

## 8. Phases

### Phase 0: real reference set and evaluation (~1 week)
- Tag reference photos (airport, campus, ...): sign type, placement, material, pictograms,
  rough distance and angle, lighting.
- Fix held-out venues (section 6). Write the scorer. Run baselines.
- **Done when** one command scores any model on real photos.

### Phase 1: assets (~1 week)
- **Pictograms**: AIGA/DOT Symbol Signs (50 symbols, public domain) as SVG -> PNG (white on
  transparent, named by class). Fill gaps: all-gender restroom (variants: toilet symbol, the
  word RESTROOM, the circle-triangle style), family restroom, lactation room, AED, baby
  changing, classic and "dynamic" accessibility symbols, escalator up/down, lift, stairs,
  taxi, information. CC0 sources or own redraws; track any attribution licenses.
  **2-3 drawing styles per class**, including non-US styles, so models learn the meaning.
- **Arrows**: several glyph styles (thin, bold, chevron, rounded, filled triangle head,
  outlined), all 8 directions.
- **Fonts**: several free sans-serifs close to wayfinding fonts (OFL fonts may be rendered
  into images): e.g. Inter, Source Sans, Fira Sans, Overpass, Atkinson Hyperlegible.
- **Materials/HDRIs** (Phase 3+): Poly Haven, ambientCG (CC0).
- `LICENSES.md` from day one.

### Phase 2: sign generator
- `sign_config.yaml`: pictogram hierarchy (labels by meaning: `restroom -> men / women /
  all_gender / family`), style families, sampling weights from Phase 0 tags.
- **Sign types**: room plate, wall directional, two-sided projecting (blade), overhead hanging,
  long overhead board, pillar wrap (curved), directory board, exit sign (US: mostly the word
  EXIT, red or green, lit).
- **Layout conventions** (these decide which arrow belongs to which entry; random placement
  would teach the wrong pairing): arrow at the end of a row; one arrow serving a block of
  lines; symbol rows sharing an arrow; level panel on one side; way-of-travel icons between
  text and arrow; multilingual repeats of a line; destinations with a level ("@ L3").
- Room numbers in many conventions (building-room "14-100", lettered "A140", plain "1200",
  mixed "B2-105"), ranges ("Rooms 200-250").
- Outputs per sign: front image, material, physical size, label JSON (section 5).
- **Done when** a few hundred varied flat signs pass spot checks.

### Phase 3: scenes (Blender + BlenderProc)
- Modules: corridor (random width, length, door spacing), T and L junctions, lobby, concourse,
  lift lobby, stairwell.
- Mounting rules (US ADA as the default; configurable): room plates beside doors on the latch
  side, text 48-60 in above the floor; overhead signs with ~80 in clearance; projecting signs
  perpendicular to the wall; pillar wraps.
- Content consistent with the layout: room numbers increase along a corridor, arrows point to
  where things really are, so labels carry true directions.
- Distractors: posters, flyers, notice boards, extinguisher tags, capacity placards, art.
- Lighting: LED/fluorescent panels, windows behind signs, dim corridors, lit signs.

### Phase 4: camera and walking paths
- Calibrate the real glasses (checkerboard, `calibrate.py` in sign-nav) and copy FOV and
  distortion into the Blender camera.
- Paths: down the centre, wall-following (common with a cane; confirm with O&M specialists),
  stopping at junctions, turning toward a sign. Head bob per step, sideways glances, small
  tilts (no large roll). Signs often at the frame edge or cut off.
- Outputs: single frames (reading) and sequences (tracking, best-frame selection).

### Phase 5: rendering and camera degradation
- Colour, per-sign masks, depth (for distances).
- Degradation: auto-exposure clipping, motion blur at walking speed, rolling shutter, noise,
  and **re-encoding at the stream's resolution and compression** (what the reader sees).
- Cycles for realism; compare EEVEE in the pilot if too slow.
- **Never** flip images horizontally/vertically or rotate more than ~15 deg: it silently
  corrupts arrow labels (and text).

### Phase 6: labels computed from the scene
Everything in section 5, including `readable` from sign-nav's legibility thresholds
(calibrated character heights per viewing angle; store the pixel height so thresholds can be
re-applied later), and the collapsed reader-format target.

### Phase 7: pilot and realism checks (stop here before scaling)
One corridor module, ~20 sign types, ~2,000 frames. Pass thresholds fixed in advance:
1. **Matched difficulty**: reader accuracy on synthetic within **10 points** of real, at the
   same character-height and angle bins.
2. **Transfer**: a small fine-tune improves held-out real arrow or pictogram accuracy by
   **+5 points**, with text no worse than **-2**.
3. Visual side-by-side review with real photos.
If a check fails, fix the biggest gap first, then re-render.

### Phase 8: scale and train
- All modules and sign types (tens of thousands of frames).
- Detector: fine-tune YOLO-World (real-time, could replace Grounding DINO live).
- Reader: Gemma LoRA on sign crops (see risks).
- Paper experiments: synthetic only vs real only vs synthetic then real; with vs without
  degradation; with vs without readability labels; 2D renderer vs Blender.

### Phase 9: release
Generator, configs, rendered dataset, `LICENSES.md`, label-format spec.

## 9. Week 1 checklist
1. Phase 0: tag reference photos, write the scorer, record Gemma v4/v6 baselines.
2. Assets: AIGA SVG -> PNG, 4-6 fonts, arrow glyph styles, `LICENSES.md`.
3. **Track A spike**: render ~500+ arrow crops (styles x 8 directions x degradations), train a
   tiny classifier (e.g. 64x64 input), test on real arrows from SiGNgapore2D. Target: beat
   Gemma on real diagonals (~0%).
4. **LoRA spike**: fine-tune Gemma 4 E4B for a few steps on anything (needs the original
   weights, not Ollama's GGUF; e.g. Unsloth, 4-bit QLoRA fits in 16 GB), export, and run it
   in Ollama including vision. Confirms Phase 8's reader path before betting on it.

## 10. Decisions (defaults proposed)
- Held-out venues: all of SiGNgapore (+2D) and one US campus building not used for styles.
- Pilot thresholds: as in Phase 7.
- GPU: the RTX 5060 Ti for the pilot; measure seconds per frame before planning scale.

## 11. Risks
| risk | mitigation |
|---|---|
| endless realism tuning | pilot thresholds fixed before starting |
| leakage (styles inspired by a test venue) | venue holdout in Phase 0 |
| render time | measure in the pilot; EEVEE or fewer samples as fallback |
| model learns to guess unreadable text | `"unreadable"` targets |
| Gemma LoRA / GGUF / vision export doesn't work | week-1 spike; Track A classifiers don't depend on it |
| wrong arrow-entry pairing learned from unrealistic layouts | layout conventions in Phase 2 |
| US-only styles fail abroad | 2-3 styles per class incl. non-US; SiGNgapore held out |

## 12. References
- Sign Language: Towards Sign Understanding for Robot Autonomy (Agrawal, Loo, Zimmerman, Hsu;
  arXiv 2506.02556). Code and SiGNgapore2D: github.com/AdaCompNUS/Sign-Understanding. Their
  prompt separates text and symbol destinations, seeds a symbol list (taxi stand, lifts,
  up/down escalators, toilet, AED, babycare, wheelchair, ...), uses 5 directions + locational,
  and votes over several answers.
- SiGNgapore: An Interactive Dataset for Sign-based Visual Navigation (Zimmerman, Loo, Wang,
  Hsu; arXiv 2610.09488). 7 venues, 456 scenarios, 56 missions; tasks: sign parsing, visual
  sign grounding (multiple-choice path options), multi-step missions. Instructions include 8
  directions and symbolic ones like "up-escalator". Code: github.com/FullMetalNicky/SiGNgapore;
  data on Hugging Face (NickyZimmerman/SiGNgapore; check the dataset card's license).
- SignScene: Visual Sign Grounding for Mapless Navigation (Zimmerman et al., ISRR 2026; arXiv
  2602.12686): Gemini 2.5 Pro for parsing, GPT-5 for grounding over a 2D scene sketch.
- SignNav (Sun et al., RA-L 2026; arXiv 2603.16166; github.com/IMRL/SignNav): Habitat scenes
  with auto-inserted arrow-only signs (3 directions, no text); dataset not released as of
  Nov 2026.
- AIGA/DOT Symbol Signs: public domain. Poly Haven, ambientCG: CC0.
