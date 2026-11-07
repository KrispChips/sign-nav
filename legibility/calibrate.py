"""Parts C-E: the calibration experiment and the fitted model.

  # 1) run: degrade signs, send each through the live path and the reader, score the reads
  python -m legibility.calibrate run --gt <gt_annotation.json> --images <images dir> \
      --reader gemma --signs 80 --scales 8 --angles 0 30 50 70 --out calib_runs/v1
  # 2) fit: samples -> legibility_calib.json (+ plots next to it)
  python -m legibility.calibrate fit --samples calib_runs/v1/samples.jsonl --out legibility_calib.json

`run` writes one row per degraded sample to samples.jsonl. `fit` only reads those rows, so
rows collected from real footage (same fields) can replace the synthetic ones without
changing the fitting or the model code.

Every reader call is cached by a hash of the exact image sent plus the reader, model and
prompt, so reruns are free. Per sign and angle, scales run from largest to smallest and stop
after two consecutive scales with zero recall.

`--reader ocr` (RapidOCR recognition, arrows ignored) and `--synthetic N` (rendered signs)
exist to dry-run the pipeline without Gemma or the dataset. Their output is stamped
"dry_run": a calibration file from them is NOT a calibration."""
import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import facilities  # noqa: E402
from entries import canonical_arrow, normalize  # noqa: E402
from legibility.degrade import degrade, log_scales, sample_params, sight, stable_seed  # noqa: E402
from legibility.measure import box_height, line_height  # noqa: E402

TARGET_ACCURACY = 0.90        # fraction of the reader's plateau (see fit())
MAX_EXCESS_HALLUCINATION = 0.03
TEXT_MATCH = 90            # fuzzy score for "read correctly" (step 1 normalizer first)
LABEL_KEYS = {"text labels": "text", "mixed": "text", "symbol labels": "symbol"}


# ============================================================== dataset

def load_dataset(gt_path, images_dir):
    """SiGNgapore2D ground truth -> one dict per annotated sign. boundingBox is
    [centre x, centre y, width, height] in pixels (the dataset's own evaluation code calls it
    "xywh-center"; reading it as a top-left corner crops half a sign away from the board)."""
    out = []
    for img in json.loads(Path(gt_path).read_text(encoding="utf-8")):
        for ann in img.get("annotation", []):
            text, symbol = [], []
            for key, kind in LABEL_KEYS.items():
                for label, direction in (ann.get(key) or {}).items():
                    for d in (direction if isinstance(direction, list) else [direction]):
                        (text if kind == "text" else symbol).append((label, canonical_arrow(d)))
            if text or symbol:
                out.append({"id": f"{img['imagePath']}#{ann.get('objectID')}",
                            "image": str(Path(images_dir) / img["imagePath"]),
                            "bbox": ann["boundingBox"], "text": text, "symbol": symbol})
    return out


_IMAGES = {}


def crop_of(sign):
    if "crop" in sign:
        return sign["crop"]
    path = sign["image"]
    if path not in _IMAGES:
        _IMAGES.clear()   # keep one image in memory
        _IMAGES[path] = cv2.imread(path)
        if _IMAGES[path] is None:
            raise FileNotFoundError(path)
    im = _IMAGES[path]
    return crop_box(im, sign["bbox"])


def crop_box(im, bbox):
    """Crop a [centre x, centre y, width, height] box, clipped to the image."""
    cx, cy, w, h = [float(v) for v in bbox]
    x1, y1 = max(0, int(round(cx - w / 2))), max(0, int(round(cy - h / 2)))
    x2, y2 = min(im.shape[1], int(round(cx + w / 2))), min(im.shape[0], int(round(cy + h / 2)))
    return im[y1:y2, x1:x2].copy()


# ============================================================== Part C: true heights

class TruthFinder:
    """True text heights from the full-resolution, angle-0 crop: OCR detection + recognition,
    each recognized line fuzzy-matched to the sign's annotated entries. Heights use the same
    line-height estimator as live measurement, so truth and live are in the same units.
    At scale s the true height is the full-resolution height times s."""

    def __init__(self, side_len=2000):
        import logging
        from rapidocr import RapidOCR
        self.engine = RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": side_len,
                                       "Global.use_cls": False})
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)

    def __call__(self, crop, entries):
        r = self.engine(crop, use_det=True, use_cls=False, use_rec=True)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lines = []
        for box, txt in zip(r.boxes if r.boxes is not None else [], r.txts or ()):
            h = line_height(gray, box) or box_height(box)
            lines.append((normalize(txt).text, h))
        entry_h = []
        for text, _ in entries:
            t = normalize(text).text
            hs = [h for lt, h in lines if lt and (fuzz.partial_ratio(t, lt) >= 85 if len(lt) >= 3
                                                  else lt == t)]
            entry_h.append(float(np.median(hs)) if hs else None)
        heights = [h for _, h in lines]
        return {"h_p25": float(np.percentile(heights, 25)) if heights else None,
                "line_heights": heights, "entry_h": entry_h,
                "sign_px": float(min(crop.shape[:2]))}


# ============================================================== Part D: scoring reads

def _canon(text):
    """Step 1 normalizer (which also drops accents: "CAFÉ" = "CAFE")."""
    return normalize(text).text


def text_score(a, b):
    """Same entry? Also compares with spaces removed, so "TE 14" = "TE14"."""
    na, nb = _canon(a), _canon(b)
    return max(fuzz.ratio(na, nb), fuzz.token_sort_ratio(na, nb),
               fuzz.ratio(na.replace(" ", ""), nb.replace(" ", "")))


RELATED = 75      # below this a returned entry is unrelated to every annotation: invented


def is_invented(out_text, gt):
    """Hallucination = a returned entry unrelated to ANY annotated entry. Partial reads ("Bus
    Stop" for "BUS STOP 2"), formatting ("TE 14") and garbled-but-real text ("Pasar Ribs")
    are not inventions; bare arrows and symbols are skipped. Real but unannotated text still
    counts, which is why the fit compares against each angle's baseline."""
    o = _canon(out_text)
    if not re.search(r"[A-Z0-9]", o):
        return False
    rel = max((max(fuzz.partial_ratio(o, _canon(g)), fuzz.token_set_ratio(o, _canon(g))) for g, _ in gt),
              default=0)
    return rel < RELATED


def symbol_score(a, b):
    fa, fb = facilities.lookup(normalize(a).text), facilities.lookup(normalize(b).text)
    if fa and fb and fa[0] == fb[0]:
        return 100.0
    return text_score(a, b)


def score_read(gt, out, symbols=False, ignore_arrows=False):
    """gt: [(text, arrow)], out: [{"text", "arrow"}] -> per-gt correctness, hallucinations.
    One output entry can satisfy one annotated entry (greedy on text score).
    hallucination = returned entry that matches no annotation's text;
    wrong_arrow = matches an annotation's text but not its arrow."""
    sim = symbol_score if symbols else text_score
    pairs = sorted(((sim(g[0], o.get("text", "")), gi, oi) for gi, g in enumerate(gt)
                    for oi, o in enumerate(out)), reverse=True)
    correct, used_g, used_o, wrong_arrow = [False] * len(gt), set(), set(), 0
    for s, gi, oi in pairs:
        if s < TEXT_MATCH:
            break
        if gi in used_g or oi in used_o:
            continue
        arrow_ok = ignore_arrows or canonical_arrow(out[oi].get("arrow")) == gt[gi][1]
        if arrow_ok:
            used_g.add(gi); used_o.add(oi); correct[gi] = True
    for oi, o in enumerate(out):
        if oi in used_o:
            continue
        best = max((sim(g[0], o.get("text", "")) for g in gt), default=0)
        if best >= TEXT_MATCH:
            wrong_arrow += 1
    halluc = sum(1 for oi, o in enumerate(out) if oi not in used_o
                 and max((sim(g[0], o.get("text", "")) for g in gt), default=0) < TEXT_MATCH
                 and (symbols or is_invented(o.get("text", ""), gt)))
    return correct, halluc, wrong_arrow


# ============================================================== readers + cache

class GemmaReader:
    """The production reader: vlm.SignReader with its prompt and preprocessing."""
    ignore_arrows = False
    dry_run = False

    def __init__(self, model="gemma4:e4b", host=None, prompt=None):
        import vlm
        self.reader = vlm.SignReader(model, host, prompt or vlm.DEFAULT_PROMPT)
        self.tag = f"gemma|{model}|{hashlib.sha256(self.reader.prompt.encode()).hexdigest()[:16]}"

    def __call__(self, crop):
        entries, raw = self.reader(crop)
        return entries, raw


class OcrProxyReader:
    """DRY RUN ONLY: RapidOCR recognition as a stand-in for Gemma (no arrows)."""
    ignore_arrows = True
    dry_run = True
    tag = "ocr-proxy|v1"

    def __init__(self):
        from signfilter import ocr_engine
        self.engine = ocr_engine()

    def __call__(self, crop):
        r = self.engine(crop, use_det=True, use_cls=False, use_rec=True)
        lines = [t for t, s in zip(r.txts or (), r.scores or ()) if s >= 0.6]
        return [{"text": t, "arrow": "none"} for t in lines], " | ".join(lines)


class Cache:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(crop, tag):
        ok, png = cv2.imencode(".png", crop)
        return hashlib.sha256(png.tobytes() + tag.encode()).hexdigest()

    def get(self, key):
        p = self.dir / f"{key[:2]}" / f"{key}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, key, value):
        p = self.dir / f"{key[:2]}" / f"{key}.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(value), encoding="utf-8")


# ============================================================== Part D: the experiment

def run_experiment(signs, reader, out_dir, angles, scales, variants=1, stop_after=2, log=print):
    from signfilter import ocr_engine
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(out_dir / "cache")
    engine, truth = ocr_engine(), TruthFinder()
    max_calls = len(signs) * len(angles) * len(scales) * variants
    log(f"{len(signs)} signs x {len(angles)} angles x {len(scales)} scales x {variants} variant(s): "
        f"at most {max_calls} reads (early stopping cuts this)")
    calls = cached = 0
    t_reads = 0.0
    t0 = time.time()
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for n, sign in enumerate(signs, 1):
            full = crop_of(sign)
            kind = "text" if sign["text"] else "symbol"
            gt = sign["text"] if kind == "text" else sign["symbol"]
            tr = truth(full, gt)
            for angle in angles:
                zero_streak = 0
                for si, s in enumerate(scales):
                    recalls = []
                    for v in range(variants):
                        p = sample_params(angle, s, stable_seed(sign["id"], angle, si, v))
                        scene = degrade(full, p)
                        crop, info, m = sight(scene.image, scene.box, scene.K, engine)
                        key = Cache.key(crop, reader.tag)
                        hit = cache.get(key)
                        if hit is None:
                            ts = time.time()
                            entries, raw = reader(crop)
                            t_reads += time.time() - ts
                            cache.put(key, {"entries": entries, "raw": raw})
                            calls += 1
                        else:
                            entries, raw = hit["entries"], hit["raw"]
                            cached += 1
                        correct, halluc, wrong_arrow = score_read(gt, entries, kind == "symbol",
                                                                   reader.ignore_arrows)
                        tcorrect = score_read(gt, entries, kind == "symbol", True)[0]
                        recalls.append(sum(correct) / len(gt))
                        row = {
                            "sign": sign["id"], "kind": kind, "angle": angle, "scale": s,
                            "variant": v, "params": p.to_dict(),
                            "live": dict(m.to_dict(), quad_found=bool(info["quad_found"])),
                            "truth": {"h_p25": tr["h_p25"] * s if tr["h_p25"] else None,
                                      "sign_px": tr["sign_px"] * s,
                                      "full_h_p25": tr["h_p25"]},
                            "entries": [{"text": t, "arrow": a, "correct": c,
                                         "h": eh * s if eh else None}
                                        for (t, a), c, eh in zip(gt, correct, tr["entry_h"])],
                            "text_correct": tcorrect,
                            "n_out": len(entries), "hallucinated": halluc, "wrong_arrow": wrong_arrow,
                            "read": entries, "reader": reader.tag, "dry_run": reader.dry_run,
                        }
                        f.write(json.dumps(row) + "\n")
                    zero_streak = zero_streak + 1 if max(recalls) == 0 else 0
                    if zero_streak >= stop_after:
                        break
            f.flush()
            per = t_reads / calls if calls else 0
            log(f"[{n}/{len(signs)}] {sign['id']}: {calls} reads ({cached} cached), "
                f"{per:.1f} s/read, {time.time() - t0:.0f} s elapsed")
    return out_dir / "samples.jsonl"


# ============================================================== Part E: fitting

def fit_logistic(x, y, l2=1e-2, iters=100):
    """P(y=1) = sigmoid(a + b x) by Newton's method (small ridge for separable data)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (y - p) - l2 * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    return math.log(p / (1 - p))


def load_samples(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def _detection_floor(pairs, min_rate=0.9, nbins=12):
    """Smallest true height from which the detector finds text in >= min_rate of samples.
    pairs: [(true_h, live_h or None)]. Also returns per-bin rates for the plot."""
    pairs = [(t, l) for t, l in pairs if t and t > 0]
    if not pairs:
        return None, []
    lt = np.log([t for t, _ in pairs])
    edges = np.linspace(lt.min(), lt.max() + 1e-9, nbins + 1)
    bins = []
    for i in range(nbins):
        sel = [(t, l) for (t, l), x in zip(pairs, lt) if edges[i] <= x < edges[i + 1]]
        if sel:
            bins.append((float(np.exp(edges[i])), float(np.exp(edges[i + 1])),
                         sum(l is not None for _, l in sel) / len(sel), len(sel)))
    # Isotonic (non-decreasing) fit of detection vs height, read off at min_rate: exact for a
    # step, and noisy dips get pooled instead of moving the floor. No curve shape is assumed.
    srt = sorted(pairs, key=lambda p: p[0])
    fitted = _isotonic([1.0 if l is not None else 0.0 for _, l in srt])
    ok = [t for (t, _), f in zip(srt, fitted) if f >= min_rate]
    return (float(ok[0]) if ok else None), bins


def _accuracy_floor(pairs, tol=0.20, min_rate=0.9):
    """Smallest true height from which detected measurements are within `tol` of the truth
    in >= min_rate of cases (isotonic fit, like the detection floor)."""
    det = sorted(((t, l) for t, l in pairs if t and l), key=lambda p: p[0])
    if len(det) < 10:
        return None
    fitted = _isotonic([1.0 if abs(l / t - 1) <= tol else 0.0 for t, l in det])
    ok = [t for (t, _), f in zip(det, fitted) if f >= min_rate]
    return float(ok[0]) if ok else None


def _isotonic(y):
    """Pool-adjacent-violators: the best non-decreasing fit to y (equal weights)."""
    blocks = []   # [value, weight]
    for v in y:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2 = blocks.pop()
            v1, w1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    return [v for v, w in blocks for _ in range(int(w))]


def _relation(pairs):
    """log(live) = c + d log(true), least squares; identity if too few points."""
    pairs = [(t, l) for t, l in pairs if t and l]
    if len(pairs) < 8:
        return 0.0, 1.0
    x, y = np.log([t for t, _ in pairs]), np.log([l for _, l in pairs])
    d, c = np.polyfit(x, y, 1)
    return float(c), float(d)


def text_correct(r):
    """Per annotated entry: was its TEXT read (arrow ignored)? Legibility is about text: on
    real signs Gemma's arrow errors (left/right swaps, 'up' read as 'right') don't shrink as
    the text gets bigger, so they are reported separately instead of bending the curve."""
    if "text_correct" in r:
        return r["text_correct"]
    if "read" not in r:   # rows from another source without the raw read: use their scoring
        return [e["correct"] for e in r["entries"]]
    gt = [(e["text"], e["arrow"]) for e in r["entries"]]
    return score_read(gt, r.get("read", []), symbols=r["kind"] == "symbol", ignore_arrows=True)[0]


def hallucinated(r):
    """Invented entries in a row's read, under the current definition (rescored from the
    stored read, so older samples files are judged the same way)."""
    if "read" not in r:
        return r["hallucinated"]
    gt = [(e["text"], e["arrow"]) for e in r["entries"]]
    return score_read(gt, r["read"], symbols=r["kind"] == "symbol", ignore_arrows=True)[1]


def fit_plateau(x, y, grid=np.linspace(0.3, 1.0, 71)):
    """P(correct) = c * sigmoid(a + b x), by profile likelihood over the plateau c. The reader
    tops out below 100% even on big text (about 80-85% for Gemma 4 E4B on SiGNgapore2D), so
    a plain logistic, which must reach 1, mis-places the threshold."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    J = np.column_stack([np.ones_like(x), x])
    best = None
    for c in grid:
        w = np.array([0.0, 1.0])
        for _ in range(80):
            sg = sigmoid(w[0] + w[1] * x)
            p = np.clip(c * sg, 1e-6, 1 - 1e-6)
            dp = c * sg * (1 - sg)
            grad = J.T @ ((y / p - (1 - y) / (1 - p)) * dp)
            curv = (y / p ** 2 + (1 - y) / (1 - p) ** 2) * dp ** 2
            step = np.linalg.solve(J.T @ (J * curv[:, None]) + 1e-3 * np.eye(2), grad)
            w += step
            if np.abs(step).max() < 1e-7:
                break
        p = np.clip(c * sigmoid(w[0] + w[1] * x), 1e-6, 1 - 1e-6)
        ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if best is None or ll > best[0]:
            best = (ll, float(c), float(w[0]), float(w[1]))
    return best[1:]


def complete_flags(rows):
    """Per row: did it show at least as many text lines as the same sign and angle at its
    largest scale? While lines are still missing, the measured height comes from the larger
    lines that survive the distance and overstates legibility (the read trigger waits for the
    line count to settle for the same reason)."""
    ref = {}
    for r in rows:
        k = (r["sign"], r["angle"])
        if k not in ref or r["scale"] > ref[k][0]:
            ref[k] = (r["scale"], r["live"].get("n_lines", 0))
    return [r["live"].get("n_lines", 0) >= ref[(r["sign"], r["angle"])][1] for r in rows]


MIN_PLATEAU = 0.4     # below this the angle itself costs most of the reads: face the sign


def fit(rows, target=TARGET_ACCURACY, max_excess=MAX_EXCESS_HALLUCINATION):
    """`target` is a fraction of the reader's plateau: the threshold is where text accuracy
    reaches target * plateau, i.e. where getting closer stops paying off."""
    text = [r for r in rows if r["kind"] == "text"]
    symbol = [r for r in rows if r["kind"] == "symbol"]
    tcorr = {id(r): text_correct(r) for r in rows}
    complete = {id(r): f for r, f in zip(text, complete_flags(text))}
    out = {"version": 2, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "reader": sorted({r["reader"] for r in rows}),
           "dry_run": any(r.get("dry_run") for r in rows),
           "settings": {"target_fraction_of_plateau": target, "max_excess_hallucination": max_excess,
                        "scored_on": "text (arrows reported separately)",
                        "measure": "h_p25 of FWHM line heights, source px",
                        "ocr_side_len": 320},
           "n_samples": len(rows), "n_signs": len({r["sign"] for r in rows}),
           "angles": []}
    plot = {"angles": {}}

    # ---- estimator accuracy (live vs true, sign level)
    pairs = [(r["truth"]["h_p25"], r["live"]["h_p25"]) for r in text]
    # Two floors. detection_floor: below it the detector often finds nothing (so "no text" may
    # just mean "too far"). floor (accuracy floor): below it, measurements that DO come back
    # stop tracking the truth within 20%; the model only trusts measurements above it.
    det_floor, det_bins = _detection_floor(pairs)
    floor_true = _accuracy_floor(pairs) or det_floor or 8.0

    def ratio_stats(ps):
        rr = np.array([l / t for t, l in ps if t and l and t >= 8.0])
        if not len(rr):
            rr = np.array([1.0])
        return {"median_ratio": float(np.median(rr)),
                "median_abs_rel_error": float(np.median(np.abs(rr - 1))),
                "p90_abs_rel_error": float(np.percentile(np.abs(rr - 1), 90)),
                "within_20pct": float(np.mean(np.abs(rr - 1) <= 0.20)), "n": int(len(rr))}
    all_stats = ratio_stats(pairs)
    comp_pairs = [(r["truth"]["h_p25"], r["live"]["h_p25"]) for r in text if complete[id(r)]]
    comp_stats = ratio_stats(comp_pairs)
    med_ratio = comp_stats["median_ratio"]
    out["estimator"] = dict(all_stats, floor_true=floor_true, floor=floor_true * med_ratio,
                            detection_floor_true=det_floor, detection_bins=det_bins,
                            all_lines_visible=comp_stats,
                            note="live/true ratios for true >= 8 px; 'all_lines_visible' = samples "
                                 "showing as many lines as at full size (what the trigger reads)")
    plot["estimator"] = {"pairs": pairs, "floor_true": floor_true, "detection_floor_true": det_floor}

    # ---- accuracy curve per angle (text), plateau-relative threshold
    for angle in sorted({r["angle"] for r in text}):
        rs = [r for r in text if r["angle"] == angle]
        ent = [(e["h"], c) for r in rs for e, c in zip(r["entries"], tcorr[id(r)]) if e["h"]]
        res = {"angle": angle, "n_entries": len(ent), "blocked": False}
        a_pairs = [(r["truth"]["h_p25"], r["live"]["h_p25"]) for r in rs]
        a_comp = [(r["truth"]["h_p25"], r["live"]["h_p25"]) for r in rs if complete[id(r)]]
        ratio = ratio_stats(a_comp)["median_ratio"] if a_comp else med_ratio
        res["live_ratio"] = ratio
        res["floor_true"] = _accuracy_floor(a_pairs) or floor_true
        res["floor"] = res["floor_true"] * ratio
        res["detection_floor_true"] = _detection_floor(a_pairs)[0]
        # "No text found" only means "no text" once text would have been found: the sign size
        # at which text signs' text is detected 90% of the time. The no-text (pictogram)
        # fallback waits for at least this size, so far text signs don't trigger it.
        sp = [(r["truth"]["sign_px"], r["live"]["h_p25"]) for r in rs]
        nt_true = _detection_floor(sp)[0]
        sratio = float(np.median([r["live"]["sign_px"] / r["truth"]["sign_px"] for r in rs])) if rs else 1.0
        res["no_text_sign_px"] = nt_true * sratio if nt_true else None
        # arrows, reported separately: of the entries whose text was read, how many arrows were right
        arrow_ok = [e["correct"] for r in rs for e, c in zip(r["entries"], tcorr[id(r)]) if e["h"] and c]
        res["arrow_accuracy_when_text_read"] = float(np.mean(arrow_ok)) if arrow_ok else None
        if len(ent) < 20 or len({c for _, c in ent}) < 2:
            res.update(blocked=True, h_star=None, note="too little data to fit")
            out["angles"].append(res)
            continue
        c, a, b = fit_plateau(np.log([h for h, _ in ent]), [cc for _, cc in ent])
        res["curve"] = {"plateau": c, "a": a, "b": b, "x": "log(true entry height px)"}
        h_max = max(h for h, _ in ent)
        h_star = math.exp((logit(target) - a) / b) if b > 0 else math.inf
        if c < MIN_PLATEAU or not (h_star <= h_max):
            res.update(blocked=True, h_star=None, h_star_true=None,
                       note=f"plateau {c:.2f} < {MIN_PLATEAU}" if c < MIN_PLATEAU
                       else "threshold beyond the observed heights: face the sign")
        else:
            # hallucination: per returned entry, vs the sign's true p25 height
            hx, hy = [], []
            for r in rs:
                if r["truth"]["h_p25"] and r["n_out"]:
                    nh = hallucinated(r)
                    hx += [math.log(r["truth"]["h_p25"])] * r["n_out"]
                    hy += [1] * nh + [0] * (r["n_out"] - nh)
            ha, hb = fit_logistic(hx, hy) if len(set(hy)) > 1 else (-10.0, 0.0)
            base = float(sigmoid(ha + hb * max(hx))) if hx else 0.0
            at = float(sigmoid(ha + hb * math.log(h_star)))
            raised = False
            if at - base > max_excess and hb < 0:
                h_star = math.exp((logit(min(0.99, base + max_excess)) - ha) / hb)
                at, raised = float(sigmoid(ha + hb * math.log(h_star))), True
            above = [cc for h, cc in ent if h >= h_star]
            # The floor only shapes guidance below the threshold: it never sits above it.
            res["accuracy_floor_raw"] = res["floor_true"]
            res["floor_true"] = min(res["floor_true"], h_star)
            res["floor"] = res["floor_true"] * ratio
            res.update(h_star_true=h_star, h_star=h_star * ratio,
                       text_accuracy_above_h_star=float(np.mean(above)) if above else None,
                       accuracy_above_h_star=float(np.mean(above)) if above else None,
                       hallucination_at_h_star=at, hallucination_baseline=base,
                       raised_for_hallucination=raised,
                       hallucination_curve={"a": ha, "b": hb})
        out["angles"].append(res)
        plot["angles"][angle] = {"entries": ent, "rows": rs}

    # ---- pictogram fallback (same plateau-relative rule, on the sign's size)
    ent = [(r["truth"]["sign_px"], c) for r in symbol for c in tcorr[id(r)]]
    pic = {"n_entries": len(ent)}
    if len(ent) >= 20 and len({c for _, c in ent}) == 2:
        c, a, b = fit_plateau(np.log([h for h, _ in ent]), [cc for _, cc in ent])
        s_true = math.exp((logit(target) - a) / b) if b > 0 else math.inf
        live_ratio = float(np.median([r["live"]["sign_px"] / r["truth"]["sign_px"] for r in symbol]))
        pic.update(curve={"plateau": c, "a": a, "b": b}, sign_px_star_true=s_true,
                   sign_px_star=s_true * live_ratio if math.isfinite(s_true) else None)
    if not pic.get("sign_px_star"):
        pic.update(sign_px_star=60.0, note="not enough symbol-only data: conservative default")
    out["pictogram"] = pic
    plot["pictogram"] = ent

    # ---- trigger: sharpness that 95% of fully read samples had
    sharp = [r["live"]["sharpness"] for r in text if r["entries"] and all(tcorr[id(r)])]
    out["trigger"] = {"min_sharpness": float(np.percentile(sharp, 5)) if len(sharp) >= 20 else 100.0,
                      "settle_sightings": 2}
    return out, plot


# ============================================================== plots

def make_plots(calib, plot, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed: pip install matplotlib")
        return []
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    out_dir = Path(out_dir)
    files = []
    ticks = [2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200]

    def px_axis(axis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    title_note = " (DRY RUN: not a calibration)" if calib["dry_run"] else ""

    angles = [a for a in calib["angles"]]
    fig, axes = plt.subplots(1, max(1, len(angles)), figsize=(4.2 * max(1, len(angles)), 3.6), squeeze=False)
    for ax, res in zip(axes[0], angles):
        data = plot["angles"].get(res["angle"])
        ax.set_title(f"{res['angle']} deg" + (" - BLOCKED" if res["blocked"] else ""))
        ax.set_xscale("log"); px_axis(ax.xaxis); ax.set_ylim(-0.05, 1.05); ax.set_xlabel("true text height (px)")
        if not data:
            continue
        ent = data["entries"]
        hs = np.array([h for h, _ in ent]); cs = np.array([c for _, c in ent], float)
        edges = np.geomspace(hs.min(), hs.max() * 1.001, 12)
        mids = [math.sqrt(edges[i] * edges[i + 1]) for i in range(11)]
        acc = [cs[(hs >= edges[i]) & (hs < edges[i + 1])].mean() if ((hs >= edges[i]) & (hs < edges[i + 1])).any()
               else np.nan for i in range(11)]
        ax.plot(mids, acc, "o", color="#1f77b4", label="text read (binned)")
        if "curve" in res:
            xs = np.geomspace(hs.min(), hs.max(), 100)
            cv = res["curve"]
            ax.plot(xs, cv.get("plateau", 1.0) * sigmoid(cv["a"] + cv["b"] * np.log(xs)), "-", color="#1f77b4")
            ax.axhline(cv.get("plateau", 1.0) * calib["settings"]["target_fraction_of_plateau"],
                       color="gray", lw=0.8, ls=":")
        if "hallucination_curve" in res:
            xs = np.geomspace(hs.min(), hs.max(), 100)
            hc = res["hallucination_curve"]
            ax.plot(xs, sigmoid(hc["a"] + hc["b"] * np.log(xs)), "--", color="#d62728", label="hallucination")
        if res.get("h_star_true"):
            ax.axvline(res["h_star_true"], color="k", lw=1)
            ax.text(res["h_star_true"], 0.05, f" h*={res['h_star_true']:.1f}", fontsize=8)
    axes[0][0].set_ylabel("probability"); axes[0][0].legend(fontsize=7, loc="center right")
    fig.suptitle("Text accuracy vs true text height (dotted: threshold level)" + title_note, fontsize=10)
    fig.tight_layout(); p = out_dir / "legibility_accuracy.png"; fig.savefig(p, dpi=120); plt.close(fig); files.append(p)

    est = calib["estimator"]
    pairs = [(t, l) for t, l in plot["estimator"]["pairs"] if t]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    tt = [t for t, l in pairs if l]; ll = [l for t, l in pairs if l]
    a1.loglog(tt, ll, ".", alpha=0.4, ms=4); px_axis(a1.xaxis); px_axis(a1.yaxis)
    lim = [max(1, min(tt + ll)) if tt else 1, max(tt + ll) if tt else 100]
    xs = np.geomspace(*lim, 50)
    a1.plot(xs, xs, "k-", lw=1); a1.fill_between(xs, xs * 0.8, xs * 1.2, color="gray", alpha=0.15, label="+-20%")
    a1.axvline(est["floor_true"], color="#d62728", lw=1, label=f"accuracy floor {est['floor_true']:.1f} px")
    a1.set_xlabel("true text height (px)"); a1.set_ylabel("live measurement (px)"); a1.legend(fontsize=8)
    a1.set_title(f"median |error| {100 * est['median_abs_rel_error']:.0f}% (true >= 8 px)", fontsize=9)
    b = est["detection_bins"]
    a2.semilogx([math.sqrt(lo * hi) for lo, hi, _, _ in b], [r for _, _, r, _ in b], "o-"); px_axis(a2.xaxis)
    a2.axhline(0.9, color="gray", ls=":")
    if est.get("detection_floor_true"):
        a2.axvline(est["detection_floor_true"], color="#ff7f0e", lw=1,
                   label=f"detection floor {est['detection_floor_true']:.1f} px")
        a2.legend(fontsize=8)
    a2.set_xlabel("true text height (px)"); a2.set_ylabel("text detected (fraction)"); a2.set_ylim(-0.05, 1.05)
    fig.suptitle("Live measurement vs truth" + title_note, fontsize=10)
    fig.tight_layout(); p = out_dir / "legibility_estimator.png"; fig.savefig(p, dpi=120); plt.close(fig); files.append(p)

    if plot["pictogram"]:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        hs = np.array([h for h, _ in plot["pictogram"]]); cs = np.array([c for _, c in plot["pictogram"]], float)
        ax.semilogx(hs, cs + np.random.default_rng(0).uniform(-0.03, 0.03, len(cs)), ".", alpha=0.4)
        px_axis(ax.xaxis)
        pic = calib["pictogram"]
        if "curve" in pic:
            xs = np.geomspace(hs.min(), hs.max(), 100)
            ax.plot(xs, pic["curve"].get("plateau", 1.0) * sigmoid(pic["curve"]["a"] + pic["curve"]["b"] * np.log(xs)), "-")
        ax.set_xlabel("true sign size, shorter side (px)"); ax.set_ylabel("symbol read correctly")
        ax.set_title("Pictogram fallback" + title_note, fontsize=10)
        fig.tight_layout(); p = out_dir / "legibility_pictogram.png"; fig.savefig(p, dpi=120); plt.close(fig); files.append(p)
    return files


# ============================================================== CLI

def select_signs(signs, n_text, seed=0):
    rnd = random.Random(seed)
    text = [s for s in signs if s["text"]]
    symbol = [s for s in signs if not s["text"] and s["symbol"]]
    rnd.shuffle(text)
    return text[:n_text] + symbol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the degradation + read experiment")
    r.add_argument("--gt", help="SiGNgapore2D gt_annotation.json")
    r.add_argument("--images", help="SiGNgapore2D images folder")
    r.add_argument("--synthetic", type=int, help="DRY RUN: use N rendered signs instead of the dataset")
    r.add_argument("--reader", choices=("gemma", "ocr"), default="gemma")
    r.add_argument("--model", default="gemma4:e4b")
    r.add_argument("--ollama-host")
    r.add_argument("--signs", type=int, default=80, help="text signs to use (all symbol-only signs are added)")
    r.add_argument("--scales", type=int, default=8)
    r.add_argument("--smallest", type=float, default=0.08)
    r.add_argument("--angles", type=float, nargs="+", default=[0, 30, 50, 70])
    r.add_argument("--variants", type=int, default=1, help="blur/noise variants per scale")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="calib_runs/run")
    f = sub.add_parser("fit", help="fit legibility_calib.json from samples")
    f.add_argument("--samples", nargs="+", required=True)
    f.add_argument("--out", default="legibility_calib.json")
    f.add_argument("--target", type=float, default=TARGET_ACCURACY)
    f.add_argument("--max-excess-hallucination", type=float, default=MAX_EXCESS_HALLUCINATION)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        if a.synthetic:
            from legibility.synth import synthetic_signs
            signs = synthetic_signs(a.synthetic, a.seed)
        else:
            if not (a.gt and a.images):
                ap.error("--gt and --images are required (or --synthetic N for a dry run)")
            signs = select_signs(load_dataset(a.gt, a.images), a.signs, a.seed)
        reader = GemmaReader(a.model, a.ollama_host) if a.reader == "gemma" else OcrProxyReader()
        path = run_experiment(signs, reader, a.out, a.angles, log_scales(a.scales, a.smallest), a.variants)
        print(f"samples: {path}\nnext: python -m legibility.calibrate fit --samples {path}")
    else:
        rows = load_samples(a.samples)
        calib, plot = fit(rows, a.target, a.max_excess_hallucination)
        Path(a.out).write_text(json.dumps(calib, indent=1), encoding="utf-8")
        files = make_plots(calib, plot, Path(a.out).resolve().parent)
        print(f"wrote {a.out}" + "".join(f"\n      {p}" for p in files))
        report(calib)


def report(c):
    if c["dry_run"]:
        print("*** DRY RUN (stand-in reader and/or rendered signs): NOT A CALIBRATION ***")
    e, v = c["estimator"], c["estimator"]["all_lines_visible"]
    det = e.get("detection_floor_true")
    print(f"estimator (true >= 8 px): median |error| {100 * e['median_abs_rel_error']:.0f}% all samples, "
          f"{100 * v['median_abs_rel_error']:.0f}% when all lines visible (within 20%: {100 * v['within_20pct']:.0f}%, "
          f"n={v['n']}); accuracy floor {e['floor_true']:.1f} px, detection floor "
          + (f"{det:.1f} px" if det else "n/a"))
    for r in c["angles"]:
        arrow = r.get("arrow_accuracy_when_text_read")
        arrow_s = f", arrows right {100 * arrow:.0f}% when text read" if arrow is not None else ""
        if r["blocked"]:
            print(f"  {r['angle']:>4} deg: BLOCKED ({r.get('note', '')}){arrow_s}")
        else:
            cv = r["curve"]
            print(f"  {r['angle']:>4} deg: plateau {100 * cv['plateau']:.0f}% text read; threshold "
                  f"{r['h_star_true']:.1f} px true = {r['h_star']:.1f} px live; text read above it "
                  f"{100 * r['text_accuracy_above_h_star']:.0f}%; hallucination {100 * r['hallucination_at_h_star']:.1f}% "
                  f"(baseline {100 * r['hallucination_baseline']:.1f}%)"
                  + (", raised for hallucination" if r["raised_for_hallucination"] else "") + arrow_s)
    p = c["pictogram"]
    print(f"pictogram: sign_px* = {p['sign_px_star']:.1f} px" + (f" ({p['note']})" if p.get("note") else ""))
    print(f"trigger: min_sharpness {c['trigger']['min_sharpness']:.0f}, waits {c['trigger'].get('settle_sightings', 2)} "
          "sightings without new text lines")


if __name__ == "__main__":
    main()
