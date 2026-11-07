"""Compare reader prompts (and models) on SiGNgapore2D, with arrows in focus.

  python vlm_eval.py --gt data\\SiGNgapore2D\\gt\\gt_annotation.json --images data\\SiGNgapore2D\\images \
      --prompts v1 v2 v3 --cache calib_runs\\v1\\cache

Each sign is read at its closest view (full photo size) from two viewing angles, through the
live path (degrade -> rectify), with the same seeds as `legibility.calibrate run`: pointing
--cache at that run's cache makes the v1 baseline free. Every read is cached by image +
model + prompt.

Reported per prompt: text recall; arrow accuracy over entries whose text was read (exact,
and within 45 degrees, e.g. up-left read as left), split by signs whose entries share one
arrow vs point different ways, by diagonal / straight / locational ("none") arrows;
stability (same arrow from both views); invented entries per read."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from entries import canonical_arrow  # noqa: E402
from legibility.calibrate import (Cache, crop_of, load_dataset, score_read, select_signs,  # noqa: E402
                                  text_score)
from legibility.degrade import degrade, sample_params, sight, stable_seed  # noqa: E402

ANGLE_OF = {"up": 0, "up-right": 45, "right": 90, "down-right": 135, "down": 180,
            "down-left": 225, "left": 270, "up-left": 315}


def arrow_close(a, b, tol=45):
    """Same arrow, or both real arrows within `tol` degrees ("none" only matches "none")."""
    if a == b:
        return True
    if a not in ANGLE_OF or b not in ANGLE_OF:
        return False
    d = abs(ANGLE_OF[a] - ANGLE_OF[b]) % 360
    return min(d, 360 - d) <= tol


def pair_entries(gt, read):
    """Greedy text pairing -> [(gt index, read arrow)] for entries whose text was read."""
    pairs, used = [], set()
    cand = sorted(((text_score(g[0], o.get("text", "")), gi, oi) for gi, g in enumerate(gt)
                   for oi, o in enumerate(read)), reverse=True)
    done = set()
    for s, gi, oi in cand:
        if s < 90:
            break
        if gi in done or oi in used:
            continue
        done.add(gi); used.add(oi)
        pairs.append((gi, canonical_arrow(read[oi].get("arrow"))))
    return pairs


def score_sign(gt, read):
    text_ok = score_read(gt, read, ignore_arrows=True)[0]
    invented = score_read(gt, read, ignore_arrows=True)[1]
    mixed = len({a for _, a in gt}) > 1
    arrows = []
    for gi, a in pair_entries(gt, read):
        g = gt[gi][1]
        kind = "none" if g == "none" else "diagonal" if "-" in g else "straight"
        arrows.append({"gi": gi, "gt": g, "read": a, "exact": a == g, "close": arrow_close(a, g),
                       "kind": kind, "mixed": mixed})
    return {"text_ok": text_ok, "invented": invented, "n_read": len(read), "arrows": arrows}


class Reader:
    def __init__(self, version, model, host):
        import vlm
        self.version = version
        self.reader = vlm.SignReader(model, host, version)
        self.tag = f"gemma|{model}|{hashlib.sha256(self.reader.prompt.encode()).hexdigest()[:16]}"

    def __call__(self, crop, cache):
        key = Cache.key(crop, self.tag)
        hit = cache.get(key)
        if hit is not None:
            return hit["entries"], True
        entries, raw = self.reader(crop)
        cache.put(key, {"entries": entries, "raw": raw})
        return entries, False


def summarize(results):
    """results: [{"sign", "view", "score"}] for one prompt."""
    text = [c for r in results for c in r["score"]["text_ok"]]
    arr = [a for r in results for a in r["score"]["arrows"]]

    def acc(sel, key="exact"):
        sel = list(sel)
        return (float(np.mean([a[key] for a in sel])), len(sel)) if sel else (None, 0)
    by_entry = collections.defaultdict(dict)
    for r in results:
        for a in r["score"]["arrows"]:
            by_entry[(r["sign"], a["gi"])][r["view"]] = a["read"]
    two = [v for v in by_entry.values() if len(v) >= 2]
    return {
        "text_recall": float(np.mean(text)) if text else None,
        "arrow_exact": acc(arr), "arrow_within_45": acc(arr, "close"),
        "arrow_one_arrow_signs": acc(a for a in arr if not a["mixed"]),
        "arrow_mixed_signs": acc(a for a in arr if a["mixed"]),
        "arrow_straight": acc(a for a in arr if a["kind"] == "straight"),
        "arrow_diagonal": acc(a for a in arr if a["kind"] == "diagonal"),
        "arrow_none": acc(a for a in arr if a["kind"] == "none"),
        "stable_across_views": (float(np.mean([len(set(v.values())) == 1 for v in two])), len(two)) if two else (None, 0),
        "invented_per_read": float(np.mean([r["score"]["invented"] for r in results])) if results else None,
    }


def report(table):
    names = list(table)
    rows = [("text recall", "text_recall"), ("arrow exact", "arrow_exact"),
            ("arrow within 45 deg", "arrow_within_45"), ("  one-arrow signs", "arrow_one_arrow_signs"),
            ("  mixed-arrow signs", "arrow_mixed_signs"), ("  straight arrows", "arrow_straight"),
            ("  diagonal arrows", "arrow_diagonal"), ("  'none' (this place)", "arrow_none"),
            ("same arrow both views", "stable_across_views"), ("invented entries/read", "invented_per_read")]
    print(f"{'':24}" + "".join(f"{n:>16}" for n in names))
    for label, key in rows:
        cells = []
        for n in names:
            v = table[n][key]
            if isinstance(v, (list, tuple)):
                cells.append(f"{100 * v[0]:5.1f}% (n={v[1]})" if v[0] is not None else "-")
            elif key == "invented_per_read":
                cells.append(f"{v:.2f}" if v is not None else "-")
            else:
                cells.append(f"{100 * v:5.1f}%" if v is not None else "-")
        print(f"{label:24}" + "".join(f"{c:>16}" for c in cells))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--prompts", nargs="+", default=["v1", "v2", "v3"])
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--ollama-host")
    ap.add_argument("--signs", type=int, default=80, help="text signs (same order as calibration)")
    ap.add_argument("--views", type=float, nargs="+", default=[0, 30], help="viewing angles")
    ap.add_argument("--cache", default="calib_runs/v1/cache", help="reuse the calibration run's cache")
    ap.add_argument("--out", default="vlm_eval_results.json")
    a = ap.parse_args(argv)

    from signfilter import ocr_engine
    signs = [s for s in select_signs(load_dataset(a.gt, a.images), a.signs, 0) if s["text"]][:a.signs]
    cache, engine = Cache(a.cache), ocr_engine()
    readers = {v: Reader(v, a.model, a.ollama_host) for v in a.prompts}
    results = {v: [] for v in a.prompts}
    calls = 0
    for n, sign in enumerate(signs, 1):
        full = crop_of(sign)
        for angle in a.views:
            # the same view as `legibility.calibrate run` (scale 1.0 is scale index 0, variant 0)
            p = sample_params(angle, 1.0, stable_seed(sign["id"], angle, 0, 0))
            scene = degrade(full, p)
            crop, _, _ = sight(scene.image, scene.box, scene.K, engine)
            for v, rd in readers.items():
                entries, hit = rd(crop, cache)
                calls += not hit
                results[v].append({"sign": sign["id"], "view": angle, "read": entries,
                                   "score": score_sign(sign["text"], entries)})
        print(f"[{n}/{len(signs)}] {sign['id']}  ({calls} new reads)", flush=True)
    table = {f"{v}" + ("" if a.model == "gemma4:e4b" else f"@{a.model}"): summarize(r) for v, r in results.items()}
    Path(a.out).write_text(json.dumps({"model": a.model, "table": table, "results": results}, indent=1))
    print(f"\nmodel {a.model}, {len(signs)} signs x {len(a.views)} views\n")
    report(table)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
