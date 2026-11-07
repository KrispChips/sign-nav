"""Test 6: simulated approaches. Walk toward each sign at constant speed (far scale -> full
size), send every sighting through the live path and the read trigger, read when it fires,
and record reads per sign and the final recall.

  python -m legibility.simulate --calib legibility_calib.json --gt <gt.json> --images <dir> \
      --reader gemma --signs 20 --out sim_runs/v1

A read counts as OK for the trigger the way it would live: the returned entries are backed by
OCR on the same crop (the step 1 cross-check, partial ratio >= 80). The annotations are used
only afterwards, to score final recall. Signs at a blocked angle are reported separately.

Acceptance (plan): <= 2 reads per sign on average, final recall >= 90% for non-blocked signs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entries import normalize  # noqa: E402
from legibility.calibrate import (Cache, GemmaReader, OcrProxyReader, crop_of, load_dataset,  # noqa: E402
                                  score_read, select_signs)
from legibility.degrade import degrade, sample_params, sight, stable_seed  # noqa: E402
from legibility.model import LegibilityModel  # noqa: E402
from legibility.trigger import DONE, READ, UNREADABLE, ReadTrigger  # noqa: E402

OCR_AGREE = 80


def cross_check(entries, ocr_text) -> bool:
    """Live-style read check: at least half the returned entries appear in the OCR text."""
    if not entries:
        return False
    ocr = normalize(ocr_text or "").text
    hits = sum(fuzz.partial_ratio(normalize(e.get("text", "")).text, ocr) >= OCR_AGREE for e in entries)
    return hits >= max(1, len(entries) / 2)


def approach_scales(start_scale=0.06, steps=40):
    """Constant walking speed: distance falls linearly from 1/start_scale to 1 (full size)."""
    d = np.linspace(1 / start_scale, 1.0, steps)
    return [float(1 / x) for x in d]


def cached_read(reader, cache, crop):
    key = Cache.key(crop, reader.tag)
    hit = cache.get(key)
    if hit is None:
        entries, raw = reader(crop)
        cache.put(key, {"entries": entries, "raw": raw})
        return entries, True
    return hit["entries"], False


def simulate_sign(sign, angle, model, reader, cache, engine, scales, log=None):
    full = crop_of(sign)
    gt = sign["text"] or sign["symbol"]
    symbols = not sign["text"]
    # Ceiling: what the reader gets from the closest view this simulation can produce.
    near = degrade(full, sample_params(angle, 1.0, stable_seed("ceiling", sign["id"], angle)))
    near_crop, _, _ = sight(near.image, near.box, near.K, engine)
    near_entries, _ = cached_read(reader, cache, near_crop)
    ceiling = sum(score_read(gt, near_entries, symbols, reader.ignore_arrows)[0]) / len(gt)
    text_ceiling = sum(score_read(gt, near_entries, symbols, True)[0]) / len(gt)
    trig = ReadTrigger(model)
    legible = []   # per sighting, at the base threshold: could the trigger have fired by the end?
    reads, last_recall, final_recall, events = 0, 0.0, None, []
    last_text, final_text = 0.0, None
    for k, s in enumerate(scales):
        p = sample_params(angle, s, stable_seed("approach", sign["id"], angle, k))
        scene = degrade(full, p)
        crop, info, m = sight(scene.image, scene.box, scene.K, engine)
        legible.append(model.is_legible(m))
        d = trig.update(m)
        if d.action != READ:
            if d.action in (DONE, UNREADABLE):
                break
            continue
        entries, _ = cached_read(reader, cache, crop)
        reads += 1
        ocr = engine(crop, use_det=True, use_cls=False, use_rec=True)
        ok = cross_check(entries, " ".join(ocr.txts or ()))
        correct, halluc, _ = score_read(gt, entries, symbols, reader.ignore_arrows)
        last_recall = sum(correct) / len(gt)
        last_text = sum(score_read(gt, entries, symbols, True)[0]) / len(gt)
        events.append({"step": k, "scale": s, "h_p25": m.h_p25, "n_lines": m.n_lines, "ok": ok,
                       "recall": last_recall, "text_recall": last_text, "hallucinated": halluc})
        res = trig.read_result(ok)
        if ok:
            final_recall, final_text = last_recall, last_text
        if res.action in (DONE, UNREADABLE):
            break
    blocked = model.blocked_by_angle(m)
    # In range = legible in 2 of the last 3 sightings before the walk ended (or a read fired).
    in_range = reads > 0 or sum(legible[-3:]) >= 2
    category = ("blocked" if blocked else "evaluated" if in_range else "out_of_range")
    return {"sign": sign["id"], "angle": angle, "reads": reads,
            "state": trig.state or ("still approaching" if reads else "never fired"),
            "blocked": blocked, "category": category, "ceiling_recall": ceiling,
            "final_recall": final_recall if final_recall is not None else last_recall,
            "text_ceiling": text_ceiling,
            "final_text_recall": final_text if final_text is not None else last_text,
            "events": events}


def summarize(results):
    """`evaluated` = not blocked by angle and legible by the end of the approach. Signs that
    never become legible within the simulated range (the source photo is the closest view
    available) are `out_of_range` and reported separately, not counted as failures."""
    ev = [r for r in results if r["category"] == "evaluated"]
    mean = lambda xs: float(np.mean(xs)) if xs else None   # noqa: E731
    return {"signs": len(results), "mean_reads": mean([r["reads"] for r in results]) or 0.0,
            "evaluated": len(ev),
            "mean_final_recall_evaluated": mean([r["final_recall"] for r in ev]),
            "mean_ceiling_recall_evaluated": mean([r["ceiling_recall"] for r in ev]),
            "mean_final_text_recall_evaluated": mean([r.get("final_text_recall", 0) for r in ev]),
            "mean_text_ceiling_evaluated": mean([r.get("text_ceiling", 0) for r in ev]),
            "blocked": sum(r["category"] == "blocked" for r in results),
            "out_of_range": sum(r["category"] == "out_of_range" for r in results),
            "states": {s: sum(r["state"] == s for r in results) for s in sorted({r["state"] for r in results})}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--gt"); ap.add_argument("--images")
    ap.add_argument("--synthetic", type=int, help="DRY RUN: N rendered signs")
    ap.add_argument("--reader", choices=("gemma", "ocr"), default="gemma")
    ap.add_argument("--model", default="gemma4:e4b"); ap.add_argument("--ollama-host")
    ap.add_argument("--signs", type=int, default=20)
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 30, 50, 70])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--start-scale", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="sim_runs/run")
    a = ap.parse_args(argv)

    from signfilter import ocr_engine
    if a.synthetic:
        from legibility.synth import synthetic_signs
        signs = synthetic_signs(a.synthetic, a.seed)
    else:
        if not (a.gt and a.images):
            ap.error("--gt and --images are required (or --synthetic N)")
        signs = [s for s in select_signs(load_dataset(a.gt, a.images), a.signs, a.seed) if s["text"]][:a.signs]
    reader = GemmaReader(a.model, a.ollama_host) if a.reader == "gemma" else OcrProxyReader()
    model = LegibilityModel.load(a.calib)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache, engine = Cache(out / "cache"), ocr_engine()
    results = []
    for i, sign in enumerate(signs, 1):
        angle = a.angles[(i - 1) % len(a.angles)] if len(a.angles) > 1 else a.angles[0]
        r = simulate_sign(sign, angle, model, reader, cache, engine, approach_scales(a.start_scale, a.steps))
        results.append(r)
        print(f"[{i}/{len(signs)}] {sign['id']} @ {angle:.0f} deg: {r['reads']} read(s), {r['state']}, "
              f"final recall {100 * r['final_recall']:.0f}% (ceiling {100 * r['ceiling_recall']:.0f}%)"
              + ("" if r["category"] == "evaluated" else f"  [{r['category']}]"))
    s = summarize(results)
    s["dry_run"] = bool(a.synthetic) or a.reader != "gemma"
    (out / "results.json").write_text(json.dumps({"summary": s, "signs": results}, indent=1))
    if s["dry_run"]:
        print("*** DRY RUN: not an acceptance result ***")
    pct = lambda v: 100 * (v or 0)   # noqa: E731
    print(f"mean reads per sign {s['mean_reads']:.2f} (target <= 2); evaluated signs: {s['evaluated']}\n"
          f"  text read:      final {pct(s['mean_final_text_recall_evaluated']):.0f}% vs the reader's ceiling "
          f"{pct(s['mean_text_ceiling_evaluated']):.0f}% at the closest view\n"
          f"  text + arrow:   final {pct(s['mean_final_recall_evaluated']):.0f}% vs ceiling "
          f"{pct(s['mean_ceiling_recall_evaluated']):.0f}%\n"
          f"  blocked {s['blocked']}, out of range {s['out_of_range']}; states {s['states']}")


if __name__ == "__main__":
    main()
