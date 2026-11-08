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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legibility.calibrate import (Cache, GemmaReader, OcrProxyReader, crop_of, load_dataset,  # noqa: E402
                                  score_read, select_signs)
from legibility.degrade import degrade, sample_params, sight, stable_seed  # noqa: E402
from legibility.model import LegibilityModel  # noqa: E402
from legibility.trigger import (DONE, READ, UNREADABLE, ReadTrigger, cross_check,  # noqa: E402
                                cross_check_score)



def approach_scales(start_scale=0.06, steps=40, dwell=4):
    """Constant walking speed: distance falls linearly from 1/start_scale to 1 (full size),
    then `dwell` more sightings at the closest view (a person stops in front of the sign for
    a moment; each sighting gets its own blur/noise)."""
    d = np.linspace(1 / start_scale, 1.0, steps)
    return [float(1 / x) for x in d] + [1.0] * dwell


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
    best_support, best = -1, None
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
        ocr_text = " ".join(ocr.txts or ())
        ok = cross_check(entries, ocr_text)
        support = cross_check_score(entries, ocr_text)[0]
        correct, halluc, _ = score_read(gt, entries, symbols, reader.ignore_arrows)
        last_recall = sum(correct) / len(gt)
        last_text = sum(score_read(gt, entries, symbols, True)[0]) / len(gt)
        events.append({"step": k, "scale": s, "h_p25": m.h_p25, "n_lines": m.n_lines, "ok": ok,
                       "recall": last_recall, "text_recall": last_text, "hallucinated": halluc})
        res = trig.read_result(ok)
        if ok:
            final_recall, final_text = last_recall, last_text
        elif final_recall is None and support > best_support:
            # no read passed yet: keep the best-supported one (knowable live), not the last
            best_support, best = support, (last_recall, last_text)
        if res.action in (DONE, UNREADABLE):
            break
    if final_recall is None and best is not None:
        final_recall, final_text = best
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


def reachable(sign, angle, model, engine):
    """Does the sign's text get legible by the end of a walk, i.e. at full photo size? Many
    dataset photos were taken from far away; a walk toward them can't get closer than that."""
    full = crop_of(sign)
    sc = degrade(full, sample_params(angle, 1.0, stable_seed("reach", sign["id"], angle)))
    return model.is_legible(sight(sc.image, sc.box, sc.K, engine)[2])


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
    ap.add_argument("--prompt", help="reader prompt version (vlm.PROMPTS); default vlm.DEFAULT_PROMPT")
    ap.add_argument("--signs", type=int, default=20)
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 30, 50, 70])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dwell", type=int, default=4, help="extra sightings at the closest view")
    ap.add_argument("--start-scale", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--all-signs", action="store_true",
                    help="don't skip signs that never get legible within their photo")
    ap.add_argument("--out", default="sim_runs/run")
    a = ap.parse_args(argv)

    from signfilter import ocr_engine
    if a.synthetic:
        from legibility.synth import synthetic_signs
        signs = synthetic_signs(a.synthetic, a.seed)
    else:
        if not (a.gt and a.images):
            ap.error("--gt and --images are required (or --synthetic N)")
        # all text signs, shuffled; the reachable ones are picked below
        signs = [s for s in select_signs(load_dataset(a.gt, a.images), 10 ** 6, a.seed) if s["text"]]
    reader = GemmaReader(a.model, a.ollama_host, a.prompt) if a.reader == "gemma" else OcrProxyReader()
    model = LegibilityModel.load(a.calib)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache, engine = Cache(out / "cache"), ocr_engine()
    results = []
    skipped = []
    for sign in signs:
        if len(results) >= a.signs:
            break
        angle = a.angles[len(results) % len(a.angles)]
        if not a.all_signs and not reachable(sign, angle, model, engine):
            skipped.append(sign["id"])
            continue
        r = simulate_sign(sign, angle, model, reader, cache, engine, approach_scales(a.start_scale, a.steps, a.dwell))
        results.append(r)
        i = len(results)
        print(f"[{i}/{a.signs}] {sign['id']} @ {angle:.0f} deg: {r['reads']} read(s), {r['state']}, "
              f"final recall {100 * r['final_recall']:.0f}% (ceiling {100 * r['ceiling_recall']:.0f}%)"
              + ("" if r["category"] == "evaluated" else f"  [{r['category']}]"))
    s = summarize(results)
    s["skipped_unreachable"] = len(skipped)
    s["dry_run"] = bool(a.synthetic) or a.reader != "gemma"
    if skipped:
        print(f"skipped {len(skipped)} signs whose text never gets legible within their photo "
              "(the photo is the closest view a simulation has); --all-signs keeps them")
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
