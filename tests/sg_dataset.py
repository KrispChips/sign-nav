"""SiGNgapore2D (github.com/AdaCompNUS/sign-understanding, gt/gt_annotation.json) as test input.

The ground truth is not copied into this repo (it has no license file). Point to it with
SIGN_UNDERSTANDING_GT=/path/to/gt_annotation.json, or clone the repo next to this one.
Run `python tests/sg_dataset.py` for the full report.

Each annotated sign has "text labels", "symbol labels" and "mixed" mappings of label ->
direction (a few directions are lists). Every (label, direction) pair becomes one
{text, arrow} entry, through the same arrow adapter the VLM output uses."""
import collections
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import facilities  # noqa: E402
from entries import canonical_arrow, normalize, parse_entry  # noqa: E402
from matcher import MatchLevel, match_sign  # noqa: E402
from targets import load_rule, parse_target  # noqa: E402

LABEL_KEYS = ("text labels", "symbol labels", "mixed")
CANDIDATES = (
    ROOT.parent / "sign-understanding" / "gt" / "gt_annotation.json",
    ROOT.parent / "adacompnus" / "sign-understanding" / "gt" / "gt_annotation.json",
    ROOT / "data" / "sign-understanding" / "gt" / "gt_annotation.json",
)


def find_gt():
    env = os.environ.get("SIGN_UNDERSTANDING_GT")
    for p in ([Path(env)] if env else []) + list(CANDIDATES):
        if p.is_file():
            return p
    return None


def load_signs(path):
    """-> [{"id", "image", "entries": [{"text", "arrow", "kind"}]}]"""
    signs = []
    for img in json.loads(Path(path).read_text(encoding="utf-8")):
        for ann in img.get("annotation", []):
            entries = []
            for kind in LABEL_KEYS:
                for text, direction in (ann.get(kind) or {}).items():
                    for d in (direction if isinstance(direction, list) else [direction]):
                        entries.append({"text": text, "arrow": canonical_arrow(d), "kind": kind})
            if entries:
                signs.append({"id": f"{img['imagePath']}#{ann.get('objectID')}",
                              "image": img["imagePath"], "entries": entries})
    return signs


def coverage(signs, rule):
    types = collections.Counter()
    unknown = []
    for s in signs:
        for e in s["entries"]:
            for p in parse_entry(e["text"], e["arrow"], rule):
                types[p.type] += 1
                if p.type == "UNKNOWN":
                    unknown.append(e["text"])
    return types, unknown


def _norm(text):
    return normalize(text).text


def self_consistency(signs, rule):
    """For each entry, use it as the target; the sign must return it with its arrow."""
    results = []
    for s in signs:
        seen = set()
        for e in s["entries"]:
            key = _norm(e["text"])
            if key in seen:
                continue
            seen.add(key)
            arrows = {x["arrow"] for x in s["entries"] if _norm(x["text"]) == key}
            parts = parse_entry(e["text"], e["arrow"], rule)
            room = None
            if len(parts) == 1 and parts[0].type == "RANGE":
                room = parts[0].ranges[0][0]
                target = parse_target(room.text, rule)
            else:
                target = parse_target(e["text"], rule)
            m = match_sign(target, s["entries"], rule)
            pool = m.ties if m.ambiguous else ([m.best] if m.best else [])
            ok = any(_norm(r.entry.source) == key and r.entry.arrow in arrows for r in pool)
            if not ok and room is not None and m.level == MatchLevel.EXACT:
                ok = True   # the sign also names the range's low room exactly: that wins
            results.append(dict(sign=s["id"], text=e["text"], target=target.kind, ok=ok,
                                ambiguous=m.ambiguous, level=m.level.name, explain=m.explain()))
    return results


def negatives(signs, rule, per_sign=3, seed=0):
    """Targets from other images must never come out EXACT or RANGE. Pairs are skipped only
    on parser-independent grounds (see _plausibly_present)."""
    rnd = random.Random(seed)
    pool = [(s["image"], e["text"]) for s in signs for e in s["entries"]]
    out, skipped = [], 0
    for s in signs:
        texts = [_norm(e["text"]) for e in s["entries"]]
        picks = 0
        while picks < per_sign:
            image, text = rnd.choice(pool)
            if image == s["image"]:
                continue
            picks += 1
            if _plausibly_present(_norm(text), texts):
                skipped += 1
                continue
            m = match_sign(parse_target(text, rule), s["entries"], rule)
            out.append(dict(sign=s["id"], target=text, level=m.level.name, explain=m.explain(),
                            bad=m.level in (MatchLevel.EXACT, MatchLevel.RANGE)))
    return out, skipped


def _plausibly_present(target, sign_texts):
    if any(target == t or target in t or t in target for t in sign_texts):
        return True   # the same destination really is on this sign
    fac = facilities.lookup(target)
    if fac and any(facilities.lookup(t) and facilities.lookup(t)[0] == fac[0] for t in sign_texts):
        return True   # TOILET vs WASHROOM: a true match, not a negative
    unit = re.match(r"^(B?\d{1,2})-\d", target)
    if unit and any(" TO " in t and unit.group(1) + "-" in t for t in sign_texts):
        return True   # a unit number next to a range on the same floor might really be in it
    return False


def hand_labels():
    return json.loads((Path(__file__).parent / "data" / "sg_handlabels.json").read_text())


def label_accuracy(rule):
    rows = []
    for item in hand_labels()["labels"]:
        got = [p.type for p in parse_entry(item["text"], "none", rule)]
        rows.append(dict(item, got=got, ok=got == item["types"]))
    return rows


def report(path):
    rule = load_rule("singapore_units")
    signs = load_signs(path)
    n_entries = sum(len(s["entries"]) for s in signs)
    print(f"{path}\n{len(signs)} signs, {n_entries} entries (rule: {rule.name})\n")
    types, unknown = coverage(signs, rule)
    total = sum(types.values())
    print("Parsed entry types:")
    for t, c in types.most_common():
        print(f"  {t:9} {c:5}  {100 * c / total:5.1f}%")
    print(f"UNKNOWN rate: {100 * types['UNKNOWN'] / total:.1f}%  {sorted(set(unknown))}\n")

    rows = label_accuracy(rule)
    acc = sum(r["ok"] for r in rows) / len(rows)
    print(f"Hand-labelled entries: {len(rows)}, type accuracy {100 * acc:.1f}%")
    for r in rows:
        if not r["ok"]:
            print(f"  MISS {r['text']!r}: labelled {r['types']}, parsed {r['got']}  ({r.get('note', '')})")
    print()

    sc = self_consistency(signs, rule)
    ok = sum(r["ok"] for r in sc)
    amb = sum(r["ambiguous"] for r in sc)
    print(f"Self-consistency: {ok}/{len(sc)} = {100 * ok / len(sc):.1f}%  "
          f"(ambiguous results: {amb}, by target kind: "
          f"{dict(collections.Counter(r['target'] for r in sc))})")
    for r in sc:
        if not r["ok"]:
            print(f"  FAIL {r['text']!r}: {r['explain']}")
    print()

    neg, skipped = negatives(signs, rule)
    bad = [r for r in neg if r["bad"]]
    print(f"Negatives: {len(neg)} checked ({skipped} skipped as plausibly present), "
          f"false EXACT/RANGE: {len(bad)}, levels {dict(collections.Counter(r['level'] for r in neg))}")
    for r in bad:
        print(f"  BAD {r['target']!r} on {r['sign']}: {r['explain']}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else find_gt()
    if not p:
        sys.exit("ground truth not found: set SIGN_UNDERSTANDING_GT or pass the path")
    report(p)
