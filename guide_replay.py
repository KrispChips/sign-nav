"""Replay recorded sign reads (debug_out/reads.jsonl) through the guide, without video, GPU
or Gemma: what would have been said for a goal. Seconds per walk, for trying goals and
wording on real reads.

  python guide_replay.py --goal 014-190                    # every run in the log
  python guide_replay.py --goal A140 --run 7 --all          # one run, noted reads too
  python guide_replay.py --goal 014-190 --say "3:I think it's on the left" --say "40:read it"

A log holds one run after another (main.py appends); a run starts where time goes back.
Only reads that were spoken at the time (OCR-confirmed) are replayed, as in main.py.
"""
import argparse
import json
import sys

from guide import Guide, Read


def runs(path: str) -> list[list[dict]]:
    out: list[list[dict]] = [[]]
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if out[-1] and r["t"] < out[-1][-1]["t"] - 1.0:
            out.append([])
        out[-1].append(r)
    return [r for r in out if r]


def confirmed(r: dict) -> bool:
    if "confirmed" in r:
        return bool(r["confirmed"]) and bool(r["entries"])
    return bool(r["entries"]) and bool(r.get("said"))


def replay(run: list[dict], goal: str, rule=None, says=(), show_all=False, out=sys.stdout):
    g = Guide(goal, rule_override=rule)
    events = sorted([(t, "say", text) for t, text in says]
                    + [(r["t"], "read", r) for r in run if confirmed(r)], key=lambda e: e[0])
    d = g.start(run[0]["t"] if run else 0.0)
    print(f"  {d.t:6.1f}  {'question':8}  {d.text}", file=out)
    counts: dict[str, int] = {}
    for t, kind, x in events:
        if kind == "say":
            d = g.hear(x, t)
            print(f"  {t:6.1f}  {'you':8}  \"{x}\"", file=out)
        else:
            d = g.on_read(Read(x["track"], x["entries"], t, x.get("rel") or 0.0,
                               x.get("mount") or "facing", None))
        counts[d.kind] = counts.get(d.kind, 0) + 1
        if d.text or show_all:
            what = d.text or f"({d.reason}: " + "; ".join(
                f"{e['text']}" + ("" if e.get("arrow", "none") == "none" else f" [{e['arrow']}]")
                for e in (x["entries"] if kind == "read" else [])) + ")"
            print(f"  {t:6.1f}  {d.kind:8}  {what}", file=out)
    print("  totals: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), file=out)
    return g, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", default="debug_out/reads.jsonl")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--rule", help="optional building numbering override")
    ap.add_argument("--run", type=int, action="append", help="run index (default: all)")
    ap.add_argument("--say", action="append", default=[], metavar="T:TEXT",
                    help="what the user says at time T (s), e.g. '5:read it'")
    ap.add_argument("--all", action="store_true", help="also list noted reads")
    a = ap.parse_args()
    says = [(float(s.split(":", 1)[0]), s.split(":", 1)[1]) for s in a.say]
    rs = runs(a.reads)
    for i, run in enumerate(rs):
        if a.run and i not in a.run:
            continue
        n = sum(confirmed(r) for r in run)
        print(f"run {i}: {run[0]['t']:.1f}-{run[-1]['t']:.1f} s, {n} confirmed reads, goal {a.goal}")
        replay(run, a.goal, a.rule, says, a.all)


if __name__ == "__main__":
    main()
