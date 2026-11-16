"""Offline evaluation on SiGNgapore (Zimmerman et al., arXiv 2610.09488; data:
huggingface.co/datasets/NickyZimmerman/SiGNgapore, CC BY-NC-ND 4.0, test only).

Sign parsing: for each scenario (a real walk up to a sign) and each mission goal, which way
does the sign say? Labels: E*/S*/VQA/parsing_gt.json, a list of {place: direction}.

  # 1. read: run the sign pipeline (main.py, no hallway detection) on each scenario's video;
  #    reads are cached per scenario, so step 2 can be re-run in seconds
  python eval_signgapore.py read --root data/SiGNgapore --env 4
  # 2. score
  python eval_signgapore.py score --root data/SiGNgapore --env 4

Two scores, both over (scenario, mission goal) pairs whose goal is on the labelled sign:
- **official**: their metric (Scripts/EvalUtils.py EvalParsing): we write parsing.json per
  scenario, {place: direction} per sign we read, and their fuzzy goal matching
  (WRatio >= 90 on lower-cased text) picks the place; correct if the directions are equal.
  SignScene (the paper's baseline): 548/648 = 84.6% over all seven venues.
- **guide**: the same pairs, but our matcher (guide.py / matcher.py) decides which entry on the
  signs we read is the goal, as it would live.

Directions: our reader's arrows map to their words (up -> forward, none -> locational, ...);
an escalator / stairs entry gets their up-escalator / down-stairs style labels.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import subprocess
import sys

from rapidfuzz import fuzz, process, utils

ARROW_TO_SG = {"up": "forward", "up-left": "forward-left", "up-right": "forward-right",
               "left": "left", "right": "right", "down": "down", "down-left": "down-left",
               "down-right": "down-right", "none": "locational"}


def to_sg(text: str, arrow: str, via: str = None) -> str:
    """Our {text, arrow[, via]} -> their direction word."""
    a = ARROW_TO_SG.get(arrow, "locational")
    if via and arrow in ("up", "down"):   # reader said the way is an escalator / stairs
        return f"{arrow}-{via}"
    t = text.lower()
    for thing in ("escalator", "stairs"):
        if thing in t or (thing == "stairs" and "stair" in t):
            if arrow in ("up", "up-left", "up-right"):
                return f"up-{thing}"
            if arrow in ("down", "down-left", "down-right"):
                return f"down-{thing}"
    return a


def scenarios(root: str, env: int) -> list[str]:
    d = os.path.join(root, f"E{env}")
    names = [n for n in os.listdir(d) if re.fullmatch(r"S\d+", n)]
    return [os.path.join(d, n) for n in sorted(names, key=lambda n: int(n[1:]))]


def mission_goals(root: str, env: int) -> set[str]:
    out = set()
    for f in glob.glob(os.path.join(root, f"E{env}", "T*", "task.json")):
        out.add(json.load(open(f, encoding="utf-8"))["destination"])
    return out


def match_goal(goal: str, sign: dict):
    """Their matchGoal2Sign: thefuzz process.extract (WRatio, default processing), >= 90."""
    places = list(sign.keys())
    if not places:
        return None
    best = process.extractOne(goal.lower(), places, scorer=fuzz.WRatio,
                              processor=utils.default_process)
    return best[0] if best and best[1] >= 90 else None


def hfov_of(scen: str) -> float:
    import cv2
    rows = [[float(x) for x in line.split(",")]
            for line in open(os.path.join(scen, "camera_matrix.csv")) if line.strip()]
    w = cv2.VideoCapture(os.path.join(scen, "rgb.mp4")).get(cv2.CAP_PROP_FRAME_WIDTH)
    return math.degrees(2 * math.atan(w / 2 / rows[0][0]))


# ---------------------------------------------------------------- read

def cmd_read(a):
    for scen in scenarios(a.root, a.env):
        name = os.path.basename(scen)
        if a.only and name not in a.only:
            continue
        out = os.path.join(a.out, f"E{a.env}", name)
        reads = os.path.join(out, "reads.jsonl")
        if os.path.exists(reads) and not a.force:
            continue
        if not os.path.exists(os.path.join(scen, "rgb.mp4")):
            print(f"[read] E{a.env}/{name}: no rgb.mp4 yet, skipped", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        if os.path.exists(reads):
            os.remove(reads)
        cmd = [sys.executable, "main.py", "--source", os.path.join(scen, "rgb.mp4"),
               "--hfov", f"{hfov_of(scen):.1f}", "--fps", str(a.fps), "--no-intersections",
               "--no-tts", "--debug-dir", out] + a.main_args
        print(f"[read] E{a.env}/{name} ...", flush=True)
        with open(os.path.join(out, "run.txt"), "w", encoding="utf-8", errors="replace") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if r.returncode != 0:
            print(f"[read] E{a.env}/{name} FAILED (see {out}/run.txt)", flush=True)
            if os.path.exists(reads):
                os.replace(reads, reads + ".partial")


# ---------------------------------------------------------------- score

def load_reads(path: str) -> list[dict]:
    """Confirmed reads, the last per track (the closest look), in time order."""
    by_track = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("entries") and r.get("confirmed", True):
            by_track[r["track"]] = r
    return sorted(by_track.values(), key=lambda r: r["t"])


def parsing_pred(reads: list[dict]) -> list[dict]:
    return [{e["text"].lower(): to_sg(e["text"], e.get("arrow", "none"), e.get("via"))
             for e in r["entries"] if e.get("text")} for r in reads]


def guide_direction(goal: str, reads: list[dict]):
    """Our matcher: the best entry for the goal over all signs read (later reads win ties)."""
    from matcher import MatchLevel, match_sign
    from roomid import rule_for_goal
    from targets import parse_target
    rule = rule_for_goal(goal)
    target = parse_target(goal, rule)
    best = None
    for r in reads:
        m = match_sign(target, r["entries"], rule)
        if m.level > MatchLevel.NONE and (best is None or m.level >= best[0].level):
            best = (m, r)
    if best is None:
        return None, None
    m, r = best
    text = m.entry.source or m.entry.raw
    via = next((e.get("via") for e in r["entries"] if e.get("text") in (m.entry.source, m.entry.raw)),
               None)
    return to_sg(text, m.arrow or "none", via), m


def cmd_score(a):
    goals = mission_goals(a.root, a.env)
    off_ok = off_n = g_ok = g_n = 0
    no_reads = missed_goal = 0
    confusion = collections.Counter()
    rows = []
    for scen in scenarios(a.root, a.env):
        name = os.path.basename(scen)
        gt_path = os.path.join(scen, "VQA", "parsing_gt.json")
        if not os.path.exists(gt_path):
            continue
        gt = json.load(open(gt_path, encoding="utf-8"))
        out = os.path.join(a.out, f"E{a.env}", name)
        reads_path = os.path.join(out, "reads.jsonl")
        reads = load_reads(reads_path) if os.path.exists(reads_path) else None
        pred = parsing_pred(reads) if reads is not None else None
        if pred is not None:
            json.dump(pred, open(os.path.join(out, "parsing.json"), "w", encoding="utf-8"), indent=1)
        for goal in sorted(goals):
            gt_dir = gt_place = None
            for sign in gt:
                p = match_goal(goal, sign)
                if p:
                    gt_place, gt_dir = p, sign[p]
            if gt_dir is None:
                continue
            if reads is None:
                no_reads += 1
                continue
            # official
            off_n += 1
            pred_dir = pred_place = None
            for sign in pred:
                p = match_goal(goal, sign)
                if p:
                    pred_place, pred_dir = p, sign[p]
            off_ok += pred_dir == gt_dir
            # ours
            g_n += 1
            our_dir, m = guide_direction(goal, reads)
            g_ok += our_dir == gt_dir
            if our_dir is None:
                missed_goal += 1
            confusion[(gt_dir, our_dir or "-")] += 1
            rows.append(dict(scenario=name, goal=goal, gt_place=gt_place, gt=gt_dir,
                             official_place=pred_place, official=pred_dir,
                             guide_entry=(m.entry.raw if m else None), guide=our_dir))
    pct = lambda k, n: f"{k}/{n} = {100 * k / n:.1f}%" if n else f"{k}/0"
    print(f"E{a.env} sign parsing, {len(goals)} mission goals")
    print(f"  official metric (their matching): {pct(off_ok, off_n)}")
    print(f"  guide (our matching):             {pct(g_ok, g_n)}   goal not found on our reads: {missed_goal}")
    if no_reads:
        print(f"  (not counted: {no_reads} goal/scenario pairs without reads yet)")
    print("  label -> ours (count):")
    for (gt_dir, ours), k in sorted(confusion.items(), key=lambda x: -x[1])[:25]:
        print(f"    {gt_dir:18} -> {ours:18} {k}{'  ok' if gt_dir == ours else ''}")
    path = os.path.join(a.out, f"E{a.env}", "score.json")
    json.dump(dict(official=[off_ok, off_n], guide=[g_ok, g_n], rows=rows),
              open(path, "w", encoding="utf-8"), indent=1)
    print(f"  per pair: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("read", "score"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="data/SiGNgapore")
        p.add_argument("--env", type=int, required=True)
        p.add_argument("--out", default="debug_signgapore")
        if name == "read":
            p.add_argument("--fps", type=float, default=3.0, help="frames processed per second")
            p.add_argument("--force", action="store_true", help="re-read cached scenarios")
            p.add_argument("--only", nargs="*", help="scenario names, e.g. S0 S3")
            p.add_argument("--main-args", nargs=argparse.REMAINDER, default=[],
                           help="extra main.py options (last), e.g. --main-args --crop-context 0.5")
    a = ap.parse_args()
    {"read": cmd_read, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
