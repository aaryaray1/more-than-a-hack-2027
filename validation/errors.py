import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from validation import harness  # noqa: E402

IOU = 0.5


def spell_table(truth, pred, match):
    runs = pm.valid_runs(truth)
    gt, pr = pm.possession_spells(truth, runs), pm.possession_spells(pred, runs)
    hit = pm.matched_gt_flags(pm.spell_overlaps(gt, pr), len(gt), IOU)
    visible = ~np.isnan(match.ball[:, 0])
    rows = []
    for (a, b, team), ok in zip(gt, hit):
        before = truth[a - 1] if a > 0 else pm.X
        rows.append({"len_s": (b - a) / match.fps, "vis": float(visible[a:b].mean()),
                     "start": "restart" if before in (pm.D, pm.X) else "turnover",
                     "start_seen": bool(visible[max(a - 12, 0):a + 13].any()), "hit": bool(ok)})
    return rows, len(pr) - int(hit.sum())


def bucket(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        if sel:
            out.append("{}-{}: {:.0%} of {}".format(lo, hi, np.mean([r["hit"] for r in sel]), len(sel)))
    return "  ".join(out)


def main():
    parser = argparse.ArgumentParser(description="Where does a pipeline miss possessions?")
    parser.add_argument("pipeline")
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()
    seeds = tuple(range(args.seeds))
    train = harness.mixed_runs(seeds)
    runs = harness.harness_runs(seeds)
    preds = harness.predict_runs(args.pipeline, runs, train)
    rows, spurious = [], 0
    for (_, match, truth), pred in zip(runs, preds):
        r, s = spell_table(truth["control"], pred, match)
        rows += r
        spurious += s
    print("{}: {} true possessions, {:.0%} matched at IoU {}; {} spurious predicted".format(
        args.pipeline, len(rows), np.mean([r["hit"] for r in rows]), IOU, spurious))
    print("by length (s)      ", bucket(rows, "len_s", [0, 3, 6, 12, 25, 60, 1000]))
    print("by ball visible    ", bucket(rows, "vis", [0, 0.25, 0.5, 0.75, 1.01]))
    for start in ("restart", "turnover"):
        sel = [r for r in rows if r["start"] == start]
        seen = [r for r in sel if r["start_seen"]]
        hidden = [r for r in sel if not r["start_seen"]]
        print("start {:<9} {:.0%} of {:>4} | ball seen at start {:.0%} of {:>4} | hidden {:.0%} of {:>4}".format(
            start, np.mean([r["hit"] for r in sel]), len(sel), np.mean([r["hit"] for r in seen]) if seen else 0,
            len(seen), np.mean([r["hit"] for r in hidden]) if hidden else 0, len(hidden)))


if __name__ == "__main__":
    main()
