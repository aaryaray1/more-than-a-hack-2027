import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from solution import models  # noqa: E402
from validation import harness  # noqa: E402

GRID = {
    "penalty": [8.0, 64.0],
    "radius": [1.0, 1.5],
    "min_frames": [1, 3],
    "min_s": [2.0],
    "gap_switch": ["mid"],
    "margin": [0.0, 0.3, 0.6, 1.0, 1.5, 2.5],
}


TEAM_GRID = {
    "penalty": [32.0, 64.0],
    "team_penalty": [2.0, 4.0, 8.0, 16.0, 32.0],
    "min_s": [1.0, 1.5, 2.0, 2.5],
}


def main():
    parser = argparse.ArgumentParser(description="Sweep DeadAliveTeam decode settings, LOGO")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--team", action="store_true", help="sweep the learned team model (v2)")
    args = parser.parse_args()
    grid = TEAM_GRID if args.team else GRID
    cls = models.LearnedTeam if args.team else models.DeadAliveTeam

    seeds = tuple(range(args.seeds))
    train = harness.mixed_runs(seeds)
    runs = harness.mixed_runs(seeds)
    games = sorted({g for g, _, _ in runs})
    p_dead, p_home = {}, {}
    for game in games:
        model = cls().fit([(m, t["control"]) for g, m, t in train if g != game])
        for g, m, _ in runs:
            if g == game:
                p_dead[m.name] = model.dead_probability(m)
                if args.team:
                    p_home[m.name] = model.home_probability(m)

    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        model = cls(**params)
        scores = {"control": [], "raw": []}
        per_game = {g: [] for g in games}
        for g, m, truth in runs:
            pred = (model.predict(m, p_dead[m.name], p_home[m.name]) if args.team
                    else model.predict(m, p_dead[m.name]))
            for variant in scores:
                s = pm.spell_scores(truth[variant], pred, pm.valid_runs(truth[variant]))["mean_f1"]
                scores[variant].append(s)
            per_game[g].append(scores["control"][-1])
        results.append((float(np.mean(scores["control"])), float(np.mean(scores["raw"])),
                        min(float(np.mean(v)) for v in per_game.values()), params))

    results.sort(key=lambda r: -r[0])
    print("{} configs, {} runs ({} games x {} seeds)".format(len(results), len(runs), len(games),
                                                           args.seeds))
    print("{:>8} {:>8} {:>9}  params".format("control", "raw", "worst_gm"))
    for control, raw, worst, params in results[:args.top]:
        print("{:>8.4f} {:>8.4f} {:>9.4f}  {}".format(control, raw, worst, params))
    default = [r for r in results if r[3] == {"penalty": 8.0, "radius": 1.0, "min_frames": 3,
                                              "min_s": 2.0, "gap_switch": "mid", "margin": 0.0}]
    if default:
        print("default  {:.4f} {:.4f} {:.4f}".format(*default[0][:3]))
    for column, label in ((0, "control"), (1, "raw")):
        print("best per value ({})".format(label))
        for key in keys:
            marg = {}
            for row in results:
                marg.setdefault(row[3][key], []).append(row[column])
            print("  {:<11} ".format(key) + "  ".join("{}: {:.4f}".format(v, max(s)) for v, s in marg.items()))


if __name__ == "__main__":
    main()
