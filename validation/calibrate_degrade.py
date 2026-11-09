import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from validation import degrade as dg, idsse  # noqa: E402

RANGES = {"a0": (np.log(1 / 30000), np.log(1 / 100)), "a_dead": (0, 6), "a_line": (0, 5),
          "a_air": (0, 3), "a_near": (0, 4), "b0": (np.log(1 / 3000), np.log(1 / 30)), "b_dead": (-8, 1)}
LOG_KEYS = ("a0", "b0")


def sample(rng, centre=None, scale=1.0):
    out = {}
    for key, (lo, hi) in RANGES.items():
        if centre is None:
            value = rng.uniform(lo, hi)
        else:
            mid = np.log(centre[key]) if key in LOG_KEYS else centre[key]
            value = np.clip(mid + rng.normal(0, scale * (hi - lo) / 6), lo, hi)
        out[key] = float(np.exp(value)) if key in LOG_KEYS else float(value)
    return dg.BallLoss(**out)


def evaluate(matches, players, params, cache, seed=0):
    losses, stats = [], []
    for m, (xy, team), (cov, near, speed) in zip(matches, players, cache):
        present = dg.ball_present_mask(m, params, np.random.default_rng(seed), cov)
        d = replace(m, xy=xy, team=team, ball=np.where(present[:, None], m.ball, np.nan))
        s = dg.ball_gap_stats(d, near, speed)
        stats.append(s)
        losses.append(dg.stats_loss(s))
    return float(np.mean(losses)), stats


def main(n_wide=500, n_local=400):
    rng = np.random.default_rng(0)
    matches = [idsse.load(mid) for mid in idsse.available()]
    matches = [replace(m, ball=dg.unfreeze_dead_ball(m, np.random.default_rng(2))) for m in matches]
    players = [dg.degrade_players(m, dg.PlayerNoise(), np.random.default_rng(1)) for m in matches]
    cache = []
    for m, (xy, team) in zip(matches, players):
        d = replace(m, xy=xy, team=team)
        cache.append((dg.loss_covariates(m), dg.nearest_player(d), dg.player_mean_speed(d)))
    best = (np.inf, None, None)
    for i in range(n_wide + n_local):
        params = sample(rng) if i < n_wide else sample(rng, best[1].__dict__, 0.5)
        loss, stats = evaluate(matches, players, params, cache)
        if loss < best[0]:
            best = (loss, params, stats)
    print("best loss {:.4f}".format(best[0]))
    print(best[1])
    for m, s in zip(matches, best[2]):
        print("  {:<14} {}".format(m.name, {k: round(v, 3) for k, v in s.items()}))
    print("  target         {}".format(dg.TARGETS))


if __name__ == "__main__":
    main()
