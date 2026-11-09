import os
import sys
from functools import partial

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

NOBODY = -1


def nearest_team(match):
    codes = np.full(match.n_frames, NOBODY, np.int8)
    frames = np.flatnonzero(~np.isnan(match.ball[:, 0]))
    diff = match.xy[frames] - match.ball[frames, None, :]
    dist2 = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1]
    dist2 = np.where(np.isnan(dist2), np.inf, dist2)
    best = np.argmin(dist2, axis=1)
    found = np.isfinite(dist2[np.arange(len(frames)), best])
    codes[frames[found]] = match.team[best[found]]
    return codes


def majority(codes, window):
    if window <= 0:
        return np.where(codes == NOBODY, pm.D, codes).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(codes == pm.H)))
    away = np.concatenate(([0], np.cumsum(codes == pm.A)))
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - window), np.minimum(len(codes), index + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    return np.where(h + a == 0, pm.D, np.where(h >= a, pm.H, pm.A)).astype(np.int8)


def baseline(match, window=50):
    return majority(nearest_team(match), window)


PIPELINES = {
    "baseline_raw": partial(baseline, window=0),
    "baseline25": partial(baseline, window=25),
    "baseline50": partial(baseline, window=50),
    "baseline125": partial(baseline, window=125),
}


def run(name, match):
    codes = PIPELINES[name](match)
    if codes.shape != (match.n_frames,) or not np.isin(codes, (pm.H, pm.A, pm.D, pm.X)).all():
        raise ValueError("pipeline {} returned malformed codes".format(name))
    return codes
