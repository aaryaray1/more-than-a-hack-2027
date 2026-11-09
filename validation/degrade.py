import os
import sys
from dataclasses import dataclass, replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from solution.match import AWAY, HOME, Match  # noqa: E402

# BAR-ATH tactical ATD aggregates over graded frames, from ball_gap_stats() (IMPROVEMENTS.md §3)
TARGETS = {"missing": 0.407, "gap_p50": 268.0, "present_p50": 345.0, "speed_ratio": 0.54,
           "line_end": 0.289}
LINE_MARGIN_M = 2.05
AIR_M = 2.0
MAX_PLAYER_SPEED = 12.0


@dataclass(frozen=True)
class BallLoss:  # fit by validation/calibrate_degrade.py to TARGETS (loss 0.10)
    a0: float = 0.00081  # present -> lost hazard per frame
    a_dead: float = 1.68
    a_line: float = 0.0
    a_air: float = 2.0
    b0: float = 0.0060  # lost -> found hazard per frame while the ball is live
    b_dead: float = -4.60


@dataclass(frozen=True)
class PlayerNoise:
    tracklet_median: float = 886.0
    absence_median: float = 65.0
    sigma: float = 1.0
    slots_per_team: int = 23
    dup_frame_share: float = 0.024
    dup_median: float = 50.0
    pos_sigma: float = 0.20  # constant per tracklet; per-frame noise would fake ~3.5 m/s of speed


def near_line(ball, length, width, margin=LINE_MARGIN_M):
    x, y = ball[:, 0], ball[:, 1]
    return (x < margin) | (x > length - margin) | (y < margin) | (y > width - margin)


def ball_present_mask(match, params, rng):
    dead = (match.labels == pm.D).astype(float)
    line = near_line(match.ball, match.length, match.width).astype(float)
    air = (np.nan_to_num(match.ball_z) > AIR_M).astype(float)
    lose = np.clip(params.a0 * np.exp(params.a_dead * dead + params.a_line * line
                                      + params.a_air * air), 0, 1)
    find = np.clip(params.b0 * np.exp(params.b_dead * dead), 0, 1)
    u = rng.random(match.n_frames)
    present = np.zeros(match.n_frames, bool)
    for start, stop in match.halves:
        state = True
        for t in range(start, stop):
            present[t] = state
            state = (u[t] >= lose[t]) if state else (u[t] < find[t])
    return present & ~np.isnan(match.ball[:, 0])


def _segments(rng, n, median_on, median_off, sigma, start_on):
    out, t, on = [], 0, start_on
    while t < n:
        length = max(1, int(rng.lognormal(np.log(median_on if on else median_off), sigma)))
        if on:
            out.append((t, min(n, t + length)))
        t += length
        on = not on
    return out


def degrade_players(match, noise, rng):
    tracklets = []
    for slot in range(len(match.team)):
        seen = ~np.isnan(match.xy[:, slot, 0])
        for start, stop in match.halves:
            idx = np.flatnonzero(seen[start:stop])
            if not len(idx):
                continue
            lo, hi = start + idx[0], start + idx[-1] + 1
            for a, b in _segments(rng, hi - lo, noise.tracklet_median, noise.absence_median,
                                  noise.sigma, rng.random() < 0.95):
                tracklets.append((lo + a, lo + b, slot, None))

    n_dup = rng.poisson(noise.dup_frame_share * match.n_frames
                        / (noise.dup_median * np.exp(noise.sigma ** 2 / 2)))
    for _ in range(n_dup):
        slot = int(rng.integers(len(match.team)))
        start = int(rng.integers(match.n_frames))
        stop = min(match.n_frames, start + max(1, int(rng.lognormal(np.log(noise.dup_median),
                                                                     noise.sigma))))
        tracklets.append((start, stop, slot, rng.normal(0, 0.7, 2)))

    n_out = 2 * noise.slots_per_team
    out_team = np.repeat(np.array([HOME, AWAY], np.int8), noise.slots_per_team)
    xy = np.full((match.n_frames, n_out, 2), np.nan)
    busy = np.zeros(n_out, np.int64)
    for start, stop, slot, offset in sorted(tracklets, key=lambda t: (t[0], t[2])):
        free = np.flatnonzero((out_team == match.team[slot]) & (busy <= start))
        if not len(free):
            continue
        target = free[0]
        busy[target] = stop
        piece = match.xy[start:stop, slot].copy()
        if offset is not None:
            piece += offset
        xy[start:stop, target] = piece + rng.normal(0, noise.pos_sigma, 2)
    return xy, out_team


def degrade(match, seed, ball_loss=BallLoss(), noise=PlayerNoise()):
    rng = np.random.default_rng(seed)
    present = ball_present_mask(match, ball_loss, rng)
    ball = np.where(present[:, None], match.ball, np.nan)
    xy, team = degrade_players(match, noise, rng)
    return replace(match, name="{}~s{}".format(match.name, seed), xy=xy, team=team, ball=ball,
                   ball_z=None, meta=dict(match.meta, degrade_seed=seed))


def player_mean_speed(match):
    step = np.diff(match.xy, axis=0) * match.fps
    speed = np.hypot(step[..., 0], step[..., 1])
    speed = np.where(speed < MAX_PLAYER_SPEED, speed, np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(speed, axis=1)
    return np.concatenate(([np.nan], mean))


def ball_gap_stats(match):
    graded = match.in_halves()
    have = ~np.isnan(match.ball[:, 0])
    gaps, present_runs, line_end = [], [], []
    line = near_line(match.ball, match.length, match.width)
    for start, stop in match.halves:
        e = np.diff(np.concatenate(([0], (~have[start:stop]).astype(np.int8), [0])))
        gaps += (np.flatnonzero(e == -1) - np.flatnonzero(e == 1)).tolist()
        e = np.diff(np.concatenate(([0], have[start:stop].astype(np.int8), [0])))
        s, t = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
        present_runs += (t - s).tolist()
        line_end += line[start + t - 1].tolist()
    speed = player_mean_speed(match)
    with np.errstate(all="ignore"):
        ratio = np.nanmean(speed[graded & ~have]) / np.nanmean(speed[graded & have])
    return {"missing": float(np.mean(~have[graded])),
            "gap_p50": float(np.median(gaps)) if gaps else 0.0,
            "present_p50": float(np.median(present_runs)) if present_runs else 0.0,
            "speed_ratio": float(ratio),
            "line_end": float(np.mean(line_end)) if line_end else 0.0}


def stats_loss(stats, targets=TARGETS):
    return float(sum(np.log(max(stats[k], 1e-6) / targets[k]) ** 2 for k in targets))
