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
TARGETS = {"missing": 0.407, "gap_p50": 268.0, "present_p50": 345.0, "speed_ratio": 0.541,
           "line_end": 0.289, "visible_outside": 0.042, "lost_near_p50": 1.454}
LINE_MARGIN_M = 2.05
AIR_M = 2.0
NEAR_M = 1.0
MAX_PLAYER_SPEED = 12.0


@dataclass(frozen=True)
class BallLoss:  # fit by validation/calibrate_degrade.py on 4 IDSSE games to TARGETS (loss 0.43)
    a0: float = 0.00023  # present -> lost hazard per frame
    a_dead: float = 3.20
    a_line: float = 1.20
    a_air: float = 1.25
    a_near: float = 2.18  # player within NEAR_M of the ball (occlusion)
    b0: float = 0.0130  # lost -> found hazard per frame while the ball is live
    b_dead: float = -5.42


@dataclass(frozen=True)
class PlayerNoise:
    tracklet_median: float = 886.0
    absence_median: float = 65.0
    sigma: float = 1.0
    slots_per_team: int = 23
    dup_frame_share: float = 0.25  # BAR-ATH: a team shows > 11 objects on 5-8% of frames
    dup_median: float = 50.0
    burst_frame_share: float = 0.30  # BAR-ATH: < 18 objects on 9.2% of frames
    burst_median: float = 50.0
    burst_players: tuple = (3, 8)
    pos_sigma: float = 0.20  # constant per tracklet; per-frame noise would fake ~3.5 m/s of speed


def near_line(ball, length, width, margin=LINE_MARGIN_M):
    x, y = ball[:, 0], ball[:, 1]
    return (x < margin) | (x > length - margin) | (y < margin) | (y > width - margin)


def loss_covariates(match):
    return {"dead": (match.labels == pm.D).astype(float),
            "line": near_line(match.ball, match.length, match.width).astype(float),
            "air": (np.nan_to_num(match.ball_z) > AIR_M).astype(float),
            "near": (nearest_player(match) < NEAR_M).astype(float)}


def ball_present_mask(match, params, rng, cov=None):
    cov = cov if cov is not None else loss_covariates(match)
    dead = cov["dead"]
    lose = np.clip(params.a0 * np.exp(params.a_dead * dead + params.a_line * cov["line"]
                                      + params.a_air * cov["air"] + params.a_near * cov["near"]), 0, 1)
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

    mean_len = noise.burst_median * np.exp(noise.sigma ** 2 / 2)
    for _ in range(rng.poisson(noise.burst_frame_share * match.n_frames / mean_len)):
        start = int(rng.integers(match.n_frames))
        stop = min(match.n_frames, start + max(1, int(rng.lognormal(np.log(noise.burst_median),
                                                                     noise.sigma))))
        k = int(rng.integers(noise.burst_players[0], noise.burst_players[1]))
        xy[start:stop, rng.choice(n_out, size=k, replace=False)] = np.nan
    return xy, out_team


def unfreeze_dead_ball(match, rng, roll_tau_s=0.7, max_roll_s=2.0):
    ball = match.ball.copy()
    dead = match.labels == pm.D
    fps = match.fps
    for start, stop in match.halves:
        edges = np.diff(np.concatenate(([0], dead[start:stop].astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1) + start, np.flatnonzero(edges == -1) + start):
            p0 = ball[a - 1] if a > start else ball[a]
            v0 = (ball[a - 1] - ball[a - 2]) * fps if a - 2 >= start else np.zeros(2)
            v0 = v0 * min(1.0, 25.0 / max(np.hypot(*v0), 1e-6))
            p1 = ball[b] if b < stop else None
            n = b - a
            t = np.arange(1, n + 1) / fps
            path = p0 + np.outer(roll_tau_s * (1 - np.exp(-np.minimum(t, max_roll_s) / roll_tau_s)), v0)
            if p1 is not None:
                place = min(n, int(rng.uniform(1.0, 6.0) * fps))
                roll_end = min(n - place, int(max_roll_s * fps))
                carry = n - place - roll_end
                start_carry = path[roll_end - 1] if roll_end > 0 else p0
                if carry > 0:
                    w = np.arange(1, carry + 1)[:, None] / carry
                    path[roll_end:roll_end + carry] = start_carry + w * (p1 - start_carry)
                path[n - place:] = p1 + rng.normal(0, 0.05, (place, 2))
            ball[a:b] = path
    return ball


def degrade(match, seed, ball_loss=BallLoss(), noise=PlayerNoise()):
    rng = np.random.default_rng(seed)
    if match.labels is not None and match.meta.get("dead_ball_frozen", True):
        match = replace(match, ball=unfreeze_dead_ball(match, rng))
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


def nearest_player(match):
    diff = match.xy - match.ball[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist).min(axis=1)
    return np.where(np.isfinite(dist), dist, np.nan)


def ball_gap_stats(match, near=None, speed=None):
    graded = match.in_halves()
    have = ~np.isnan(match.ball[:, 0])
    gaps, present_runs, line_end, near_end = [], [], [], []
    line = near_line(match.ball, match.length, match.width)
    near = near if near is not None else nearest_player(match)
    for start, stop in match.halves:
        e = np.diff(np.concatenate(([0], (~have[start:stop]).astype(np.int8), [0])))
        gaps += (np.flatnonzero(e == -1) - np.flatnonzero(e == 1)).tolist()
        e = np.diff(np.concatenate(([0], have[start:stop].astype(np.int8), [0])))
        s, t = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
        present_runs += (t - s).tolist()
        line_end += line[start + t - 1].tolist()
        near_end += near[start + t - 1].tolist()
    speed = speed if speed is not None else player_mean_speed(match)
    with np.errstate(all="ignore"):
        ratio = np.nanmean(speed[graded & ~have]) / np.nanmean(speed[graded & have])
    return {"missing": float(np.mean(~have[graded])),
            "gap_p50": float(np.median(gaps)) if gaps else 0.0,
            "present_p50": float(np.median(present_runs)) if present_runs else 0.0,
            "speed_ratio": float(ratio),
            "line_end": float(np.mean(line_end)) if line_end else 0.0,
            "visible_outside": float(np.mean(outside(match.ball[graded & have], match.length, match.width))),
            "lost_near_p50": float(np.nanmedian(near_end)) if near_end else 0.0}


def outside(ball, length, width):
    return (ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width)


def stats_loss(stats, targets=TARGETS):
    return float(sum(np.log(max(stats[k], 1e-6) / targets[k]) ** 2 for k in targets))
