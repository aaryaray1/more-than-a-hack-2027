import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

UNKNOWN = -1


def viterbi_binary(p_on, penalty, halves, n_frames, eps=1e-4):
    p = np.clip(p_on, eps, 1 - eps)
    gain = (np.log(p) - np.log(1 - p)).tolist()
    cost = np.broadcast_to(np.asarray(penalty, float), (n_frames,)).tolist()
    out = np.zeros(n_frames, bool)
    for start, stop in halves:
        score_off, score_on = 0.0, 0.0
        back_on, back_off = np.zeros(stop - start, bool), np.zeros(stop - start, bool)
        for i, t in enumerate(range(start, stop)):
            stay_off, jump_off = score_off, score_on - cost[t]
            stay_on, jump_on = score_on, score_off - cost[t]
            back_off[i] = jump_off > stay_off
            back_on[i] = jump_on > stay_on
            score_off = max(stay_off, jump_off)
            score_on = max(stay_on, jump_on) + gain[t]
        state = score_on > score_off
        for i in range(stop - start - 1, -1, -1):
            out[start + i] = state
            state = (not state) if (back_on[i] if state else back_off[i]) else state
    return out


def runs_of(mask, halves):
    out = []
    for start, stop in halves:
        edges = np.diff(np.concatenate(([0], mask[start:stop].astype(np.int8), [0])))
        out += [(start + a, start + b) for a, b in zip(np.flatnonzero(edges == 1),
                                                      np.flatnonzero(edges == -1))]
    return out


def control_events(team_near, dist_near, alive, radius, min_frames):
    ctrl = np.where(alive & (dist_near < radius), team_near, UNKNOWN)
    edges = np.flatnonzero(np.diff(np.concatenate(([UNKNOWN - 1], ctrl, [UNKNOWN - 1]))) != 0)
    events = []
    for a, b in zip(edges[:-1], edges[1:]):
        if ctrl[a] != UNKNOWN and b - a >= min_frames:
            events.append((a, int(ctrl[a])))
    return events


def absorb_short(codes, fps, min_s):
    out = codes.copy()
    min_frames = int(round(min_s * fps))
    for start, stop in pm.valid_runs(np.where(codes == UNKNOWN, pm.D, codes)):
        owner = None
        for a, b, code in pm.segments(codes[start:stop], [(0, stop - start)]):
            if code == pm.D:
                owner = None
            elif owner is None or (code != owner and b - a >= min_frames):
                owner = code
            elif code != owner:
                out[start + a:start + b] = owner
    return out


def team_track(events, alive_runs, visible, n_frames, gap_switch="mid"):
    codes = np.full(n_frames, pm.D, np.int8)
    times = np.array([e[0] for e in events], np.int64)
    teams = np.array([e[1] for e in events], np.int8)
    for start, stop in alive_runs:
        lo, hi = np.searchsorted(times, start), np.searchsorted(times, stop)
        if lo == hi:
            codes[start:stop] = UNKNOWN
            continue
        seq_t, seq_team = times[lo:hi].copy(), teams[lo:hi]
        if gap_switch == "mid":
            for i in range(1, len(seq_t)):
                if seq_team[i] != seq_team[i - 1]:
                    seen = np.flatnonzero(visible[seq_t[i - 1]:seq_t[i]])
                    if len(seen):
                        last_seen = seq_t[i - 1] + seen[-1]
                        if seq_t[i] - last_seen > 1:
                            seq_t[i] = (last_seen + seq_t[i]) // 2
        idx = np.searchsorted(seq_t, np.arange(start, stop), side="right") - 1
        codes[start:stop] = seq_team[np.clip(idx, 0, None)]
    return codes


def fill_unknown(codes, halves):
    out = codes.copy()
    for start, stop in halves:
        window = out[start:stop]
        known = np.flatnonzero((window == pm.H) | (window == pm.A))
        if not len(known):
            window[window == UNKNOWN] = pm.H
            continue
        carry = np.maximum.accumulate(np.where((window == pm.H) | (window == pm.A),
                                               np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        window[window == UNKNOWN] = window[carry][window == UNKNOWN]
    return out


def own_half_episodes(split, n_seen, lo, hi, threshold=0.9, min_frames=50, min_players=16):
    mask = (split[lo:hi] >= threshold) & (n_seen[lo:hi] >= min_players)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return [(lo + a, lo + b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if b - a >= min_frames]


def kickoff_dead(split, n_seen, halves, n_frames, fps, kickoff_tags=None, search_s=300):
    dead = np.zeros(n_frames, bool)
    for index, (start, stop) in enumerate(halves, 1):
        for a, b in own_half_episodes(split, n_seen, start, stop):
            dead[a:b] = True
        tagged = [f for f in (kickoff_tags or {}).get(str(index), [])
                  if start < f <= min(stop, start + search_s * fps)]
        if tagged:
            dead[start:tagged[0]] = True
    return dead


def phase_prior(p_home, phases, trusted, n_frames, min_frames=1500, clip=0.03):
    logit = np.zeros(n_frames)
    base = float(np.clip(np.mean(p_home[trusted] > 0.5), clip, 1 - clip))
    groups = {}
    for start, stop, key in phases:
        groups.setdefault(key, []).append((start, stop))
    for key in sorted(groups):
        mask = np.zeros(n_frames, bool)
        for start, stop in groups[key]:
            mask[start:stop] = True
        seen = mask & trusted
        if seen.sum() < min_frames:
            continue
        rate = float(np.clip(np.mean(p_home[seen] > 0.5), clip, 1 - clip))
        bonus = np.log(rate / (1 - rate)) - np.log(base / (1 - base))
        logit[mask] = np.where(np.abs(bonus) > np.abs(logit[mask]), bonus, logit[mask])
    return logit


def shift_probability(p, logit, eps=1e-4):
    q = np.clip(p, eps, 1 - eps)
    return 1.0 / (1.0 + np.exp(-(np.log(q) - np.log(1 - q) + logit)))


def switch_cost(p_turn, weight, low, high, eps=1e-4):
    return np.clip(-weight * np.log(np.clip(p_turn, eps, 1.0)), low, high)
