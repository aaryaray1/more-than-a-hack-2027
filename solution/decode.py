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
    out = np.zeros(n_frames, bool)
    for start, stop in halves:
        score_off, score_on = 0.0, 0.0
        back_on, back_off = np.zeros(stop - start, bool), np.zeros(stop - start, bool)
        for i, t in enumerate(range(start, stop)):
            stay_off, jump_off = score_off, score_on - penalty
            stay_on, jump_on = score_on, score_off - penalty
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
