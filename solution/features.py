import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

MAX_PLAYER_SPEED = 12.0
TIME_CAP = 3000


def centred_mean(x, half):
    valid = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(x))
    low, high = np.maximum(0, index - half), np.minimum(len(x), index + half + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def shifted(x, lag):
    out = np.full_like(x, np.nan)
    if lag > 0:
        out[lag:] = x[:-lag]
    elif lag < 0:
        out[:lag] = x[-lag:]
    else:
        out[:] = x
    return out


def player_speeds(match):
    step = np.diff(match.xy, axis=0) * match.fps
    speed = np.hypot(step[..., 0], step[..., 1])
    speed = np.where(speed < MAX_PLAYER_SPEED, speed, np.nan)
    return np.concatenate((np.full((1, speed.shape[1]), np.nan), speed))


def line_distance(match, points):
    x, y = points[:, 0], points[:, 1]
    return np.minimum(np.minimum(x, match.length - x), np.minimum(y, match.width - y))


def gap_index(visible, halves):
    last = np.full(len(visible), -1, np.int64)
    nxt = np.full(len(visible), -1, np.int64)
    for start, stop in halves:
        idx = np.where(visible[start:stop], np.arange(start, stop), -1)
        last[start:stop] = np.maximum.accumulate(idx)
        rev = np.where(visible[start:stop], np.arange(start, stop), stop + 10 ** 9)[::-1]
        nx = np.minimum.accumulate(rev)[::-1]
        nxt[start:stop] = np.where(nx >= stop, -1, nx)
    return last, nxt


def at(values, index):
    return np.where(index >= 0, values[np.clip(index, 0, None)], np.nan)


def nearest_distances(match):
    diff = match.xy - match.ball[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    out = {}
    for name, team in (("home", 0), ("away", 1)):
        d = np.where(match.team[None, :] == team, dist, np.nan)
        with np.errstate(all="ignore"):
            out[name] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    for name in out:
        out[name] = np.where(np.isfinite(out[name]), out[name], np.nan)
    return out["home"], out["away"]


def kickoff_split(match):
    left = match.xy[..., 0] < match.length / 2
    seen = ~np.isnan(match.xy[..., 0])
    home, away = (match.team == 0)[None, :] & seen, (match.team == 1)[None, :] & seen
    with np.errstate(all="ignore"):
        a = ((left & home).sum(1) + (~left & away).sum(1)) / seen.sum(1)
        b = ((~left & home).sum(1) + (left & away).sum(1)) / seen.sum(1)
    return np.fmax(a, b)


def players_within(match, radius):
    diff = match.xy - match.ball[:, None, :]
    close = (np.hypot(diff[..., 0], diff[..., 1]) < radius).sum(axis=1).astype(float)
    return np.where(np.isnan(match.ball[:, 0]), np.nan, close)


def frame_features(match):
    t = np.arange(match.n_frames)
    visible = ~np.isnan(match.ball[:, 0])
    speeds = player_speeds(match)
    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(speeds, axis=1)
        moving = np.nanmean(np.where(np.isnan(speeds), np.nan, speeds > 2.0), axis=1)

    ball_step = np.vstack(([np.nan, np.nan], np.diff(match.ball, axis=0)))
    ball_speed = centred_mean(np.hypot(ball_step[:, 0], ball_step[:, 1]) * match.fps, 2)
    line = line_distance(match, match.ball)
    home_d, away_d = nearest_distances(match)
    near = np.fmin(home_d, away_d)
    corner = np.min([np.hypot(match.ball[:, 0] - cx, match.ball[:, 1] - cy)
                     for cx in (0, match.length) for cy in (0, match.width)], axis=0)
    centre = np.hypot(match.ball[:, 0] - match.length / 2, match.ball[:, 1] - match.width / 2)

    last, nxt = gap_index(visible, match.halves)
    since = np.where(last >= 0, t - last, TIME_CAP).clip(0, TIME_CAP)
    until = np.where(nxt >= 0, nxt - t, TIME_CAP).clip(0, TIME_CAP)
    gap_len = np.where(visible, 0, np.minimum(since + until, TIME_CAP))
    jump = np.hypot(*(at(match.ball[:, k], nxt) - at(match.ball[:, k], last) for k in (0, 1)))

    hidden = ~visible
    lost, found = gap_index(hidden, match.halves)
    vis_since = np.where(visible, np.where(lost >= 0, t - lost, TIME_CAP), 0).clip(0, TIME_CAP)
    vis_until = np.where(visible, np.where(found >= 0, found - t, TIME_CAP), 0).clip(0, TIME_CAP)

    s12, s50, s125 = (centred_mean(mean_speed, h) for h in (12, 50, 125))
    columns = {
        "visible": visible.astype(float),
        "ball_line": line, "ball_speed": ball_speed, "ball_still": centred_mean(ball_speed, 25),
        "near": near, "home_d": home_d, "away_d": away_d, "corner": corner, "centre": centre,
        "since": since, "until": until, "gap_len": gap_len,
        "last_line": at(line, last), "next_line": at(line, nxt), "last_speed": at(ball_speed, last),
        "next_near": at(near, nxt), "last_near": at(near, last), "jump": jump,
        "vis_since": vis_since, "vis_until": vis_until,
        "speed12": s12, "speed50": s50, "speed125": s125,
        "speed_drop": s50 - shifted(s50, 125), "speed_rise": shifted(s50, -125) - s50,
        "moving": centred_mean(moving, 25),
        "split": centred_mean(kickoff_split(match), 12),
        "within5": centred_mean(players_within(match, 5.0), 12),
        "within10": centred_mean(players_within(match, 10.0), 12),
    }
    names = list(columns)
    return np.column_stack([columns[n] for n in names]).astype(np.float32), names


def nearest_slot(match, team):
    diff = match.xy - match.ball[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where((match.team[None, :] == team) & ~np.isnan(dist), dist, np.inf)
    slot = np.argmin(dist, axis=1)
    return slot, np.isfinite(dist[np.arange(len(slot)), slot])


def time_since(mask, cap=TIME_CAP):
    idx = np.where(mask, np.arange(len(mask)), -10 ** 9)
    return np.minimum(np.arange(len(mask)) - np.maximum.accumulate(idx), cap).astype(float)


def time_until(mask, cap=TIME_CAP):
    return time_since(mask[::-1], cap)[::-1]


def team_features(match, radius=1.0, margin=0.3):
    home_d, away_d = nearest_distances(match)
    ball_v = np.vstack(([np.nan, np.nan], np.diff(match.ball, axis=0))) * match.fps
    player_v = np.concatenate((np.full((1,) + match.xy.shape[1:], np.nan),
                               np.diff(match.xy, axis=0))) * match.fps
    cols = {}
    for name, team, own, opp in (("h", 0, home_d, away_d), ("a", 1, away_d, home_d)):
        slot, ok = nearest_slot(match, team)
        rel = ball_v - player_v[np.arange(match.n_frames), slot]
        co = np.where(ok, np.hypot(rel[:, 0], rel[:, 1]), np.nan)
        ctrl = np.nan_to_num(own, nan=np.inf) < radius
        ctrl &= (np.nan_to_num(opp, nan=np.inf) - np.nan_to_num(own, nan=np.inf)) > margin
        cols[name + "_d"] = own
        cols[name + "_co"] = centred_mean(co, 3)
        cols[name + "_since"] = time_since(ctrl)
        cols[name + "_until"] = time_until(ctrl)
        cols[name + "_ctrl25"] = centred_mean(ctrl.astype(float), 25)
    diff = home_d - away_d
    cols["diff"] = diff
    cols["diff12"] = centred_mean(diff, 12)
    cols["diff50"] = centred_mean(diff, 50)
    names = sorted(cols)
    return np.column_stack([cols[n] for n in names]).astype(np.float32), names


def swap_teams(x, names):
    index = {n: i for i, n in enumerate(names)}
    order = []
    for n in names:
        if n.startswith("h_"):
            order.append(index["a_" + n[2:]])
        elif n.startswith("a_"):
            order.append(index["h_" + n[2:]])
        else:
            order.append(index[n])
    out = x[:, order].copy()
    for n in ("diff", "diff12", "diff50"):
        out[:, index[n]] = -out[:, index[n]]
    return out


OFFSETS = (-50, -25, -12, -6, -3, 0, 3, 6, 12, 25, 50)
FAR = 30.0


def window_mean(x, lo, hi):
    total = np.concatenate(([0.0], np.cumsum(np.nan_to_num(x, nan=0.0))))
    index = np.arange(len(x))
    low, high = np.clip(index + lo, 0, len(x)), np.clip(index + hi + 1, 0, len(x))
    return (total[high] - total[low]) / max(hi - lo + 1, 1)


def window_min(x, lo, hi):
    centre = (lo + hi) // 2
    padded = minimum_filter1d(x, size=hi - lo + 1, mode="constant", cval=FAR)
    return shifted(padded, -centre) if centre else padded


def attack_sign(match):
    sign = np.ones(match.n_frames)
    home, away = match.team == 0, match.team == 1
    for start, stop in match.halves:
        with np.errstate(all="ignore"):
            hx = np.nanmean(match.xy[start:stop][:, home, 0])
            ax = np.nanmean(match.xy[start:stop][:, away, 0])
        sign[start:stop] = 1.0 if hx < ax else -1.0
    return sign


def kick_strength(match, half=6):
    v = np.vstack(([np.nan, np.nan], np.diff(match.ball, axis=0))) * match.fps
    return np.hypot(*(shifted(v, -half) - shifted(v, half)).T)


def team_features2(match, radius=1.0, margin=0.3):
    n = match.n_frames
    visible = ~np.isnan(match.ball[:, 0])
    home_d, away_d = nearest_distances(match)
    ball_v = np.vstack(([np.nan, np.nan], np.diff(match.ball, axis=0))) * match.fps
    ball_speed = centred_mean(np.hypot(ball_v[:, 0], ball_v[:, 1]), 1)
    player_v = np.concatenate((np.full((1,) + match.xy.shape[1:], np.nan),
                               np.diff(match.xy, axis=0))) * match.fps
    player_v = np.where(np.abs(player_v) < MAX_PLAYER_SPEED, player_v, np.nan)
    sign = attack_sign(match)
    kick_raw = kick_strength(match)
    kick = np.nan_to_num(kick_raw, nan=0.0)
    cols = {}
    for name, team, own, opp, direction in (("h", 0, home_d, away_d, sign), ("a", 1, away_d, home_d, -sign)):
        own_f, opp_f = np.nan_to_num(own, nan=FAR).clip(0, FAR), np.nan_to_num(opp, nan=FAR).clip(0, FAR)
        slot, ok = nearest_slot(match, team)
        rel = ball_v - player_v[np.arange(n), slot]
        co = np.where(ok, np.hypot(rel[:, 0], rel[:, 1]), np.nan)
        ctrl = (own_f < radius) & (opp_f - own_f > margin)
        close = own_f < 2.0
        for k in OFFSETS:
            cols["{}_d{:+d}".format(name, k)] = shifted(own, -k)
        for lo, hi in ((-12, -1), (-50, -13), (-125, -51), (1, 12), (13, 50), (51, 125)):
            cols["{}_min{:+d}{:+d}".format(name, lo, hi)] = window_min(own_f, lo, hi)
        for lo, hi in ((-25, -1), (-75, -26), (1, 25), (26, 75)):
            cols["{}_ctrl{:+d}{:+d}".format(name, lo, hi)] = window_mean(ctrl.astype(float), lo, hi)
        cols[name + "_since"], cols[name + "_until"] = time_since(ctrl), time_until(ctrl)
        cols[name + "_since2"], cols[name + "_until2"] = time_since(close), time_until(close)
        kicked = kick * ((own_f < 2.5) & (own_f < opp_f))
        for lo, hi in ((-12, 12), (-50, -13), (-125, -51), (13, 50), (51, 125)):
            cols["{}_kick{:+d}{:+d}".format(name, lo, hi)] = window_mean(kicked, lo, hi)
        cols[name + "_co"] = centred_mean(co, 3)
        with np.errstate(all="ignore"):
            vx = np.nanmean(player_v[:, match.team == team, 0], axis=1) * direction
            cx = np.nanmean(match.xy[:, match.team == team, 0], axis=1)
        adv = centred_mean(vx, 37)
        cols[name + "_adv25"], cols[name + "_adv75"] = centred_mean(vx, 12), adv
        cols[name + "_adv_past"], cols[name + "_adv_next"] = shifted(adv, 75), shifted(adv, -75)
        cols[name + "_depth"] = np.where(direction > 0, cx, match.length - cx)
        cols[name + "_ballx"] = np.where(direction > 0, match.ball[:, 0], match.length - match.ball[:, 0])
        cols[name + "_ballv"] = centred_mean(ball_v[:, 0], 6) * direction
    diff = home_d - away_d
    for k in OFFSETS:
        cols["diff{:+d}".format(k)] = shifted(diff, -k)
    cols["diff_mean25"], cols["diff_mean75"] = centred_mean(diff, 12), centred_mean(diff, 37)
    for k in (-12, -3, 0, 3, 12):
        cols["g_speed{:+d}".format(k)] = shifted(ball_speed, -k)
    for k in (-25, -12, 0, 12, 25):
        cols["g_kick{:+d}".format(k)] = shifted(kick_raw, -k)
    last, nxt = gap_index(visible, match.halves)
    t = np.arange(n)
    cols["g_visible"] = visible.astype(float)
    cols["g_since_seen"] = np.where(last >= 0, t - last, TIME_CAP).clip(0, TIME_CAP).astype(float)
    cols["g_until_seen"] = np.where(nxt >= 0, nxt - t, TIME_CAP).clip(0, TIME_CAP).astype(float)
    cols["g_line"] = line_distance(match, match.ball)
    cols["g_within3"] = players_within(match, 3.0)
    jump = (at(match.ball[:, 0], nxt) - at(match.ball[:, 0], last)) * sign
    cols["h_jump"], cols["a_jump"] = jump, -jump
    names = sorted(cols)
    return np.column_stack([cols[c] for c in names]).astype(np.float32), names


def swap_teams2(x, names):
    index = {c: i for i, c in enumerate(names)}
    order = [index[("a_" if c[0] == "h" else "h_") + c[2:]] if c[:2] in ("h_", "a_") else index[c]
             for c in names]
    out = x[:, order].copy()
    for c in names:
        if c.startswith("diff"):
            out[:, index[c]] = -out[:, index[c]]
    return out


CHAIN_CAP = 250


def kick_peaks(match, home_d, away_d, floor=3.0, half=6, reach=3.0):
    kick = np.nan_to_num(kick_strength(match), nan=0.0)
    peak = (kick > floor) & (kick == maximum_filter1d(kick, size=2 * half + 1, mode="constant"))
    own_h, own_a = np.nan_to_num(home_d, nan=FAR), np.nan_to_num(away_d, nan=FAR)
    side = np.where((own_h < own_a) & (own_h < reach), 1.0, np.where((own_a < own_h) & (own_a < reach), -1.0, 0.0))
    frames = np.flatnonzero(peak & (side != 0))
    return frames, side[frames], kick[frames], (own_a - own_h)[frames]


def chain_features(match, home_d, away_d):
    frames, side, strength, margin = kick_peaks(match, home_d, away_d)
    t = np.arange(match.n_frames)
    after = np.searchsorted(frames, t, side="right")
    cols = {}
    for label, index in (("prev1", after - 1), ("prev2", after - 2), ("next1", after), ("next2", after + 1)):
        ok = (index >= 0) & (index < len(frames))
        safe = np.clip(index, 0, max(len(frames) - 1, 0))
        if len(frames):
            gap = np.abs(t - frames[safe])
            ok &= gap <= CHAIN_CAP
            cols["diff_k" + label] = np.where(ok, side[safe], 0.0)
            cols["diff_km" + label] = np.where(ok, margin[safe].clip(-10, 10), 0.0)
            cols["g_kdt" + label] = np.where(ok, gap, CHAIN_CAP).astype(float)
            cols["g_ks" + label] = np.where(ok, strength[safe], 0.0)
        else:
            cols["diff_k" + label] = cols["diff_km" + label] = cols["g_ks" + label] = np.zeros(match.n_frames)
            cols["g_kdt" + label] = np.full(match.n_frames, float(CHAIN_CAP))
    cols["diff_kpair"] = cols["diff_kprev1"] + cols["diff_knext1"]
    return cols


def crowd_features(match):
    diff = match.xy - match.ball[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    hidden = np.isnan(match.ball[:, 0])
    cols = {}
    for name, team in (("h", 0), ("a", 1)):
        own = np.where((match.team[None, :] == team) & ~np.isnan(dist), dist, np.inf)
        second = np.partition(own, 1, axis=1)[:, 1]
        cols[name + "_d2"] = np.where(np.isfinite(second) & ~hidden, second, np.nan)
        cols[name + "_n5"] = np.where(hidden, np.nan, (own < 5.0).sum(axis=1))
        for lag in (12, 25, -25):
            target = shifted(match.ball, -lag)
            away = match.xy - target[:, None, :]
            reach = np.hypot(away[..., 0], away[..., 1])
            reach = np.where((match.team[None, :] == team) & ~np.isnan(reach), reach, np.inf).min(axis=1)
            cols["{}_reach{:+d}".format(name, lag)] = np.where(np.isfinite(reach), reach, np.nan)
    return cols


def team_features3(match):
    x, names = team_features2(match)
    home_d, away_d = nearest_distances(match)
    cols = dict(zip(names, x.T))
    cols.update(chain_features(match, home_d, away_d))
    cols.update(crowd_features(match))
    names = sorted(cols)
    return np.column_stack([cols[c] for c in names]).astype(np.float32), names


def frame_features2(match):
    x, names = frame_features(match)
    cols = dict(zip(names, x.T))
    speeds = player_speeds(match)
    seen = ~np.isnan(match.xy[..., 0])
    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(speeds, axis=1)
        fast = np.nanpercentile(speeds, 90, axis=1)
        slow = np.nanmean(np.where(np.isnan(speeds), np.nan, speeds < 1.0), axis=1)
        spread = []
        for team in (0, 1):
            pos = match.xy[:, match.team == team]
            centre = np.nanmean(pos, axis=1, keepdims=True)
            spread.append(np.nanmean(np.hypot(*(pos - centre).transpose(2, 0, 1)), axis=1))
    spread = centred_mean(np.nanmean(spread, axis=0), 12)
    px, py = match.xy[..., 0], match.xy[..., 1]
    edge = np.minimum(np.minimum(px, match.length - px), np.minimum(py, match.width - py))
    edge_min = centred_mean(np.where(seen, edge, np.inf).min(axis=1).clip(-5, 20), 12)
    outside = centred_mean((np.where(seen, edge, 1.0) < 0).sum(axis=1).astype(float), 12)
    s12, fast25, slow25 = centred_mean(mean_speed, 12), centred_mean(fast, 12), centred_mean(slow, 12)
    for lag in (-200, -100, -50, -25, 25, 50, 100, 200):
        cols["speed12{:+d}".format(lag)] = shifted(s12, -lag)
    for lag in (-100, -50, 0, 50, 100):
        cols["fast{:+d}".format(lag)] = shifted(fast25, -lag)
        cols["slow{:+d}".format(lag)] = shifted(slow25, -lag)
        cols["edge{:+d}".format(lag)] = shifted(edge_min, -lag)
    cols["spread"], cols["spread_change"] = spread, shifted(spread, -100) - shifted(spread, 100)
    cols["outside"] = outside
    names = list(cols)
    return np.column_stack([cols[c] for c in names]).astype(np.float32), names
