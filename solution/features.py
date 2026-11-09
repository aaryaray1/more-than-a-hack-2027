import numpy as np

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
