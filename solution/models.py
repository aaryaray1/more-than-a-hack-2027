import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from solution import decode, features  # noqa: E402
from solution.pipelines import nearest_team  # noqa: E402

TRAIN_STRIDE = 5
_FEATURES = {}


def cached_features(match):
    key = (match.name, match.n_frames, "features")
    if key not in _FEATURES:
        _FEATURES[key] = features.frame_features(match)[0]
    return _FEATURES[key]


def cached_nearest(match):
    key = (match.name, match.n_frames, "nearest")
    if key not in _FEATURES:
        home_d, away_d = features.nearest_distances(match)
        _FEATURES[key] = (nearest_team(match), np.nan_to_num(np.fmin(home_d, away_d), nan=np.inf))
    return _FEATURES[key]


class DeadAliveTeam:
    learned = True

    def __init__(self, penalty=8.0, radius=1.0, min_frames=3, min_s=2.0, gap_switch="mid"):
        self.penalty, self.radius, self.min_frames = penalty, radius, min_frames
        self.min_s, self.gap_switch = min_s, gap_switch
        self.model = None

    def fit(self, runs):
        xs, ys = [], []
        for match, truth in runs:
            x = cached_features(match)
            graded = np.flatnonzero(truth != pm.X)[::TRAIN_STRIDE]
            xs.append(x[graded])
            ys.append(truth[graded] == pm.D)
        self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                    max_leaf_nodes=31, l2_regularization=1.0,
                                                    early_stopping=False, random_state=0)
        self.model.fit(np.vstack(xs), np.concatenate(ys))
        return self

    def dead_probability(self, match):
        return self.model.predict_proba(cached_features(match))[:, 1]

    def predict(self, match, p_dead=None):
        if p_dead is None:
            p_dead = self.dead_probability(match)
        dead = decode.viterbi_binary(p_dead, self.penalty, match.halves, match.n_frames)
        alive = match.in_halves() & ~dead
        team_near, dist = cached_nearest(match)
        events = decode.control_events(team_near, dist, alive, self.radius, self.min_frames)
        visible = ~np.isnan(match.ball[:, 0])
        codes = decode.team_track(events, decode.runs_of(alive, match.halves), visible,
                                  match.n_frames, self.gap_switch)
        codes = decode.fill_unknown(codes, match.halves)
        codes[~alive] = pm.D
        return decode.absorb_short(codes, match.fps, self.min_s)


LEARNED = {
    "v1": lambda: DeadAliveTeam(),
    "v1t": lambda: DeadAliveTeam(penalty=64.0, radius=1.0, min_frames=1, min_s=2.0, gap_switch="mid"),
}
