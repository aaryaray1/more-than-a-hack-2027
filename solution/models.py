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


def cached_team_features(match):
    key = (match.name, match.n_frames, "team")
    if key not in _FEATURES:
        _FEATURES[key] = features.team_features(match)
    return _FEATURES[key]


def cached_kickoff(match):
    key = (match.name, match.n_frames, "kickoff")
    if key not in _FEATURES:
        split = features.centred_mean(features.kickoff_split(match), 12)
        n_seen = (~np.isnan(match.xy[..., 0])).sum(axis=1)
        _FEATURES[key] = decode.kickoff_dead(split, n_seen, match.halves, match.n_frames,
                                             match.fps, match.meta.get("kickoff_tags"))
    return _FEATURES[key]


def cached_nearest(match):
    key = (match.name, match.n_frames, "nearest")
    if key not in _FEATURES:
        home_d, away_d = features.nearest_distances(match)
        home_d, away_d = np.nan_to_num(home_d, nan=np.inf), np.nan_to_num(away_d, nan=np.inf)
        team = nearest_team(match)
        _FEATURES[key] = (team, np.fmin(home_d, away_d), np.where(team == pm.H, away_d, home_d))
    return _FEATURES[key]


class DeadAliveTeam:
    learned = True

    def __init__(self, penalty=8.0, radius=1.0, min_frames=3, min_s=2.0, gap_switch="mid",
                 margin=0.0, kickoff=False):
        self.penalty, self.radius, self.min_frames = penalty, radius, min_frames
        self.min_s, self.gap_switch, self.margin, self.kickoff = min_s, gap_switch, margin, kickoff
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
        if self.kickoff:
            dead |= cached_kickoff(match)
        alive = match.in_halves() & ~dead
        team_near, dist, other = cached_nearest(match)
        events = decode.control_events(team_near, dist, alive & (other - dist > self.margin),
                                       self.radius, self.min_frames)
        visible = ~np.isnan(match.ball[:, 0])
        codes = decode.team_track(events, decode.runs_of(alive, match.halves), visible,
                                  match.n_frames, self.gap_switch)
        codes = decode.fill_unknown(codes, match.halves)
        codes[~alive] = pm.D
        return decode.absorb_short(codes, match.fps, self.min_s)


class LearnedTeam(DeadAliveTeam):
    def __init__(self, team_penalty=16.0, **kwargs):
        super().__init__(**kwargs)
        self.team_penalty = team_penalty
        self.team_model = None

    def fit(self, runs):
        super().fit(runs)
        xs, ys = [], []
        for match, truth in runs:
            x, names = cached_team_features(match)
            alive = np.flatnonzero((truth == pm.H) | (truth == pm.A))[::TRAIN_STRIDE]
            xs += [x[alive], features.swap_teams(x[alive], names)]
            ys += [truth[alive] == pm.H, truth[alive] == pm.A]
        self.team_model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                         max_leaf_nodes=31, l2_regularization=1.0,
                                                         early_stopping=False, random_state=0)
        self.team_model.fit(np.vstack(xs), np.concatenate(ys))
        return self

    def home_probability(self, match):
        x, names = cached_team_features(match)
        p = self.team_model.predict_proba(x)[:, 1]
        q = self.team_model.predict_proba(features.swap_teams(x, names))[:, 1]
        return 0.5 * (p + 1 - q)

    def predict(self, match, p_dead=None, p_home=None):
        if p_dead is None:
            p_dead = self.dead_probability(match)
        if p_home is None:
            p_home = self.home_probability(match)
        dead = decode.viterbi_binary(p_dead, self.penalty, match.halves, match.n_frames)
        alive = match.in_halves() & ~dead
        home = decode.viterbi_binary(p_home, self.team_penalty, decode.runs_of(alive, match.halves),
                                     match.n_frames)
        codes = np.where(home, pm.H, pm.A).astype(np.int8)
        codes[~alive] = pm.D
        return decode.absorb_short(codes, match.fps, self.min_s)


LEARNED = {
    "v1": lambda: DeadAliveTeam(),
    "v2": lambda: LearnedTeam(penalty=64.0),
    "v1m": lambda: DeadAliveTeam(penalty=64.0, radius=1.0, min_frames=1, min_s=2.0, gap_switch="mid",
                                 margin=0.3),
    "v1k": lambda: DeadAliveTeam(kickoff=True),
    "v1t": lambda: DeadAliveTeam(penalty=64.0, radius=1.0, min_frames=1, min_s=2.0, gap_switch="mid"),
}
