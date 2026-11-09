import numpy as np
import pytest

import possession_metrics as pm
from solution import decode
from validation import degrade, idsse


def test_viterbi_ignores_flicker_and_finds_step():
    rng = np.random.default_rng(0)
    p = np.where(np.arange(2000) < 1000, 0.2, 0.8) + rng.normal(0, 0.15, 2000)
    out = decode.viterbi_binary(np.clip(p, 0.01, 0.99), 20.0, [(0, 2000)], 2000)
    assert out[:950].sum() == 0 and out[1050:].all()
    assert 950 <= np.flatnonzero(out)[0] <= 1050


def test_viterbi_respects_halves():
    p = np.full(300, 0.9)
    out = decode.viterbi_binary(p, 5.0, [(0, 100), (200, 300)], 300)
    assert out[:100].all() and out[200:].all() and not out[100:200].any()


def test_control_events_need_min_frames():
    team = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0], np.int8)
    dist = np.full(9, 0.5)
    events = decode.control_events(team, dist, np.ones(9, bool), 1.0, 3)
    assert events == [(0, 0), (5, 0)]


def test_team_track_places_gap_switch_mid():
    visible = np.ones(100, bool)
    visible[40:60] = False
    codes = decode.team_track([(10, pm.H), (60, pm.A)], [(0, 100)], visible, 100, "mid")
    assert codes[48] == pm.H and codes[49] == pm.A
    codes = decode.team_track([(10, pm.H), (60, pm.A)], [(0, 100)], visible, 100, "end")
    assert codes[59] == pm.H and codes[60] == pm.A


def test_absorb_short_keeps_restart_owner():
    H, A, D = pm.H, pm.A, pm.D
    codes = np.array([D] * 5 + [A] * 10 + [H] * 100 + [A] * 5 + [H] * 50 + [D] * 5, np.int8)
    out = decode.absorb_short(codes, 25, 1.0)
    assert out[5:15].tolist() == [A] * 10
    assert out[115:120].tolist() == [H] * 5


@pytest.mark.skipif(not idsse.available(), reason="IDSSE not downloaded")
def test_unfrozen_dead_ball_moves():
    m = idsse.load(idsse.available()[0])
    ball = degrade.unfreeze_dead_ball(m, np.random.default_rng(0))
    dead = m.labels == pm.D
    speed = np.concatenate(([np.nan], np.hypot(*np.diff(ball, axis=0).T)))
    frozen = float(np.mean(speed[dead] == 0))
    assert frozen < 0.05
    alive_same = bool(np.array_equal(ball[~dead], m.ball[~dead], equal_nan=True))
    assert alive_same
