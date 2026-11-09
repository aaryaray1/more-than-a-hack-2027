import glob
import os

import numpy as np
import pytest

import possession_metrics as pm
from conftest import ROOT

N_FRAMES = 30000
HEAD, TAIL = 1000, 1000


def synthetic_gt(seed=0):
    rng = np.random.default_rng(seed)
    codes = np.full(N_FRAMES, pm.X, dtype=np.int8)
    pos, team = HEAD, pm.H
    while pos < N_FRAMES - TAIL:
        length = int(rng.integers(60, 600))
        codes[pos:min(pos + length, N_FRAMES - TAIL)] = team
        pos += length
        if rng.random() < 0.3:
            dead = int(rng.integers(50, 300))
            codes[pos:min(pos + dead, N_FRAMES - TAIL)] = pm.D
            pos += dead
            team = pm.H if rng.random() < 0.5 else pm.A
            continue
        team = pm.A if team == pm.H else pm.H
    return codes


def spell_f1(gt, pred):
    return pm.evaluate(gt, pred)["spell_f1_mean"]


def shifted(gt, frames):
    pred = gt.copy()
    graded = gt != pm.X
    inner = np.flatnonzero(graded)
    pred[inner[frames:]] = gt[inner[:-frames]] if frames else gt[inner]
    return pred


def spells(gt):
    return pm.possession_spells(gt, pm.valid_runs(gt))


@pytest.fixture(scope="module")
def gt():
    return synthetic_gt()


def test_perfect_scores_one(gt):
    assert spell_f1(gt, gt.copy()) == 1.0


def test_one_frame_late_is_free(gt):
    assert spell_f1(gt, shifted(gt, 1)) == 1.0


def test_lateness_cost_is_monotone(gt):
    scores = [spell_f1(gt, shifted(gt, s)) for s in (0, 3, 8, 25, 50)]
    assert all(a >= b for a, b in zip(scores, scores[1:]))
    assert scores[-1] < 1.0


@pytest.mark.parametrize("label", [pm.H, pm.A, pm.D])
def test_constant_guess_scores_zero(gt, label):
    pred = np.where(gt == pm.X, pm.X, label).astype(np.int8)
    assert spell_f1(gt, pred) == 0.0


def test_x_inside_possession_is_free(gt):
    pred = gt.copy()
    for start, stop, _ in spells(gt):
        quarter = (stop - start) // 4
        pred[start + quarter:stop - quarter] = pm.X
    assert spell_f1(gt, pred) == 1.0


def test_x_across_turnover_costs(gt):
    pred = gt.copy()
    for start, stop, _ in spells(gt)[1:]:
        pred[start:start + (stop - start) // 2] = pm.X
    assert spell_f1(gt, pred) < spell_f1(gt, gt.copy())


def test_x_cannot_stand_in_for_dead_ball(gt):
    pred = np.where(gt == pm.D, pm.X, gt).astype(np.int8)
    assert spell_f1(gt, pred) < 1.0


def test_missed_dead_ball_merges_spells_and_costs(gt):
    pred = gt.copy()
    s = spells(gt)
    merged = 0
    for (a0, a1, ta), (b0, b1, tb) in zip(s, s[1:]):
        if ta == tb and np.all(gt[a1:b0] == pm.D):
            pred[a1:b0] = ta
            merged += 1
    assert merged > 0
    result = pm.evaluate(gt, pred)
    assert result["n_pred_spells"] == result["n_gt_spells"] - merged
    assert result["spell_f1_mean"] < 1.0


def test_flicker_splits_spell_and_costs(gt):
    pred = gt.copy()
    start, stop, team = max(spells(gt), key=lambda sp: sp[1] - sp[0])
    middle = (start + stop) // 2
    pred[middle:middle + 5] = pm.A if team == pm.H else pm.H
    result = pm.evaluate(gt, pred)
    assert result["n_pred_spells"] == result["n_gt_spells"] + 2
    assert result["spell_f1_mean"] < 1.0


def test_scorer_is_deterministic(gt):
    pred = shifted(gt, 8)
    first, second = pm.evaluate(gt, pred), pm.evaluate(gt, pred)
    assert first["spell_f1_mean"] == second["spell_f1_mean"]
    assert first["macro_f1"] == second["macro_f1"]


@pytest.mark.parametrize("path", sorted(glob.glob(
    os.path.join(ROOT, "evaluation", "results", "baseline_*.csv"))))
def test_committed_baselines_validate(path):
    assert pm.validate_submission(pm.read_label_csv(path), 199319) is None


def test_public_gt_meta_matches_file():
    import json
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    with open(gt_path.replace(".csv", "_meta.json")) as handle:
        meta = json.load(handle)
    codes = pm.label_codes(pm.read_label_csv(gt_path))
    assert len(codes) == meta["n_video_frames"]
    assert int(np.sum(codes != pm.X)) == meta["n_graded_frames"]
    runs = pm.valid_runs(codes)
    assert len(pm.possession_spells(codes, runs)) == meta["n_possessions"]
