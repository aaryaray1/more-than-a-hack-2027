import os

import numpy as np
import pytest

import possession_metrics as pm
from conftest import ROOT, needs_data
from solution import io_atd, pipelines
from validation import degrade, idsse, labels

have_idsse = pytest.mark.skipif(not idsse.available(), reason="IDSSE not downloaded")


@needs_data
@pytest.mark.parametrize("name,file", [("baseline_raw", "baseline_raw"),
                                       ("baseline25", "baseline_smooth25"),
                                       ("baseline50", "baseline_smooth50"),
                                       ("baseline125", "baseline_smooth125")])
def test_ported_baseline_matches_official_file(name, file):
    codes = pipelines.run(name, io_atd.load())
    official = pm.label_codes(pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "results", file + ".csv")))
    same = bool(np.array_equal(codes, official))
    assert same


def test_control_labels_absorb_short_flips():
    H, A, D = pm.H, pm.A, pm.D
    gt = np.array([pm.X] + [D] * 5 + [H] * 100 + [A] * 10 + [H] * 100 + [A] * 80 + [A] * 10
                  + [H] * 10 + [D] * 5 + [A] * 20 + [pm.X], np.int8)
    out = labels.controlled(gt, fps=25, min_s=2.0)
    runs = pm.valid_runs(out)
    assert [c for _, _, c in pm.possession_spells(out, runs)] == [H, A, A]
    assert out[106:116].tolist() == [H] * 10
    assert out[306:316].tolist() == [A] * 10
    assert out[321:341].tolist() == [A] * 20


@have_idsse
def test_idsse_labels_and_shapes():
    m = idsse.load(idsse.available()[0])
    graded = m.labels != pm.X
    assert len(m.halves) == 2
    assert set(np.unique(m.labels[graded]).tolist()) == {pm.H, pm.A, pm.D}
    assert 0.25 < np.mean(m.labels[graded] == pm.D) < 0.6
    per_frame = (~np.isnan(m.xy[..., 0])).sum(1)[graded]
    assert np.median(per_frame) == 22


@have_idsse
def test_degrade_is_deterministic_and_close_to_targets():
    m = idsse.load(idsse.available()[0])
    a, b = degrade.degrade(m, 7), degrade.degrade(m, 7)
    assert np.array_equal(a.ball, b.ball, equal_nan=True)
    assert np.array_equal(a.xy, b.xy, equal_nan=True)
    stats = degrade.ball_gap_stats(a)
    assert degrade.stats_loss(stats) < 0.5
    assert a.ball_z is None
