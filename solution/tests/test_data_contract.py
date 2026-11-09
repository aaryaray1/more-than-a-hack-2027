from collections import Counter

import numpy as np
import pytest

import baseline
import load_atd
import load_homography
import load_smart_tagging
import possession_metrics as pm
from conftest import ROOT, needs_data
from games import GAME, GAMES, asset_paths, half_video_blocks

CODE_COUNTS = {
    "PLAYERS IN THE BOX": 97, "BALL IN FINAL THIRD": 93, "BALL IN THE BOX": 72,
    "PROGRESSION": 66, "DEFENDING IN MIDDLE THIRD": 66, "SET PIECES": 59,
    "CREATING CHANCES": 30, "DEFENDING IN DEFENSIVE THIRD": 30, "LONG BALL": 29,
    "ATTACKING TRANSITION": 24, "DEFENSIVE TRANSITION": 24, "BUILD UP": 22,
    "DEFENDING IN ATTACKING THIRD": 22, "GOALS": 4,
}
SET_PIECE_TYPES = {"Throw-in": 26, "Corner Kick": 12, "Goal Kick": 10, "Free Kick": 5,
                   "Kick Off (After Goal)": 4, "Kick Off (Start)": 2}


def test_published_half_blocks():
    assert half_video_blocks(GAME) == [(1, 11185, 90651), (2, 110808, 190197)]


@needs_data
def test_atd_one_record_per_video_frame():
    agrees = bool(load_atd.check_alignment_raw()["agrees"])
    assert agrees


@needs_data
def test_atd_slot_teams():
    teams = baseline.team_of_each_track(asset_paths()["atd_meta"])
    assert len(teams) == 46
    assert set(teams.tolist()) == {baseline.HOME, baseline.AWAY}
    home, _ = baseline.team_names(asset_paths()["atd_meta"])
    assert "barcelona" in home.lower()


@needs_data
def test_kloppy_reads_the_epts_pair():
    from kloppy import metrica
    meta, raw = load_atd.atd_paths()
    dataset = metrica.load_tracking_epts(meta_data=meta, raw_data=raw, limit=50)
    assert len(dataset.records) == 50
    assert len(dataset.metadata.teams) == 2


@needs_data
def test_homography_tactical_valid_everywhere():
    assert load_homography.check_alignment()["agrees"]
    _, error, valid = load_homography.load()
    all_valid, all_finite = bool(valid.all()), bool(np.isfinite(error).all())
    assert all_valid and all_finite


@needs_data
def test_homography_broadcast_shape():
    report = load_homography.check_alignment(source="broadcast")
    assert report["agrees"]
    _, _, valid = load_homography.load(source="broadcast")
    assert abs(valid.mean() - 0.658) < 0.005


@needs_data
def test_smart_tagging_time_base_and_counts():
    events = load_smart_tagging.load()
    assert len(events) == 638
    agrees = bool(load_smart_tagging.check_time_base(events)["agrees"])
    assert agrees
    assert dict(Counter(e["code"] for e in events)) == CODE_COUNTS
    types = Counter(e["tags"].get("Type") for e in events if e["code"] == "SET PIECES")
    assert dict(types) == SET_PIECE_TYPES
    assert {e["tags"].get("Team") for e in events} == {"Barcelona", "Atletic Club", "N/A"}


@needs_data
@pytest.mark.slow
def test_baseline_regenerates_committed_files():
    codes = baseline.nearest_player_labels(
        asset_paths()["atd_raw"], baseline.team_of_each_track(asset_paths()["atd_meta"]),
        GAMES[GAME]["n_video_frames"], GAMES[GAME]["pitch"])
    for name, window in baseline.VARIANTS:
        committed = pm.read_label_csv(
            "{}/evaluation/results/{}.csv".format(ROOT, name))["label"].values.astype(str)
        assert np.array_equal(baseline.smooth(codes, window), committed), name
