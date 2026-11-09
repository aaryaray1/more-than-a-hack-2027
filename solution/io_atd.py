import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from solution.match import AWAY, HOME, Match  # noqa: E402

CACHE_DIR = os.path.join(DATA_DIR, "cache")


def slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def parse_raw(raw_path, n_slots, n_frames, length, width):
    xy = np.full((n_frames, n_slots, 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip().split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if parts[2][0] != "N":
                values = parts[2].split(",")
                ball[frame] = float(values[0]) * length, float(values[1]) * width
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] != "N":
                    values = group.split(",")
                    xy[frame, slot] = float(values[0]) * length, float(values[1]) * width
    return xy, ball


def load(source="tactical", game=GAME, refresh=False):
    cache = os.path.join(CACHE_DIR, "{}_{}.npz".format(game, source))
    if os.path.isfile(cache) and not refresh:
        return Match.load(cache)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    length, width = GAMES[game]["pitch"]
    team = slot_teams(meta_path)
    xy, ball = parse_raw(raw_path, len(team), source_frames(source, game), length, width)
    halves = [(first, last + 1) for _, first, last in half_video_blocks(game)]
    match = Match(name="{}_{}".format(game, source), fps=GAMES[game]["fps"], length=length,
                  width=width, xy=xy, team=team, ball=ball, halves=halves)
    match.save(cache)
    return match


if __name__ == "__main__":
    m = load(refresh="--refresh" in sys.argv)
    print("{}: {} frames, {} slots, ball present {:.1%} in halves".format(
        m.name, m.n_frames, len(m.team), np.mean(~np.isnan(m.ball[m.in_halves(), 0]))))
