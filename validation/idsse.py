import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from solution.match import AWAY, HOME, Match  # noqa: E402

SOURCE_DIR = os.path.join(ROOT, "validation", "cache", "idsse")
FPS = 25
HALFTIME_GAP = 250
SECTIONS = ("firstHalf", "secondHalf")

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="(-?[\d.]+)" Y="(-?[\d.]+)"(?: Z="(-?[\d.]+)")?')
BALL_FLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')
PITCH = re.compile(r'<PitchSize X="([\d.]+)" Y="([\d.]+)"')


def files(match_id):
    found = {kind: glob.glob(os.path.join(SOURCE_DIR, "DFL_{}_*{}.xml".format(prefix, match_id)))
             for kind, prefix in (("info", "02_01_matchinformation"), ("pos", "04_03_positions"))}
    missing = [kind for kind, paths in found.items() if len(paths) != 1]
    if missing:
        raise IOError("IDSSE {}: missing {} in {}".format(match_id, missing, SOURCE_DIR))
    return found["info"][0], found["pos"][0]


def available():
    return sorted({re.search(r"(J03W\w+)\.xml$", p).group(1)
                   for p in glob.glob(os.path.join(SOURCE_DIR, "DFL_04_03_positions_*.xml"))})


def read_positions(pos_path):
    tracks, pitch, current = {}, (105.0, 68.0), None
    with open(pos_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = FRAME.match(line)
                row = [int(m.group(1)), float(m.group(2)), float(m.group(3))]
                if current[1] == "BALL":
                    flags = BALL_FLAGS.search(line)
                    row += [float(m.group(4) or 0.0), int(flags.group(1)), int(flags.group(2))]
                tracks[current].append(row)
            elif line.startswith("<FrameSet "):
                current = FRAMESET.match(line).groups()
                tracks[current] = []
            elif line.startswith("<PitchSize"):
                p = PITCH.search(line)
                pitch = (float(p.group(1)), float(p.group(2)))
    return {key: np.array(rows) for key, rows in tracks.items()}, pitch


def build(match_id):
    info_path, pos_path = files(match_id)
    with open(info_path, encoding="utf-8") as handle:
        text = handle.read()
    home_id = re.search(r'HomeTeamId="([^"]+)"', text).group(1)
    guest_id = re.search(r'GuestTeamId="([^"]+)"', text).group(1)
    tracks, (length, width) = read_positions(pos_path)
    tracks = {k: v for k, v in tracks.items() if k[1] in ("BALL", home_id, guest_id)}

    offsets, spans, cursor = {}, [], 0
    for section in SECTIONS:
        ball = tracks[(section, "BALL", next(k[2] for k in tracks if k[:2] == (section, "BALL")))]
        first, last = int(ball[:, 0].min()), int(ball[:, 0].max())
        offsets[section] = cursor - first
        spans.append((cursor, cursor + last - first + 1))
        cursor += last - first + 1 + HALFTIME_GAP
    n_frames = cursor - HALFTIME_GAP

    people = sorted({k[2] for k in tracks if k[1] != "BALL"})
    slot = {person: i for i, person in enumerate(people)}
    team = np.zeros(len(people), np.int8)
    xy = np.full((n_frames, len(people), 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    ball_z = np.full(n_frames, np.nan)
    labels = np.full(n_frames, pm.X, np.int8)

    for (section, team_id, person), rows in tracks.items():
        frames = rows[:, 0].astype(np.int64) + offsets[section]
        x, y = rows[:, 1] + length / 2, rows[:, 2] + width / 2
        if team_id == "BALL":
            ball[frames] = np.column_stack([x, y])
            ball_z[frames] = rows[:, 3]
            owner = np.where(rows[:, 4] == 1, pm.H, pm.A)
            labels[frames] = np.where(rows[:, 5] == 1, owner, pm.D)
        else:
            team[slot[person]] = HOME if team_id == home_id else AWAY
            xy[frames, slot[person]] = np.column_stack([x, y])

    return Match(name="IDSSE_{}".format(match_id), fps=FPS, length=length, width=width, xy=xy,
                 team=team, ball=ball, halves=spans, labels=labels, ball_z=ball_z,
                 meta={"source": "IDSSE CC-BY 4.0", "match_id": match_id})


def load(match_id, refresh=False):
    cache = os.path.join(SOURCE_DIR, "{}.npz".format(match_id))
    if os.path.isfile(cache) and not refresh:
        return Match.load(cache)
    match = build(match_id)
    match.save(cache)
    return match


if __name__ == "__main__":
    for match_id in available():
        m = load(match_id, refresh="--refresh" in sys.argv)
        graded = m.labels != pm.X
        runs = pm.valid_runs(m.labels)
        stats = pm.possession_stats(m.labels, runs)
        print("{}: {} frames, {} players, halves {}, graded {}".format(
            m.name, m.n_frames, len(m.team), m.halves, int(graded.sum())))
        print("  H {:.1%} A {:.1%} D {:.1%}  possessions {} mean {:.1f} s  per min {:.2f}  ball present {:.1%}".format(
            *(np.mean(m.labels[graded] == c) for c in (pm.H, pm.A, pm.D)),
            stats["n_spells"], stats["mean_duration_s"], stats["n_spells"] / (graded.sum() / FPS / 60),
            np.mean(~np.isnan(m.ball[graded, 0]))))
