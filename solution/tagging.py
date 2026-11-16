import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402

HALF_NAMES = {1: "1st half", 2: "2nd half"}


def kickoff_start_frames(game=None):
    events = load_smart_tagging.load(game) if game else load_smart_tagging.load()
    out = {}
    for e in events:
        tags = {k.lower(): (v or "").strip().lower() for k, v in e["tags"].items()}
        if (e["code"] or "").strip().lower() == "set pieces" and tags.get("type") == "kick off (start)":
            for half, name in HALF_NAMES.items():
                if tags.get("half") == name:
                    out.setdefault(half, []).append(e["start_frame"])
    return {half: sorted(frames) for half, frames in out.items()}


def team_phases(game=None):
    events = load_smart_tagging.load(game) if game else load_smart_tagging.load()
    out = []
    for e in events:
        team = (e["tags"].get("Team") or "").strip()
        if team and team.upper() != "N/A" and e["end_frame"] > e["start_frame"]:
            out.append([e["start_frame"], e["end_frame"], "{}|{}".format((e["code"] or "").strip(), team)])
    return out
