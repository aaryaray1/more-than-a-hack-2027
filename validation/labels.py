import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

CONTROL_MIN_S = 2.0  # CHALLENGE.md: control = "keeps the ball for a few seconds"


def controlled(labels, fps, min_s=CONTROL_MIN_S):
    out = labels.copy()
    min_frames = int(round(min_s * fps))
    for start, stop in pm.valid_runs(labels):
        owner = None
        for a, b, code in pm.segments(labels[start:stop], [(0, stop - start)]):
            if code == pm.D:
                owner = None
            elif owner is None or (code != owner and b - a >= min_frames):
                owner = code
            elif code != owner:
                out[start + a:start + b] = owner
    return out
