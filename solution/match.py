import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

HOME, AWAY = 0, 1


@dataclass
class Match:
    name: str
    fps: int
    length: float
    width: float
    xy: np.ndarray  # (T, S, 2) metres, origin at a corner, NaN where not tracked
    team: np.ndarray  # (S,) HOME / AWAY
    ball: np.ndarray  # (T, 2) metres, NaN where not tracked
    halves: List[Tuple[int, int]]  # half-open [start, stop) spans of play
    labels: Optional[np.ndarray] = None  # (T,) possession_metrics codes; X outside halves
    ball_z: Optional[np.ndarray] = None  # (T,) metres, only clean other-game data has it
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return self.xy.shape[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = {"xy": self.xy, "team": self.team, "ball": self.ball,
                  "halves": np.array(self.halves, dtype=np.int64)}
        if self.labels is not None:
            arrays["labels"] = self.labels
        if self.ball_z is not None:
            arrays["ball_z"] = self.ball_z
        header = {"name": self.name, "fps": self.fps, "length": self.length,
                  "width": self.width, "meta": self.meta}
        np.savez_compressed(path, header=np.array(json.dumps(header)), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            return cls(name=header["name"], fps=header["fps"], length=header["length"],
                       width=header["width"], xy=z["xy"], team=z["team"], ball=z["ball"],
                       halves=[tuple(map(int, h)) for h in z["halves"]],
                       labels=z["labels"] if "labels" in z else None,
                       ball_z=z["ball_z"] if "ball_z" in z else None,
                       meta=header.get("meta", {}))

    def in_halves(self):
        mask = np.zeros(self.n_frames, bool)
        for start, stop in self.halves:
            mask[start:stop] = True
        return mask
