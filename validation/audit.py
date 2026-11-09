import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import features, io_atd  # noqa: E402
from validation import harness  # noqa: E402

QUANTILES = (5, 25, 50, 75, 95)
SAMPLE = 20000


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                               - np.searchsorted(b, grid, side="right") / len(b))))


def sample(x, mask, rng):
    idx = np.flatnonzero(mask)
    return x[rng.choice(idx, size=min(SAMPLE, len(idx)), replace=False)]


def main():
    rng = np.random.default_rng(0)
    bar = io_atd.load()
    xb, names = features.frame_features(bar)
    xb = sample(xb, bar.in_halves(), rng)
    xh = np.vstack([sample(features.frame_features(m)[0], m.in_halves(), rng)
                    for _, m, _ in harness.harness_runs((0,))])
    rows = []
    for j, name in enumerate(names):
        a, b = xh[:, j], xb[:, j]
        na, nb = float(np.mean(np.isnan(a))), float(np.mean(np.isnan(b)))
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        rows.append((ks(a, b) if len(a) and len(b) else 1.0, name, na, nb,
                     np.percentile(a, QUANTILES) if len(a) else None,
                     np.percentile(b, QUANTILES) if len(b) else None))
    rows.sort(key=lambda r: -r[0])
    print("feature distributions: harness (degraded IDSSE) vs BAR-ATH, graded frames, aggregates")
    print("{:<12} {:>5} {:>13}  {:<34} {}".format("feature", "KS", "NaN h/B", "harness p5/25/50/75/95",
                                                  "BAR-ATH p5/25/50/75/95"))
    for d, name, na, nb, qa, qb in rows:
        fmt = lambda q: "-" if q is None else "/".join("{:.3g}".format(v) for v in q)  # noqa: E731
        print("{:<12} {:>5.2f} {:>6.2f}/{:<6.2f}  {:<34} {}".format(name, d, na, nb, fmt(qa), fmt(qb)))


if __name__ == "__main__":
    main()
