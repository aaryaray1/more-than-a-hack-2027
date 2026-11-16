import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from solution import io_atd, models, registry  # noqa: E402
from validation import harness  # noqa: E402

PIPELINE = "v4"
TRAIN_SEEDS = (0, 1)
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def predict(pipeline=PIPELINE, seeds=TRAIN_SEEDS):
    match = io_atd.load()
    if not registry.is_learned(pipeline):
        return match, registry.predict(pipeline, match)
    model = registry.make(pipeline)
    if isinstance(model, models.RichTeam):
        train = ((m, truth["control"]) for _, m, truth in harness.iter_mixed_runs(seeds))
        return match, model.fit(train, cache=False).predict(match)
    train = [(m, truth["control"]) for _, m, truth in harness.mixed_runs(seeds)]
    return match, model.fit(train).predict(match)


def write_csv(codes, match, path):
    labels = np.full(match.n_frames, "X", dtype="<U1")
    graded = match.in_halves()
    labels[graded] = np.array(pm.LABELS)[codes[graded]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.write("".join("{},{}\n".format(i, label) for i, label in enumerate(labels)))


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="data/ -> evaluation/results/<name>.csv")
    parser.add_argument("--name", default=PIPELINE, help="file name without .csv (the team name)")
    parser.add_argument("--pipeline", default=PIPELINE, choices=registry.names())
    args = parser.parse_args()

    match, codes = predict(args.pipeline)
    path = os.path.join(RESULTS_DIR, args.name + ".csv")
    write_csv(codes, match, path)
    stats = harness.label_free(codes, match)
    print("label-free: {poss} poss ({poss_per_min:.2f}/min), mean {mean_s:.1f} s, "
          "D {d_share:.1%}, spells < 2 s: {short}".format(**stats))
    print("sha256 {}  {}".format(sha256(path), os.path.relpath(path, ROOT)))
    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    sys.exit(subprocess.call([sys.executable, validator, path]))


if __name__ == "__main__":
    main()
