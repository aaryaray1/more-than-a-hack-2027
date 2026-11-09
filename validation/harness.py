import argparse
import csv
import datetime
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from solution import io_atd, registry  # noqa: E402
from validation import degrade, idsse, labels  # noqa: E402

LOG_PATH = os.path.join(ROOT, "planning", "experiments.csv")
SEEDS = (0, 1, 2)
VARIANTS = ("control", "raw")


def harness_runs(seeds=SEEDS, ball_loss=degrade.BallLoss(), tag=""):
    runs = []
    for match_id in idsse.available():
        clean = idsse.load(match_id)
        truth = {"raw": clean.labels, "control": labels.controlled(clean.labels, clean.fps)}
        for seed in seeds:
            run = degrade.degrade(clean, seed, ball_loss=ball_loss)
            runs.append((match_id, replace(run, name=run.name + tag), truth))
    return runs


TRAIN_MIX = ("", "a_dead=1.0,b_dead=-1.0")


def mixed_runs(seeds, shifts=TRAIN_MIX):
    runs = []
    for text in shifts:
        ball_loss, tag = parse_shift(text)
        runs += harness_runs(seeds, ball_loss, tag)
    return runs


def parse_shift(text):
    if not text:
        return degrade.BallLoss(), ""
    values = dict(item.split("=") for item in text.split(","))
    return replace(degrade.BallLoss(), **{k: float(v) for k, v in values.items()}), "~" + text


def predict_runs(name, runs, train_runs=None):
    if not registry.is_learned(name):
        return [registry.predict(name, match) for _, match, _ in runs]
    train_runs = train_runs if train_runs is not None else runs
    out = [None] * len(runs)
    for game in sorted({g for g, _, _ in runs}):
        model = registry.make(name).fit([(m, t["control"]) for g, m, t in train_runs if g != game])
        for i, (g, match, _) in enumerate(runs):
            if g == game:
                out[i] = model.predict(match)
    return out


def predict_bar(name, runs, bar_match):
    if not registry.is_learned(name):
        return registry.predict(name, bar_match)
    return registry.make(name).fit([(m, t["control"]) for _, m, t in runs]).predict(bar_match)


def spell_flags(gt, pred):
    runs = pm.valid_runs(gt)
    gt_spells, pred_spells = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), t) for t in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def spell_f1_from(flags, n_pred, picks):
    n_gt = len(picks)
    f1s = []
    for row in flags:
        tp = int(row[picks].sum())
        fp, fn = max(0, n_pred - tp), n_gt - tp
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def paired_delta(pairs, n_draws, seed=0):
    rng = np.random.RandomState(seed)
    draws = np.zeros(n_draws)
    for (flags_a, n_a), (flags_b, n_b) in pairs:
        n_gt = flags_a.shape[1]
        for d in range(n_draws):
            picks = rng.randint(0, n_gt, n_gt)
            draws[d] += spell_f1_from(flags_b, n_b, picks) - spell_f1_from(flags_a, n_a, picks)
    draws /= len(pairs)
    return tuple(np.percentile(draws, [5, 95]).tolist())


def label_free(codes, match):
    runs = match.halves
    stats = pm.possession_stats(codes, runs)
    spells = pm.possession_spells(codes, runs)
    graded = match.in_halves()
    minutes = graded.sum() / match.fps / 60
    return {"poss": stats["n_spells"], "poss_per_min": stats["n_spells"] / minutes,
            "mean_s": stats["mean_duration_s"], "d_share": float(np.mean(codes[graded] == pm.D)),
            "short": sum(1 for a, b, _ in spells if b - a < 2 * match.fps)}


def git_version():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate(name, versus=None, seeds=SEEDS, n_draws=1000, runs=None, train_runs=None):
    runs = runs if runs is not None else harness_runs(seeds)
    preds = predict_runs(name, runs, train_runs)
    others = predict_runs(versus, runs, train_runs) if versus else [None] * len(runs)
    rows, pairs = [], {v: [] for v in VARIANTS}
    for (_, match, truth), pred, other in zip(runs, preds, others):
        row = {"run": match.name}
        for variant in VARIANTS:
            result = pm.evaluate(truth[variant], pred)
            row[variant] = result["spell_f1_mean"]
            if variant == "control":
                row.update(macro=result["macro_f1"], n_pred=result["n_pred_spells"],
                           n_gt=result["n_gt_spells"], alive_dead=result["alive_dead_f1"],
                           team_acc=result["team_acc_both_alive"])
            if other is not None:
                pairs[variant].append((spell_flags(truth[variant], other),
                                       spell_flags(truth[variant], pred)))
                row[variant + "_vs"] = pm.evaluate(truth[variant], other)["spell_f1_mean"]
        rows.append(row)
    deltas = {v: paired_delta(pairs[v], n_draws) for v in VARIANTS} if versus else {}
    return rows, deltas


def report(name, rows, deltas, versus, bar):
    print("pipeline {}{}".format(name, "  vs " + versus if versus else ""))
    print("{:<22} {:>8} {:>8} {:>7} {:>11} {:>9} {:>8}".format(
        "run", "spellF1", "raw", "macro", "poss p/gt", "aliveF1", "teamAcc"))
    for r in rows:
        print("{:<22} {:>8.4f} {:>8.4f} {:>7.4f} {:>5}/{:<5} {:>9.4f} {:>8.4f}".format(
            r["run"], r["control"], r["raw"], r["macro"], r["n_pred"], r["n_gt"],
            r["alive_dead"], r["team_acc"]))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("control", "raw", "macro")}
    print("{:<22} {:>8.4f} {:>8.4f} {:>7.4f}".format("MEAN", mean["control"], mean["raw"],
                                                       mean["macro"]))
    for variant, (low, high) in deltas.items():
        diff = [r[variant] - r[variant + "_vs"] for r in rows]
        wins = sum(d > 0 for d in diff)
        print("delta {:<8} {:+.4f}  90% CI [{:+.4f}, {:+.4f}]  better on {}/{} runs".format(
            variant, float(np.mean(diff)), low, high, wins, len(diff)))
    print("BAR-ATH label-free: {poss} poss ({poss_per_min:.2f}/min), mean {mean_s:.1f} s, "
          "D {d_share:.1%}, spells < 2 s: {short}".format(**bar))
    return mean


def log(name, versus, mean, rows, deltas, bar, note):
    header = ["date", "version", "pipeline", "versus", "spellF1_control", "spellF1_raw",
              "macroF1", "delta_control", "delta_control_ci", "delta_raw", "per_run_control",
              "bar_poss", "bar_mean_s", "bar_d_share", "note"]
    delta = {v: float(np.mean([r[v] - r[v + "_vs"] for r in rows])) if versus else "" for v in VARIANTS}
    row = [datetime.date.today().isoformat(), git_version(), name, versus or "",
           "{:.4f}".format(mean["control"]), "{:.4f}".format(mean["raw"]),
           "{:.4f}".format(mean["macro"]),
           "{:+.4f}".format(delta["control"]) if versus else "",
           "[{:+.4f},{:+.4f}]".format(*deltas["control"]) if versus else "",
           "{:+.4f}".format(delta["raw"]) if versus else "",
           " ".join("{:.3f}".format(r["control"]) for r in rows),
           bar["poss"], "{:.1f}".format(bar["mean_s"]), "{:.3f}".format(bar["d_share"]), note]
    new = not os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if new:
            writer.writerow(header)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Score a pipeline on the degraded IDSSE harness")
    parser.add_argument("pipeline", choices=registry.names())
    parser.add_argument("--vs", choices=registry.names())
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--shift", metavar="K=V,...",
                        help="degrade the test games with other BallLoss values (train stays default)")
    parser.add_argument("--train-mix", action="store_true",
                        help="train learned pipelines on the default and the looser degrader")
    parser.add_argument("--log", metavar="NOTE", help="append the result to planning/experiments.csv")
    args = parser.parse_args()

    seeds = tuple(range(args.seeds))
    runs = mixed_runs(seeds) if args.train_mix else harness_runs(seeds)
    ball_loss, tag = parse_shift(args.shift)
    test_runs = harness_runs(seeds, ball_loss, tag) if (args.shift or args.train_mix) else runs
    rows, deltas = evaluate(args.pipeline, args.vs, n_draws=args.draws, runs=test_runs,
                            train_runs=runs)
    bar_match = io_atd.load()
    bar = label_free(predict_bar(args.pipeline, runs, bar_match), bar_match)
    mean = report(args.pipeline, rows, deltas, args.vs, bar)
    if args.log is not None:
        log(args.pipeline, args.vs, mean, rows, deltas, bar, args.log)
        print("logged to {}".format(os.path.relpath(LOG_PATH, ROOT)))


if __name__ == "__main__":
    main()
