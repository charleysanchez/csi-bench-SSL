"""
Compare domain_aware CPC vs regular CPC results.

Usage:
  python scripts/analyze_domain_aware.py
  python scripts/analyze_domain_aware.py --metric f1_score
  python scripts/analyze_domain_aware.py --trials baseline dann dann_mixup kitchen_sink
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

TASKS = ["HumanActivityRecognition", "HumanIdentification", "ProximityRecognition"]
SPLITS = ["test_cross_user", "test_cross_env", "test_cross_device"]
SEEDS = [42, 43, 44]
MODEL = "cpc"

ROOT = Path(__file__).resolve().parents[1]
REGULAR_DIR = ROOT / "results"
DOMAIN_AWARE_DIR = ROOT / "results" / "domain_aware"


def load_result(base_dir, task, trial, seed):
    path = base_dir / task / MODEL / f"{trial}_s{seed}" / f"{MODEL}_{task}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect(base_dir, tasks, trials, seeds, metric):
    """Returns {trial: {task: {split: [values across seeds]}}}"""
    data = {}
    for trial in trials:
        data[trial] = {}
        for task in tasks:
            data[trial][task] = {s: [] for s in SPLITS}
            for seed in seeds:
                res = load_result(base_dir, task, trial, seed)
                if res is None:
                    continue
                for split in SPLITS:
                    if split in res:
                        data[trial][task][split].append(res[split][metric])
    return data


def mean_std(vals):
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def print_comparison(regular, domain_aware, trials, metric):
    col_w = 14

    for trial in trials:
        print(f"\n{'='*72}")
        print(f"  Trial: {trial}   (metric: {metric})")
        print(f"{'='*72}")
        header = f"{'Task':<28} {'Split':<18} {'Regular':>{col_w}} {'DomainAware':>{col_w}} {'Delta':>{col_w}}"
        print(header)
        print("-" * 72)

        for task in TASKS:
            short_task = task.replace("HumanActivityRecognition", "HAR") \
                             .replace("HumanIdentification", "HumanID") \
                             .replace("ProximityRecognition", "ProxRec")
            for split in SPLITS:
                short_split = split.replace("test_cross_", "X-")
                reg_vals = regular.get(trial, {}).get(task, {}).get(split, [])
                da_vals = domain_aware.get(trial, {}).get(task, {}).get(split, [])
                reg_mean, reg_std = mean_std(reg_vals)
                da_mean, da_std = mean_std(da_vals)
                delta = da_mean - reg_mean if not (np.isnan(reg_mean) or np.isnan(da_mean)) else float("nan")
                sign = "+" if delta > 0 else ""
                delta_str = f"{sign}{delta*100:.1f}%" if not np.isnan(delta) else "  N/A"
                reg_str = f"{reg_mean*100:.1f}±{reg_std*100:.1f}" if not np.isnan(reg_mean) else "  N/A"
                da_str = f"{da_mean*100:.1f}±{da_std*100:.1f}" if not np.isnan(da_mean) else "  N/A"
                marker = " <<" if delta > 0.01 else (" >>" if delta < -0.01 else "")
                print(f"{short_task:<28} {short_split:<18} {reg_str:>{col_w}} {da_str:>{col_w}} {delta_str:>{col_w}}{marker}")


def print_summary(regular, domain_aware, trials, metric):
    """Average delta across all tasks and seeds per trial per OOD split."""
    print(f"\n{'='*72}")
    print(f"  SUMMARY: Avg delta (domain_aware - regular) across all tasks")
    print(f"  metric: {metric}")
    print(f"{'='*72}")
    header = f"{'Trial':<28} {'X-User':>10} {'X-Env':>10} {'X-Device':>10} {'OOD Avg':>10}"
    print(header)
    print("-" * 72)

    for trial in trials:
        split_deltas = {s: [] for s in SPLITS}
        for task in TASKS:
            for split in SPLITS:
                reg_vals = regular.get(trial, {}).get(task, {}).get(split, [])
                da_vals = domain_aware.get(trial, {}).get(task, {}).get(split, [])
                reg_mean, _ = mean_std(reg_vals)
                da_mean, _ = mean_std(da_vals)
                if not (np.isnan(reg_mean) or np.isnan(da_mean)):
                    split_deltas[split].append(da_mean - reg_mean)

        avgs = {s: np.mean(v) if v else float("nan") for s, v in split_deltas.items()}
        all_vals = [v for vals in split_deltas.values() for v in vals]
        overall = np.mean(all_vals) if all_vals else float("nan")

        def fmt(v):
            if np.isnan(v):
                return "   N/A"
            sign = "+" if v > 0 else ""
            return f"{sign}{v*100:.1f}%"

        marker = " <<" if overall > 0.005 else (" >>" if overall < -0.005 else "")
        print(f"{trial:<28} {fmt(avgs['test_cross_user']):>10} {fmt(avgs['test_cross_env']):>10} {fmt(avgs['test_cross_device']):>10} {fmt(overall):>10}{marker}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="f1_score", choices=["f1_score", "accuracy"])
    p.add_argument("--trials", nargs="+", default=None,
                   help="Subset of trials to compare (default: all found in regular results)")
    args = p.parse_args()

    # Auto-detect available trials
    trial_dirs = sorted(
        d.name.rsplit("_s", 1)[0]
        for d in (REGULAR_DIR / "HumanActivityRecognition" / MODEL).iterdir()
        if d.is_dir() and not d.name.startswith("best")
    )
    all_trials = sorted(set(trial_dirs))
    trials = args.trials if args.trials else all_trials

    print(f"Comparing {len(trials)} trials across {len(TASKS)} tasks, {len(SEEDS)} seeds")
    print(f"Regular:      {REGULAR_DIR}")
    print(f"Domain-aware: {DOMAIN_AWARE_DIR}")

    regular = collect(REGULAR_DIR, TASKS, trials, SEEDS, args.metric)
    domain_aware = collect(DOMAIN_AWARE_DIR, TASKS, trials, SEEDS, args.metric)

    print_comparison(regular, domain_aware, trials, args.metric)
    print_summary(regular, domain_aware, trials, args.metric)


if __name__ == "__main__":
    main()
