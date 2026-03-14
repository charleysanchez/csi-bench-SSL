#!/usr/bin/env python3
"""Run multiple seeds of the best config and aggregate results.

Since the experiment_id is derived from a hash of params (not seed),
all seeds overwrite the same dir. We capture results immediately after
each run completes.

Usage:
    python scripts/run_multi_seed.py --task FallDetection --model timesformer1d --config configs/paper_low_lr.yaml
    python scripts/run_multi_seed.py --task FallDetection --model timesformer1d --encoder path/to/encoder.pt --seed_start 42 --seed_end 57
"""
import subprocess
import json
import os
import sys
import glob
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Run multi-seed supervised training and aggregate results.")
    p.add_argument("--task", type=str, default="FallDetection", help="Task name")
    p.add_argument("--model", type=str, default="timesformer1d", help="Model architecture")
    p.add_argument("--config", type=str, default="configs/paper_low_lr.yaml", help="Path to YAML config")
    p.add_argument("--encoder", type=str, default="pretrain_results/all_tasks/timesformer1d/fe5af1b0dc/encoder_weights.pt",
                   help="Path to pretrained encoder .pt (omit or empty to train from scratch)")
    p.add_argument("--results_base", type=str, default=None,
                   help="Base dir for results (default: results/<task>/<model>)")
    p.add_argument("--multi_seed_dir", type=str, default=None,
                   help="Dir for multi-seed output JSON (default: <results_base>/multi_seed_results_zeropad)")
    p.add_argument("--seed_start", type=int, default=42, help="First seed (inclusive)")
    p.add_argument("--seed_end", type=int, default=57, help="Last seed (exclusive); e.g. 57 gives seeds 42..56")
    p.add_argument("--save_dir", type=str, default="results", help="Passed to train_supervised --save_dir")
    p.add_argument("--output_dir", type=str, default="results", help="Passed to train_supervised --output_dir")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    return p.parse_args()


args = parse_args()
SEEDS = list(range(args.seed_start, args.seed_end))
CONFIG = args.config
ENCODER = args.encoder if args.encoder else None
TASK = args.task
MODEL = args.model
RESULTS_BASE = args.results_base or os.path.join("results", TASK, MODEL)
MULTI_SEED_DIR = args.multi_seed_dir or os.path.join(RESULTS_BASE, "multi_seed_results_zeropad")

os.makedirs(MULTI_SEED_DIR, exist_ok=True)

all_results = []

for i, seed in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"  SEED {seed} ({i+1}/{len(SEEDS)})")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "scripts/train_supervised.py",
        "--task", TASK,
        "--model", MODEL,
        "--config", CONFIG,
        "--save_dir", args.save_dir,
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
        "--seed", str(seed),
    ]
    if ENCODER:
        cmd.extend(["--pretrained_encoder", ENCODER])

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"  SEED {seed} FAILED (exit code {result.returncode})")
        continue

    # Find the most recently modified results JSON
    pattern = os.path.join(RESULTS_BASE, "params_*", f"{MODEL}_{TASK}_results.json")
    result_files = glob.glob(pattern)
    if not result_files:
        print(f"  WARNING: No results file found for seed {seed}")
        continue

    # Get the most recently modified one
    latest = max(result_files, key=os.path.getmtime)
    
    with open(latest) as f:
        res = json.load(f)
    
    seed_result = {"seed": seed}
    for split_name, split_data in res.items():
        if isinstance(split_data, dict):
            seed_result[f"{split_name}_acc"] = split_data.get("accuracy", 0)
            seed_result[f"{split_name}_f1"] = split_data.get("f1_score", 0)
    
    all_results.append(seed_result)
    
    test_id_acc = seed_result.get("test_id_acc", seed_result.get("test_acc", 0))
    test_hard_acc = seed_result.get("test_hard_acc", 0)
    print(f"\n  >> SEED {seed}: test_id={test_id_acc*100:.2f}%, test_hard={test_hard_acc*100:.2f}%")
    
    # Save intermediate results after each seed
    intermediate = {
        "config": CONFIG,
        "encoder": ENCODER,
        "completed_seeds": len(all_results),
        "total_seeds": len(SEEDS),
        "per_seed": all_results,
    }
    with open(os.path.join(MULTI_SEED_DIR, "multi_seed_progress.json"), "w") as f:
        json.dump(intermediate, f, indent=2)

# Final aggregation
print(f"\n{'='*60}")
print(f"  AGGREGATE RESULTS ({len(all_results)}/{len(SEEDS)} seeds)")
print(f"{'='*60}\n")

if all_results:
    # The results JSON uses "test" for test_id
    splits = [
        ("test_id", ["test_id_acc", "test_acc"]),
        ("test_easy", ["test_easy_acc"]),
        ("test_medium", ["test_medium_acc"]),
        ("test_hard", ["test_hard_acc"]),
    ]
    
    summary = {}
    for split_label, acc_keys in splits:
        accs = []
        f1s = []
        for r in all_results:
            for key in acc_keys:
                if key in r:
                    accs.append(r[key])
                    f1_key = key.replace("_acc", "_f1")
                    if f1_key in r:
                        f1s.append(r[f1_key])
                    break
        
        if accs:
            summary[split_label] = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "f1_mean": float(np.mean(f1s)) if f1s else 0,
                "f1_std": float(np.std(f1s)) if f1s else 0,
                "n_seeds": len(accs),
                "all_accs": [float(a) for a in accs],
            }
            print(f"  {split_label:15s}  Acc: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%  "
                  f"F1: {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}%  (n={len(accs)})")

    # Save final results
    output = {
        "config": CONFIG,
        "encoder": ENCODER,
        "num_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "per_seed": all_results,
        "aggregate": summary,
    }
    
    output_path = os.path.join(MULTI_SEED_DIR, "multi_seed_summary.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {output_path}")
else:
    print("  No results collected!")
