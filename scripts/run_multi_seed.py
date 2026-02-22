#!/usr/bin/env python3
"""Run multiple seeds of the best config and aggregate results.

Since the experiment_id is derived from a hash of params (not seed),
all seeds overwrite the same dir. We capture results immediately after
each run completes.
"""
import subprocess
import json
import os
import sys
import glob
import numpy as np

# Configuration
SEEDS = list(range(42, 57))  # 15 seeds: 42-56
CONFIG = "configs/paper_low_lr.yaml"
ENCODER = "pretrain_results/all_tasks/timesformer1d/fe5af1b0dc/encoder_weights.pt"
TASK = "FallDetection"
MODEL = "timesformer1d"
RESULTS_BASE = "results/FallDetection/timesformer1d"
MULTI_SEED_DIR = "results/FallDetection/timesformer1d/multi_seed_results"

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
        "--pretrained_encoder", ENCODER,
        "--config", CONFIG,
        "--save_dir", "results",
        "--output_dir", "results",
        "--num_workers", "8",
        "--seed", str(seed),
    ]

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
