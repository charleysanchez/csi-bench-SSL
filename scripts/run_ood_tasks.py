#!/usr/bin/env python3
"""Run multi-seed evaluation on the 3 true OOD tasks and aggregate results."""
import subprocess
import json
import os
import sys
import glob
import numpy as np

import argparse
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_supervised import MODEL_TYPES

def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed evaluation on OOD tasks.")
    parser.add_argument("--encoder", type=str, default=None, help="Path to encoder_weights.pt file.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for.")
    return parser.parse_args()

args = parse_args()

if os.path.isdir(args.encoder):
    search_pattern = os.path.join(args.encoder, "**", "*.pt")
    encoder_files = glob.glob(search_pattern, recursive=True)
    if not encoder_files:
        print(f"Error: No .pt files found in directory {args.encoder}")
        sys.exit(1)
    ENCODER = max(encoder_files, key=os.path.getmtime)
    dir_name = os.path.dirname(ENCODER)
    CONFIG = os.path.join(dir_name, "config.yaml")
    print(f"Directory passed to --encoder. Automatically found latest encoder weights: {ENCODER}")
else:
    ENCODER = args.encoder

valid_models = list(MODEL_TYPES.keys())
parts = Path(ENCODER).parts

# Check if any part of the path matches a valid model name
for part in reversed(parts):
    if part in valid_models:
        MODEL = part
        break
        
if MODEL:
    print(f"Derived model name '{MODEL}' from encoder path.")
elif len(parts) >= 3:
    MODEL = parts[-3]
    print(f"Derived model name '{MODEL}' from encoder path (fallback).")
else:
    MODEL = "timesformer1d"
    print(f"Could not derive model name, defaulting to {MODEL}.")

# Configuration
SEEDS = [42, 43, 44]  # 3 seeds for speed
TASKS = ["HumanActivityRecognition", "HumanIdentification", "ProximityRecognition"]

all_results = {}

for task in TASKS:
    print(f"\n{'='*80}")
    print(f"  STARTING TASK: {task}")
    print(f"{'='*80}\n")
    
    task_results = []
    RESULTS_BASE = f"results/{task}/{MODEL}"
    MULTI_SEED_DIR = f"{RESULTS_BASE}/multi_seed_results"
    os.makedirs(MULTI_SEED_DIR, exist_ok=True)
    
    for i, seed in enumerate(SEEDS):
        print(f"\n{'-'*60}")
        print(f"  {task} - SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'-'*60}\n")
        
        cmd = [
            sys.executable, "-u", "scripts/train_supervised.py",
            "--task", task,
            "--model", MODEL,
            "--pretrained_encoder", ENCODER,
            "--config", CONFIG,
            "--save_dir", "results",
            "--output_dir", "results",
            "--num_workers", "8",
            "--seed", str(seed),
            "--epochs", str(args.epochs),
        ]
        
        result = subprocess.run(cmd, text=True)
        
        if result.returncode != 0:
            print(f"  SEED {seed} FAILED (exit code {result.returncode})")
            continue
            
        pattern = os.path.join(RESULTS_BASE, "params_*", f"{MODEL}_{task}_results.json")
        result_files = glob.glob(pattern)
        if not result_files:
            print(f"  WARNING: No results file found for seed {seed}")
            continue
            
        latest = max(result_files, key=os.path.getmtime)
        with open(latest) as f:
            res = json.load(f)
            
        seed_result = {"seed": seed}
        for split_name, split_data in res.items():
            if isinstance(split_data, dict):
                seed_result[f"{split_name}_acc"] = split_data.get("accuracy", 0)
                seed_result[f"{split_name}_f1"] = split_data.get("f1_score", 0)
                
        task_results.append(seed_result)
        
        test_id_acc = seed_result.get("test_id_acc", seed_result.get("test_acc", 0))
        test_env = seed_result.get("test_cross_env_acc", 0)
        test_user = seed_result.get("test_cross_user_acc", 0)
        test_dev = seed_result.get("test_cross_device_acc", 0)
        
        print(f"\n  >> {task} SEED {seed}: id={test_id_acc*100:.2f}%, env={test_env*100:.2f}%, user={test_user*100:.2f}%, dev={test_dev*100:.2f}%")
        
    all_results[task] = task_results
    
    # Aggregate for task
    if task_results:
        splits = [
            ("test_id", ["test_id_acc", "test_acc"]),
            ("test_cross_env", ["test_cross_env_acc"]),
            ("test_cross_user", ["test_cross_user_acc"]),
            ("test_cross_device", ["test_cross_device_acc"]),
        ]
        
        summary = {}
        print(f"\n  {task} AGGREGATE RESULTS ({len(task_results)} seeds):")
        for split_label, acc_keys in splits:
            accs = []
            f1s = []
            for r in task_results:
                for key in acc_keys:
                    if key in r and r[key] > 0:
                        accs.append(r[key])
                        f1_key = key.replace("_acc", "_f1")
                        if f1_key in r:
                            f1s.append(r[f1_key])
                        break
            if accs:
                summary[split_label] = {
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
                    "f1_std": float(np.std(f1s)) if f1s else 0.0,
                }
                print(f"    {split_label:18s}  Acc: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%")
                
        output = {
            "config": CONFIG,
            "encoder": ENCODER,
            "aggregate": summary,
            "per_seed": task_results
        }
        with open(os.path.join(MULTI_SEED_DIR, "multi_seed_summary.json"), "w") as f:
            json.dump(output, f, indent=2)

print("\nAll OOD tasks completed!")
