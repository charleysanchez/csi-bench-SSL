#!/usr/bin/env python3
"""Run multi-seed evaluation on the 3 true OOD tasks and aggregate results."""
import subprocess
import json
import os
import sys
import glob
import numpy as np

# Configuration
SEEDS = [42, 43, 44]  # 3 seeds for speed
CONFIG = "configs/paper_low_lr.yaml"
ENCODER = "pretrain_results/all_tasks/timesformer1d/fe5af1b0dc/encoder_weights.pt"
MODEL = "timesformer1d"
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
