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

# Ensure we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_supervised import MODEL_TYPES

def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed evaluation on OOD tasks.")
    parser.add_argument("--encoder", type=str, default=None, help="Path to encoder_weights.pt file. Omit to train from scratch.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml. Auto-inferred if encoder is provided.")
    parser.add_argument("--model", type=str, default=None, help="Model architecture. Auto-inferred if encoder is provided.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for.")
    parser.add_argument("--pipeline", type=str, default="supervised", choices=["supervised", "multitask"], help="Choose training approach.")
    parser.add_argument('--freeze_backbone', action="store_true", default=False,
                        help='freeze backbone to allow pretraining to work on its own.')
    # wandb parameters
    parser.add_argument("--use_wandb", action="store_true", help="Enable tracking with Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="cs8803hsi", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name")
    return parser.parse_args()

args = parse_args()
valid_models = list(MODEL_TYPES.keys())

MODEL = args.model
ENCODER = None
CONFIG = args.config

# Only parse encoder paths if an encoder was actually provided
if args.encoder is not None:
    if os.path.isdir(args.encoder):
        search_pattern = os.path.join(args.encoder, "**", "*.pt")
        encoder_files = glob.glob(search_pattern, recursive=True)
        if not encoder_files:
            print(f"Error: No .pt files found in directory {args.encoder}")
            sys.exit(1)
        ENCODER = max(encoder_files, key=os.path.getmtime)
        dir_name = os.path.dirname(ENCODER)
        if CONFIG is None:
            CONFIG = os.path.join(dir_name, "config.yaml")
        print(f"Directory passed. Found latest encoder weights: {ENCODER}")
    else:
        ENCODER = args.encoder
        if CONFIG is None:
            CONFIG = os.path.join(os.path.dirname(ENCODER), "config.yaml")
            
    # Try to derive the model name from the path if not provided
    if MODEL is None:
        parts = Path(ENCODER).parts
        for part in reversed(parts):
            if part in valid_models:
                MODEL = part
                break
        if MODEL is None and len(parts) >= 3:
            MODEL = parts[-3]

# Fallback if model still isn't set
if MODEL is None:
    MODEL = "transformer"
    print(f"Could not derive model name, defaulting to {MODEL}.")
else:
    print(f"Using model architecture: {MODEL}")

# Configuration
SEEDS = [42, 43, 44]  # 3 seeds for speed
TASKS = ["HumanActivityRecognition", "ProximityRecognition", "HumanIdentification"]


# ==============================================================================
# MULTITASK PIPELINE (Joint Training)
# ==============================================================================
if args.pipeline == "multitask":
    print(f"\n{'='*80}")
    print(f"  STARTING JOINT MULTITASK TRAINING FOR: {TASKS}")
    print(f"{'='*80}\n")
    
    all_task_results = {task: [] for task in TASKS}
    tasks_str = ",".join(TASKS)
    
    for i, seed in enumerate(SEEDS):
        print(f"\n{'-'*60}")
        print(f"  MULTITASK JOINT TRAINING - SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'-'*60}\n")
        
        cmd = [
            sys.executable, "-u", "scripts/train_multitask_adapter.py",
            "--tasks", tasks_str,
            "--model", MODEL,
            "--save_dir", "results",
            "--num_workers", "8",
            "--seed", str(seed),
            "--epochs", str(args.epochs)
        ]
        
        if ENCODER is not None: cmd.extend(["--pretrained_encoder", ENCODER])
        if CONFIG is not None: cmd.extend(["--config", CONFIG])
        if args.freeze_backbone:
            cmd.extend(["--freeze_backbone"])
            
        if args.use_wandb:
            cmd.extend(["--use_wandb"])
            cmd.extend(["--wandb_project", args.wandb_project])
            if args.wandb_entity:
                cmd.extend(["--wandb_entity", args.wandb_entity])
                
        result = subprocess.run(cmd, text=True)
        
        if result.returncode != 0:
            print(f"  SEED {seed} FAILED (exit code {result.returncode})")
            continue
            
        # Collect results for ALL tasks from this single run
        print("\n  Extracting results...")
        for task in TASKS:
            RESULTS_BASE = f"results/{task}/{MODEL}"
            pattern = os.path.join(RESULTS_BASE, "*", f"{MODEL}_{task}_test_summary.json")
            result_files = glob.glob(pattern)
            
            if not result_files:
                print(f"  WARNING: No results file found for {task} seed {seed}")
                continue
                
            latest = max(result_files, key=os.path.getmtime)
            with open(latest) as f:
                res = json.load(f)
                
            if "test_results" in res: 
                res = res["test_results"]
                
            seed_result = {"seed": seed}
            for split_name, split_data in res.items():
                if isinstance(split_data, dict):
                    seed_result[f"{split_name}_acc"] = split_data.get("accuracy", 0)
                    seed_result[f"{split_name}_f1"] = split_data.get("f1_score", 0)
                    
            all_task_results[task].append(seed_result)
            
            test_id_acc = seed_result.get("test_id_acc", seed_result.get("test_acc", 0))
            test_env = seed_result.get("test_cross_env_acc", 0)
            print(f"  >> {task} SEED {seed}: id={test_id_acc*100:.2f}%, env={test_env*100:.2f}%")

    # Aggregate Multitask Results
    for task, task_results in all_task_results.items():
        if not task_results: continue
            
        MULTI_SEED_DIR = f"results/{task}/{MODEL}/multi_seed_results"
        os.makedirs(MULTI_SEED_DIR, exist_ok=True)
        
        splits = [
            ("test_id", ["test_id_acc", "test_acc"]),
            ("test_cross_env", ["test_cross_env_acc"]),
            ("test_cross_user", ["test_cross_user_acc"]),
            ("test_cross_device", ["test_cross_device_acc"]),
        ]
        
        summary = {}
        print(f"\n  {task} (MULTITASK) AGGREGATE RESULTS ({len(task_results)} seeds):")
        for split_label, acc_keys in splits:
            accs, f1s = [], []
            for r in task_results:
                for key in acc_keys:
                    if key in r and r[key] > 0:
                        accs.append(r[key])
                        f1_key = key.replace("_acc", "_f1")
                        if f1_key in r: f1s.append(r[f1_key])
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
            "config": CONFIG, "encoder": ENCODER if ENCODER else "Scratch", "pipeline": "multitask",
            "aggregate": summary, "per_seed": task_results
        }
        with open(os.path.join(MULTI_SEED_DIR, "multitask_multi_seed_summary.json"), "w") as f:
            json.dump(output, f, indent=2)

# ==============================================================================
# SUPERVISED PIPELINE (Single Task Training)
# ==============================================================================
elif args.pipeline == "supervised":
    all_results = {}
    for task in TASKS:
        print(f"\n{'='*80}")
        print(f"  STARTING SUPERVISED TASK: {task}")
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
                "--save_dir", "results",
                "--output_dir", "results",
                "--num_workers", "8",
                "--seed", str(seed),
                "--epochs", str(args.epochs)
            ]
            
            if ENCODER is not None: cmd.extend(["--pretrained_encoder", ENCODER])
            if CONFIG is not None: cmd.extend(["--config", CONFIG])
            
            if args.use_wandb:
                cmd.extend(["--use_wandb"])
                cmd.extend(["--wandb_project", args.wandb_project])
                if args.wandb_entity:
                    cmd.extend(["--wandb_entity", args.wandb_entity])
            
            result = subprocess.run(cmd, text=True)
            
            if result.returncode != 0:
                print(f"  SEED {seed} FAILED")
                continue
                
            pattern = os.path.join(RESULTS_BASE, "params_*", f"{MODEL}_{task}_results.json")
            result_files = glob.glob(pattern)
            
            if not result_files:
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
            print(f"\n  >> {task} SEED {seed}: id={test_id_acc*100:.2f}%")
            
        all_results[task] = task_results
        
        # Aggregate Supervised Results
        if task_results:
            splits = [
                ("test_id", ["test_id_acc", "test_acc"]),
                ("test_cross_env", ["test_cross_env_acc"]),
                ("test_cross_user", ["test_cross_user_acc"]),
                ("test_cross_device", ["test_cross_device_acc"]),
            ]
            
            summary = {}
            print(f"\n  {task} (SUPERVISED) AGGREGATE RESULTS ({len(task_results)} seeds):")
            for split_label, acc_keys in splits:
                accs, f1s = [], []
                for r in task_results:
                    for key in acc_keys:
                        if key in r and r[key] > 0:
                            accs.append(r[key])
                            f1_key = key.replace("_acc", "_f1")
                            if f1_key in r: f1s.append(r[f1_key])
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
                "config": CONFIG, "encoder": ENCODER if ENCODER else "Scratch", "pipeline": "supervised",
                "aggregate": summary, "per_seed": task_results
            }
            with open(os.path.join(MULTI_SEED_DIR, "supervised_multi_seed_summary.json"), "w") as f:
                json.dump(output, f, indent=2)

print("\nAll OOD tasks completed!")