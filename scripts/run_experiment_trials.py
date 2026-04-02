#!/usr/bin/env python3
"""
Run the recommended CPC finetune experiment trials (baseline, freeze, DANN, mixup, SWA)
across tasks and seeds, then optionally evaluate with TTA.

Usage:
  # Run all trials with your CPC encoder (single task, 3 seeds)
  python scripts/run_experiment_trials.py --encoder pretrain_results/all_tasks/cpc/6b8fd6c79b

  # Run on all 3 OOD tasks
  python scripts/run_experiment_trials.py --encoder pretrain_results/all_tasks/cpc/6b8fd6c79b --tasks all

  # Custom seeds and output dir
  python scripts/run_experiment_trials.py --encoder path/to/encoder_weights.pt --seeds 42 43 44 45 --output_dir results/exp1

  # Skip specific trials (e.g. only baseline + DANN)
  python scripts/run_experiment_trials.py --encoder path/to/encoder.pt --only baseline dann
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"

# Trial definitions: (name, config_basename, freeze_backbone, use_dann, extra_flags)
TRIALS = [
    ("baseline", "cpc_baseline.yaml", False, False, []),
    ("freeze", "cpc_baseline.yaml", True, False, []),
    ("dann", "cpc_baseline.yaml", False, True, []),
    ("freeze_dann", "cpc_baseline.yaml", True, True, []),
    ("mixup", "cpc_mixup.yaml", False, False, []),
    ("dann_mixup", "cpc_mixup.yaml", False, True, []),
    ("swa", "cpc_swa.yaml", False, False, []),
    ("dann_swa", "cpc_swa.yaml", False, True, []),
    # --- New OOD techniques ---
    ("sam", "cpc_sam.yaml", False, False, ["--use_sam"]),
    ("dann_sam", "cpc_sam.yaml", False, True, ["--use_sam"]),
    ("manifold_mixup", "cpc_manifold_mixup.yaml", False, False, ["--manifold_mixup"]),
    ("dann_manifold_mixup", "cpc_manifold_mixup.yaml", False, True, ["--manifold_mixup"]),
    ("dann_coral", "cpc_coral.yaml", False, True, []),
    ("dann_xdomain_mixup", "cpc_mixup.yaml", False, True, ["--cross_domain_mixup"]),
    ("kitchen_sink", "cpc_kitchen_sink.yaml", False, True, ["--use_sam", "--manifold_mixup", "--cross_domain_mixup"]),
    # --- Freeze combos (freeze backbone + domain alignment on heads) ---
    ("freeze_coral", "cpc_coral.yaml", True, True, []),
    ("freeze_xdomain_mixup", "cpc_mixup.yaml", True, True, ["--cross_domain_mixup"]),
    ("freeze_sam", "cpc_sam.yaml", True, False, ["--use_sam"]),
]

DEFAULT_TASKS = ["HumanActivityRecognition", "ProximityRecognition", "HumanIdentification"]
DEFAULT_SEEDS = [42, 43, 44]
MODEL = "cpc"


def parse_args():
    p = argparse.ArgumentParser(description="Run CPC finetune experiment trials.")
    p.add_argument("--encoder", type=str, required=True,
                   help="Path to CPC encoder dir or encoder_weights.pt file.")
    p.add_argument("--config_dir", type=str, default=None,
                   help=f"Directory of experiment configs (default: {CONFIG_DIR}).")
    p.add_argument("--tasks", nargs="+", default=["HumanActivityRecognition"],
                   help="Task names or 'all' for HumanActivityRecognition, ProximityRecognition, HumanIdentification.")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Random seeds per run.")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    p.add_argument("--output_dir", type=str, default="results",
                   help="Base directory for results.")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional tag (e.g. 'domain_aware') appended as a subdirectory under output_dir.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only these trial names (e.g. --only baseline dann).")
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip final TTA evaluation step.")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers.")
    return p.parse_args()


def resolve_encoder(encoder_path: str):
    """Return path to encoder .pt file and config if present."""
    p = Path(encoder_path)
    if not p.exists():
        return None, None
    if p.is_file() and p.suffix in (".pt", ".pth"):
        config = p.parent / "config.yaml"
        return str(p), str(config) if config.exists() else None
    # Directory: find latest .pt
    pts = list(p.rglob("*.pt"))
    if not pts:
        return None, None
    latest = max(pts, key=lambda x: x.stat().st_mtime)
    config = latest.parent / "config.yaml"
    return str(latest), str(config) if config.exists() else None


def run_trial(args, trial_name, config_path, freeze, use_dann, task, seed, extra_flags=None):
    """Run one train_supervised.py invocation. Returns success and result path."""
    cmd = [
        sys.executable, "-u", "scripts/train_supervised.py",
        "--task", task,
        "--model", MODEL,
        "--pretrained_encoder", args.encoder_path,
        "--config", config_path,
        "--save_dir", args.output_dir,
        "--output_dir", args.output_dir,
        "--num_workers", str(args.num_workers),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--experiment_id", f"{trial_name}_s{seed}",
    ]
    if freeze:
        cmd.append("--freeze_backbone")
    if use_dann:
        cmd.append("--use_dann")
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if result.returncode != 0:
        return False, None

    # Find latest results JSON for this task
    base = Path(args.output_dir) / task / MODEL
    pattern = str(base / "*" / f"{MODEL}_{task}_results.json")
    import glob
    files = glob.glob(pattern)
    if not files:
        return True, None
    latest = max(files, key=os.path.getmtime)
    return True, latest


def main():
    args = parse_args()
    args.encoder_path, default_config = resolve_encoder(args.encoder)
    if not args.encoder_path:
        print(f"Error: encoder path not found or no .pt file: {args.encoder}")
        sys.exit(1)
    if args.tag:
        args.output_dir = str(Path(args.output_dir) / args.tag)
        print(f"Tag: {args.tag} -> output_dir: {args.output_dir}")
    print(f"Using encoder: {args.encoder_path}")
    if default_config:
        print(f"Encoder dir config: {default_config}")

    config_dir = Path(args.config_dir or CONFIG_DIR)
    if not config_dir.exists():
        print(f"Error: config dir not found: {config_dir}")
        sys.exit(1)

    tasks = DEFAULT_TASKS if args.tasks == ["all"] else args.tasks
    trials = TRIALS
    if args.only:
        trials = [t for t in TRIALS if t[0] in args.only]
        if len(trials) != len(args.only):
            unknown = set(args.only) - {t[0] for t in trials}
            print(f"Warning: unknown trial names ignored: {unknown}")

    summary = {
        "encoder": args.encoder_path,
        "tasks": tasks,
        "seeds": args.seeds,
        "trials": [],
        "by_trial": {},
    }

    for trial_name, config_basename, freeze, use_dann, extra_flags in trials:
        config_path = config_dir / config_basename
        if not config_path.exists():
            print(f"Skipping trial {trial_name}: config not found {config_path}")
            continue
        config_path = str(config_path)
        summary["by_trial"][trial_name] = {"freeze": freeze, "use_dann": use_dann, "config": config_basename, "extra_flags": extra_flags, "runs": []}

        for task in tasks:
            for seed in args.seeds:
                run_id = f"{trial_name}/{task}/seed{seed}"
                print(f"\n{'='*60}")
                print(f"  {run_id}")
                print(f"{'='*60}\n")
                ok, result_path = run_trial(args, trial_name, config_path, freeze, use_dann, task, seed, extra_flags)
                summary["by_trial"][trial_name]["runs"].append({
                    "task": task, "seed": seed, "ok": ok, "results_file": result_path
                })
                if not ok:
                    print(f"  >> FAILED {run_id}")
                elif result_path:
                    print(f"  >> OK {run_id} -> {result_path}")

    # Write summary
    out_path = Path(args.output_dir) / "experiment_trials_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")

    if not args.skip_eval:
        print("\nTo evaluate with TTA, run evaluate_ood.py for each checkpoint, e.g.:")
        print("  python scripts/evaluate_ood.py --weights <best_model.pt> --config <config.yaml> --pipeline supervised --tasks <Task> --tta --tta_rounds 10")

    print("\nAll trials completed.")


if __name__ == "__main__":
    main()
