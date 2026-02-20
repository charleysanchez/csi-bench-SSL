#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import json
import hashlib
import yaml
import time

from load.dataloader import get_loaders
from engine.masked_trainer import MaskedTrainer
from utils.config import update_args_with_yaml, save_config
from model.models import (
    MaskedLSTM,
    MaskedPatchTST,
    MaskedTimesFormer1D,
    MaskedTransformer
)

# -------------------------------------------------
# MODEL REGISTRY (must output reconstruction)
# -------------------------------------------------

MODEL_TYPES = {
    "lstm": MaskedLSTM,
    "patchtst": MaskedPatchTST,
    "timesformer1d": MaskedTimesFormer1D,
    "transformer": MaskedTransformer,
}

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Masked CSI Pretraining")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, default="lstm",
                        choices=list(MODEL_TYPES.keys()))

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--win_len", type=int, default=500)
    parser.add_argument("--feature_size", type=int, default=232)

    parser.add_argument("--save_dir", type=str, default="pretrain_results")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file to override args")

    args = parser.parse_args()
    
    # Override args with YAML config if provided
    if args.config is not None:
        args = update_args_with_yaml(args, args.config)

    # -------------------------------------------------
    # SEED
    # -------------------------------------------------

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------------------------------
    # EXPERIMENT ID
    # -------------------------------------------------

    param_str = f"{args.model}_{args.learning_rate}_{args.batch_size}_{args.epochs}"
    experiment_id = hashlib.md5(param_str.encode()).hexdigest()[:10]

    results_dir = os.path.join(
        args.save_dir, args.task, args.model, experiment_id
    )
    os.makedirs(results_dir, exist_ok=True)

    print(f"Experiment ID: {experiment_id}")
    print(f"Results: {results_dir}")

    # -------------------------------------------------
    # DEVICE
    # -------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------
    # LOAD DATA (NO LABELS USED)
    # -------------------------------------------------

    print("Loading dataset...")
    data = get_loaders(
        root=args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    train_loader = data["loaders"]["train"]
    val_loader = data["loaders"].get("val", None)

    if val_loader is None:
        print("No validation split found — using training loader for validation.")
        val_loader = train_loader

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------

    ModelClass = MODEL_TYPES[args.model]

    model = ModelClass(
        win_len=args.win_len,
        feature_size=args.feature_size,
    )

    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # -------------------------------------------------
    # CONFIG SAVE
    # -------------------------------------------------

    save_config(args, os.path.join(results_dir, "config.yaml"))

    # -------------------------------------------------
    # TRAINER
    # -------------------------------------------------

    trainer = MaskedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_path=results_dir,
        config=args,
    )

    print("\nStarting masked pretraining...\n")

    model, training_results = trainer.train()

    # -------------------------------------------------
    # SAVE FINAL SUMMARY
    # -------------------------------------------------

    summary = {
        "best_epoch": training_results["best_epoch"],
        "best_val_loss": float(training_results["best_val_loss"]),
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    encoder_state_dict = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("head")
    }

    torch.save(encoder_state_dict, os.path.join(results_dir, "encoder_weights.pt"))

    print("\nPretraining complete.")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best val loss: {summary['best_val_loss']:.6f}")

    return summary, model


if __name__ == "__main__":
    main()