#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import torch
import numpy as np
import json
import hashlib
import yaml
import time

from torch.utils.data import DataLoader, ConcatDataset
from load.dataloader import get_loaders
from load.dataset import CSIDataset
from load.pretrain_dataset import PretrainDataset
from engine.masked_trainer import MaskedTrainer
from engine.cpc_trainer import CPCTrainer
from utils.config import update_args_with_yaml, save_config
from model.models import (
    MaskedLSTM,
    MaskedPatchTST,
    MaskedTimesFormer1D,
    MaskedTransformer,
    CPCModel
)

# -------------------------------------------------
# MODEL REGISTRY (must output reconstruction)
# -------------------------------------------------

MODEL_TYPES = {
    "lstm": MaskedLSTM,
    "patchtst": MaskedPatchTST,
    "timesformer1d": MaskedTimesFormer1D,
    "transformer": MaskedTransformer,
    "cpc": CPCModel,
}

# -------------------------------------------------
# MAIN
# -------------------------------------------------

class CollateSkipNone:
    def __init__(self, win_len, feature_size):
        self.win_len = win_len
        self.feature_size = feature_size

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return torch.zeros(0, 1, self.win_len, self.feature_size), torch.zeros(0), []
        return torch.utils.data.dataloader.default_collate(batch)

def main():

    parser = argparse.ArgumentParser(description="Masked & CPC CSI Pretraining")

    parser.add_argument("--pretrain_method", type=str, default="masked", choices=["masked", "cpc"],
                        help="Pretraining method to use.")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--task", type=str, default=None,
                        help="Single task to pretrain on. Omit to use --all_tasks.")
    parser.add_argument("--all_tasks", action="store_true",
                        help="Pretrain on all available tasks combined.")
    parser.add_argument("--model", type=str, default="lstm",
                        choices=list(MODEL_TYPES.keys()))

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cpc_k_steps", type=int, default=4, help="Number of steps for CPC future prediction")

    parser.add_argument("--win_len", type=int, default=500)
    parser.add_argument("--feature_size", type=int, default=232)

    parser.add_argument("--save_dir", type=str, default="pretrain_results")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file to override args")

    args = parser.parse_args()

    # require either --task or --all_tasks
    if args.task is None and not args.all_tasks:
        parser.error("Provide either --task <task_name> or --all_tasks")

    # Override args with YAML config if provided
    if args.config is not None:
        args = update_args_with_yaml(args, args.config)

    # -------------------------------------------------
    # SEED
    # -------------------------------------------------

    if args.pretrain_method == "cpc":
        args.model = "cpc"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------------------------------
    # EXPERIMENT ID
    # -------------------------------------------------

    task_label = "all_tasks" if args.all_tasks else args.task
    param_str = f"{args.model}_{args.learning_rate}_{args.batch_size}_{args.epochs}"
    experiment_id = hashlib.md5(param_str.encode()).hexdigest()[:10]

    results_dir = os.path.join(
        args.save_dir, task_label, args.model, experiment_id
    )
    os.makedirs(results_dir, exist_ok=True)

    print(f"Experiment ID: {experiment_id}")
    print(f"Results: {results_dir}")

    # -------------------------------------------------
    # DEVICE
    # -------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------
    # COLLATE FN
    # -------------------------------------------------

    def collate_skip_none(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return torch.zeros(0, 1, args.win_len, args.feature_size), torch.zeros(0)
            
        collated = torch.utils.data.dataloader.default_collate(batch)
        
        # Check if the dataset returned (inputs, labels)
        if isinstance(collated, (list, tuple)):
            inputs = collated[0]
            # If [Batch, Time, Features], add the dummy Channel dimension
            if len(inputs.shape) == 3: 
                inputs = inputs.unsqueeze(1) # -> [Batch, 1, Time, Features]
            
            # Reconstruct and return the tuple
            return (inputs,) + tuple(collated[1:])
            
        # Check if the dataset returned just inputs
        else:
            inputs = collated
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            return inputs

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------

    if args.all_tasks:
        print("\nLoading datasets for all tasks...")

        task_dirs = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
            and d not in ["RawContinuousRecording"]#, "ProximityRecognition", "Localization"]
        ])
        print(f"Found tasks: {task_dirs}")

        train_datasets = []
        val_datasets = []

        for task in task_dirs:
            task_dir = os.path.join(args.data_dir, task)
            splits_dir = os.path.join(task_dir, "splits")

            if not os.path.exists(splits_dir):
                print(f"  Skipping {task} — no splits directory")
                continue

            try:
                train_ds = PretrainDataset(
                    root=args.data_dir,
                    task=task,
                    split="train_id",
                    task_dir=task_dir,
                )
                train_datasets.append(train_ds)
                print(f"  {task} train: {len(train_ds)} samples")
            except Exception as e:
                print(f"  Skipping {task} train: {e}")

            try:
                val_ds = PretrainDataset(
                    root=args.data_dir,
                    task=task,
                    split="val_id",
                    task_dir=task_dir,
                )
                val_datasets.append(val_ds)
                print(f"  {task} val:   {len(val_ds)} samples")
            except Exception as e:
                print(f"  Skipping {task} val: {e}")

        if not train_datasets:
            raise RuntimeError("No training datasets found.")

        combined_train = ConcatDataset(train_datasets)
        combined_val = ConcatDataset(val_datasets) if val_datasets else None

        print(f"\nTotal train samples: {len(combined_train)}")
        print(f"Total val samples:   {len(combined_val) if combined_val else 0}")

        train_loader = DataLoader(
            combined_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_skip_none,
        )

        val_loader = DataLoader(
            combined_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_skip_none,
        ) if combined_val else None

    else:
        print(f"Loading dataset for task: {args.task}")
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
    
    model_kwargs = {}

    # Map parameters to model_kwargs based on model type
    if args.model == 'lstm':
        model_kwargs.update({'feature_size': args.feature_size})
        
    elif args.model == 'transformer':
        # Transformers usually take d_model/emb_dim, dropout, etc.
        model_kwargs.update({
            'feature_size': args.feature_size,
            'd_model': getattr(args, 'emb_dim', getattr(args, 'd_model', 128)),
            'dropout': getattr(args, 'dropout', 0.1)
        })
        # Add these if your MaskedTransformer specifically expects them
        if hasattr(args, 'num_heads'): model_kwargs['nhead'] = args.num_heads
        if hasattr(args, 'depth'): model_kwargs['num_layers'] = args.depth
        
    elif args.model == 'patchtst':
        model_kwargs.update({
            'win_len': args.win_len,
            'feature_size': args.feature_size,
            'patch_len': getattr(args, 'patch_len', 16),
            'stride': getattr(args, 'stride', 8),
            'emb_dim': getattr(args, 'emb_dim', 128),
            'depth': getattr(args, 'depth', 4),
            'num_heads': getattr(args, 'num_heads', 4),
            'dropout': getattr(args, 'dropout', 0.1)
        })
        
    elif args.model == 'timesformer1d':
        model_kwargs.update({
            'win_len': args.win_len,
            'feature_size': args.feature_size,
            'patch_size': getattr(args, 'patch_size', 4),
            'emb_dim': getattr(args, 'emb_dim', 128),
            'depth': getattr(args, 'depth', 4),
            'num_heads': getattr(args, 'num_heads', 4),
            'dropout': getattr(args, 'dropout', 0.1)
        })

    print(f"Creating {args.model} model with kwargs: {model_kwargs}")
    model = ModelClass(**model_kwargs)

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
    # STEP-LEVEL LR SCHEDULER
    # -------------------------------------------------

    num_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    min_lr_ratio = getattr(args, 'min_lr_ratio', 0.0)

    def warmup_cosine_schedule(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    print(f"Using step-level warmup cosine LR scheduler: {warmup_steps} warmup steps, {num_steps} total steps")

    # -------------------------------------------------
    # CONFIG SAVE
    # -------------------------------------------------

    save_config(args, os.path.join(results_dir, "config.yaml"))

    # -------------------------------------------------
    # TRAINER
    # -------------------------------------------------

    if args.pretrain_method == "masked":
        TrainerClass = MaskedTrainer
    elif args.pretrain_method == "cpc":
        TrainerClass = CPCTrainer
    else:
        raise ValueError(f"Unknown pretrain_method: {args.pretrain_method}")

    trainer = TrainerClass(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=results_dir,
        config=args,
    )

    print(f"\nStarting {args.pretrain_method} pretraining...\n")

    model, training_results = trainer.train()

    # -------------------------------------------------
    # SAVE FINAL SUMMARY
    # -------------------------------------------------

    summary = {
        "best_epoch": training_results["best_epoch"],
        "best_val_loss": float(training_results["best_val_loss"]),
        "experiment_id": experiment_id,
        "task": task_label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    encoder_state_dict = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("head")
    }

    encoder_path = os.path.join(results_dir, "encoder_weights.pt")
    torch.save(encoder_state_dict, encoder_path)
    print(f"Encoder weights saved to {encoder_path}")

    print("\nPretraining complete.")
    print(f"Best epoch:    {summary['best_epoch']}")
    print(f"Best val loss: {summary['best_val_loss']:.6f}")

    return summary, model


if __name__ == "__main__":
    main()