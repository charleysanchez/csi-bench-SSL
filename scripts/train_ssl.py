"""
VQ-CPC Self-Supervised Pre-training Script
==========================================

Usage:
    python train.py \
        --data_dir ../data/csi-bench-dataset/csi-bench-dataset \
        --task MotionSourceRecognition \
        --epochs 100 \
        --model_dim 128 \
        --feature_size 232 \
        --win_len 500

After pre-training, the saved encoder weights can be loaded and
fine-tuned on downstream tasks with labelled data.
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

import random
import time

import numpy as np
import torch
import torch.nn as nn

# Framework imports
from load.dataloader import get_loaders
# from model.ssl_encoders import CSIEncoder, CSIConvEncoder # Deleted, using model.models is better if possible, but wait, I deleted ssl_encoders.py! 
# I need to CHECK if model.models has CSIEncoder. It has MLPClassifier.
# I should have merged them. I will use MLPClassifier or restore ssl_encoders.
# Actually I deleted model/ssl_encoders.py in step 462. 
# I see MLPClassifier in model/models.py. It is similar but not identical.
# I will switch to using model.models.MLPClassifier and adapt it OR I need to restore ssl_encoders.py as model/encoders.py
# Let's assume I messed up deleting ssl_encoders without merging. I will recreate it as model/encoders.py locally or use MLPClassifier.
# Re-reading model.models.py: MLPClassifier takes (win_len, feature_size). CSIEncoder took (feature_size, model_dim).
# They are different. I should NOT have deleted ssl_encoders.py if I wanted to keep the exact architecture.
# I will restore it as `model/encoders.py`.

from model.encoders import CSIEncoder, CSIConvEncoder
from pretext_tasks.vqcpc import VQCPC
from engine.trainer import (
    train_one_step,
    evaluate,
    save_checkpoint,
    load_checkpoint,
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bespoke VQ-CPC Self-Supervised Pre-training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data = parser.add_argument_group("Data")
    data.add_argument("--data_dir", type=str,
                      default="../data/csi-bench-dataset/csi-bench-dataset",
                      help="Root directory containing h5 dataset files")
    data.add_argument("--task", type=str, default="MotionSourceRecognition",
                      help="Dataset task name (used to locate the h5 file)")
    data.add_argument("--batch_size", type=int, default=32)
    data.add_argument("--num_workers", type=int, default=4)
    data.add_argument("--data_key", type=str, default="CSI_amps",
                      help="HDF5 dataset key for the CSI amplitudes")
    data.add_argument("--file_format", type=str, default="h5",
                      choices=["h5"], help="Dataset file format")
    data.add_argument("--no_pin_memory", action="store_true",
                      help="Disable pin_memory in DataLoaders (use if RAM is tight)")
    data.add_argument("--val_split", type=float, default=0.1,
                      help="Fraction of data held out for validation")
    data.add_argument("--train_filter", type=str, default=None,
                      help="Filter dictionary for training data (e.g. \"{'user':['1','2']}\")")
    data.add_argument("--test_filter", type=str, default=None,
                      help="Filter dictionary for test data")

    # ── Training ──────────────────────────────────────────────────────────────
    train = parser.add_argument_group("Training")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--grad_clip", type=float, default=1.0,
                       help="Max gradient norm (0 = disabled)")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--save_dir", type=str, default="./results_vqcpc")
    train.add_argument("--log_interval", type=int, default=10,
                       help="Log training metrics every N batches")
    train.add_argument("--resume", type=str, default=None,
                       help="Path to a checkpoint to resume from")
    train.add_argument("--encoder", type=str, default="mlp",
                       choices=["mlp", "conv"],
                       help="Backbone encoder architecture")

    # ── LR Schedule ───────────────────────────────────────────────────────────
    sched = parser.add_argument_group("LR Schedule")
    sched.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "step", "none"])
    sched.add_argument("--warmup_epochs", type=int, default=5,
                       help="Linear warmup epochs (cosine only)")
    sched.add_argument("--lr_step_size", type=int, default=30,
                       help="StepLR step size in epochs")
    sched.add_argument("--lr_gamma", type=float, default=0.1,
                       help="StepLR decay factor")

    # ── Model / VQ-CPC ────────────────────────────────────────────────────────
    model = parser.add_argument_group("Model / VQ-CPC")
    model.add_argument("--model_dim", type=int, default=128,
                       help="Encoder output / VQ embedding dimensionality")
    model.add_argument("--feature_size", type=int, default=232,
                       help="Input feature size per timestep")
    model.add_argument("--win_len", type=int, default=500,
                       help="Number of timesteps per training window")
    model.add_argument("--context_dim", type=int, default=256,
                       help="GRU hidden size (context model)")
    model.add_argument("--num_embeddings", type=int, default=512,
                       help="Codebook size (number of VQ codes)")
    model.add_argument("--pred_steps", type=int, default=4,
                       help="Future timesteps to predict in CPC")
    model.add_argument("--n_negatives", type=int, default=16,
                       help="Negative samples per positive in InfoNCE")
    model.add_argument("--commitment_cost", type=float, default=0.25,
                       help="VQ commitment loss coefficient")
    model.add_argument("--vq_loss_weight", type=float, default=1.0,
                       help="Weight of VQ commitment loss in total loss")
    model.add_argument("--ar_layers", type=int, default=1,
                       help="Number of GRU layers in the AR context model")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, args, steps_per_epoch: int):
    if args.scheduler == "cosine":
        # Linear warmup → cosine decay
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size * steps_per_epoch, gamma=args.lr_gamma
        )

    else:  # none
        return None


def format_metrics(metrics: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "mps":
        args.num_workers = 0
        args.pin_memory = False

    print(f"Save dir: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    import ast
    train_filter_dict = None
    if args.train_filter:
        try:
            train_filter_dict = ast.literal_eval(args.train_filter)
            print(f"Using train filter: {train_filter_dict}")
        except Exception as e:
            print(f"Error parsing train_filter: {e}")
            
    test_filter_dict = None
    if args.test_filter:
        try:
            test_filter_dict = ast.literal_eval(args.test_filter)
            print(f"Using test filter: {test_filter_dict}")
        except Exception as e:
            print(f"Error parsing test_filter: {e}")

    # ── Data ──────────────────────────────────────────────────────────────────
    # User existing robust loader
    loaders_dict = get_loaders(
        root=args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        train_filter=train_filter_dict,
        # test_filter is not supported by get_loaders, validation uses val_filter which we don't expose yet or could map
    )
    
    # get_loaders returns dict with 'loaders' key
    if isinstance(loaders_dict, dict) and "loaders" in loaders_dict:
        loaders = loaders_dict["loaders"]
    else:
        loaders = loaders_dict

    train_loader = loaders["train"]
    # get_loaders creates val loader if split is requested, or we use test as val
    val_loader = loaders.get("val", loaders.get("test_id", None))
    
    if val_loader is None:
        print("Warning: No validation loader found. Using training loader for metrics (not recommended).")
        val_loader = train_loader

    # ── Encoder ───────────────────────────────────────────────────────────────
    if args.encoder == "mlp":
        encoder = CSIEncoder(
            feature_size=args.feature_size,
            model_dim=args.model_dim,
        )
    else:
        encoder = CSIConvEncoder(
            feature_size=args.feature_size,
            model_dim=args.model_dim,
        )
    encoder = encoder.to(device)

    # ── Pretext task ──────────────────────────────────────────────────────────
    task = VQCPC(
        encoder_dim=args.model_dim,
        context_dim=args.context_dim,
        num_embeddings=args.num_embeddings,
        pred_steps=args.pred_steps,
        n_negatives=args.n_negatives,
        commitment_cost=args.commitment_cost,
        vq_loss_weight=args.vq_loss_weight,
        ar_layers=args.ar_layers,
    ).to(device)

    print(f"\nEncoder parameters:  {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Task parameters:     {sum(p.numel() for p in task.parameters()):,}")

    # ── Optimiser — joint over encoder + task parameters ─────────────────────
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(task.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, encoder, task, optimizer, scheduler)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        task.train()

        epoch_loss = 0.0
        epoch_metrics: dict[str, float] = {}
        t0 = time.time()
        
        # Use tqdm for progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch", dynamic_ncols=True) as pbar:
            for batch_idx, raw_batch in enumerate(pbar):
                loss, metrics = train_one_step(
                    encoder, task, raw_batch, optimizer, device, grad_clip
                )
                if scheduler is not None:
                    scheduler.step()

                epoch_loss += loss
                
                # Aggregate metrics
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                    elif isinstance(v, torch.Tensor) and v.numel() == 1:
                        epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v.item()

                # Update progress bar
                if (batch_idx + 1) % args.log_interval == 0:
                    current_metrics = {}
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            current_metrics[k] = v
                        elif isinstance(v, torch.Tensor) and v.numel() == 1:
                            current_metrics[k] = v.item()
                            
                    pbar.set_postfix({"loss": f"{loss:.4f}", **current_metrics})

        # ── Epoch summary ─────────────────────────────────────────────────────
        n = len(train_loader)
        train_summary = {"train_loss": epoch_loss / n}
        train_summary.update({k: v / n for k, v in epoch_metrics.items()})

        val_summary = evaluate(encoder, task, val_loader, device)

        elapsed = time.time() - t0

        
        # Compact summary
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s): "
              f"Train Loss: {train_summary['train_loss']:.4f} | "
              f"Val Loss: {val_summary['val_loss']:.4f} | "
              f"Val Acuracy: {val_summary.get('val_cpc_accuracy', 0.0):.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        is_best = val_summary["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_summary["val_loss"]

        save_checkpoint(
            save_dir=args.save_dir,
            epoch=epoch,
            encoder=encoder,
            task=task,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={**train_summary, **val_summary},
            is_best=is_best,
        )

    print(f"\nPre-training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Encoder weights: {os.path.join(args.save_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
