#!/usr/bin/env python3
"""
Few-shot linear probe evaluation for pretrained CPC encoders.

After pretraining (pretrain.py or pretrain_raw.py), this script measures how
classification performance on downstream tasks scales with the number of labeled
examples available.  It answers: "How much labeled data do we need after
pretraining before the model is useful?"

Methodology:
  1. Load a frozen CPCModel encoder (g_enc + g_ar).
  2. Extract fixed-size representations by mean-pooling the GRU context c over time.
  3. For each task × each k_shot value:
       - Sample exactly k_shot examples per class from the train split.
       - Fit a logistic regression classifier (no fine-tuning of the encoder).
       - Evaluate on test_id, test_xenv, test_xuser, test_xdevice.
  4. Repeat each k_shot over multiple seeds to estimate variance.
  5. Save results CSV + learning-curve plots.

Data leakage note:
  If the encoder was pretrained on raw data that includes users / environments
  present in the OOD test splits, those metrics may be slightly optimistic.
  Always compare the pretrain metadata (pretrain_metadata.csv) against the
  benchmark splits before drawing conclusions from xuser / xenv numbers.

Usage:
  python scripts/linear_probe.py \\
      --encoder pretrain_results/raw/<id>/encoder_weights.pt \\
      --data_dir data \\
      --tasks HumanActivityRecognition ProximityRecognition HumanIdentification \\
      --k_shots 1 2 4 8 16 32 64 full \\
      --seeds 42 43 44
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

from load.dataset import CSIDataset
from load.collate import CollateSkipNone
from model.models import CPCModel


# -----------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------

@torch.no_grad()
def extract_features(model: CPCModel, loader: DataLoader, device: torch.device):
    """
    Run the frozen encoder over a dataloader and return (features, labels).

    Representation: mean of the GRU context c over the time dimension.
    This gives a (hidden_size,) vector per sample — a global summary of the
    temporal context learned by CPC.

    Returns:
        feats:  np.ndarray  (N, hidden_size)
        labels: np.ndarray  (N,)
    """
    model.eval()
    all_feats, all_labels = [], []

    for batch in loader:
        x = batch[0].to(device)          # (B, 1, T, F) or (B, T, F)
        y = batch[1]                      # (B,) integer labels

        if x.dim() == 3:
            x = x.unsqueeze(1)

        _, c, _ = model(x)               # c: (B, T, hidden_size)
        feats = c.mean(dim=1).cpu().numpy()   # (B, hidden_size)

        all_feats.append(feats)
        all_labels.append(y.numpy() if torch.is_tensor(y) else np.array(y))

    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)


# -----------------------------------------------------------------------
# Few-shot sampling
# -----------------------------------------------------------------------

def sample_k_shot(feats, labels, k, rng: np.random.Generator):
    """
    Return indices for a balanced k-shot sample (k examples per class).
    Classes with fewer than k examples use all available examples.

    Args:
        feats:  (N, D) features (unused here, but kept for signature symmetry)
        labels: (N,) integer class labels
        k:      shots per class; -1 means use all data
        rng:    numpy Generator for reproducibility

    Returns:
        train_idx: np.ndarray of selected indices
    """
    if k == -1:
        return np.arange(len(labels))

    classes = np.unique(labels)
    idx = []
    for c in classes:
        class_idx = np.where(labels == c)[0]
        n = min(k, len(class_idx))
        chosen = rng.choice(class_idx, size=n, replace=False)
        idx.append(chosen)

    return np.concatenate(idx)


# -----------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------

SPLITS = ["test_id", "test_cross_env", "test_cross_user", "test_cross_device"]


def load_split(task_dir, split, win_len, feature_size, batch_size, num_workers):
    """Load a CSIDataset split; return None if the split file is missing."""
    split_path = os.path.join(task_dir, "splits", f"{split}.json")
    if not os.path.exists(split_path):
        return None

    try:
        ds = CSIDataset(task_dir=task_dir, split=split)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=CollateSkipNone(win_len, feature_size, for_supervised=True),
        )
        return loader
    except Exception as e:
        print(f"    Warning: could not load {split}: {e}")
        return None


def fit_and_score(train_feats, train_labels, test_feats, test_labels):
    """
    Fit a logistic regression on train_feats and score on test_feats.
    Returns (accuracy, macro_f1).
    """
    if len(np.unique(train_labels)) < 2:
        # Only one class in training — degenerate; return chance performance.
        return 0.0, 0.0

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feats)
    X_test  = scaler.transform(test_feats)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
        )
        clf.fit(X_train, train_labels)

    preds = clf.predict(X_test)
    acc  = accuracy_score(test_labels, preds)
    f1   = f1_score(test_labels, preds, average="macro", zero_division=0)
    return float(acc), float(f1)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Few-shot linear probe for pretrained CPC encoders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--encoder", type=str, required=True,
                        help="Path to encoder_weights.pt from pretrain.py / pretrain_raw.py")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--tasks", nargs="+",
                        default=["HumanActivityRecognition",
                                 "ProximityRecognition",
                                 "HumanIdentification"])

    # k_shot: integer values + the keyword "full"
    parser.add_argument("--k_shots", nargs="+",
                        default=["1", "2", "4", "8", "16", "32", "64", "full"],
                        help="Number of labeled examples per class. 'full' = all training data.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                        help="Random seeds for k-shot sampling (results averaged).")

    parser.add_argument("--win_len",      type=int, default=500)
    parser.add_argument("--feature_size", type=int, default=232)
    parser.add_argument("--hidden_size",  type=int, default=256)
    parser.add_argument("--cpc_k_steps",  type=int, default=4)

    parser.add_argument("--batch_size",   type=int, default=256)
    parser.add_argument("--num_workers",  type=int, default=4)

    parser.add_argument("--save_dir", type=str, default="results/linear_probe",
                        help="Directory for CSV results and plots")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Parse k_shots: convert to int or -1 for "full"
    k_shot_values = []
    for s in args.k_shots:
        if s == "full":
            k_shot_values.append(-1)
        else:
            k_shot_values.append(int(s))

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load encoder
    # -----------------------------------------------------------------------
    model = CPCModel(
        feature_size=args.feature_size,
        hidden_size=args.hidden_size,
        cpc_k_steps=args.cpc_k_steps,
    ).to(device)

    state = torch.load(args.encoder, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys (expected if head not saved): {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    # Freeze encoder entirely — we are training only a linear probe.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    print(f"Encoder loaded from {args.encoder}")

    # -----------------------------------------------------------------------
    # Per-task evaluation
    # -----------------------------------------------------------------------
    all_records = []

    for task in args.tasks:
        task_dir = os.path.join(args.data_dir, task)
        if not os.path.isdir(task_dir):
            print(f"\nSkipping {task} — directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        # Load train split to extract features once (expensive)
        train_loader = load_split(
            task_dir, "train_id",
            args.win_len, args.feature_size, args.batch_size, args.num_workers,
        )
        if train_loader is None:
            print(f"  No train_id split found — skipping {task}")
            continue

        print("  Extracting train features...")
        train_feats, train_labels = extract_features(model, train_loader, device)
        print(f"  Train: {train_feats.shape}, classes: {np.unique(train_labels).tolist()}")

        # Load all test splits once
        test_data = {}
        for split in SPLITS:
            loader = load_split(
                task_dir, split,
                args.win_len, args.feature_size, args.batch_size, args.num_workers,
            )
            if loader is not None:
                print(f"  Extracting {split} features...")
                feats, labels = extract_features(model, loader, device)
                test_data[split] = (feats, labels)

        if not test_data:
            print(f"  No test splits found — skipping {task}")
            continue

        # -----------------------------------------------------------------------
        # Few-shot sweep
        # -----------------------------------------------------------------------
        for k in k_shot_values:
            k_label = "full" if k == -1 else str(k)

            seed_accs  = {split: [] for split in test_data}
            seed_f1s   = {split: [] for split in test_data}

            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                idx = sample_k_shot(train_feats, train_labels, k, rng)

                if len(idx) == 0:
                    continue

                X_k = train_feats[idx]
                y_k = train_labels[idx]

                for split, (t_feats, t_labels) in test_data.items():
                    acc, f1 = fit_and_score(X_k, y_k, t_feats, t_labels)
                    seed_accs[split].append(acc)
                    seed_f1s[split].append(f1)

            # Aggregate over seeds
            for split in test_data:
                if not seed_accs[split]:
                    continue
                record = {
                    "task": task,
                    "split": split,
                    "k_shot": k_label,
                    "k_shot_int": k if k != -1 else len(train_feats),
                    "acc_mean":  np.mean(seed_accs[split]),
                    "acc_std":   np.std(seed_accs[split]),
                    "f1_mean":   np.mean(seed_f1s[split]),
                    "f1_std":    np.std(seed_f1s[split]),
                    "n_seeds":   len(seed_accs[split]),
                }
                all_records.append(record)

                print(
                    f"  k={k_label:>4}  {split:<20}  "
                    f"acc={record['acc_mean']:.3f}±{record['acc_std']:.3f}  "
                    f"f1={record['f1_mean']:.3f}±{record['f1_std']:.3f}"
                )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    if not all_records:
        print("\nNo results to save.")
        return

    results_df = pd.DataFrame(all_records)
    csv_path = os.path.join(args.save_dir, "few_shot_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # -----------------------------------------------------------------------
    # Plots: one figure per task, one line per split
    # -----------------------------------------------------------------------
    _plot_learning_curves(results_df, args.save_dir)
    print(f"Plots saved → {args.save_dir}/")


def _plot_learning_curves(df: pd.DataFrame, save_dir: str):
    """
    For each task, plot F1 vs. k_shot for each test split.
    x-axis is log-scaled to show few-shot regime clearly.
    """
    split_styles = {
        "test_id":           dict(color="#2196F3", linestyle="-",  label="In-dist"),
        "test_cross_env":    dict(color="#FF9800", linestyle="--", label="Cross-env"),
        "test_cross_user":   dict(color="#4CAF50", linestyle="-.", label="Cross-user"),
        "test_cross_device": dict(color="#F44336", linestyle=":",  label="Cross-device"),
    }

    for task in df["task"].unique():
        task_df = df[df["task"] == task].copy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Linear Probe — {task}", fontsize=13)

        for metric, ax in zip(["acc_mean", "f1_mean"], axes):
            std_col = metric.replace("mean", "std")
            ylabel  = "Accuracy" if metric == "acc_mean" else "Macro F1"

            for split in SPLITS:
                split_df = task_df[task_df["split"] == split].sort_values("k_shot_int")
                if split_df.empty:
                    continue

                style = split_styles.get(split, {})
                xs = split_df["k_shot_int"].values
                ys = split_df[metric].values
                es = split_df[std_col].values

                ax.plot(xs, ys, marker="o", **style)
                ax.fill_between(xs, ys - es, ys + es,
                                color=style.get("color", "gray"), alpha=0.15)

            ax.set_xscale("log")
            ax.set_xlabel("Labeled examples per class (log scale)")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        safe_name = task.replace(" ", "_")
        fig_path = os.path.join(save_dir, f"few_shot_{safe_name}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Plot: {fig_path}")


if __name__ == "__main__":
    main()
