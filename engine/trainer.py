"""
Generic training loop for any PretextTask.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pretext_tasks.base import PretextTask, PretextBatch


def train_one_step(
    encoder: nn.Module,
    task: PretextTask,
    raw_batch,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> tuple[float, dict]:
    """
    Single training step. Handles both SinglePassTask and MultiPassTask.

    Returns:
        loss_value: float
        metrics:    dict of task-specific diagnostics (from batch.metadata)
    """
    # Move data to device
    if isinstance(raw_batch, (tuple, list)):
        raw_batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in raw_batch)
    elif torch.is_tensor(raw_batch):
        raw_batch = raw_batch.to(device)

    # Transform raw batch → PretextBatch
    batch: PretextBatch = task(raw_batch)

    optimizer.zero_grad()

    if task.uses_model_directly():
        loss = task.compute_loss_with_model(encoder, batch)
    else:
        output = encoder(batch.inputs.to(device))
        loss = task.compute_loss(output, batch)

    loss.backward()

    if grad_clip is not None:
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(task.parameters()),
            grad_clip,
        )

    optimizer.step()

    return loss.item(), batch.metadata


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    task: PretextTask,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run a full validation pass and return averaged metrics.
    """
    encoder.eval()
    task.eval()

    total_loss = 0.0
    agg_metrics: dict[str, float] = {}
    n_batches = 0

    for raw_batch in loader:
        if isinstance(raw_batch, (tuple, list)):
            raw_batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in raw_batch)
        elif torch.is_tensor(raw_batch):
            raw_batch = raw_batch.to(device)

        batch: PretextBatch = task(raw_batch)

        if task.uses_model_directly():
            loss = task.compute_loss_with_model(encoder, batch)
        else:
            output = encoder(batch.inputs.to(device))
            loss = task.compute_loss(output, batch)

        total_loss += loss.item()
        for k, v in batch.metadata.items():
            if isinstance(v, (int, float)):
                agg_metrics[k] = agg_metrics.get(k, 0.0) + v
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                agg_metrics[k] = agg_metrics.get(k, 0.0) + v.item()
        n_batches += 1

    results = {"val_loss": total_loss / max(n_batches, 1)}
    for k, v in agg_metrics.items():
        results[f"val_{k}"] = v / max(n_batches, 1)

    encoder.train()
    task.train()

    return results


def save_checkpoint(
    save_dir: str,
    epoch: int,
    encoder: nn.Module,
    task: PretextTask,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics: dict,
    is_best: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "encoder_state": encoder.state_dict(),
        "task_state": task.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
    }
    path = os.path.join(save_dir, "latest.pt")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(save_dir, "best.pt"))


def load_checkpoint(
    path: str,
    encoder: nn.Module,
    task: PretextTask,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
) -> int:
    """Load a checkpoint and return the starting epoch."""
    state = torch.load(path, map_location="cpu")
    encoder.load_state_dict(state["encoder_state"])
    task.load_state_dict(state["task_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    epoch = state["epoch"] + 1
    print(f"Resumed from checkpoint: {path}  (epoch {state['epoch']})")
    return epoch
