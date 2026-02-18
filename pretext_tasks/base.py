"""
Base classes for the pretext task framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PretextBatch:
    """
    Standardized container passed between transform() and compute_loss().
    The training loop only ever sees this type — task-specific details
    live in `metadata`.
    """
    inputs: torch.Tensor
    targets: Any = None                          # tensor, dict, list — unconstrained
    metadata: dict = field(default_factory=dict) # escape hatch for anything extra


class PretextTask(ABC, nn.Module):
    """
    Abstract base for all pretext tasks.

    Inherits nn.Module so auxiliary parameters (projection heads,
    codebooks, AR models, etc.) are automatically tracked by the optimizer.

    Subclasses implement ONE of two contracts:
      - SinglePassTask  → implement compute_loss(model_output, batch)
      - MultiPassTask   → implement compute_loss_with_model(model, batch)
    """

    @abstractmethod
    def transform(self, raw_batch: Any) -> PretextBatch:
        """
        Convert a raw dataloader batch into a PretextBatch.
        Augmentations, masking, view generation, etc. live here.
        """
        ...

    def uses_model_directly(self) -> bool:
        """
        Return True if this task needs to call the encoder itself
        (e.g. for multiple forward passes, momentum encoders, etc.)
        """
        return False

    def compute_loss(self, model_output: Any, batch: PretextBatch) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_loss() "
            "or set uses_model_directly()=True and implement compute_loss_with_model()."
        )

    def compute_loss_with_model(self, model: nn.Module, batch: PretextBatch) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} declared uses_model_directly()=True "
            "but did not implement compute_loss_with_model()."
        )

    def forward(self, raw_batch: Any) -> PretextBatch:
        return self.transform(raw_batch)


class SinglePassTask(PretextTask):
    """
    Convenience base for tasks where the pipeline handles the forward pass.
    Implement transform() and compute_loss() only.
    """

    @abstractmethod
    def compute_loss(self, model_output: Any, batch: PretextBatch) -> torch.Tensor:
        ...


class MultiPassTask(PretextTask):
    """
    Convenience base for tasks that need to call the encoder themselves.
    Implement transform() and compute_loss_with_model() only.
    """

    def uses_model_directly(self) -> bool:
        return True

    @abstractmethod
    def compute_loss_with_model(self, model: nn.Module, batch: PretextBatch) -> torch.Tensor:
        ...


class MultiTaskWrapper(PretextTask):
    """
    Combines multiple pretext tasks with per-task loss weights.
    Each sub-task's transform is called independently; losses are summed.

    Usage:
        task = MultiTaskWrapper({
            "cpc": (cpc_task, 1.0),
            "rotation": (rotation_task, 0.5),
        })
    """

    def __init__(self, tasks: dict[str, tuple["PretextTask", float]]):
        super().__init__()
        self.tasks = nn.ModuleDict({k: v[0] for k, v in tasks.items()})
        self.weights = {k: v[1] for k, v in tasks.items()}

    def transform(self, raw_batch: Any) -> PretextBatch:
        sub_batches = {name: task.transform(raw_batch) for name, task in self.tasks.items()}
        # Use first task's inputs as the nominal batch inputs
        first = next(iter(sub_batches.values()))
        return PretextBatch(
            inputs=first.inputs,
            targets=None,
            metadata={"sub_batches": sub_batches},
        )

    def uses_model_directly(self) -> bool:
        return any(task.uses_model_directly() for task in self.tasks.values())

    def compute_loss(self, model_output: Any, batch: PretextBatch) -> torch.Tensor:
        total = torch.tensor(0.0, device=model_output.device if hasattr(model_output, 'device') else 'cpu')
        for name, task in self.tasks.items():
            if not task.uses_model_directly():
                sub = batch.metadata["sub_batches"][name]
                total = total + self.weights[name] * task.compute_loss(model_output, sub)
        return total

    def compute_loss_with_model(self, model: nn.Module, batch: PretextBatch) -> torch.Tensor:
        device = next(model.parameters()).device
        total = torch.tensor(0.0, device=device)
        for name, task in self.tasks.items():
            sub = batch.metadata["sub_batches"][name]
            if task.uses_model_directly():
                loss = task.compute_loss_with_model(model, sub)
            else:
                output = model(sub.inputs.to(device))
                loss = task.compute_loss(output, sub)
            total = total + self.weights[name] * loss
        return total
