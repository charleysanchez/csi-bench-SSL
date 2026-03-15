"""
Shared optimizer and LR scheduler setup for supervised and multitask training.
Use these helpers so param groups and step-based warmup cosine are defined in one place.
"""
import math
import torch
from torch.optim.lr_scheduler import LambdaLR


# Default head parameter name substrings (supervised / DANN single model)
DEFAULT_HEAD_KEYWORDS = (
    "classifier.",
    "head.",
    "fc.",
    "main_head.",
    "user_head.",
    "env_head.",
    "device_head.",
)


def get_param_groups(
    model,
    lr,
    backbone_lr_scale=0.1,
    head_keywords=None,
):
    """
    Split parameters into backbone and head groups for discriminative learning rates.
    Returns a list of param-group dicts for use with AdamW(param_groups=...).

    Args:
        model: nn.Module (must have named_parameters()).
        lr: Base learning rate (used for head).
        backbone_lr_scale: Scale for backbone LR (backbone gets lr * backbone_lr_scale).
        head_keywords: Tuple of substrings; if any appears in param name, it's head. Default: DEFAULT_HEAD_KEYWORDS.

    Returns:
        list of dicts: [{"params": backbone_params, "lr": backbone_lr}, {"params": head_params, "lr": head_lr}]
    """
    if head_keywords is None:
        head_keywords = DEFAULT_HEAD_KEYWORDS

    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = any(k in name for k in head_keywords)
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    backbone_lr = lr * backbone_lr_scale
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": lr},
    ]


def build_optimizer(
    model,
    lr,
    weight_decay,
    backbone_lr_scale=None,
    head_keywords=None,
):
    """
    Build AdamW optimizer. Optionally use discriminative LR (backbone at lr * backbone_lr_scale, head at lr).

    Args:
        model: nn.Module.
        lr: Learning rate.
        weight_decay: Weight decay.
        backbone_lr_scale: If not None, use param groups with backbone at lr * backbone_lr_scale, head at lr.
        head_keywords: Passed to get_param_groups when backbone_lr_scale is set.

    Returns:
        torch.optim.AdamW
    """
    if backbone_lr_scale is not None:
        param_groups = get_param_groups(model, lr, backbone_lr_scale=backbone_lr_scale, head_keywords=head_keywords)
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def step_warmup_cosine_scheduler(
    optimizer,
    num_steps,
    warmup_steps,
    min_lr_ratio=0.0,
):
    """
    Step-based (per-batch) warmup + cosine decay. Use when the trainer calls scheduler.step() every batch.

    Args:
        optimizer: Optimizer.
        num_steps: Total number of steps (e.g. len(train_loader) * epochs).
        warmup_steps: Number of warmup steps.
        min_lr_ratio: Minimum LR as a fraction of base LR (default 0).

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def build_optimizer_and_scheduler_for_supervised(
    model,
    train_loader,
    epochs,
    lr,
    weight_decay,
    warmup_epochs,
    backbone_lr_scale=None,
    head_keywords=None,
):
    """
    One-shot builder for supervised training: AdamW (optionally with backbone/head groups) + step warmup cosine.
    Returns (optimizer, scheduler). Trainer should call scheduler.step() every batch.
    """
    optimizer = build_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        backbone_lr_scale=backbone_lr_scale,
        head_keywords=head_keywords,
    )
    num_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    scheduler = step_warmup_cosine_scheduler(optimizer, num_steps, warmup_steps)
    return optimizer, scheduler
