from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Training loop
    epochs: int = 3
    patience: int = 15

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Scheduler
    scheduler: str = "warmup_cosine"
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.0

    # Misc
    grad_clip: float = 1.0
    log_interval: int = 1
