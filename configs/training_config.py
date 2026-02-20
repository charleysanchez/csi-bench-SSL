from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class TrainingConfig:
    # Training loop
    epochs: int = 30
    patience: int = 15
    grad_clip: float = 1.0
    log_interval: int = 1
    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    # Scheduler
    scheduler: str = "warmup_cosine"
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.0
    # Model
    emb_dim: int = 128
    dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        flat = {}
        flat.update(raw.get("training", {}))
        flat.update(raw.get("model_params", {}))

        opt = raw.get("optimizer", {})
        if "name" in opt:
            flat["optimizer"] = opt.pop("name")
        if "lr" in opt:
            flat["lr"] = float(opt.pop("lr"))
        if "weight_decay" in opt:
            flat["weight_decay"] = float(opt.pop("weight_decay"))

        sched = raw.get("scheduler", {})
        if "name" in sched:
            flat["scheduler"] = sched.pop("name")
        flat.update(sched)

        # Cast numeric fields
        float_fields = {"lr", "weight_decay", "min_lr_ratio", "grad_clip", "dropout"}
        int_fields = {"epochs", "patience", "warmup_epochs", "log_interval", "emb_dim"}

        for k in float_fields:
            if k in flat:
                flat[k] = float(flat[k])
        for k in int_fields:
            if k in flat:
                flat[k] = int(flat[k])

        # Only pass keys that exist in the dataclass
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in flat.items() if k in valid_keys}

        return cls(**filtered)