import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
from tqdm import tqdm

from engine.base_trainer import BaseTrainer
from load.data_augmentation import CSIAugmentation

torch.backends.cudnn.benchmark = True

def warmup_cosine_lr(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.0):
    """Warmup learning rate scheduler with cosine annealing."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(
                1, total_epochs - warmup_epochs
            )
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


class CPCTrainer(BaseTrainer):
    """Trainer for Contrastive Predictive Coding (CPC) self-supervised CSI pretraining."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device="cuda:0",
        save_path="./results",
        checkpoint_path=None,
        config=None,
        distributed=False,
        local_rank=0,
        k_steps=4,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.distributed = distributed
        self.local_rank = local_rank
        self.k_steps = getattr(config, "cpc_k_steps", k_steps)

        # Domain-aware CPC: use domain labels to create harder negatives
        self.domain_aware = getattr(config, "domain_aware", False) if config else False
        # Max fraction of negatives to sample from same domain — linearly ramped from 0.0
        self.domain_neg_ratio_max = getattr(config, "domain_neg_ratio", 0.8) if config else 0.8
        self.domain_neg_ratio = 0.0  # starts at 0, ramped up each epoch
        if self.domain_aware:
            print(f"Domain-aware CPC enabled (domain_neg_ratio ramps 0.0 -> {self.domain_neg_ratio_max})")

        if not distributed or local_rank == 0:
            os.makedirs(save_path, exist_ok=True)

        self.model.to(self.device)

        if optimizer is None:
            lr = getattr(config, "lr", 1e-3)
            wd = getattr(config, "weight_decay", 0.0)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
            )

        if scheduler is None:
            warmup_epochs = getattr(config, "warmup_epochs", 5)
            total_epochs = getattr(config, "epochs", 100)
            self.scheduler = warmup_cosine_lr(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
            )

        # CSI augmentation during pretraining — stronger than finetune defaults
        self.augment = CSIAugmentation(
            amp_scale_range=(0.3, 3.0),
            noise_std=0.15,
            time_shift_max=50,
            subcarrier_drop_max=30,
            subcarrier_drop_prob=0.5,
            time_shift_prob=0.5,
            noise_prob=0.5,
            freq_jitter_prob=0.3,
            freq_jitter_std=0.1,
        )

        self.train_losses = []
        self.best_epoch = 0

        # AMP: bf16 on Blackwell/Ampere, fp16 on older CUDA, no-op on CPU/MPS
        self.use_amp = getattr(config, "amp", False) and torch.cuda.is_available()
        if self.use_amp:
            has_bf16 = torch.cuda.is_bf16_supported()
            self.amp_dtype = torch.bfloat16 if has_bf16 else torch.float16
            # GradScaler only needed for fp16 (bf16 doesn't underflow)
            self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp_dtype == torch.float16))
            print(f"AMP enabled: {self.amp_dtype}")
        else:
            self.scaler = None

    # -------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------

    def train(self):

        epochs = getattr(self.config, "epochs", 30)
        best_train_loss = float("inf")
        best_model = None
        records = []

        max_train_hours = getattr(self.config, "max_train_hours", None)
        train_start = time.time()

        for epoch in range(epochs):
            # Time-budget check: stop before the wall-clock limit is hit
            if max_train_hours is not None:
                elapsed_h = (time.time() - train_start) / 3600
                if elapsed_h >= max_train_hours:
                    print(f"\nTime budget reached ({elapsed_h:.2f}h >= {max_train_hours}h) — stopping after epoch {epoch}.")
                    break
            self.current_epoch = epoch

            # Linear ramp: domain_neg_ratio goes 0.0 -> domain_neg_ratio_max over training
            if self.domain_aware:
                self.domain_neg_ratio = self.domain_neg_ratio_max * (epoch / max(1, epochs - 1))

            if not self.distributed or self.local_rank == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                if self.domain_aware:
                    print(f"domain_neg_ratio: {self.domain_neg_ratio:.3f}")

            train_loss, train_time = self.train_epoch()
            self.train_losses.append(train_loss)

            # NaN detection — stop and keep best checkpoint
            if np.isnan(train_loss) or train_loss == 0.0:
                print(f"NaN/zero loss at epoch {epoch + 1}, stopping. Best epoch: {self.best_epoch}")
                break

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_model = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1

            if not self.distributed or self.local_rank == 0:
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Time/sample: {train_time:.6f}s")

                record = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Time per sample": train_time,
                }
                records.append(record)
                if _WANDB and wandb.run is not None:
                    wandb.log({
                        "train/loss": train_loss,
                        "train/time_per_sample": train_time,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/domain_neg_ratio": self.domain_neg_ratio,
                        "epoch": epoch + 1,
                    })

        # Restore best checkpoint
        if best_model is not None:
            self.model.load_state_dict(best_model)

        if not self.distributed or self.local_rank == 0:
            results_df = pd.DataFrame(records)
            results_df.to_csv(
                os.path.join(self.save_path, "training_results.csv"),
                index=False,
            )
            self.plot_training_results()

        return self.model, {
            "train_loss_history": self.train_losses,
            "best_epoch": self.best_epoch,
            "best_train_loss": best_train_loss,
        }

    # -------------------------------------------------
    # TRAIN ONE EPOCH
    # -------------------------------------------------

    def train_epoch(self):

        self.model.train()
        total_loss = 0.0
        total_samples = 0
        total_time = 0.0

        loader = tqdm(self.train_loader, leave=False)
        for batch in loader:
            inputs = batch[0]
            users = batch[2] if len(batch) > 2 else None
            envs = batch[3] if len(batch) > 3 else None
            devices = batch[4] if len(batch) > 4 else None

            if inputs.size(0) == 0:
                continue

            start_time = time.perf_counter()

            inputs = inputs.to(self.device)
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            inputs = self.augment(inputs)
            B = inputs.size(0)

            self.optimizer.zero_grad()

            amp_ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else torch.amp.autocast("cpu", enabled=False)
            with amp_ctx:
                outputs = self.model(inputs)

                domain_labels = None
                if self.domain_aware and (envs is not None or devices is not None):
                    domain_labels = {"users": users, "envs": envs, "devices": devices}

                loss = self.compute_cpc_loss(outputs, self.k_steps, users, domain_labels=domain_labels)

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                continue

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * B
            total_samples += B
            total_time += time.perf_counter() - start_time

            loader.set_postfix(loss=loss.item())

        if total_samples == 0:
            return 0.0, 0.0

        if self.distributed and torch.distributed.is_initialized():
            metrics = torch.tensor(
                [total_loss, total_time, total_samples],
                device=self.device,
            )
            torch.distributed.all_reduce(
                metrics, op=torch.distributed.ReduceOp.SUM
            )
            total_loss, total_time, total_samples = metrics.tolist()

        epoch_loss = total_loss / total_samples
        time_per_sample = total_time / total_samples

        return epoch_loss, time_per_sample

    # -------------------------------------------------
    # EVALUATE (not used for SSL, but required by BaseTrainer)
    # -------------------------------------------------

    def evaluate(self, loader):
        return 0.0

    # -------------------------------------------------
    # PLOTTING
    # -------------------------------------------------

    def plot_training_results(self):

        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CPC Pretraining Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.save_path, "training_curve.png"))
        plt.close()

    # -------------------------------------------------
    # CPC LOSS
    # -------------------------------------------------

    def _labels_to_ids(self, labels, B):
        """Convert a list/tensor of labels to integer IDs on self.device."""
        if labels is None:
            return torch.arange(B, device=self.device)
        if torch.is_tensor(labels):
            labels_list = labels.cpu().tolist()
        else:
            labels_list = [int(u) if torch.is_tensor(u) else u for u in labels]
        unique = set(labels_list)
        if len(unique) <= 1:
            return torch.zeros(B, dtype=torch.long, device=self.device)
        label_to_id = {u: i for i, u in enumerate(sorted(unique))}
        return torch.tensor([label_to_id[u] for u in labels_list], device=self.device)

    def compute_cpc_loss(self, outputs, k_steps=None, users=None, domain_labels=None):
        """
        Compute InfoNCE loss for CPC.

        When domain_aware=True, uses domain-aware hard negative mining:
        preferentially samples negatives from the SAME domain (user/env/device)
        as the anchor. This forces the encoder to learn features that distinguish
        content (activity) from domain (who/where/what device), because same-domain
        negatives only differ in content — not in domain-specific signal characteristics.

        Standard CPC negatives are random, so the model can "cheat" by using
        domain-specific patterns (e.g., device-specific noise floor) to reject
        negatives without learning semantic content. Hard negatives prevent this.
        """
        if k_steps is None:
            k_steps = getattr(self.config, "cpc_k_steps", self.k_steps)

        z, c, preds = outputs
        B, T, feat_dim = z.shape

        loss = 0.0

        # Pre-compute user IDs for same-user masking (always active)
        user_ids = self._labels_to_ids(users, B)

        # Pre-compute domain IDs for hard negative sampling (only when domain_aware)
        domain_ids = None
        if self.domain_aware and domain_labels is not None:
            # Build a composite domain ID from all available domain axes
            # e.g., (user=U01, env=E03, device=HP) -> unique integer
            env_ids = self._labels_to_ids(domain_labels.get("envs"), B)
            dev_ids = self._labels_to_ids(domain_labels.get("devices"), B)
            # Composite: treat (env, device) as the domain to sample hard negatives from.
            # We exclude user because user is already used for false-negative masking.
            n_envs = env_ids.max().item() + 1
            domain_ids = env_ids * (dev_ids.max().item() + 1) + dev_ids

        num_negatives = getattr(self.config, "cpc_num_negatives", 256)

        for k in range(1, k_steps + 1):
            if T - k <= 0:
                continue

            z_t_k = z[:, k:, :]          # [B, T-k, feat_dim]
            c_t_pred = preds[k - 1][:, :-k, :]  # [B, T-k, feat_dim]

            c_flat = c_t_pred.flatten(0, 1)  # [N, d]
            z_flat = z_t_k.flatten(0, 1)     # [N, d]

            c_norm = F.normalize(c_flat, dim=-1)
            z_norm = F.normalize(z_flat, dim=-1)

            N = c_flat.shape[0]
            K_neg = min(num_negatives, N)

            # --- Negative sampling ---
            if domain_ids is not None and K_neg > 0 and K_neg > 0 and self.domain_neg_ratio > 0:
                # Domain-aware hard negative sampling:
                # For each anchor, sample domain_neg_ratio of negatives from the
                # same (env, device) domain and the rest randomly.
                domain_ids_flat = domain_ids.repeat_interleave(T - k)  # [N]
                K_hard = int(K_neg * self.domain_neg_ratio)
                K_rand = K_neg - K_hard

                # Sample hard negatives: pick from same domain as a random anchor
                # (batch-level approximation for efficiency)
                anchor_domain = domain_ids_flat[torch.randint(N, (1,))].item()
                same_domain_mask = (domain_ids_flat == anchor_domain)
                same_domain_idx = same_domain_mask.nonzero(as_tuple=False).squeeze(-1)

                if same_domain_idx.numel() >= K_hard:
                    perm = torch.randperm(same_domain_idx.numel(), device=self.device)[:K_hard]
                    hard_idx = same_domain_idx[perm]
                else:
                    # Fewer same-domain samples than requested; take all, fill rest randomly
                    hard_idx = same_domain_idx
                    K_hard = hard_idx.numel()
                    K_rand = K_neg - K_hard

                # Random negatives for diversity
                rand_idx = torch.randperm(N, device=self.device)[:K_rand]
                neg_idx = torch.cat([hard_idx, rand_idx])
            else:
                neg_idx = torch.randperm(N, device=self.device)[:K_neg]

            z_neg = z_norm[neg_idx]  # [K_neg, d]

            tau = 0.1
            sim_pos = (c_norm * z_norm).sum(dim=-1, keepdim=True) / tau
            sim_neg = torch.matmul(c_norm, z_neg.T) / tau

            # Mask out negatives from the same user (false negative prevention)
            user_ids_flat = user_ids.repeat_interleave(T - k)
            neg_users = user_ids_flat[neg_idx]
            mask = (user_ids_flat.unsqueeze(1) == neg_users.unsqueeze(0))
            sim_neg.masked_fill_(mask, -float('inf'))

            logits = torch.cat([sim_pos, sim_neg], dim=1)
            step_labels = torch.zeros(N, dtype=torch.long, device=self.device)

            step_loss = F.cross_entropy(logits, step_labels)
            loss += step_loss

        return loss / max(1, k_steps)
