import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from engine.base_trainer import BaseTrainer


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


class MaskedTrainer(BaseTrainer):
    """Trainer for masked self-supervised CSI pretraining."""

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

        self.train_losses = []
        self.val_losses = []
        self.best_epoch = 0

    # -------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------

    def train(self):

        epochs = getattr(self.config, "epochs", 30)
        patience = getattr(self.config, "patience", 15)

        best_val_loss = float("inf")
        best_model = None
        epochs_no_improve = 0
        records = []

        for epoch in range(epochs):
            self.current_epoch = epoch

            if not self.distributed or self.local_rank == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss, train_time = self.train_epoch()
            val_loss = (
                self.evaluate(self.val_loader)
                if self.val_loader
                else train_loss
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step()

            if not self.distributed or self.local_rank == 0:
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss:   {val_loss:.6f}")
                print(f"Time/sample: {train_time:.6f}s")

                records.append(
                    {
                        "Epoch": epoch + 1,
                        "Train Loss": train_loss,
                        "Val Loss": val_loss,
                        "Time per sample": train_time,
                    }
                )

            # ----- Early Stopping -----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break

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
            "val_loss_history": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": best_val_loss,
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

        for inputs, _ in loader:
            if inputs.size(0) == 0:
                continue

            start_time = time.perf_counter()

            inputs = inputs.to(self.device)
            B = inputs.size(0)
            total_samples += B

            masked_inputs, mask = self.apply_mask(inputs)

            self.optimizer.zero_grad()
            outputs = self.model(masked_inputs)

            loss = self.compute_masked_loss(outputs, inputs, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * B
            total_time += time.perf_counter() - start_time

            loader.set_postfix(loss=loss.item())

        if total_samples == 0:
            return 0.0, 0.0

        # Distributed reduction
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
    # EVALUATE
    # -------------------------------------------------

    def evaluate(self, loader):

        if loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, _ in loader:
                if inputs.size(0) == 0:
                    continue

                inputs = inputs.to(self.device)
                B = inputs.size(0)
                total_samples += B

                masked_inputs, mask = self.apply_mask(inputs)
                outputs = self.model(masked_inputs)

                loss = self.compute_masked_loss(outputs, inputs, mask)
                total_loss += loss.item() * B

        if total_samples == 0:
            return 0.0

        if self.distributed and torch.distributed.is_initialized():
            metrics = torch.tensor(
                [total_loss, total_samples],
                device=self.device,
            )
            torch.distributed.all_reduce(
                metrics, op=torch.distributed.ReduceOp.SUM
            )
            total_loss, total_samples = metrics.tolist()

        return total_loss / total_samples

    # -------------------------------------------------
    # PLOTTING
    # -------------------------------------------------

    def plot_training_results(self):

        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train")
        plt.plot(self.val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Masked Pretraining Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.save_path, "training_curve.png"))
        plt.close()

    # -------------------------------------------------
    # MASKING
    # -------------------------------------------------

    def apply_mask(self, inputs):
        B, T, F = inputs.shape
        mask = torch.ones_like(inputs)

        t0 = torch.randint(0, T - 20, (B,), device=inputs.device)

        for i in range(B):
            mask[i, t0[i] : t0[i] + 20, :] = 0

        masked_inputs = inputs * mask
        return masked_inputs, 1 - mask

    def compute_masked_loss(self, outputs, targets, mask):
        diff = (outputs - targets) ** 2
        loss = (diff * mask).sum() / (mask.sum() + 1e-8)
        return loss
