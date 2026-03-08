import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from engine.base_trainer import BaseTrainer

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
        for batch in loader:
            inputs = batch[0]
            if len(batch) > 2:
                users = batch[2]
            else:
                users = None
                
            if inputs.size(0) == 0:
                continue

            start_time = time.perf_counter()

            inputs = inputs.to(self.device)
            B = inputs.size(0)
            total_samples += B

            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            
            loss = self.compute_cpc_loss(outputs, self.k_steps, users)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item() * B
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
    # EVALUATE
    # -------------------------------------------------

    def evaluate(self, loader):

        if loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                if len(batch) > 2:
                    users = batch[2]
                else:
                    users = None

                if inputs.size(0) == 0:
                    continue

                inputs = inputs.to(self.device)
                B = inputs.size(0)
                total_samples += B

                outputs = self.model(inputs)
                loss = self.compute_cpc_loss(outputs, self.k_steps, users)

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
        plt.title("CPC Pretraining Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.save_path, "training_curve.png"))
        plt.close()

    # -------------------------------------------------
    # CPC LOSS
    # -------------------------------------------------

    def compute_cpc_loss(self, outputs, k_steps=None, users=None):
        """
        Compute InfoNCE loss for CPC.
        We rely on randomized negative sampling across the batch and time dimensions.
        This provides temporal negative diversity without the massive memory accumulation 
        and quadratic overhead of O(B^2 * T^2) all-to-all comparisons.
        """
        if k_steps is None:
            k_steps = getattr(self.config, "cpc_k_steps", self.k_steps)
            
        z, c, preds = outputs
        B, T, feat_dim = z.shape
        
        loss = 0.0
        
        # Pre-compute user mappings outside the loop
        if users is not None:
            # Check edge case: The entire batch is from a single user
            if len(set(users)) <= 1:
                # Fallback: treat each sequence in the batch as an independent identity
                user_ids = torch.arange(B, device=self.device)
            else:
                user_to_id = {u: i for i, u in enumerate(set(users))}
                user_ids = torch.tensor([user_to_id[u] for u in users], device=self.device)
        else:
            user_ids = torch.arange(B, device=self.device)
            
        # We will use random negative sampling to get cross-time negatives
        # num_negatives determines how many random negatives to draw from the flattened N pool.
        num_negatives = getattr(self.config, "cpc_num_negatives", 256)
        
        for k in range(1, k_steps + 1):
            if T - k <= 0:
                continue
                
            # z_t_k is the true target vector at time t+k
            z_t_k = z[:, k:, :] # [B, T-k, feat_dim]
            
            # c_t_pred is the predicted target from time t using W_k
            c_t_pred = preds[k - 1][:, :-k, :] # [B, T-k, feat_dim]
            
            c_flat = c_t_pred.flatten(0, 1) # [N, d]
            z_flat = z_t_k.flatten(0, 1) # [N, d]
            
            # Normalize embeddings for cosine similarity
            c_norm = F.normalize(c_flat, dim=-1)
            z_norm = F.normalize(z_flat, dim=-1)
            
            N = c_flat.shape[0]
            K_neg = min(num_negatives, N)
            
            # Randomly sample K_neg indices for the shared negative pool
            neg_idx = torch.randperm(N, device=self.device)[:K_neg]
            z_neg = z_norm[neg_idx] # [K_neg, d]
            
            # Calculate positive pairs (diagonal equivalent)
            tau = 0.1
            sim_pos = (c_norm * z_norm).sum(dim=-1, keepdim=True) / tau # [N, 1]
            
            # Calculate negative pairs
            sim_neg = torch.matmul(c_norm, z_neg.T) / tau # [N, K_neg]
            
            # Mask out negatives that come from the same user context
            user_ids_flat = user_ids.repeat_interleave(T - k) # [N]
            neg_users = user_ids_flat[neg_idx] # [K_neg]
            
            mask = (user_ids_flat.unsqueeze(1) == neg_users.unsqueeze(0)) # [N, K_neg]
            sim_neg.masked_fill_(mask, -float('inf'))
            
            # Concatenate positives and negatives
            logits = torch.cat([sim_pos, sim_neg], dim=1) # [N, 1 + K_neg]
            
            # Target is always index 0
            step_labels = torch.zeros(N, dtype=torch.long, device=self.device)
            
            step_loss = F.cross_entropy(logits, step_labels)
            loss += step_loss
            
        return loss / max(1, k_steps)
