import os
import json
import torch
import torch.nn as nn
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from load.data_augmentation import CSIAugmentation
import wandb


def compute_padding_mask(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Detect zero-padded subcarrier columns in CSI data.

    Args:
        x: Input tensor of shape [B, 1, T, F] or [B, T, F]
        eps: Threshold below which a column is considered padding

    Returns:
        pad_mask: BoolTensor of shape [B, T, F]
                  True  = padding (should be IGNORED)
                  False = real signal (should be used)
    """
    if x.dim() == 4:
        x = x.squeeze(1)  # [B, T, F]

    # A subcarrier column is padding if ALL time steps are near-zero
    # Sum absolute values over the time axis
    col_energy = x.abs().sum(dim=1)  # [B, F]

    # Expand back to [B, T, F] so we can use it as a per-position mask
    pad_cols = col_energy < eps  # [B, F]  True = padding column
    pad_mask = pad_cols.unsqueeze(1).expand_as(x)  # [B, T, F]

    return pad_mask  # True where padding


def create_padding_aware_mask(
    x: torch.Tensor,
    mask_ratio: float = 0.75,
    block_size: int = 4,
) -> torch.Tensor:
    """Vectorized padding-aware mask — no Python loops over patches."""
    if x.dim() == 4:
        x_3d = x.squeeze(1)
    else:
        x_3d = x

    B, T, F = x_3d.shape
    device = x_3d.device

    # Which subcarrier columns are real signal? [B, F]
    valid_f = (x_3d.abs().sum(dim=1) > 1e-6)  # [B, F]

    # Collapse time into blocks: [B, n_blocks, F]
    n_blocks = T // block_size
    T_blocked = n_blocks * block_size
    x_blocked = x_3d[:, :T_blocked, :].reshape(B, n_blocks, block_size, F)
    valid_blocks = (x_blocked.abs().sum(dim=(2, 3)) > 1e-6)  # [B, n_blocks] — blocks with any signal

    # For each sample, randomly mask mask_ratio of VALID (block, subcarrier) pairs
    # Generate uniform noise and set invalid positions to 2.0 (they'll never be top-k)
    noise = torch.rand(B, n_blocks, F, device=device)
    
    # Invalid = padding block OR padding subcarrier
    invalid = (~valid_blocks.unsqueeze(2)) | (~valid_f.unsqueeze(1))  # [B, n_blocks, F]
    noise[invalid] = 2.0  # push padding positions out of the random selection

    # Count valid patches per sample, compute how many to mask
    n_valid = (~invalid).float().sum(dim=(1, 2))  # [B]
    n_mask = (n_valid * mask_ratio).long().clamp(min=1)  # [B]

    # Flatten noise to [B, n_blocks*F], argsort to get ranking
    noise_flat = noise.reshape(B, -1)  # [B, n_blocks*F]
    ids_shuffle = torch.argsort(noise_flat, dim=1)  # ascending: lowest noise = masked first

    # Build mask: positions with rank < n_mask are masked
    ranks = torch.argsort(ids_shuffle, dim=1)  # [B, n_blocks*F]
    mask_flat = (ranks < n_mask.unsqueeze(1)).float()  # [B, n_blocks*F]
    mask_blocked = mask_flat.reshape(B, n_blocks, F)  # [B, n_blocks, F]

    # Expand blocks back to full time resolution [B, T, F]
    signal_mask = mask_blocked.unsqueeze(2).expand(B, n_blocks, block_size, F)
    signal_mask = signal_mask.reshape(B, T_blocked, F)

    # Pad remaining time steps with zeros if T wasn't divisible by block_size
    if T_blocked < T:
        pad = torch.zeros(B, T - T_blocked, F, device=device)
        signal_mask = torch.cat([signal_mask, pad], dim=1)

    return signal_mask  # [B, T, F]

def padding_aware_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    signal_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    norm_pix_loss: bool = True,
) -> torch.Tensor:
    """
    Compute MSE reconstruction loss ONLY over:
      - positions that were masked during pretraining  (signal_mask == 1)
      - positions that are NOT padding                 (pad_mask == False)

    Args:
        pred:        [B, T, F]  model output
        target:      [B, T, F]  original (unmasked) input
        signal_mask: [B, T, F]  1 = was masked
        pad_mask:    [B, T, F]  True = is padding
        norm_pix_loss: normalize target patch variance (helps training stability)

    Returns:
        Scalar loss
    """
    # Only reconstruct masked, non-padding positions
    valid = signal_mask * (~pad_mask).float()  # [B, T, F]

    if norm_pix_loss:
        # Normalize target values by the mean/var of the valid region per sample
        # This prevents large-amplitude subcarriers from dominating the loss
        mean = (target * valid).sum(dim=(1, 2), keepdim=True) / \
               (valid.sum(dim=(1, 2), keepdim=True) + 1e-8)
        var  = ((target - mean) ** 2 * valid).sum(dim=(1, 2), keepdim=True) / \
               (valid.sum(dim=(1, 2), keepdim=True) + 1e-8)
        target = (target - mean) / (var + 1e-8).sqrt()

    loss = ((pred - target) ** 2 * valid).sum() / (valid.sum() + 1e-8)
    return loss


def build_transformer_key_padding_mask(
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a per-position padding mask [B, T, F] to a per-timestep key_padding_mask
    suitable for nn.TransformerEncoder.

    A timestep is considered padding if ALL its subcarrier values are padded.

    Args:
        pad_mask: [B, T, F]  True = padding

    Returns:
        key_padding_mask: [B, T]  True = this time position should be ignored
    """
    # A time step is "all padding" if every subcarrier is padding
    return pad_mask.all(dim=-1)  # [B, T]


class MaskedTrainer:
    """
    Trainer for masked CSI autoencoder pretraining.

    Key features vs. naive implementation:
    - Padding-aware masking: only masks real signal, not zero-padded subcarriers
    - Padding-aware loss: only backprops on masked non-padding positions
    - Passes a key_padding_mask to the transformer so attention ignores padding
    - Optional mixed-precision training
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        save_path: str,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.config = config

        self.mask_ratio   = getattr(config, 'mask_ratio',   0.75)
        self.block_size   = getattr(config, 'block_size',   4)
        self.patience     = getattr(config, 'patience',     20)
        self.norm_pix_loss = getattr(config, 'norm_pix_loss', True)
        self.use_amp      = getattr(config, 'use_amp', torch.cuda.is_available())
        self.use_wandb    = getattr(config, 'use_wandb', False)

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.history = []
        self.best_val_loss = float('inf')
        self.best_epoch    = 0
        self.no_improve    = 0
        self.augment = CSIAugmentation()


    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _step(self, batch, train: bool = True):
        """
        Single forward + (optionally) backward pass.

        Returns scalar loss value.
        """
        # Support datasets that return (x, label) or just x
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(self.device, non_blocking=True)
        if train:
            x = self.augment(x)

        # ---- Build masks ----
        pad_mask    = compute_padding_mask(x)          # [B, T, F]  True=padding
        signal_mask = create_padding_aware_mask(
            x,
            mask_ratio=self.mask_ratio,
            block_size=self.block_size,
        )                                               # [B, T, F]  1=masked

        # ---- Apply mask to input (zero out masked positions) ----
        x_3d = x.squeeze(1) if x.dim() == 4 else x    # [B, T, F]
        x_masked = x_3d * (1 - signal_mask)            # zero out masked positions

        # Rebuild [B, 1, T, F] for models that expect a channel dim
        x_masked_4d = x_masked.unsqueeze(1)

        # ---- Build key_padding_mask for transformer attention ----
        # Shape [B, T] — True means "ignore this timestep"
        key_pad = build_transformer_key_padding_mask(pad_mask)  # [B, T]

        # ---- Forward pass ----
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Pass key_padding_mask only if the model accepts it
            try:
                pred = self.model(x_masked_4d, key_padding_mask=key_pad)
            except TypeError:
                pred = self.model(x_masked_4d)

            # pred shape: [B, T, F]
            loss = padding_aware_reconstruction_loss(
                pred        = pred,
                target      = x_3d,
                signal_mask = signal_mask,
                pad_mask    = pad_mask,
                norm_pix_loss = self.norm_pix_loss,
            )

        # ---- Backward pass ----
        if train:
            self.optimizer.zero_grad()
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches  = 0


        for batch in self.train_loader:
            if isinstance(batch, (list, tuple)) and batch[0].shape[0] == 0:
                continue  # skip empty batches from collate_skip_none

            loss_val = self._step(batch, train=True)
            total_loss += loss_val
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)) and batch[0].shape[0] == 0:
                continue

            loss_val = self._step(batch, train=False)
            total_loss += loss_val
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        epochs = getattr(self.config, 'epochs', 100)

        print(f"Padding-aware masked pretraining | mask_ratio={self.mask_ratio} "
              f"block_size={self.block_size} norm_pix_loss={self.norm_pix_loss}")
        print(f"Mixed precision: {self.use_amp}\n")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()

            row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
            self.history.append(row)

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                })

            improved = val_loss < self.best_val_loss
            marker   = " ✓" if improved else ""
            print(f"Epoch {epoch:>3}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}{marker}")

            if improved:
                self.best_val_loss = val_loss
                self.best_epoch    = epoch
                self.no_improve    = 0
                # Save best checkpoint
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_path, "best_model.pt"))
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        # Reload best weights
        best_ckpt = os.path.join(self.save_path, "best_model.pt")
        if os.path.exists(best_ckpt):
            self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
            print(f"\nReloaded best model from epoch {self.best_epoch}")

        self._save_history()
        self._plot_loss()

        return self.model, {
            'best_epoch':    self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'history':       self.history,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _save_history(self):
        path = os.path.join(self.save_path, "train_history.json")
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _plot_loss(self):
        epochs     = [r['epoch']      for r in self.history]
        train_loss = [r['train_loss'] for r in self.history]
        val_loss   = [r['val_loss']   for r in self.history]

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss,   label='Val Loss')
        plt.axvline(self.best_epoch, color='red', linestyle='--',
                    label=f'Best epoch ({self.best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Masked CSI Pretraining (Padding-Aware)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "loss_curve.png"), dpi=120)
        plt.close()
        print(f"Loss curve saved to {self.save_path}/loss_curve.png")