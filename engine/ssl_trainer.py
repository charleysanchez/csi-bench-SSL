import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from engine.base_trainer import BaseTrainer
from utils.logging import log_epoch

class SSLBackboneTaskWrapper(nn.Module):
    """
    A specific implementation of an SSL model that is composed of:
    1. A Backbone Encoder (architecture)
    2. A Pretext Task (loss/logic)
    
    This wraps them into a single nn.Module that outputs (loss, metrics), 
    satisfying the interface expected by SSLTrainer.
    """
    def __init__(self, encoder, task):
        super().__init__()
        self.encoder = encoder
        self.task = task
        
    def forward(self, raw_batch):
        # 1. Transform batch (augmentations, etc.)
        batch = self.task.transform(raw_batch)
        
        # 2. Compute loss
        if self.task.uses_model_directly():
            # Task controls the forward pass (e.g. for momentum encoders, multi-view)
            loss = self.task.compute_loss_with_model(self.encoder, batch)
        else:
            # Standard: encode -> compute loss
            output = self.encoder(batch.inputs)
            loss = self.task.compute_loss(output, batch)
            
        return loss, batch.metadata

class SSLTrainer(BaseTrainer):
    """
    Generic Trainer for Self-Supervised Learning.
    
    This trainer is agnostic to the model architecture. 
    It expects `model` to return a tuple `(loss, metrics_dict)` in its forward pass.
    
    For models composed of an Encoder + PretextTask, use `SSLBackboneTaskWrapper`.
    For other architectures (e.g. MAE, SimCLR with custom logic), simply provide
    an nn.Module that satisfies the release interface.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        optimizer=None,
        scheduler=None,
        device=None,
        config=None,
        save_path=None,
    ):
        """
        Initialize the SSL Trainer.
        
        Args:
            model: The generic SSL model. Must return (loss, metrics) during forward().
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            device: torch.device.
            config: Config object or dict.
            save_path: Directory to save checkpoints.
        """
        super().__init__(model=model, data_loader=train_loader, config=config, device=device)
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path or getattr(config, 'save_dir', './results_ssl')
        
        os.makedirs(self.save_path, exist_ok=True)
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        
    def train(self):
        """
        Main training loop.
        """
        epochs = getattr(self.config, 'epochs', 100)
        log_interval = getattr(self.config, 'log_interval', 10)
        grad_clip = getattr(self.config, 'grad_clip', 1.0)
        
        best_val_loss = float('inf')
        start_epoch = self.current_epoch
        
        print(f"Starting SSL training for {epochs} epochs...")
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_epoch(epoch, log_interval, grad_clip)
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader)
            
            # Update history
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['val_loss'])
            
            # Log to console
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val F1: {val_metrics.get('val_cpc_f1', 0.0):.4f}")

            # Checkpoint
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(epoch, train_metrics, val_metrics, is_best)
            
        print(f"SSL Training complete. Best Val Loss: {best_val_loss:.4f}")

    def train_epoch(self, epoch, log_interval, grad_clip):
        """
        Train for one epoch.
        """
        self.model.train()
        
        total_loss = 0.0
        agg_metrics = {}
        n_batches = 0
        
        # TQDM
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", unit="batch", dynamic_ncols=True)
        
        for batch_idx, raw_batch in enumerate(pbar):
            # Move data
            if isinstance(raw_batch, (tuple, list)):
                raw_batch = tuple(x.to(self.device) if torch.is_tensor(x) else x for x in raw_batch)
            elif torch.is_tensor(raw_batch):
                raw_batch = raw_batch.to(self.device)
            # dicts are handled by SSLLearner/Task logic usually

            self.optimizer.zero_grad()
            
            # Forward pass through generic model
            # Expected to return (loss, metrics_dict)
            loss, metrics = self.model(raw_batch)
            
            loss.backward()
            
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()

            # Metrics
            loss_val = loss.item()
            total_loss += loss_val
            n_batches += 1
            
            # Aggregate task-specific metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v.item()
            
            # Log to pbar
            if (batch_idx + 1) % log_interval == 0:
                 pbar.set_postfix({'loss': f"{loss_val:.4f}"})

        # Average metrics
        avg_loss = total_loss / max(n_batches, 1)
        dataset_metrics = {k: v / max(n_batches, 1) for k, v in agg_metrics.items()}
        dataset_metrics['loss'] = avg_loss
        
        return dataset_metrics

    def evaluate(self, loader):
        """
        Evaluate on validation set.
        """
        if loader is None:
            return {'val_loss': float('inf')}
            
        self.model.eval()
        
        total_loss = 0.0
        agg_metrics = {}
        n_batches = 0
        
        with torch.no_grad():
            for raw_batch in loader:
                if isinstance(raw_batch, (tuple, list)):
                    raw_batch = tuple(x.to(self.device) if torch.is_tensor(x) else x for x in raw_batch)
                elif torch.is_tensor(raw_batch):
                    raw_batch = raw_batch.to(self.device)
                
                loss, metrics = self.model(raw_batch)
                
                total_loss += loss.item()
                n_batches += 1
                
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        agg_metrics[k] = agg_metrics.get(k, 0.0) + v
                    elif isinstance(v, torch.Tensor) and v.numel() == 1:
                        agg_metrics[k] = agg_metrics.get(k, 0.0) + v.item()
        
        results = {'val_loss': total_loss / max(n_batches, 1)}
        for k, v in agg_metrics.items():
            results[f"val_{k}"] = v / max(n_batches, 1)
            
        return results

    def save_checkpoint(self, epoch, train_metrics, val_metrics, is_best=False):
        """
        Save checkpoint.
        """
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": {**train_metrics, **val_metrics}
        }
        
        # Save latest
        torch.save(state, os.path.join(self.save_path, "latest.pt"))
        
        # Save best
        if is_best:
            torch.save(state, os.path.join(self.save_path, "best.pt"))
            
    def load_checkpoint(self, path):
        """
        Load checkpoint.
        """
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return
            
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state and self.optimizer:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state and self.scheduler:
            self.scheduler.load_state_dict(state["scheduler_state"])
            
        self.current_epoch = state.get("epoch", 0) + 1
        print(f"Loaded checkpoint from {path} (Epoch {self.current_epoch})")
