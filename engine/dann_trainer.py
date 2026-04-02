import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from engine.base_trainer import BaseTrainer
from utils.labels import normalize_labels
from utils.logging import log_epoch
from utils.training import predict_from_outputs, warmup_cosine_lr

from load.data_augmentation import CSIAugmentation
from utils.sam import SAM


class DannTrainer(BaseTrainer):
    """Trainer for supervised learning tasks with CSI data."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        test_loader=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device="cuda:0",
        save_path="./results",
        checkpoint_path=None,
        num_classes=None,
        label_mapper=None,
        config=None,
        distributed=False,
        local_rank=0,
    ):
        """
        Initialize the task trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            save_path: Path to save results
            checkpoint_path: Path to load checkpoint
            num_classes: Number of classes for the model
            label_mapper: LabelMapper for mapping between class indices and names
            config: Configuration object with training parameters
            distributed: Whether this is a distributed training run
            local_rank: Local rank of this process in distributed training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.criterion_domain = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.label_mapper = label_mapper
        self.config = config
        self.distributed = distributed
        self.local_rank = local_rank
        self.augment = CSIAugmentation()
        self.mixup_alpha = getattr(config, 'mixup_alpha', 0.0) if config else 0.0
        if self.mixup_alpha > 0:
            print(f"Mixup enabled with alpha={self.mixup_alpha}")
        self.use_sam = getattr(config, 'use_sam', False) if config else False
        self.manifold_mixup = getattr(config, 'manifold_mixup', False) if config else False
        # Configurable per-domain loss weights (default 0.1 each)
        self.lambda_user = getattr(config, 'lambda_user', 0.1) if config else 0.1
        self.lambda_env = getattr(config, 'lambda_env', 0.1) if config else 0.1
        self.lambda_device = getattr(config, 'lambda_device', 0.1) if config else 0.1
        # CORAL loss weight (0 = disabled)
        self.coral_weight = getattr(config, 'coral_weight', 0.0) if config else 0.0
        if self.coral_weight > 0:
            print(f"CORAL loss enabled with weight={self.coral_weight}")

        # create directory if it doesn't exist
        if not distributed or (distributed and local_rank == 0):
            os.makedirs(save_path, exist_ok=True)

        # move model to device
        self.model.to(self.device)

        # load checkpoint if specified
        if self.checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        # create optimizer if necessary
        if optimizer is None:
            assert config is not None, "Config required to create optimizer"
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=getattr(config, "lr", 1e-3),
                weight_decay=float(getattr(config, "weight_decay", 0.0)),
            )
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.setup_scheduler()
        else:
            self.scheduler = scheduler


        # log
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

        # training tracking
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

        # DANN specific tracking
        self.grl_alpha = 0.0
        self.domain_metrics = {
            "user_acc": [],
            "env_acc": [],
            "device_acc": []
        }

    def setup_scheduler(self):
        warmup_epochs = getattr(self.config, "warmup_epochs", 5)
        total_epochs = getattr(self.config, "epochs", 100)
        
        self.scheduler = warmup_cosine_lr(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
        )

    def train(self):
        """Train the model."""
        if not self.distributed or (self.distributed and self.local_rank == 0):
            print("Starting supervised training phase...")

        # records for tracking progress
        records = []

        # set default configuration values if config is None
        if self.config is None:
            epochs = 30
            patience = 15
        else:
            epochs = getattr(self.config, "epochs", 30)
            patience = getattr(self.config, "patience", 15)
        # best model state
        best_model = None
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # only print from rank 0 in distributed mode
            if not self.distributed or (
                self.distributed and self.local_rank == 0
            ):
                print(f"Epoch {epoch + 1}/{epochs}")

            # set epoch for distributed sampler
            if (
                self.distributed
                and hasattr(self.train_loader, "sampler")
                and hasattr(self.train_loader.sampler, "set_epoch")
            ):
                self.train_loader.sampler.set_epoch(epoch)

            # train one epoch
            train_loss, train_acc, train_time = self.train_epoch()

            # evaluate
            val_loss, val_acc = self.evaluate(self.val_loader)

            # update records
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # step scheduler is now done per batch in train_epoch()

            # only log from rank 0 in distributed mode
            if not self.distributed or (
                self.distributed and self.local_rank == 0
            ):
                # record for this epoch
                record = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Train Accuracy": train_acc,
                    "Val Accuracy": val_acc,
                    "Time per sample": train_time,
                }
                records.append(record)

                print(
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
                print(f"Time per sample: {train_time:.6f} seconds")

                lr = self.optimizer.param_groups[0]["lr"]
                log_epoch(
                    epoch + 1, train_loss, train_acc, val_loss, val_acc, lr
                )

            # early stopping check
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())

                # save the best model - only from rank 0 in distributed mode
                if not self.distributed or (
                    self.distributed and self.local_rank == 0
                ):
                    best_model_path = os.path.join(
                        self.save_path, "best_model.pt"
                    )
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                            "epoch": epoch,
                        },
                        best_model_path,
                    )
                    print(f"Best model saved to {best_model_path}")

                self.best_epoch = epoch + 1
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if not self.distributed or (
                        self.distributed and self.local_rank == 0
                    ):
                        print(
                            f"Early stopping triggered after {patience} epochs without improvement."
                        )
                    self.model.load_state_dict(best_model)
                    break

            # only perform final steps from rank 0 in distributed mode
            if not self.distributed or (
                self.distributed and self.local_rank == 0
            ):
                # create results dataframe
                results_df = pd.DataFrame(records)

                # save results
                results_df.to_csv(
                    os.path.join(self.save_path, "training_results.csv"),
                    index=False,
                )

            # get the best validation accuracy and its corresponding epoch
            if len(self.val_accuracies) > 0:
                best_idx = np.argmax(self.val_accuracies)
                best_epoch = best_idx + 1
                best_val_accuracy = self.val_accuracies[best_idx]
            else:
                best_epoch = epochs
                best_val_accuracy = 0.0

            # create a dictionary with unified information as the return value
            training_results = {
                "train_loss_history": self.train_losses,
                "val_loss_history": self.val_losses,
                "train_accuracy_history": self.train_accuracies,
                "val_accuracy_history": self.val_accuracies,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
            }

            # only include dataframe if non-distributed or rank 0
            if not self.distributed or (
                self.distributed and self.local_rank == 0
            ):
                training_results["training_dataframe"] = results_df

        if best_model is not None:
            self.model.load_state_dict(best_model)
        self.plot_training_results()
        return self.model, training_results

    def train_epoch(self):
        """Train the model for a single epoch.

        Returns:
            A tuple of (loss, accuracy, time_per_sample).
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_samples = 0
        total_time = 0.0

        loader = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for inputs, labels, user_labels, env_labels, device_labels in loader:
            # skip empty batches (from custom_collate_fn if all samples were None)
            if inputs.size(0) == 0:
                continue

            start_time = time.perf_counter()

            batch_size = inputs.size(0)
            total_samples += batch_size

            # transfer to device
            inputs = inputs.to(self.device)
            
            inputs = self.augment(inputs)

            labels = normalize_labels(
                labels=labels,
                batch_size=batch_size,
                device=self.device,
                criterion=self.criterion,
            )

            # forward pass
            self.optimizer.zero_grad()

            # GRL schedule: alpha ramps from 0 (no reversal) to 1 (full reversal) over training.
            # Ganin & Lempitsky (2015): alpha = 2/(1+exp(-10*p))-1, p in [0,1].
            p = float(self.current_epoch * len(self.train_loader) + total_samples // batch_size) / (
                getattr(self.config, "epochs", 100) * len(self.train_loader)
            )
            self.grl_alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            if hasattr(self.model, 'grl'):
                self.model.grl.alpha = self.grl_alpha

            # When Mixup is enabled we skip domain losses this batch (main task only).
            if self.mixup_alpha > 0 and self.model.training:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                lam = max(lam, 1 - lam)  # ensure lam >= 0.5 so original dominates

                cross_domain_mixup = getattr(self.config, 'cross_domain_mixup', False) if self.config else False
                if cross_domain_mixup:
                    # Cross-domain Mixup: pair each sample with one from a *different* domain.
                    # Xu et al., "Adversarial Domain Adaptation with Domain Mixup" (AAAI 2020).
                    idx = self._cross_domain_indices(
                        batch_size, user_labels.to(self.device).long(),
                        env_labels.to(self.device).long(), device_labels.to(self.device).long()
                    )
                else:
                    idx = torch.randperm(batch_size, device=self.device)

                if self.manifold_mixup:
                    outputs = self.model(inputs, manifold_mixup={'lam': lam, 'idx': idx})
                else:
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[idx]
                    outputs = self.model(mixed_inputs)
                loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(outputs, labels[idx])
            else:
                user_labels = user_labels.to(self.device).long()
                env_labels = env_labels.to(self.device).long()
                device_labels = device_labels.to(self.device).long()

                main_logits, user_logits, env_logits, device_logits = self.model(inputs, return_domains=True)
                outputs = main_logits
                loss_main = self.criterion(main_logits, labels)
                loss_user = self.criterion_domain(user_logits, user_labels)
                loss_env = self.criterion_domain(env_logits, env_labels)
                loss_device = self.criterion_domain(device_logits, device_labels)

                loss = loss_main + (self.lambda_user * loss_user) + (self.lambda_env * loss_env) + (self.lambda_device * loss_device)

                # CORAL loss: align feature distributions across domains
                if self.coral_weight > 0:
                    coral_loss = self._compute_coral_loss(main_logits, user_labels, env_labels, device_labels)
                    loss = loss + self.coral_weight * coral_loss
                
                # Track domain accuracies
                with torch.no_grad():
                    u_preds = torch.argmax(user_logits, dim=1)
                    e_preds = torch.argmax(env_logits, dim=1)
                    d_preds = torch.argmax(device_logits, dim=1)
                    
                    # Only count samples where labels are not -1
                    u_mask = user_labels != -1
                    if u_mask.any():
                        u_acc = (u_preds[u_mask] == user_labels[u_mask]).float().mean().item()
                        self.domain_metrics["user_acc"].append(u_acc)
                    
                    e_mask = env_labels != -1
                    if e_mask.any():
                        e_acc = (e_preds[e_mask] == env_labels[e_mask]).float().mean().item()
                        self.domain_metrics["env_acc"].append(e_acc)
                        
                    d_mask = device_labels != -1
                    if d_mask.any():
                        d_acc = (d_preds[d_mask] == device_labels[d_mask]).float().mean().item()
                        self.domain_metrics["device_acc"].append(d_acc)

            # backward pass
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            if self.use_sam and isinstance(self.optimizer, SAM):
                self.optimizer.first_step(zero_grad=True)
                # Second forward-backward at perturbed parameters
                if self.mixup_alpha > 0 and self.model.training:
                    if self.manifold_mixup:
                        outputs2 = self.model(inputs, manifold_mixup={'lam': lam, 'idx': idx})
                    else:
                        outputs2 = self.model(mixed_inputs)
                    loss2 = lam * self.criterion(outputs2, labels) + (1 - lam) * self.criterion(outputs2, labels[idx])
                else:
                    main2, user2, env2, dev2 = self.model(inputs, return_domains=True)
                    loss2 = self.criterion(main2, labels)
                    loss2 = loss2 + self.lambda_user * self.criterion_domain(user2, user_labels)
                    loss2 = loss2 + self.lambda_env * self.criterion_domain(env2, env_labels)
                    loss2 = loss2 + self.lambda_device * self.criterion_domain(dev2, device_labels)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.step()

            # step scheduler per batch (step-level cosine warmup)
            if self.scheduler is not None:
                self.scheduler.step()

            # accumulate loss and accuracy
            epoch_loss += loss.item() * batch_size

            preds = predict_from_outputs(outputs, criterion=self.criterion)
            correct = (preds.view(-1) == labels.view(-1).long()).sum().item()
            epoch_accuracy += correct

            # measure elapsed time
            total_time += time.perf_counter() - start_time

            # update progress bar
            avg_u = np.mean(self.domain_metrics["user_acc"][-10:]) if self.domain_metrics["user_acc"] else 0
            avg_e = np.mean(self.domain_metrics["env_acc"][-10:]) if self.domain_metrics["env_acc"] else 0
            avg_d = np.mean(self.domain_metrics["device_acc"][-10:]) if self.domain_metrics["device_acc"] else 0
            loader.set_postfix(loss=f"{loss.item():.3f}", main_acc=f"{epoch_accuracy/max(1, total_samples):.3f}", u_acc=f"{avg_u:.2f}", e_acc=f"{avg_e:.2f}", d_acc=f"{avg_d:.2f}", alpha=f"{self.grl_alpha:.2f}")

        # calculate averages
        epoch_loss /= total_samples
        epoch_accuracy /= total_samples
        time_per_sample = total_time / total_samples

        # Synchronize metrics across processes in distributed training
        if self.distributed and torch.distributed.is_initialized():
            # Create tensors for each metric
            metrics = torch.tensor(
                [
                    epoch_loss * total_samples,
                    epoch_accuracy * total_samples,
                    total_time,
                    total_samples,
                ],
                device=self.device,
            )

            # All-reduce to compute mean across processes
            torch.distributed.all_reduce(
                metrics, op=torch.distributed.ReduceOp.SUM
            )

            # Get world size for averaging
            world_size = torch.distributed.get_world_size()
            metrics /= world_size

            # Extract metrics
            epoch_loss = metrics[0] / metrics[3]
            epoch_accuracy = metrics[1] / metrics[3]
            time_per_sample = metrics[2] / metrics[3]

        return epoch_loss, epoch_accuracy, time_per_sample

    def _cross_domain_indices(self, batch_size, user_labels, env_labels, device_labels):
        """
        For each sample, find a partner from a *different* domain to mix with.

        Picks the domain axis with the most unique values, then for each sample
        tries to pair it with a sample from a different domain. Falls back to
        random permutation if no cross-domain partner is available.
        """
        # Pick domain axis with most diversity
        best_labels = None
        for labels in [user_labels, env_labels, device_labels]:
            valid = labels[labels != -1]
            if valid.numel() == 0:
                continue
            if best_labels is None or valid.unique().numel() > best_labels[best_labels != -1].unique().numel():
                best_labels = labels

        if best_labels is None or best_labels[best_labels != -1].unique().numel() < 2:
            return torch.randperm(batch_size, device=user_labels.device)

        idx = torch.arange(batch_size, device=user_labels.device)
        for i in range(batch_size):
            # Find indices where domain differs
            candidates = (best_labels != best_labels[i]) & (best_labels != -1)
            if candidates.any():
                pool = candidates.nonzero(as_tuple=False).squeeze(-1)
                idx[i] = pool[torch.randint(len(pool), (1,))]
            else:
                # Fallback: random other sample
                others = torch.arange(batch_size, device=user_labels.device)
                others = others[others != i]
                idx[i] = others[torch.randint(len(others), (1,))]
        return idx

    def _compute_coral_loss(self, features, user_labels, env_labels, device_labels):
        """
        CORrelation ALignment (CORAL) loss.

        Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (ECCV 2016).

        Minimises the distance between second-order statistics (covariance matrices)
        of feature distributions from different domains. This encourages the encoder
        to produce domain-invariant representations without needing a discriminator.

        We compute pairwise CORAL across all unique domain values for the most
        populated domain axis in the batch (user > env > device).
        """
        # Pick the domain axis with the most unique labels (best signal)
        domain_labels = None
        for labels in [user_labels, env_labels, device_labels]:
            valid = labels[labels != -1]
            if valid.numel() == 0:
                continue
            if domain_labels is None or valid.unique().numel() > domain_labels.unique().numel():
                domain_labels = valid
                domain_full = labels

        if domain_labels is None or domain_labels.unique().numel() < 2:
            return torch.tensor(0.0, device=features.device)

        unique_domains = domain_labels.unique()
        coral = torch.tensor(0.0, device=features.device)
        n_pairs = 0

        # Pre-compute per-domain covariance matrices
        cov_cache = {}
        for d in unique_domains:
            mask = (domain_full == d)
            feats_d = features[mask]
            if feats_d.size(0) < 2:
                continue
            # Center features
            feats_d = feats_d - feats_d.mean(dim=0, keepdim=True)
            cov_d = (feats_d.T @ feats_d) / (feats_d.size(0) - 1)
            cov_cache[d.item()] = cov_d

        domain_keys = list(cov_cache.keys())
        for i in range(len(domain_keys)):
            for j in range(i + 1, len(domain_keys)):
                diff = cov_cache[domain_keys[i]] - cov_cache[domain_keys[j]]
                coral = coral + (diff * diff).sum()
                n_pairs += 1

        if n_pairs > 0:
            d = features.size(1)
            coral = coral / (4.0 * d * d * n_pairs)

        return coral

    def evaluate(self, data_loader):
        """
        Evaluate the model.

        Args:
            data_loader: the data loader to use for the evaluation.

        Returns:
            A tuple of (loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                # Robust unpacking for domain-aware batches (data, labels, user_labels, ...)
                inputs, labels = batch[0], batch[1]

                # skip empy batches
                if inputs.size(0) == 0:
                    continue

                batch_size = inputs.size(0)
                total_samples += batch_size

                # transfer to device
                inputs = inputs.to(self.device)
                labels = normalize_labels(
                    labels=labels,
                    batch_size=batch_size,
                    device=self.device,
                    criterion=self.criterion,
                )

                # forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # accumulate loss
                total_loss += loss.item() * batch_size

                # calculate accuracy
                preds = predict_from_outputs(
                    outputs=outputs, criterion=self.criterion
                )
                correct = (preds.view(-1) == labels.view(-1).long()).sum().item()

                total_correct += correct
                
                # free memory immediately 
                del inputs, labels, outputs, preds, loss

        # calculate averages
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Synchronize metrics across processes in distributed training
        if self.distributed and torch.distributed.is_initialized():
            # Create tensors for each metric
            metrics = torch.tensor(
                [avg_loss, accuracy, total_samples],
                dtype=torch.float,
                device=self.device,
            )

            # All-reduce to compute mean across processes
            torch.distributed.all_reduce(
                metrics, op=torch.distributed.ReduceOp.SUM
            )

            # Get world size for averaging
            world_size = torch.distributed.get_world_size()

            # For loss and accuracy we want the average, but we need to account for the
            # different number of samples each process may have processed
            world_samples = metrics[2].item()
            if world_samples > 0:
                avg_loss = metrics[0].item() * world_size / world_samples
                accuracy = metrics[1].item() * world_size / world_samples

        return avg_loss, accuracy

    def plot_training_results(self):
        """Plot the training results."""
        # create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # plot the training loss
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True)

        # plot the validation loss
        axs[0, 1].plot(self.val_losses)
        axs[0, 1].set_title("Validation Loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].grid(True)

        # plot the training accuracy
        axs[1, 0].plot(self.train_accuracies)
        axs[1, 0].set_title("Training Accuracy")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Accuracy")
        axs[1, 0].grid(True)

        # plot the validation accuracy
        axs[1, 1].plot(self.val_accuracies)
        axs[1, 1].set_title("Validation Accuracy")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Accuracy")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "training_results.png"))
        plt.close()

        # also plot the confusion matrix
        self.plot_confusion_matrix()

    def plot_confusion_matrix(self, data_loader=None, epoch=None, mode="val"):
        """
        Plot the confusion matrix and save the figure.

        Args:
            data_loader: Dataloader to use for the evaluation
            epoch: Current epoch
            mode: 'val or 'test' mode
        """
        # set evaluation mode
        self.model.eval()

        # use validation loader if not specified
        if data_loader is None:
            if mode == "val" and self.val_loader is not None:
                data_loader = self.val_loader
            elif mode == "test" and self.test_loader is not None:
                data_loader = self.test_loader
            else:
                raise ValueError(f"No data loader available for mode {mode}")
            
        if self.label_mapper is None:
            self.label_mapper = data_loader.dataset.label_mapper


        # collect all predictions and labels
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                # get data and labels
                if isinstance(batch, dict):
                    data = batch["data"]
                    labels = batch["labels"]
                else:
                    # Robust unpacking for domain-aware batches (data, labels, ...)
                    data, labels = batch[0], batch[1]

                # handle different label formats
                if isinstance(labels, tuple):
                    labels = labels[0]

                # move data to device
                data = data.to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)
                elif isinstance(labels, (list, np.ndarray)):
                    labels = torch.tensor(labels).to(self.device)

                # forward pass
                outputs = self.model(data)
                preds = predict_from_outputs(outputs=outputs, criterion=self.criterion)

                # collect all predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # free memory 
                del data, labels, outputs, preds

        # convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # get class names if available
        class_names = None
        labels_list = None
        if self.label_mapper is not None:
            class_names = list(self.label_mapper["label_to_idx"].keys())
            labels_list = list(range(len(class_names)))

        # plot confusion matrix
        if labels_list is not None:
            cm = confusion_matrix(all_labels, all_preds, labels=labels_list)
        else:
            cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({mode})")

        # save figure
        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        plt.savefig(
            os.path.join(
                self.save_path, f"confusion_matrix{mode}{epoch_str}.png"
            )
        )
        plt.close()

        # generate and save classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # replace indices with class names if available
        if class_names is not None:
            # create a mapping dictionary from indices to class names
            index_to_name = {}
            for i, name in enumerate(class_names):
                index_to_name[str(i)] = name

            # replace indices with class names
            new_index = []
            for idx in report_df.index:
                if idx in index_to_name:
                    new_index.append(index_to_name[idx])
                else:
                    new_index.append(idx)

            report_df.index = new_index

        # save report
        report_df.to_csv(
            os.path.join(
                self.save_path, f"classification_report_{mode}{epoch_str}.csv"
            )
        )

        return report_df

    def calculate_metrics(self, data_loader, epoch=None):
        """
        Calculate overall performance metrics, including weighted F1 score.

        Args:
            data_loader: Dataloader for evaluation
            epoch: Current epoch (optional)

        Returns:
            Tuple of (weighted_f1_score, per_class_f1_scores)
        """
        # set model to evaluation mode
        self.model.eval()

        # initialize lists to store predictions and ground_truth
        all_preds = []
        all_labels = []

        # no gradient during evaluation
        with torch.no_grad():
            for batch in data_loader:
                # get inputs and move to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    labels = batch[1]
                else:
                    inputs = batch["input"]
                    labels = batch["label"]

                # skip empty batches
                if inputs.shape[0] == 0:
                    continue

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                outputs = self.model(inputs)
                if outputs.ndim == 2 and outputs.size(1) > 1:
                    # multi-class
                    preds = torch.argmax(outputs, dim=1)
                else:
                    # binary
                    if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                        preds = (outputs > 0 ).long().squeeze()
                    else:
                        preds = (outputs > 0.5).long().squeeze()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # free memory 
                del inputs, labels, outputs, preds

        # convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # validate data before calculating metrics
        if len(all_preds) == 0 or len(all_labels) == 0:
            print(
                "Warning: Empty prediction or target arrays, skipping F1 calculation."
            )
            return 0.0, pd.DataFrame()

        if len(all_preds) != len(all_labels):
            print(
                f"Warning: Prediction and Label array lengths don't match: {len(all_preds)} vs {len(all_labels)}"
            )
            return 0.0, pd.DataFrame()

        # print some debug information
        print(
            f"Predictions shape: {all_preds.shape}, unique values: {np.unique(all_preds)}"
        )
        print(
            f"Labels shape: {all_labels.shape}, unique values: {np.unique(all_labels)}"
        )

        # calculate weighted f1 score
        from sklearn.metrics import classification_report, f1_score

        weighted_f1 = f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        # calculate per-class f1 scores
        per_class_f1 = f1_score(
            all_labels, all_preds, average=None, zero_division=0
        )

        # get detailed classification report
        report = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )

        # save the report to a CSV file if epoch is None (final evaluation)
        if epoch is None and hasattr(self, "save_path"):
            # convert report to DataFrame
            report_df = pd.DataFrame(report).transpose()

            # determine split name from data_loader (assuming it's in the dataloader's dataset attributes)
            split_name = getattr(data_loader.dataset, "split", "unknown")

            # save to CSV
            report_path = os.path.join(
                self.save_path, f"classification_report_{split_name}.csv"
            )
            report_df.to_csv(report_path)
            print(f"Classification report saved to {report_path}")

            return weighted_f1, report_df

        return weighted_f1, pd.DataFrame(report).transpose()
