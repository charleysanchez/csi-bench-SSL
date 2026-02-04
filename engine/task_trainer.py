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
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from engine.base_trainer import BaseTrainer
from utils.labels import normalize_labels
from utils.logging import log_epoch
from utils.training import predict_from_outputs


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


class TaskTrainer(BaseTrainer):
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

        # create directory if it doesn't exist
        if not distributed or (distributed and local_rank == 0):
            os.makedirs(save_path, exist_ok=True)

        # move model to device
        self.model.to(self.device)

        # load checkpoint if specified
        if self.checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

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

    def setup_scheduler(self):
        """Set up learning rate scheduler."""
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
            # number of epochs and patience from config
            # handle both object attributes and dictionary keys
            if isinstance(self.config, dict):
                epochs = self.config.get("epochs", 30)
                patience = self.config.get("epochs", 15)
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

            # step scheduler
            self.scheduler.step()

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

        for inputs, labels in loader:
            # skip empty batches (from custom_collate_fn if all samples were None)
            if inputs.size(0) == 0:
                continue

            start_time = time.perf_counter()

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
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            # backward pass
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            # accumulate loss and accuracy
            epoch_loss += loss.item() * batch_size

            preds = predict_from_outputs(outputs, criterion=self.criterion)
            correct = (preds.view(-1) == labels.view(-1).long()).sum().item()
            epoch_accuracy += correct

            # measure elapsed time
            total_time += time.perf_counter() - start_time

            # update progress bar
            loader.set_postfix(loss=loss.item())

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
            for inputs, labels in data_loader:
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
                    data, labels = batch

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
                if outputs.ndim == 2 and outputs.size(1) > 1:
                    # multi-class
                    preds = torch.argmax(outputs, dim=1)
                else:
                    # binary
                    if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                        preds = (outputs > 0).long().squeeze()
                    else:
                        preds = (outputs > 0.5).long().squeeze()

                # collect all predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # get class names if available
        class_names = None
        if self.label_mapper is not None:
            class_names = list(self.label_mapper["label_to_idx"].keys())

        # plot confusion matrix
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
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, labels = batch
                else:
                    inputs = batch["input"]
                    labels = batch["label"]

                # skip empty batches
                if inputs.size(0) == 0:
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
                        preds = (outputs > 0).long().squeeze()
                    else:
                        preds = (outputs > 0.5).long().squeeze()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

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
