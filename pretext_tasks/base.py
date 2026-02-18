import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class PretextTask(nn.Module, ABC):
    """
    Abstract base class for self-supervised learning pretext tasks.
    
    All SSL methods (e.g., VQ-CPC, Masked Autoencoding, Contrastive Learning)
    must implement this interface to be compatible with the generic training script.
    """
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

    @abstractmethod
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Compute the training loss for a batch.
        
        Args:
            batch: The output from the DataLoader. 
                   Usually a tuple (data, label, metadata) or dictionary.
                   The task should handle whatever format the dataset returns.
            batch_idx: The index of the current batch.
            optimizer_idx: The index of the optimizer to use (for GANs/Adversarial).
                           Defaults to 0.
                           
        Returns:
            torch.Tensor: The loss value to backpropagate.
            dict: (Optional) A dictionary of metrics to log (e.g. {'acc': 0.9, 'rec_loss': 0.1})
                  If returning a dict, the loss must be under the key 'loss'.
                  If returning a Tensor, it is assumed to be the loss.
        """
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        Compute validation metrics for a batch.
        
        Args:
            batch: The output from the DataLoader.
            batch_idx: The index of the current batch.
            
        Returns:
            dict: A dictionary of metrics to log.
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            list: A list of optimizers.
            list: (Optional) A list of schedulers.
        """
        pass
    
    @abstractmethod
    def get_encoder(self):
        """
        Return the trained backbone encoder for downstream tasks.
        
        Returns:
            nn.Module: The feature extractor.
        """
        pass
        
    def on_epoch_end(self, current_epoch, logs=None):
        """
        Optional hook called at the end of each epoch.
        """
        pass
