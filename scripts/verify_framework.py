import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pretext_tasks.vqcpc import VQCPC
from model.encoders import CSIEncoder
from load.dataset import CSIDataset

def verify_dataset():
    print("Verifying Dataset Metadata...")
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/csi-bench-dataset/csi-bench-dataset")
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found. Skipping dataset verification.")
        return

    try:
        dataset = CSIDataset(
            root=data_path,
            task="MotionSourceRecognition",
            split="train_id",
            debug=True
        )
        
        if len(dataset) == 0:
             print("Dataset empty.")
             return

        item = dataset[0]
        if item is None:
             print("Item 0 is None.")
             return

        if len(item) != 3:
            print(f"FAILED: Dataset returned {len(item)} items, expected 3.")
            return
            
        data, label, metadata = item
        if not isinstance(metadata, dict):
            print(f"FAILED: Metadata is {type(metadata)}, expected dict.")
            return
            
        print("PASSED: Dataset returns (data, label, metadata).")
        print(f"Metadata keys: {list(metadata.keys())}")
        
    except Exception as e:
        print(f"FAILED: Dataset verification error: {e}")

def verify_vqcpc():
    print("\nVerifying VQ-CPC Model...")
    
    # config = type('Config', (), {'feature_size': 30, 'model_dim': 64, 'lr': 1e-3, 'weight_decay': 0})()
    device = "cpu"
    
    feature_size = 30
    model_dim = 64
    
    try:
        # Instantiate Encoder and Task separately
        encoder = CSIEncoder(feature_size=feature_size, model_dim=model_dim)
        task = VQCPC(encoder_dim=model_dim, context_dim=64)
        
        print("Model initialized.")
        
        B, T, F = 2, 100, 30
        data = torch.randn(B, 1, T, F) # 4D input
        # data = torch.randn(B, T, F) # 3D input also supported by my fix?
        
        # Mock metadata
        metadata = [{'user': f'user_{i%2}'} for i in range(B)]
        
        # Mock default_collate behavior: simple list of dicts -> dict of lists?
        # Actually default_collate transforms list of dicts into dict of lists/tensors
        batch_metadata = {'user': [f'user_{i%2}' for i in range(B)]}
        
        # Raw batch from dataloader
        raw_batch = (data, torch.zeros(B), batch_metadata)
        
        # 1. Transform
        batch = task(raw_batch)
        print(f"PretextBatch inputs shape: {batch.inputs.shape}")
        
        # 2. Compute Loss
        loss = task.compute_loss_with_model(encoder, batch)
        
        print(f"PASSED: Forward step successful. Loss: {loss.item()}")
        
    except Exception as e:
        print(f"FAILED: VQ-CPC verification error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataset()
    verify_vqcpc()
