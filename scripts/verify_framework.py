import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pretext_tasks.vqcpc import VQCPC
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
        
        item = dataset[0]
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
    config = type('Config', (), {'feature_size': 30, 'model_dim': 64, 'lr': 1e-3, 'weight_decay': 0})()
    device = "cpu"
    
    try:
        model = VQCPC(config, device)
        print("Model initialized.")
        
        B, T, F = 2, 100, 30
        data = torch.randn(B, 1, T, F)
        
        batch = (data, torch.zeros(B), [{} for _ in range(B)])
        
        outputs = model.training_step(batch, 0)
        
        if 'loss' not in outputs:
            print("FAILED: 'loss' key missing from outputs.")
            return
            
        print(f"PASSED: Forward step successful. Loss: {outputs['loss'].item()}")
        
    except Exception as e:
        print(f"FAILED: VQ-CPC verification error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataset()
    verify_vqcpc()
