
import sys
import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from scripts, but since this file is copied to project root,
# we need to be careful with imports.
# Assuming verify_transfer.py is placed in project root.
try:
    from scripts.train_supervised import main
except ImportError:
    # If not found, manipulate path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    print(f"Path added: {current_dir}")
    try:
        from scripts.train_supervised import main
    except ImportError as e:
        print(f"Could not import main from scripts.train_supervised: {e}")
        # Try importing from file directly
        # But wait, if this runs in project root, scripts.train_supervised SHOULD be importable
        # if __init__.py exists in scripts.
        pass

# Mock get_loaders
def mock_get_loaders(*args, **kwargs):
    """Returns mock data loaders with random data."""
    batch_size = kwargs.get('batch_size', 4)
    feature_size = 232 # Mock feature size
    win_len = 500
    num_classes = 3
    
    # Create random data: (B, C, T, F) or (B, T, F)
    # train_supervised expects collate_fn to handle it.
    # Standard loader returns list of items.
    
    # Let's create a dummy Dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.x = torch.randn(size, 1, win_len, feature_size) # (B, C, T, F)
            self.y = torch.randint(0, num_classes, (size,))
            
        def __len__(self):
            return len(self.x)
            
        def __getitem__(self, idx):
            # Dataloader expects item tuple
            return self.x[idx], self.y[idx], {"user": 0}

    dataset = DummyDataset(size=batch_size * 2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=None)
    
    # Mock dictionary return
    return {
        "loaders": {
            "train": loader,
            "val": loader,
            "test_id": loader
        },
        "num_classes": num_classes,
        "label_mapper": {i: str(i) for i in range(num_classes)}
    }

def run_test():
    print("Starting Transfer Learning Verification...")
    
    # Mock args object
    class Args:
        pass
        
    args = Args()
    # Populate generic args
    args.data_dir = 'dummy_data'
    args.task = 'dummy_task'
    args.output_dir = 'test_transfer_verify'
    args.save_dir = 'test_transfer_verify'
    args.batch_size = 4
    args.epochs = 1
    args.learning_rate = 0.001
    args.weight_decay = 1e-4
    args.warmup_epochs = 0
    args.patience = 5
    args.seed = 42
    args.num_workers = 0
    args.no_pin_memory = True
    args.train_filter = None
    args.val_filter = None
    args.test_splits = 'all'
    args.data_key = 'CSI_amps'
    args.file_format = 'h5'
    args.feature_size = 232
    args.win_len = 500
    args.in_channels = 1
    args.debug = True
    args.emb_dim = 128
    args.d_model = 128 # For transformer/context
    args.pool = 'mean'
    args.head_dropout = 0.1
    args.dropout = 0.1
    args.depth = 1
    args.num_heads = 2
    
    # Test 1: SSL Linear
    print("\n--- Testing SSL Linear (CSIEncoder) ---")
    args.model = 'ssl_linear'
    args.ssl_backbone = 'csi_encoder'
    args.ssl_task = 'vqcpc' # Default
    args.pretrained_path = None
    args.freeze_encoder = False
    
    # Patch get_loaders in scripts.train_supervised
    # We need to target where it is IMPORTED in train_supervised
    # train_supervised imports get_loaders from load.dataloader
    # So we patch 'scripts.train_supervised.get_loaders'
    
    with patch('scripts.train_supervised.get_loaders', side_effect=mock_get_loaders):
        try:
            from scripts.train_supervised import main
            main(args)
            print(">> SSL Linear Test Passed!")
        except Exception as e:
            print(f">> SSL Linear Test Failed: {e}")
            import traceback
            traceback.print_exc()

    # Test 2: SSL Context (VQCPC)
    print("\n--- Testing SSL Context (VQCPC) ---")
    args.model = 'ssl_context'
    args.ssl_backbone = 'csi_encoder'
    args.ssl_task = 'vqcpc'
    args.freeze_encoder = True
    
    # Need to clear previous run potentially?
    # Or just re-run with new args.
    
    with patch('scripts.train_supervised.get_loaders', side_effect=mock_get_loaders):
        try:
             # Need to reload or re-import? main function is re-entrant if args passed.
            main(args)
            print(">> SSL Context Test Passed!")
        except Exception as e:
            print(f">> SSL Context Test Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_test()
