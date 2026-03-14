import os
print("Starting imports...")
import time
start = time.time()
from load.dataset import CSIDataset
print(f"Imports done in {time.time() - start:.2f}s")

try:
    print("Pre-CSIDataset init check cwd:", os.getcwd())
    print("Init start")
    ds = CSIDataset(
        root="./data/",
        task="MotionSourceRecognition",
        split="train_id",
    )
    print(f"Successfully loaded dataset with {len(ds)} samples.")

    indices_to_test = [0, 100, 500, len(ds) - 1]
    
    for i in indices_to_test:
        print(f"Accessing {i}...")
        sample = ds[i]
        print(f"Sample {i} domain IDs:", sample[-1])

    print("Success! No crashes.")
except Exception as e:
    print(f"FAILED with error: {e}")
    import traceback
    traceback.print_exc()
