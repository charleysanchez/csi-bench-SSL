import os, json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CSIDataset(Dataset):
    """
    Docstring for CSIDataset
    """
    def __init__(self, 
                 root="../data/",
                 task="BreathingDetection",
                 split="train_id",
                 transform=None,
                 file_format="h5",
                 data_column="file_path",
                 label_column="label",
                 data_key="CSI_amps",
                 label_mapper=None,
                 task_dir=None,
                 debug=False
            ):
        
        self.root = root
        self.task = task
        self.split = split
        self.transform = transform
        self.file_format = file_format
        self.data_column = data_column
        self.label_column = label_column
        self.data_key = data_key
        self.debug = debug

        # if task directory provided then try to use it
        if task_dir is not None and os.path.isdir(task_dir):
            self.task_dir = task_dir
            print(f"Using provided task directory: {self.task_dir}")
        # otherwise try and figure out based on task
        else:
            task_dir = os.path.join(root, task)
            if os.path.isdir(task_dir):
                self.task_dir = task_dir
                print(f"Successfully found task directory: {task_dir}")
            else:
                print(f"Could not find task directory: {task_dir}")

        # define split and metadata paths
        self.split_path = os.path.join(self.task_dir, "splits", f"{split}.json")
        print(f"Using split: {split}")

        self.metadata_dir = os.path.join(self.task_dir, "metadata")
        self.metadata_path = os.path.join(self.metadata_dir, "sample_metadata.csv")
        self.label_mapper = os.path.join(self.metadata_dir, "label_mapping.json")

        with open(self.split_path, "r") as f:
            self.split_ids = set(json.load(f))

        self.metadata = pd.read_csv(self.metadata_path)
        


        