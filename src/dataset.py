import json
import os

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn.Functional as F


class CSIDataset(Dataset):
    """
    Docstring for CSIDataset
    """

    def __init__(
        self,
        root="../data/",
        task="BreathingDetection",
        split="train_id",
        transform=None,
        target_transform=None,
        file_format="h5",
        data_column="file_path",
        label_column="label",
        data_key="CSI_amps",
        label_mapper=None,
        task_dir=None,
        debug=False,
    ):
        self.root = root
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
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
        self.split_path = os.path.join(
            self.task_dir, "splits", f"{split}.json"
        )
        print(f"Using split: {split}")

        self.metadata_dir = os.path.join(self.task_dir, "metadata")
        self.metadata_path = os.path.join(
            self.metadata_dir, "sample_metadata.csv"
        )
        self.label_mapper_path = os.path.join(
            self.metadata_dir, "label_mapping.json"
        )

        # load the split.json to create a set of the IDs to use
        with open(self.split_path, "r") as f:
            self.split_ids = set(json.load(f))

        # read in the metadata.csv file
        self.metadata = pd.read_csv(self.metadata_path)

        # filter metadata for this split
        self.split_metadata = self.metadata[
            self.metadata["id"].isin(self.split_ids)
        ].reset_index(drop=True)

        # make sure required columns are in the dataset
        if data_column not in self.split_metadata.columns:
            raise ValueError(
                f"Data Column '{data_column}' not found in metadata"
            )
        if label_column not in self.split_metadata.columns:
            raise ValueError(
                f"Label Column '{label_column}' not found in metadata"
            )

        print(
            f"Loaded {len(self.split_metadata)} samples for {task} - {split}"
        )

        # load label_mapper
        with open(self.label_mapper_path, "r") as f:
            self.label_mapper = json.load(f)

        self.num_classes = self.label_mapper["num_classes"]

    def __len__(self):
        return len(self.split_metadata)

    def __getitem__(self, index):
        try:
            row = self.split_metadata.iloc[index]

            # path within metadata is relative to task directory
            subPath = row["file_path"]
            fullPath = os.path.join(self.task_dir, subPath)

            # load in .h5 files (only working with this rn)
            with h5py.File(fullPath, "r") as f:
                if len(f.keys()) == 0:
                    print(
                        f"Skipping file {fullPath} due to error: 'Empty H5 file: {fullPath}'"
                    )
                    return None

            if self.data_key in f:
                csi_data = np.array(f[self.data_key])

            # check for valid data
            if csi_data is None or csi_data.size == 0:
                print(f"Skipping file {fullPath} due to error: Empty data")
                return None
            
            # convert to torch tensor
            csi_tensor = torch.from_numpy(csi_data).float()

            # reshape to (1, time_index, feature_size)
            if len(csi_tensor.shape) == 3:  # (time_index, feature_size, 1)
                # Permute to get (1, time_index, feature_size)
                csi_tensor = csi_tensor.permute(2, 0, 1)

            # standardize along time and feature dimensions
            mean = csi_tensor.mean(dim=(1, 2), keepdim=True)
            std = csi_tensor.std(dim=(1, 2), keepdim=True)
            csi_standardized = (csi_tensor - mean) / (std + 1e-8)

            # scale to (1, 500, 232) using interpolation
            csi_standardized =  csi_standardized.unsqueeze(1)

            csi_standardized = F.interpolate(
                csi_standardized,
                size=(500, 232),
                mode="bilinear",
                align_corners=False
            )

            csi_standardized = csi_standardized.squeeze(1)

            if self.transform:
                csi_standardized = self.transform(csi_standardized)

            # get label
            label = row[self.label_column]
            if self.target_transform:
                label = self.target_transform(label)

            label_idx = self.label_mapper["label_to_idx"][label]

            return csi_standardized, label_idx
        except Exception as e:
            print(f"Error processing sample {index}: {str(e)}")
            return None
        
    def get_label_counts(self):
        return self.split_metadata[self.label_column].value_counts().to_dict()
    
    def get_label_names(self):
        return self.split_metadata[self.label_column].unique().to_list()
