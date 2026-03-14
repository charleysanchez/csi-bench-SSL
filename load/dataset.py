import json
import os

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from pathlib import Path

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

        # Global deterministic domain mappings to ensure stable IDs across independent sets
        # Mappings contain all known domains across all 8 sub-datasets.
        GLOBAL_USER_MAPPING = [
            'F01', 'F02', 'F03', 'F04', 'IR01', 'IR02', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 
            'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 
            'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11', 'U12', 
            'U13', 'U14', 'U15', 'U16', 'U17', 'U18', 'U19', 'U20', 'U21', 'U22', 'U23', 'U24', 
            'U25', 'U26', 'U27', 'U28', 'U29', 'U30', 'U31', 'U32', 'U33', 'U34', 'UM01', 'UM02', 
            'UM03', 'UM04', 'UM05', 'UM06'
        ]

        GLOBAL_ENV_MAPPING = [
            'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 
            'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24', 
            'E25', 'E26'
        ]

        GLOBAL_DEVICE_MAPPING = [
            '1B31WA000383', '1B31WA000430', '1B31WA000514', 'AmazonPlug', 'AppleHomePod', 'ESP32', 
            'EchoDot_2gen', 'EchoDot_3gen', 'EchoPlus', 'EchoShow8', 'EchoSpot', 'Echoplus', 
            'EightreePlug', 'GoogleNestHub', 'Googlenest', 'GoveePlug', 'HP', 'Hex_1cd6be198931', 
            'Hex_1cd6be198999', 'Hex_1cd6be198a27', 'Hex_1cd6be198a5f', 'Hex_1cd6be198a63', 
            'Hex_1cd6be198a91', 'Hex_1cd6be198a97', 'Hex_1cd6be1df30d', 'Hex_1cd6be1df323', 
            'Hex_1cd6be1df32b', 'Hex_1cd6be1df335', 'Hex_1cd6be1df36d', 'Hex_1cd6be1df3db', 
            'Hex_1cd6be1df3e1', 'Hex_1cd6be1df4c7', 'Hex_1cd6be1df583', 'Hex_1cd6be1df605', 
            'Lyra', 'WyzePlug', 'Unknown'
        ]

        self.user_to_idx = {name: idx for idx, name in enumerate(GLOBAL_USER_MAPPING)}
        self.user_to_idx["Unknown"] = len(self.user_to_idx)
        
        self.env_to_idx = {name: idx for idx, name in enumerate(GLOBAL_ENV_MAPPING)}
        self.env_to_idx["Unknown"] = len(self.env_to_idx)
        
        self.device_to_idx = {name: idx for idx, name in enumerate(GLOBAL_DEVICE_MAPPING)}
        if "Unknown" not in self.device_to_idx:
            self.device_to_idx["Unknown"] = len(self.device_to_idx)


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

        # filter for missing samples from split file
        self._filter_existing_files()

        print(
            f"Loaded {len(self.split_metadata)} samples for {task} - {split}"
        )

        # load label_mapper
        with open(self.label_mapper_path, "r") as f:
            self.label_mapper = json.load(f)

        self.num_classes = self.label_mapper.get("num_classes", len(self.label_mapper.get("label_to_idx", {})))

    def __len__(self):
        return len(self.split_metadata)

    def __getitem__(self, index):
        try:
            row = self.split_metadata.iloc[index]

            

            # path within metadata is relative to task directory
            subPath = row["file_path"]
            fullPath = os.path.normpath(os.path.join(self.task_dir, subPath))

            # domain initialization directly from metadata columns (more robust than parsing paths)
            domain = {"user": None, "env": None, "device": None}

            if 'user' in row and pd.notna(row['user']):
                usr_str = str(row['user']).replace('user_', '')
                domain["user"] = self.user_to_idx.get(usr_str, self.user_to_idx["Unknown"])

            if 'environment' in row and pd.notna(row['environment']):
                env_str = str(row['environment']).replace('env_', '')
                domain["env"] = self.env_to_idx.get(env_str, self.env_to_idx["Unknown"])

            if 'device' in row and pd.notna(row['device']):
                dev_str = str(row['device']).replace('device_', '')
                domain["device"] = self.device_to_idx.get(dev_str, self.device_to_idx["Unknown"])


            # load in .h5 files (only working with this rn)
            csi_data = None
            with h5py.File(fullPath, "r") as f:
                if len(f.keys()) == 0:
                    print(
                        f"Skipping file {fullPath} due to error: 'Empty H5 file: {fullPath}'"
                    )
                    return None

                if self.data_key in f.keys():
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
                csi_tensor = csi_tensor.permute(2, 1, 0)

            # standardize along time and feature dimensions
            mean = csi_tensor.mean(dim=(1, 2), keepdim=True)
            std = csi_tensor.std(dim=(1, 2), keepdim=True)
            std = torch.clamp(std, min=1e-8)
            csi_standardized = (csi_tensor - mean) / std

            # Instead of interpolation, we want to pad/clip dynamically like the official repo
            batch_size, time_index, feature_size = csi_standardized.shape
            target_time_index, target_feature_size = 500, 232

            # Create a tensor of zeros with the target shape
            padded_data = torch.zeros((batch_size, target_time_index, target_feature_size), dtype=csi_standardized.dtype)

            # Calculate dimensions for copying (clip or use the smaller of original and target)
            copy_time = min(time_index, target_time_index)
            copy_feature = min(feature_size, target_feature_size)

            # Copy data to the standardized tensor (handles both clipping and partial filling)
            padded_data[:, :copy_time, :copy_feature] = csi_standardized[:, :copy_time, :copy_feature]
            csi_standardized = padded_data.squeeze(0) # Squeeze dummy batch dim

            if self.transform:
                csi_standardized = self.transform(csi_standardized)

            # get label
            label = row[self.label_column]
            if self.target_transform:
                label = self.target_transform(label)

            label_key = str(label)
            if label_key not in self.label_mapper["label_to_idx"]:
                # Try zero-padding to 3 digits first (e.g. "10" → "010" for Localization)
                label_key_padded = label_key.zfill(3)
                if label_key_padded in self.label_mapper["label_to_idx"]:
                    label_key = label_key_padded
                else:
                    # Fall back to decimal → binary map
                    dec_to_bin = self.label_mapper.get("decimal_to_binary", {})
                    label_key = dec_to_bin.get(label_key, label_key)
            label_idx = self.label_mapper["label_to_idx"][label_key]

            u_id = domain["user"] if domain["user"] is not None else -1
            e_id = domain["env"] if domain["env"] is not None else -1
            d_id = domain["device"] if domain["device"] is not None else -1

            return csi_standardized, label_idx, u_id, e_id, d_id
        except Exception as e:
            print(f"Error processing sample {index}: {str(e)}")
            return None
        
    def get_label_counts(self):
        return self.split_metadata[self.label_column].value_counts().to_dict()
    
    def get_label_names(self):
        return self.split_metadata[self.label_column].unique().to_list()
    
    def _filter_existing_files(self):
        keep_rows = []
        missing = 0

        for _, row in self.split_metadata.iterrows():
            fullPath = os.path.normpath(os.path.join(self.task_dir, row["file_path"]))
            if os.path.exists(fullPath):
                keep_rows.append(row)
            else:
                missing += 1

        self.split_metadata = pd.DataFrame(keep_rows).reset_index(drop=True)

        if missing > 0:
            print(f"[CSIDataset] dropped {missing} missing samples")
