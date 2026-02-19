import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset

class PretrainingCSI(Dataset):
    """
    Dataset class for the RawContinuousRecording CSI
    data to perform pretraining of a model.
    """
    def __init__(
        self,
        root="../data/RawContinuousRecording",
        file_format=".mat",
        window_size=256,
        stride=128,
    ):
        self.root = Path(root)
        self.file_format = file_format
        self.window_size = window_size
        self.stride = stride

        # collect file paths
        self.files = list(self.root.rglob(f"*{self.file_format}"))

        # build index of (file_path, start_time)
        self.index = self._build_index()

    def build_index(self):
        index = []

        for path in self.files:
            # load to get n samples
            mat = sio.loadmat(path)
            csi = mat["csi_trace"]["csi"][0, 0]
            n_samples = csi.shape[3]

            for start in range(0, n_samples - self.window_size, self.stride):
                index.append((path, start))

        return index
    
    def __len__(self):
        return len(self.index)
    
    def __get_item__(self, idx):
        path, start = self.index[idx]

        # load file
        mat = sio.loadmat(path)
        csi = mat["csi_trace"]["csi"][0, 0]

    def load_mat_files(self):
        records = []
        
        for mat_file in self.root.rglob(f"*{self.file_format}"):
            user_env = mat_file.parts[-3]
            user, env = user_env.split("_")
            device = mat_file.parts[-2]

            mat = sio.loadmat(mat_file)
            csi = mat["csi_trace"]["csi"][0, 0]
            shape = csi.shape

            records.append({
                "user": user,
                "env": env,
                "device": device,
                "path": str(mat_file),
                "N_tx": shape[0],
                "N_rx": shape[1],
                "N_subcarriers": shape[2],
                "N_samples": shape[3]
            })

