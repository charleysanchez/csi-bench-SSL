import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

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
        representation="real_imag",
        target_sc=58
    ):
        self.root = Path(root)
        self.file_format = file_format
        self.window_size = window_size
        self.stride = stride
        self.representation = representation
        self.target_sc = target_sc

        # collect file paths
        self.files = list(self.root.rglob(f"*{self.file_format}"))

        # build index of (file_path, start_time)
        self.index = self._build_index()

    def _build_index(self):
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
    
    def __getitem__(self, idx):
        path, start = self.index[idx]

        # load file
        mat = sio.loadmat(path)
        csi = mat["csi_trace"]["csi"][0, 0]

        # extract window
        window = csi[:, :, :, start:start+self.window_size]

        # flatten antennas
        n_tx, n_rx, n_sc, n_t = window.shape
        window = window.reshape(n_tx * n_rx, n_sc, n_t)

        # update to match representation
        window = self._format_representation(window)

        # resize carriers
        if n_sc != self.target_sc:
            window = self._resize_carriers(window)

        # normalize
        window = self._normalize_csi(window)

    def _format_representation(self, csi):
        if self.representation == "real_imag":
            real = np.real(csi)
            imag = np.imag(csi)
            out = np.stack([real, imag], axis=0)
        elif self.representation == "mag_phase":
            mag = np.abs(csi)
            phase = np.angle(csi)
            out = np.stack([mag, phase], axis=0)
        elif self.representation == "log_mag":
            out = np.log(np.abs(csi) + 1e-6)[None, ...]
        else:
            raise ValueError("Representation input as: {self.representation}. " \
                             "Must be one of: [real_imag, mag_phase, log_phase]")
        
        return torch.tensor(out, dtype=torch.float32)
        
    def _resize_carriers(self, csi, target_sc=58):
        csi = torch.tensor(csi).float()

        csi = F.interpolate(
            csi.unsqueeze(0),
            size=(target_sc, csi.shape[-1]),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return csi
    
    def _normalize_csi(self, csi):
        mean_csi = csi.mean()
        std_csi = csi.std()

        return (csi - mean_csi) / (std_csi + 1e-6)


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

