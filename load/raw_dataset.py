import pickle
import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset
import torch


class PretrainingCSI(Dataset):
    """
    Dataset for raw continuous CSI recordings for self-supervised pretraining.

    Preprocessing follows CSI-Bench (arXiv:2505.21866):
      - Amplitude only — phase discarded (hardware-induced phase noise is unreliable)
      - Per-sample global z-score normalization per antenna pair
      - Zero-pad / clip to target_feature_size in the frequency dimension
      - Sliding window segmentation along time

    Returns (csi, -1, user_idx, env_idx, device_idx) so it is compatible with
    CPCTrainer's domain-aware negative mining (same interface as PretrainDataset).

    Path structure assumed: root/{user}_{env}/{device}/*.mat

    Args:
        root:                path to RawContinuousRecording directory
        file_format:         file extension to glob for (default ".mat")
        window_size:         number of time steps per window
        stride:              step between consecutive windows
        target_feature_size: output feature dimension (pad/clip); set to match
                             the benchmark's feature_size for encoder transfer
        exclude_users:       collection of user-ID strings to skip entirely
                             (use to prevent leakage into downstream test users)
        exclude_envs:        collection of environment-ID strings to skip entirely
        cache_index:         if True, save/load the window index to disk to avoid
                             re-scanning .mat files on repeated runs
    """

    def __init__(
        self,
        root="../data/RawContinuousRecording",
        file_format=".mat",
        window_size=256,
        stride=128,
        target_feature_size=232,
        exclude_users=None,
        exclude_envs=None,
        cache_index=True,
    ):
        self.root = Path(root)
        self.file_format = file_format
        self.window_size = window_size
        self.stride = stride
        self.target_feature_size = target_feature_size
        self.exclude_users = set(exclude_users or [])
        self.exclude_envs = set(exclude_envs or [])
        self.cache_index = cache_index

        self.files = sorted(self.root.rglob(f"*{self.file_format}"))
        if not self.files:
            raise FileNotFoundError(
                f"No {file_format} files found under {self.root}"
            )

        self._parse_metadata()
        self._build_domain_maps()
        self.index = self._load_or_build_index()

        print(
            f"PretrainingCSI: {len(self.index)} windows from "
            f"{len(set(p for p, *_ in self.index))} files"
        )
        print(
            f"  Users: {len(self.user_map)}, "
            f"Envs: {len(self.env_map)}, "
            f"Devices: {len(self.device_map)}"
        )
        if self.exclude_users:
            print(f"  Excluded users: {sorted(self.exclude_users)}")
        if self.exclude_envs:
            print(f"  Excluded envs:  {sorted(self.exclude_envs)}")

    # ------------------------------------------------------------------
    # Metadata & domain maps
    # ------------------------------------------------------------------

    def _parse_metadata(self):
        """Parse user, env, device from path: root/{user}_{env}/{device}/file.mat"""
        self.file_meta = {}
        for path in self.files:
            try:
                user_env = path.parts[-3]
                user, env = user_env.split("_", 1)
                device = path.parts[-2]
            except (IndexError, ValueError):
                user, env, device = "unknown", "unknown", "unknown"
            self.file_meta[path] = (user, env, device)

    def _build_domain_maps(self):
        """Integer ID mappings for user / env / device (local to this dataset)."""
        users   = sorted({m[0] for m in self.file_meta.values()})
        envs    = sorted({m[1] for m in self.file_meta.values()})
        devices = sorted({m[2] for m in self.file_meta.values()})
        self.user_map   = {u: i for i, u in enumerate(users)}
        self.env_map    = {e: i for i, e in enumerate(envs)}
        self.device_map = {d: i for i, d in enumerate(devices)}

    # ------------------------------------------------------------------
    # Index building (with optional disk cache)
    # ------------------------------------------------------------------

    def _cache_path(self):
        key = f"ws{self.window_size}_st{self.stride}_tf{self.target_feature_size}"
        excl = "_".join(sorted(self.exclude_users | self.exclude_envs))
        return self.root / f".index_cache_{key}_{excl or 'none'}.pkl"

    def _load_or_build_index(self):
        cache = self._cache_path()
        if self.cache_index and cache.exists():
            try:
                with open(cache, "rb") as f:
                    index = pickle.load(f)
                print(f"  Loaded window index from cache ({cache.name})")
                return index
            except Exception:
                pass  # rebuild if cache is corrupt

        index = self._build_index()

        if self.cache_index:
            try:
                with open(cache, "wb") as f:
                    pickle.dump(index, f)
                print(f"  Saved window index to cache ({cache.name})")
            except Exception as e:
                print(f"  Warning: could not save index cache: {e}")

        return index

    def _build_index(self):
        """Scan each .mat file to count samples; build (path, start, u, e, d) tuples."""
        index = []
        for path in self.files:
            user, env, device = self.file_meta[path]

            if user in self.exclude_users or env in self.exclude_envs:
                continue

            try:
                mat = sio.loadmat(str(path))
                csi = mat["csi_trace"]["csi"][0, 0]
                n_samples = csi.shape[3]
            except Exception as exc:
                print(f"  Warning: skipping {path.name}: {exc}")
                continue

            user_idx   = self.user_map[user]
            env_idx    = self.env_map[env]
            device_idx = self.device_map[device]

            for start in range(0, n_samples - self.window_size, self.stride):
                index.append((path, start, user_idx, env_idx, device_idx))

        if not index:
            raise RuntimeError(
                "Index is empty after exclusions. "
                "Check exclude_users/exclude_envs or data root."
            )
        return index

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, start, user_idx, env_idx, device_idx = self.index[idx]

        mat = sio.loadmat(str(path))
        csi = mat["csi_trace"]["csi"][0, 0]

        # Extract window: (n_tx, n_rx, n_sc, window_size)
        window = csi[:, :, :, start : start + self.window_size]

        # Flatten antenna pairs: (n_tx*n_rx, n_sc, window_size)
        n_tx, n_rx, n_sc, n_t = window.shape
        window = window.reshape(n_tx * n_rx, n_sc, n_t)

        # ---- Amplitude only (CSI-Bench §3.2) ----
        # Discard phase — phase is unreliable across hardware due to CFO and STO.
        window = np.abs(window).astype(np.float32)

        # ---- Per-sample z-score normalization (CSI-Bench §3.2) ----
        # Normalize each antenna pair independently across its subcarrier×time matrix.
        # Shape: (n_ch, n_sc, n_t) → mean/std over axes (1, 2) = per-channel global.
        mean = window.mean(axis=(1, 2), keepdims=True)  # (n_ch, 1, 1)
        std  = window.std(axis=(1, 2),  keepdims=True)
        window = (window - mean) / (std + 1e-6)

        # ---- Reshape to (T, F) ----
        # Transpose to (n_t, n_ch, n_sc) then flatten channels+subcarriers.
        window = window.transpose(2, 0, 1).reshape(n_t, -1)  # (T, n_ch*n_sc)

        # ---- Pad / clip to target_feature_size (CSI-Bench §3.2) ----
        # Do NOT interpolate — zero-pad preserves the original signal.
        T, F = window.shape
        if F < self.target_feature_size:
            pad = np.zeros((T, self.target_feature_size - F), dtype=np.float32)
            window = np.concatenate([window, pad], axis=1)
        else:
            window = window[:, : self.target_feature_size]

        csi_tensor = torch.from_numpy(window)  # (T, target_feature_size)

        # Return -1 as a dummy label; CPCTrainer ignores batch[1].
        return csi_tensor, -1, user_idx, env_idx, device_idx

    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------

    def get_metadata(self) -> pd.DataFrame:
        """
        Return a DataFrame of unique (user, env, device) present in the index.
        Useful for checking overlap with downstream task splits before pretraining.

        Example::
            ds = PretrainingCSI(root="data/RawContinuousRecording")
            print(ds.get_metadata())
        """
        seen = set()
        records = []
        for path, _, u, e, d in self.index:
            user, env, device = self.file_meta[path]
            key = (user, env, device)
            if key not in seen:
                seen.add(key)
                records.append(
                    dict(user=user, env=env, device=device,
                         user_idx=u, env_idx=e, device_idx=d)
                )
        return pd.DataFrame(records).sort_values(["user", "env", "device"])

    def list_users(self) -> list:
        return sorted(self.user_map.keys())

    def list_envs(self) -> list:
        return sorted(self.env_map.keys())
