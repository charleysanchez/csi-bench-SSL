import torch


class CSIAugmentation:
    """
    Light data augmentation for WiFi CSI data during training.
    Kept deliberately mild — aggressive augmentation was shown to hurt OOD performance.
    """

    def __init__(
        self,
        amp_scale_range=(0.8, 1.2),
        noise_std=0.05,
        time_shift_max=15,
        subcarrier_drop_max=5,
        subcarrier_drop_prob=0.3,
        time_shift_prob=0.3,
        noise_prob=0.3,
        freq_jitter_prob=0.0,
        freq_jitter_std=0.0,
    ):
        self.amp_scale_range = amp_scale_range
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.subcarrier_drop_max = subcarrier_drop_max
        self.subcarrier_drop_prob = subcarrier_drop_prob
        self.time_shift_prob = time_shift_prob
        self.noise_prob = noise_prob
        self.freq_jitter_prob = freq_jitter_prob
        self.freq_jitter_std = freq_jitter_std

    def __call__(self, x):
        # x shape: (B, T, F) or (B, 1, T, F)
        B = x.shape[0]

        # 1. Per-sample amplitude scaling — simulates different device sensitivity
        lo, hi = self.amp_scale_range
        scale = lo + (hi - lo) * torch.rand(B, *([1] * (x.dim() - 1)), device=x.device)
        x = x * scale

        # 2. Additive Gaussian noise — simulates environmental interference
        if torch.rand(1).item() < self.noise_prob:
            noise_level = self.noise_std * torch.rand(1).item()
            noise = noise_level * torch.randn_like(x)
            x = x + noise

        # 3. Random time shift with zero-padding
        if torch.rand(1).item() < self.time_shift_prob:
            shift = torch.randint(1, self.time_shift_max + 1, (1,)).item()
            x = torch.roll(x, shift, dims=-2)
            x[..., :shift, :] = 0.0

        # 4. Random subcarrier dropout
        if torch.rand(1).item() < self.subcarrier_drop_prob:
            n_drop = torch.randint(1, self.subcarrier_drop_max + 1, (1,)).item()
            drop_idx = torch.randperm(x.shape[-1], device=x.device)[:n_drop]
            x[..., drop_idx] = 0.0

        # 5. Frequency-domain jitter (optional, off by default)
        if self.freq_jitter_prob > 0 and torch.rand(1).item() < self.freq_jitter_prob:
            jitter = self.freq_jitter_std * torch.randn(
                1, *([1] * (x.dim() - 2)), x.shape[-1], device=x.device
            )
            x = x + x * jitter

        return x