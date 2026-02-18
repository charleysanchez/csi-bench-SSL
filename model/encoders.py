"""
Backbone encoder for CSI sequences.

Input:  (B, feature_size)  — a single flattened timestep
Output: (B, model_dim)     — latent representation

The VQ-CPC task handles the temporal loop (encoding B*T frames at once),
so the encoder only needs to process one frame at a time. This keeps the
encoder architecture simple and task-agnostic — swap it for a CNN, Transformer,
etc. as needed.
"""

import torch.nn as nn


class CSIEncoder(nn.Module):
    """
    MLP encoder for flattened CSI amplitude snapshots.

    Architecture: Linear → BN → ReLU → Linear → BN → ReLU → Linear

    Args:
        feature_size: Input dimensionality (e.g. subcarriers × antennas).
        model_dim:    Output embedding dimensionality.
        hidden_dim:   Width of intermediate layers (defaults to 4× model_dim).
    """

    def __init__(self, feature_size: int, model_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or model_dim * 4

        self.net = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, x):
        # x: (B, feature_size)
        return self.net(x)  # (B, model_dim)


class CSIConvEncoder(nn.Module):
    """
    1-D convolutional encoder that treats the feature dimension as channels.

    Expects input (B, feature_size) — same interface as CSIEncoder.
    Useful when feature_size has spatial structure (e.g. subcarrier axis).

    Args:
        feature_size: Input dimensionality.
        model_dim:    Output embedding dimensionality.
        channels:     Conv channel sizes for each layer.
        kernel_size:  Kernel size for all conv layers.
    """

    def __init__(
        self,
        feature_size: int,
        model_dim: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        channels = channels or [64, 128, model_dim]

        layers = []
        in_ch = 1  # treat feature dim as sequence with 1 channel
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, feature_size) → treat as (B, 1, feature_size)
        x = x.unsqueeze(1)
        x = self.conv(x)       # (B, model_dim, L)
        x = self.pool(x)       # (B, model_dim, 1)
        return x.squeeze(-1)   # (B, model_dim)
