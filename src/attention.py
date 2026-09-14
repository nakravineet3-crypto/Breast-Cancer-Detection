"""
Channel + spatial attention (CBAM-style), applied to a spiking layer's
pre-activation "current" tensor (continuous-valued, before the LIF neuron
turns it into 0/1 spikes) — not to raw binary spikes directly.
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Learns 'which of the C channels matter', one weight per channel."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        squeezed = self.avg_pool(x).view(b, c)      # (batch, channels) — one number per channel
        weights = self.sigmoid(self.mlp(squeezed))  # (batch, channels), each in [0, 1]
        return x * weights.view(b, c, 1, 1)          # scale every pixel in each channel by its weight


class SpatialAttention(nn.Module):
    """Learns 'which pixel locations matter', one weight per (H, W) position."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)          # (batch, 1, H, W) — average across channels
        max_map, _ = x.max(dim=1, keepdim=True)         # (batch, 1, H, W) — strongest channel at each pixel
        combined = torch.cat([avg_map, max_map], dim=1)  # (batch, 2, H, W)
        weight_map = self.sigmoid(self.conv(combined))   # (batch, 1, H, W), each in [0, 1]
        return x * weight_map                             # broadcasts across all channels


class CBAM(nn.Module):
    """Channel attention, then spatial attention, applied in sequence."""
    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
