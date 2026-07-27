"""
  - PGPRDT: Physics-Guided Patch Routing Dual Transformer
    for Temporal-Spectral Fusion in Wearable Human Activity Recognition
  - Author: JiminKim and Myung-Kyu Yi
  - Model Architecture Only
"""

import torch
import torch.nn as nn


class TimePatchEmbed(nn.Module):
    """Patch Flattening and Linear Projection for the Time Branch"""
    def __init__(self, num_channels, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * num_channels, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, N, D]
        """
        B, T, C = x.shape
        N = T // self.patch_size

        x = x[:, :N * self.patch_size, :]
        x = x.reshape(B, N, self.patch_size * C)
        x = self.proj(x)
        return self.norm(x)


class SpectralFilterbankEmbed(nn.Module):
    """Learnable Channel-wise Spectral Filterbank for the Frequency Branch"""
    def __init__(self, num_channels, patch_size, num_bands, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.num_bins = patch_size // 2 + 1
        self.num_bands = num_bands

        self.filterbank = nn.Parameter(
            torch.randn(num_channels, self.num_bins, num_bands) * 0.02
        )
        self.proj = nn.Linear(num_channels * num_bands, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, N, D]
        """
        B, T, C = x.shape
        N = T // self.patch_size

        x = x[:, :N * self.patch_size, :]
        x = x.reshape(B, N, self.patch_size, C)

        mag = torch.abs(torch.fft.rfft(x, dim=2))
        mag = mag.permute(0, 1, 3, 2)

        bands = torch.einsum("bncf,cfk->bnck", mag, self.filterbank)
        bands = bands.reshape(B, N, C * self.num_bands)

        z = self.proj(bands)
        return self.norm(z)


class TransformerBlock(nn.Module):
    """Pre-Normalized Residual Transformer Block"""
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class BranchEncoder(nn.Module):
    """Lightweight Transformer Encoder for a Single Representation Branch"""
    def __init__(self, d_model, nhead, d_ff, num_layers=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class PatchPhysicsGate(nn.Module):
    """Physics-Guided Gate for Patch-Level Temporal-Spectral Routing"""
    def __init__(self, phys_dim, hidden_dim=64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(phys_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_phys):
        """
        z_phys: [B, N, P]
        return: [B, N, 1]
        """
        return self.gate_net(z_phys)


class PGPRDT(nn.Module):
    """
    PGPRDT: Physics-Guided Patch Routing Dual Transformer
    """
    def __init__(self, num_channels=9, phys_dim=9, num_classes=6,
                 patch_size=8, num_patches=16, d_model=64, nhead=4,
                 num_layers=2, d_ff=128, dropout=0.1,
                 gate_hidden=64, num_freq_bands=8):
        super().__init__()

        self.time_embed = TimePatchEmbed(
            num_channels=num_channels,
            patch_size=patch_size,
            d_model=d_model,
        )

        self.freq_embed = SpectralFilterbankEmbed(
            num_channels=num_channels,
            patch_size=patch_size,
            num_bands=num_freq_bands,
            d_model=d_model,
        )

        self.time_pos = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.freq_pos = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.freq_pos, std=0.02)

        self.time_encoder = BranchEncoder(d_model, nhead, d_ff, num_layers, dropout)
        self.freq_encoder = BranchEncoder(d_model, nhead, d_ff, num_layers, dropout)

        self.patch_gate = PatchPhysicsGate(phys_dim=phys_dim, hidden_dim=gate_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, z_phys):
        """
        x: [B, T, C] normalized inertial window
        z_phys: [B, N, P] standardized patch-level physics descriptors
                [acc_energy, gyro_energy, jerk_energy, spectral_entropy,
                 dominant_freq_ratio, low_high_ratio, gravity_body_consistency,
                 acc_gyro_coupling, axis_correlation]
        """
        time_out = self.time_encoder(self.time_embed(x) + self.time_pos)
        freq_out = self.freq_encoder(self.freq_embed(x) + self.freq_pos)

        alpha = self.patch_gate(z_phys)
        patch_fused = alpha * time_out + (1.0 - alpha) * freq_out

        pooled = patch_fused.mean(dim=1)
        logits = self.classifier(pooled)

        return logits, alpha.squeeze(-1)
