"""
Physics-Guided Patch-Level Temporal-Spectral Dual Transformer (PGPRDT)

Model definition only. Patch-level physics descriptors are computed from the
raw signal during preprocessing and passed to forward() as `z_phys`.
"""

import torch
import torch.nn as nn


class TimePatchEmbed(nn.Module):
    """Flatten each patch and project it into a D-dimensional token."""

    def __init__(self, num_channels, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * num_channels, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, C) -> (B, N, D)
        B, T, C = x.shape
        N = T // self.patch_size
        x = x[:, :N * self.patch_size, :].reshape(B, N, self.patch_size * C)
        return self.norm(self.proj(x))


class SpectralFilterbankEmbed(nn.Module):
    """Patch rFFT magnitude -> learnable channel-wise filterbank -> token."""

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
        # x: (B, T, C) -> (B, N, D)
        B, T, C = x.shape
        N = T // self.patch_size
        x = x[:, :N * self.patch_size, :].reshape(B, N, self.patch_size, C)

        mag = torch.abs(torch.fft.rfft(x, dim=2))       # (B, N, F, C)
        mag = mag.permute(0, 1, 3, 2)                   # (B, N, C, F)

        bands = torch.einsum('bncf,cfk->bnck', mag, self.filterbank)
        bands = bands.reshape(B, N, C * self.num_bands)
        return self.norm(self.proj(bands))


class TransformerBlock(nn.Module):
    """Pre-normalized residual Transformer block."""

    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
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
    """Stack of Transformer blocks followed by a final layer normalization."""

    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1):
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
    """Map a patch physics descriptor to the time-branch weight alpha."""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_phys):
        # z_phys: (B, N, P) -> (B, N, 1)
        return torch.sigmoid(self.mlp(z_phys))


class PhysicsGuidedPatchDualTransformer(nn.Module):
    """
    Dual-branch Transformer with physics-guided patch-level routing.

    forward(x, z_phys) -> (logits, alpha)
        x       : (B, T, C) normalized inertial window
        z_phys  : (B, N, P) standardized patch physics descriptors
        logits  : (B, num_classes)
        alpha   : (B, N) time-branch weight per patch
    """

    def __init__(
        self,
        num_channels,
        num_classes,
        phys_dim,
        patch_size=8,
        num_patches=16,
        d_model=64,
        nhead=4,
        num_layers=2,
        d_ff=128,
        dropout=0.1,
        gate_hidden=64,
        num_freq_bands=8,
    ):
        super().__init__()
        self.time_embed = TimePatchEmbed(num_channels, patch_size, d_model)
        self.freq_embed = SpectralFilterbankEmbed(
            num_channels, patch_size, num_freq_bands, d_model
        )

        self.time_pos = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.freq_pos = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.freq_pos, std=0.02)

        self.time_encoder = BranchEncoder(d_model, nhead, d_ff, num_layers, dropout)
        self.freq_encoder = BranchEncoder(d_model, nhead, d_ff, num_layers, dropout)

        self.gate = PatchPhysicsGate(phys_dim, gate_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, z_phys):
        ht = self.time_encoder(self.time_embed(x) + self.time_pos)
        hf = self.freq_encoder(self.freq_embed(x) + self.freq_pos)

        alpha = self.gate(z_phys)                    # (B, N, 1)
        h = alpha * ht + (1.0 - alpha) * hf          # patch-wise fusion

        logits = self.classifier(h.mean(dim=1))
        return logits, alpha.squeeze(-1)
