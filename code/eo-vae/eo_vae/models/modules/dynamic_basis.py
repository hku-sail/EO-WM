import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

# -----------------------------------------------------------------------
# Shared Utilities
# -----------------------------------------------------------------------


def get_1d_sincos_pos_embed(embed_dim: int, pos: Tensor) -> Tensor:
    """Generates sinusoidal embeddings for wavelengths."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


class ScalableHyperNet(nn.Module):
    """High-Capacity HyperNetwork."""

    def __init__(self, in_dim: int, rank_dim: int, out_dim: int, depth: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(in_dim * 2, in_dim * 2), nn.GELU())
                for _ in range(depth)
            ],
            nn.Linear(in_dim * 2, rank_dim),
        )

        self.expansion = nn.Linear(rank_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        # Initialize expansion to be small to start with a "mean" basis
        init.normal_(self.expansion.weight, std=0.001)

    def forward(self, wvs_emb: Tensor) -> Tensor:
        latent = self.backbone(wvs_emb)
        coeffs = self.expansion(latent)
        return coeffs


class DynamicInputLayer(nn.Module):
    """Scalable Input Layer: Compresses N variable bands into C fixed channels.
    Uses GLOBAL Shared Basis with PER-CHANNEL Modulation.
    """

    def __init__(
        self,
        out_channels: int,
        num_bases: int = 64,
        rank_dim: int = 64,  # Increased rank for better capacity
        kernel_size: int = 3,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_bases = num_bases
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.wv_dim = 128

        # 1. GLOBAL Basis Bank
        # Shape: [Num_Bases, 1, K, K]
        self.basis_bank = nn.Parameter(
            torch.empty(num_bases, 1, kernel_size, kernel_size)
        )
        init.kaiming_uniform_(self.basis_bank, a=math.sqrt(5))

        # 2. Scalable HyperNetwork
        # Predicts coefficients for EVERY output channel individually
        # Output: [N_in, Out_Channels * Num_Bases]
        self.hypernet = ScalableHyperNet(
            in_dim=self.wv_dim, rank_dim=rank_dim, out_dim=out_channels * num_bases
        )

        self.wv_proj = nn.Linear(self.wv_dim, self.wv_dim)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def get_distillation_weight(self, wvs: Tensor):
        device = self.basis_bank.device
        wvs = wvs.to(device)

        # 1. Embed
        w_emb = get_1d_sincos_pos_embed(self.wv_dim, wvs * 1000).to(device)
        w_emb = self.wv_proj(w_emb)

        # 2. Coefficients: [N_in, Out_Channels, Num_Bases]
        coeffs = self.hypernet(w_emb)
        coeffs = coeffs.view(-1, self.out_channels, self.num_bases)

        # 3. Reconstruct Weights
        # Einsum:
        # coeffs: n (input bands), o (out channels), b (bases)
        # bank:   b (bases), 1, x, y
        # result: n, o, 1, x, y
        w_generated = torch.einsum('nob, bixy -> noixy', coeffs, self.basis_bank)

        # Squeeze the '1' from basis bank and permute for Conv2d
        # [N_in, Out, K, K] -> [Out, N_in, K, K]
        w_dynamic = w_generated.squeeze(2).permute(1, 0, 2, 3)

        return w_dynamic, self.bias

    def forward(self, x: Tensor, wvs: Tensor) -> Tensor:
        device = x.device
        w_emb = get_1d_sincos_pos_embed(self.wv_dim, wvs * 1000).to(device)
        w_emb = self.wv_proj(w_emb)

        coeffs = self.hypernet(w_emb)
        coeffs = coeffs.view(-1, self.out_channels, self.num_bases)

        # [N_in, Out, 1, K, K]
        w_generated = torch.einsum('nob, bixy -> noixy', coeffs, self.basis_bank)

        # [Out, N_in, K, K]
        w_dynamic = w_generated.squeeze(2).permute(1, 0, 2, 3)

        return F.conv2d(x, w_dynamic, bias=self.bias, padding=self.padding)


class DynamicOutputLayer(nn.Module):
    """Scalable Output Layer: Expands C fixed channels into N variable bands.
    Uses GLOBAL Shared Basis with PER-CHANNEL Modulation.
    """

    def __init__(
        self,
        in_channels: int,
        num_bases: int = 64,
        rank_dim: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_bases = num_bases
        self.padding = kernel_size // 2
        self.kernel_size = kernel_size

        self.wv_dim = 128

        # 1. GLOBAL Basis Bank
        self.basis_bank = nn.Parameter(
            torch.empty(num_bases, 1, kernel_size, kernel_size)
        )
        init.kaiming_uniform_(self.basis_bank, a=math.sqrt(5))

        # 2. Scalable HyperNetwork
        # Predicts coefficients for EVERY input channel individually
        # Output: [N_out, In_Channels * Num_Bases]
        self.hypernet = ScalableHyperNet(
            in_dim=self.wv_dim, rank_dim=rank_dim, out_dim=in_channels * num_bases
        )

        self.wv_proj = nn.Linear(self.wv_dim, self.wv_dim)

        self.bias_generator = nn.Sequential(
            nn.Linear(self.wv_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.weight = None

    def get_distillation_weight(self, wvs: Tensor):
        device = self.basis_bank.device
        wvs = wvs.to(device)

        w_emb = get_1d_sincos_pos_embed(self.wv_dim, wvs * 1000).to(device)
        w_emb = self.wv_proj(w_emb)

        # Coefficients: [N_out, In_Channels, Num_Bases]
        coeffs = self.hypernet(w_emb)
        coeffs = coeffs.view(-1, self.in_channels, self.num_bases)

        # Reconstruct
        # coeffs: n (out bands), i (in channels), b (bases)
        # bank:   b (bases), j (singleton=1), x, y
        # result: n, i, j, x, y
        w_generated = torch.einsum('nib, bjxy -> nijxy', coeffs, self.basis_bank)

        # [N_out, In, K, K]
        w_dynamic = w_generated.squeeze(2)

        bias_dynamic = self.bias_generator(w_emb).flatten()
        return w_dynamic, bias_dynamic

    def forward(self, x: Tensor, wvs: Tensor) -> Tensor:
        device = x.device
        w_emb = get_1d_sincos_pos_embed(self.wv_dim, wvs * 1000).to(device)
        w_emb = self.wv_proj(w_emb)

        coeffs = self.hypernet(w_emb)
        coeffs = coeffs.view(-1, self.in_channels, self.num_bases)

        w_generated = torch.einsum('nib, bjxy -> nijxy', coeffs, self.basis_bank)
        w_dynamic = w_generated.squeeze(2)

        bias_dynamic = self.bias_generator(w_emb).flatten()
        self.weight = w_dynamic

        # 5. Standard Conv2d
        return F.conv2d(x, w_dynamic, bias=bias_dynamic, padding=self.padding)
