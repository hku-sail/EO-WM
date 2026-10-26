# Apache-2.0 License

# Copyright (c) https://github.com/black-forest-labs/flux2

# Based on code: https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


# =============================================================================
# TEMPORAL RoPE (Rotary Position Encoding) over the time axis
# =============================================================================
#
# Motivated by TerraFlow (continuous temporal pretraining after TerraMind): we
# inject temporal structure into EO-VAE's *only* attention block by rotating
# the query/key projections by a time-index-dependent rotation matrix R^d_{Θ,m}.
# Because the rotation depends only on the RELATIVE temporal offset (n - m), and
# because at a single timestep (m=0, or any shared m) the rotation is an
# identity / cancels in the q·k dot product, the original (single-frame)
# attention is reproduced bit-exactly when T=1. RoPE adds NO learnable
# parameters, so the pretrained EO-VAE checkpoint loads strictly.
#
# Convention: input to AttnBlock is [B*T, C, H, W] (time folded into the batch
# axis). The temporal path regroups to batch=B with a combined token sequence
# T*H*W, applies RoPE by the time component of each token's index, then runs
# SDPA exactly like the spatial path.


def build_rope_freqs(d: int, base: float = 10000.0) -> Tensor:
    """Angular frequencies for RoPE: θ_i = base^{-2i/d}, i = 0..d//2-1.

    Returns a float32 tensor of shape [ceil(d/2)] (kept in fp32 for trig
    precision under mixed precision; callers cast the rotated output back).
    """
    half = (d + 1) // 2  # handle odd d by generating one extra pair slot
    return 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32)[:half] / d))


def _rotate_half_pairs(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary rotation to the last dim of x using interleaved (even/odd)
    channel pairs.

    x: [..., d] (d even here; odd d is padded by caller). cos/sin: broadcastable
    to [..., d//2]. Performs the standard interleaved RoPE
        out[2i]   = x[2i]*cos_i - x[2i+1]*sin_i
        out[2i+1] = x[2i]*sin_i + x[2i+1]*cos_i
    which realizes the block-diagonal 2D rotation R^d_{Θ,m}.
    """
    x1 = x[..., 0::2]  # even channels
    x2 = x[..., 1::2]  # odd channels
    rot_even = x1 * cos - x2 * sin
    rot_odd = x1 * sin + x2 * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


def apply_temporal_rope(
    x: Tensor, t_index: Tensor, inv_freq: Tensor
) -> Tensor:
    """Rotate x by R_{Θ, m} where m is the per-token time index.

    Args:
        x: [B, T, H*W, d] -- the query or key tensor, already regrouped so the
            time axis is explicit (dim 1). d is the channel/feature dim.
        t_index: [T] (shared across batch) or [B, T] integer time indices. For
            single-frame inheritance, this is all zeros -> identity rotation.
        inv_freq: [d//2] float32 angular frequencies (from build_rope_freqs).

    Returns:
        x rotated, same shape/dtype as input.
    """
    *_, T, HW, d = x.shape  # x: [B, T, HW, d]

    # Normalize t_index to [B, T] in fp32 for the angle computation
    if t_index.dim() == 1:
        t_index_b = t_index.view(1, T).to(torch.float32)
    else:
        t_index_b = t_index.to(torch.float32)
    B = t_index_b.shape[0]

    # angles: [B, T, d//2] = outer(t_index, inv_freq)
    angles = torch.einsum('bt,d->btd', t_index_b, inv_freq.to(torch.float32))
    # Broadcast to [B, T, 1, d//2] over the HW axis
    cos = angles.cos().unsqueeze(2)  # [B, T, 1, d//2]
    sin = angles.sin().unsqueeze(2)

    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)

    # Handle odd d: pad last channel with a pass-through zero so we have an even
    # number of channels; drop the pad after rotation.
    if d % 2 == 1:
        x_fp32 = F.pad(x_fp32, (0, 1))  # [B,T,HW,d+1]

    out = _rotate_half_pairs(x_fp32, cos, sin)  # [B,T,HW,d] (or d+1)

    if d % 2 == 1:
        out = out[..., :d]

    return out.to(orig_dtype)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.Module:
    """Group normalization with default of 32 groups."""
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode='constant', value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # --- NEW: Optional AdaIN Projection ---
        if self.cond_dim is not None:
            # Project embedding to [scale, shift] for out_channels
            self.emb_proj = nn.Linear(cond_dim, out_channels * 2)
            # Initialize to identity (scale=1, shift=0)
            nn.init.zeros_(self.emb_proj.bias)
            self.emb_proj.weight.data.zero_()
            # Initialize scale part of bias to 1
            self.emb_proj.bias.data[:out_channels] = 1.0

        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x, emb=None):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        # --- NEW: Apply AdaIN if embedding is provided ---
        if self.cond_dim is not None and emb is not None:
            # emb: [B, cond_dim] -> [B, 2*out_channels]
            style = self.emb_proj(emb)
            # Reshape for broadcasting: [B, 2*C, 1, 1]
            style = style.unsqueeze(-1).unsqueeze(-1)
            scale, shift = style.chunk(2, dim=1)

            h = self.norm2(h)
            h = h * scale + shift  # AdaIN modulation
        else:
            h = self.norm2(h)

        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """Spatial (self-)attention block with an optional *temporal* RoPE path.

    By default (enable_temporal=False, t_index=None) this is bit-identical to
    the original single-frame EO-VAE attention: SDPA over H*W spatial tokens
    per batch element. When enable_temporal=True and t_index is supplied, the
    T frames folded into the batch axis are regrouped so attention runs over a
    combined T*H*W token sequence per sample, and temporal RoPE rotates q/k by
    the per-token time index. At T=1 (or any shared time index) the rotation is
    identity and the output matches the non-temporal path exactly, so pretrained
    weights load without perturbation. RoPE adds no learnable parameters.
    """

    def __init__(self, in_channels: int, enable_temporal: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.enable_temporal = enable_temporal
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        if enable_temporal:
            # Deterministic from (d, base); not learned. Non-persistent so it
            # never appears as a missing/unexpected key when loading a pretrained
            # checkpoint that predates the temporal path.
            self.register_buffer(
                '_rope_inv_freq', build_rope_freqs(in_channels), persistent=False
            )

    def forward(self, x, t_index=None):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Reshape for SDPA: B C H W -> B 1 (H W) C
        bt, c, h, w = q.shape

        if not self.enable_temporal or t_index is None:
            # Non-temporal path: spatial self-attention per batch element.
            # (When time was folded into the batch axis, frames already do not
            # attend across each other here -- identical to original behavior.)
            q = rearrange(q, 'bt c h w -> bt 1 (h w) c').contiguous()
            k = rearrange(k, 'bt c h w -> bt 1 (h w) c').contiguous()
            v = rearrange(v, 'bt c h w -> bt 1 (h w) c').contiguous()

            h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            h_ = rearrange(h_, 'bt 1 (h w) c -> bt c h w', h=h, w=w)
            return x + self.proj_out(h_)

        # Temporal path. Convention: x came from [B, T, C, H, W] merged as
        # 'b t c h w -> (b t) c h w', so element n = (b=n//T, t=n%T). We regroup
        # to batch=B with sequence T*H*W, apply RoPE by the time index, SDPA.
        if t_index.dim() == 1:
            T = t_index.shape[0]
        else:
            T = t_index.shape[1]
        assert bt % T == 0, (
            f'temporal AttnBlock: batch*time {bt} not divisible by T={T}'
        )

        q = rearrange(q, '(b t) c h w -> b 1 (t h w) c', t=T).contiguous()
        k = rearrange(k, '(b t) c h w -> b 1 (t h w) c', t=T).contiguous()
        v = rearrange(v, '(b t) c h w -> b 1 (t h w) c', t=T).contiguous()

        # RoPE on the time component: view as [B, T, H*W, c] then back to [B,1,T*HW,c]
        q = rearrange(q, 'b 1 (t hw) c -> b t hw c', t=T, hw=h * w)
        k = rearrange(k, 'b 1 (t hw) c -> b t hw c', t=T, hw=h * w)
        q = apply_temporal_rope(q, t_index, self._rope_inv_freq)
        k = apply_temporal_rope(k, t_index, self._rope_inv_freq)
        q = rearrange(q, 'b t hw c -> b 1 (t hw) c', t=T, hw=h * w)
        k = rearrange(k, 'b t hw c -> b 1 (t hw) c', t=T, hw=h * w)

        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, 'b 1 (t h w) c -> (b t) c h w', t=T, h=h, w=w)
        return x + self.proj_out(h_)
