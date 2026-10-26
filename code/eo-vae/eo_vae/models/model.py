from __future__ import annotations
# Apache-2.0 License
# Copyright (c) https://github.com/black-forest-labs/flux2
# Modified for Unified EO-VAE / Flux Support

import torch
import torch.nn as nn
from torch import Tensor

from .modules.dynamic_conv import DynamicConv, DynamicConv_decoder
from .modules.layers import AttnBlock, Downsample, ResnetBlock, Upsample


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def get_1d_sincos_pos_embed(embed_dim: int, pos: Tensor) -> Tensor:
    """Generates sinusoidal embeddings for wavelengths."""
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)  # [1, N]

    half_dim = embed_dim // 2
    omega = torch.arange(half_dim, dtype=torch.float32, device=pos.device)
    omega /= half_dim / 1.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = torch.einsum('bn,d->bnd', pos, omega)  # (B, N, D/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    embedding = torch.cat([emb_sin, emb_cos], dim=2)  # (B, N, D)
    return embedding


class WavelengthConditioner(nn.Module):
    """Encodes a set of wavelengths into a global style vector for AdaIN."""

    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        # MLP to process the averaged wavelength embedding
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, wvs: Tensor, batch_size: int) -> Tensor:
        # wvs: [N] or [B, N]
        if wvs.dim() == 1:
            wvs = wvs.unsqueeze(0).repeat(batch_size, 1)  # [B, N]

        # 1. Embed each wavelength: [B, N, D]
        emb = get_1d_sincos_pos_embed(self.embed_dim, wvs)

        # 2. Global Average Pooling over spectral dimension: [B, D]
        # This creates a "fingerprint" of the active modality
        emb = emb.mean(dim=1)

        # 3. Project to Style Vector
        style = self.mlp(emb)
        return style


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        use_dynamic_ops: bool = False,
        dynamic_conv_kwargs: dict = None,
        enable_temporal: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels

        self.use_dynamic_ops = use_dynamic_ops
        self.enable_temporal = enable_temporal

        # --- NEW: Setup Conditioning ---
        # Check if AdaIN is requested via kwargs
        self.use_adain = False
        self.cond_dim = None

        if self.use_dynamic_ops:
            dynamic_kwargs = dynamic_conv_kwargs.copy() if dynamic_conv_kwargs else {}
            self.use_adain = dynamic_kwargs.pop('use_adain', False)

            if self.use_adain:
                self.cond_dim = 512
                self.conditioner = WavelengthConditioner(embed_dim=self.cond_dim)

            dynamic_kwargs.pop('mode', 'conv')
            wv_planes = dynamic_kwargs.pop('wv_planes', 128)
            inter_dim = dynamic_kwargs.pop('inter_dim', 128)

            self.conv_in = DynamicConv(
                wv_planes=wv_planes,
                inter_dim=inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                embed_dim=self.ch,
                **dynamic_kwargs,
            )
        else:
            self.conv_in = nn.Conv2d(
                in_channels, self.ch, kernel_size=3, stride=1, padding=1
            )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                # Pass cond_dim to ResnetBlock
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        cond_dim=self.cond_dim,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        # Pass cond_dim to Mid Blocks
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, cond_dim=self.cond_dim
        )
        self.mid.attn_1 = AttnBlock(block_in, enable_temporal=enable_temporal)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, cond_dim=self.cond_dim
        )

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)

    def forward(self, x: Tensor, wvs: Tensor = None, t_index: Tensor = None) -> Tensor:
        emb = None
        if self.use_dynamic_ops:
            assert wvs is not None, 'wvs must be provided for Dynamic Encoder'
            hs = [self.conv_in(x, wvs)]

            if self.use_adain:
                emb = self.conditioner(wvs, x.shape[0])
        else:
            hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # Pass embedding to ResnetBlock
                h = self.down[i_level].block[i_block](hs[-1], emb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, emb)
        h = self.mid.attn_1(h, t_index=t_index)
        h = self.mid.block_2(h, emb)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h

    def load_flux_weights(self, state_dict, strict=True):
        own_state = self.state_dict()
        ignore_layers = ['conv_in'] if self.use_dynamic_ops else []
        # Also ignore conditioner and projection layers if using AdaIN
        if self.use_adain:
            ignore_layers.extend(['conditioner', 'emb_proj'])

        for name, param in state_dict.items():
            if self.use_dynamic_ops and any(x in name for x in ignore_layers):
                # Skip dynamic/adain layers
                continue
            if name not in own_state:
                if strict:
                    raise KeyError(f'Unexpected key {name} in state_dict')
                continue
            try:
                own_state[name].copy_(param)
            except RuntimeError as e:
                print(f'Error loading {name}: {e}')
        print(
            f'Weights loaded. Dynamic Mode: {self.use_dynamic_ops}, AdaIN: {self.use_adain}'
        )


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        resolution: int,
        z_channels: int,
        use_dynamic_ops: bool = False,
        dynamic_conv_kwargs: dict = None,
        enable_temporal: bool = False,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.resolution = resolution
        self.use_dynamic_ops = use_dynamic_ops
        self.enable_temporal = enable_temporal

        # --- NEW: Setup Conditioning ---
        self.use_adain = False
        self.cond_dim = None

        if self.use_dynamic_ops:
            dynamic_kwargs = dynamic_conv_kwargs.copy() if dynamic_conv_kwargs else {}
            self.use_adain = dynamic_kwargs.pop('use_adain', False)

            if self.use_adain:
                self.cond_dim = 512
                self.conditioner = WavelengthConditioner(embed_dim=self.cond_dim)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, cond_dim=self.cond_dim
        )
        self.mid.attn_1 = AttnBlock(block_in, enable_temporal=enable_temporal)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, cond_dim=self.cond_dim
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        cond_dim=self.cond_dim,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )

        if self.use_dynamic_ops:
            # Re-fetch kwargs because we might have popped from a copy above
            dynamic_kwargs = dynamic_conv_kwargs.copy() if dynamic_conv_kwargs else {}
            # Remove use_adain from kwargs passed to dynamic layers
            dynamic_kwargs.pop('use_adain', None)

            dynamic_kwargs.pop('mode', 'conv')

            wv_planes = dynamic_kwargs.pop('wv_planes', 128)
            inter_dim = dynamic_kwargs.pop('inter_dim', 128)

            self.conv_out = DynamicConv_decoder(
                wv_planes=wv_planes,
                inter_dim=inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                embed_dim=block_in,
                **dynamic_kwargs,
            )
        else:
            self.conv_out = nn.Conv2d(
                block_in, out_ch, kernel_size=3, stride=1, padding=1
            )

    def forward(self, z: Tensor, wvs: Tensor = None, t_index: Tensor = None) -> Tensor:
        z = self.post_quant_conv(z)
        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)

        emb = None
        if self.use_dynamic_ops and self.use_adain:
            assert wvs is not None
            emb = self.conditioner(wvs, z.shape[0])

        h = self.mid.block_1(h, emb)
        h = self.mid.attn_1(h, t_index=t_index)
        h = self.mid.block_2(h, emb)

        h = h.to(upscale_dtype)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, emb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)

        if self.use_dynamic_ops:
            assert wvs is not None, 'wvs must be provided for Dynamic Decoder'
            h = self.conv_out(h, wvs)
        else:
            h = self.conv_out(h)

        return h

    def load_flux_weights(self, state_dict, strict=True):
        own_state = self.state_dict()
        ignore_layers = ['conv_out'] if self.use_dynamic_ops else []
        if self.use_adain:
            ignore_layers.extend(['conditioner', 'emb_proj'])

        for name, param in state_dict.items():
            if self.use_dynamic_ops and any(x in name for x in ignore_layers):
                continue
            if name not in own_state:
                continue
            try:
                own_state[name].copy_(param)
            except RuntimeError as e:
                print(f'Error loading {name}: {e}')

        print(f'Weights loaded. Dynamic Mode: {self.use_dynamic_ops}')
