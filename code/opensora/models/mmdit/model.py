# Modified from Flux
#
# Copyright 2024 Black Forest Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.models.mmdit.layers import DoubleStreamBlock, EmbedND, LastLayer, LigerEmbedND, MLPEmbedder, SingleStreamBlock, timestep_embedding
from opensora.registry import MODELS
from opensora.utils.ckpt import load_checkpoint
from einops import rearrange


@dataclass
class MMDiTConfig:
    model_type = 'MMDiT'
    from_pretrained: str
    cache_dir: str
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    cond_embed: bool = False
    fused_qkv: bool = True
    grad_ckpt_settings: tuple[int, int] | None = None
    use_liger_rope: bool = False
    patch_size: int = 2
    cond_reinject_every_n: int = 0

    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def __contains__(self, attribute_name):
        return hasattr(self, attribute_name)


class MMDiTModel(nn.Module):
    config_class = MMDiTConfig

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        self.patch_size = config.patch_size
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f'Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}')
        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(f'Got {config.axes_dim} but expected positional dim {pe_dim}')
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
        self.pe_embedder = pe_embedder_cls(dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256,
                                       hidden_dim=self.hidden_size) if config.guidance_embed else nn.Identity()
        self.cond_in = nn.Linear(self.in_channels + self.patch_size**2, self.hidden_size,
                                 bias=True) if config.cond_embed else nn.Identity()
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)
        self.eo_embedder = nn.Sequential(nn.Linear(6, self.hidden_size), nn.SiLU(),
                                         nn.Linear(self.hidden_size, self.hidden_size))
        nn.init.normal_(self.eo_embedder[-1].weight, std=0.02)
        nn.init.zeros_(self.eo_embedder[-1].bias)
        self.eo_time_embedder = nn.Sequential(nn.Linear(3, self.hidden_size), nn.SiLU(),
                                              nn.Linear(self.hidden_size, self.hidden_size))
        nn.init.normal_(self.eo_time_embedder[-1].weight, std=0.02)
        nn.init.zeros_(self.eo_time_embedder[-1].bias)
        self.dem_encoder = nn.Sequential(nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                                         nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                                         nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1))
        self.dem_in = nn.Linear(16 * self.patch_size**2, self.hidden_size, bias=True)
        self.era5_encoder = nn.Sequential(nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                                          nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                                          nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1))
        self.era5_in = nn.Linear(32 * self.patch_size**2, self.hidden_size, bias=True)
        self.era5_clim_in = nn.Linear(5, self.hidden_size, bias=True)
        self.cumstress_in = nn.Linear(3, self.hidden_size, bias=True)
        nn.init.normal_(self.dem_in.weight, std=0.02)
        nn.init.zeros_(self.dem_in.bias)
        nn.init.normal_(self.era5_in.weight, std=0.02)
        nn.init.zeros_(self.era5_in.bias)
        nn.init.zeros_(self.era5_clim_in.weight)
        nn.init.zeros_(self.era5_clim_in.bias)
        nn.init.zeros_(self.cumstress_in.weight)
        nn.init.zeros_(self.cumstress_in.bias)
        self.reinject_gates = nn.ParameterList()
        if config.cond_reinject_every_n > 0:
            num_inject = config.depth // config.cond_reinject_every_n
            for _ in range(num_inject):
                self.reinject_gates.append(nn.Parameter(torch.zeros(1)))
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(self.hidden_size,
                              self.num_heads,
                              mlp_ratio=config.mlp_ratio,
                              qkv_bias=config.qkv_bias,
                              fused_qkv=config.fused_qkv) for _ in range(config.depth)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio, fused_qkv=config.fused_qkv)
            for _ in range(config.depth_single_blocks)
        ])
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.initialize_weights()
        if self.config.grad_ckpt_settings:
            self.forward = self.forward_selective_ckpt
        else:
            self.forward = self.forward_ckpt
        self._input_requires_grad = False

    def initialize_weights(self):
        if self.config.cond_embed:
            nn.init.normal_(self.cond_in.weight, std=0.02)
            nn.init.zeros_(self.cond_in.bias)

    def encode_eo_spatial(self, dem, dem_mask, era5, era5_mask, T_lat):
        B = dem.shape[0]
        dem_in = torch.cat([dem, dem_mask], dim=1)
        dem_feat = self.dem_encoder(dem_in)
        dem_feat = dem_feat.unsqueeze(2).expand(-1, -1, T_lat, -1, -1)
        era5_in = torch.cat([era5, era5_mask], dim=1)
        _, _, _, H, W = era5_in.shape
        n_era5_ch = era5_in.shape[1]
        era5_in_reshaped = era5_in.permute(0, 2, 1, 3, 4).reshape(B * T_lat, n_era5_ch, H, W)
        era5_feat = self.era5_encoder(era5_in_reshaped)
        _, _, H_lat, W_lat = era5_feat.shape
        era5_feat = era5_feat.view(B, T_lat, 32, H_lat, W_lat).permute(0, 2, 1, 3, 4)
        return (dem_feat, era5_feat)

    def _fuse_cond_components(self, components: list[Tensor], reference: Tensor) -> Tensor:
        """Sum the projected spatial conditions."""
        if not components:
            return torch.zeros_like(reference)
        fused = components[0]
        for component in components[1:]:
            fused = fused + component
        return fused

    def prepare_block_inputs(self,
                             img: Tensor,
                             img_ids: Tensor,
                             txt: Tensor,
                             txt_ids: Tensor,
                             timesteps: Tensor,
                             y_vec: Tensor,
                             cond: Tensor = None,
                             guidance: Tensor | None = None,
                             eo_info: Tensor | None = None,
                             eo_time_info: Tensor | None = None,
                             dem: Tensor | None = None,
                             dem_mask: Tensor | None = None,
                             era5: Tensor | None = None,
                             era5_mask: Tensor | None = None,
                             era5_anom: Tensor | None = None,
                             era5_clim: Tensor | None = None,
                             cumstress: Tensor | None = None):
        """
        obtain the processed:
            img: projected noisy img latent,
            txt: text context (from t5),
            vec: clip encoded vector,
            pe: the positional embeddings for concatenated img and txt
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError('Input img and txt tensors must have 3 dimensions.')
        img = self.img_in(img)
        cond_components = []
        if self.config.cond_embed:
            if cond is None:
                raise ValueError("Didn't get conditional input for conditional model.")
            cond_emb = self.cond_in(cond)
            img = img + cond_emb
            cond_components.append(cond_emb)
        if dem is not None and era5 is not None:
            T_lat = era5.shape[2]
            p = self.patch_size
            dem_feat, _ = self.encode_eo_spatial(dem, dem_mask, era5, era5_mask, T_lat)
            dem_packed = rearrange(dem_feat, 'b c t (h ph) (w pw) -> b (t h w) (c ph pw)', ph=p, pw=p)
            dem_emb = self.dem_in(dem_packed)
            if era5_anom is not None and era5_clim is not None:
                B_a, C_a, T_a, H_a, W_a = era5_anom.shape
                anom_padded = torch.cat(
                    [era5_anom, era5_anom.new_zeros(B_a, era5.shape[1] - C_a, T_a, H_a, W_a)], dim=1)
                anom_with_mask = torch.cat([anom_padded, era5_mask], dim=1)
                B_e = anom_with_mask.shape[0]
                anom_2d = anom_with_mask.permute(0, 2, 1, 3, 4).reshape(B_e * T_lat, -1, H_a, W_a)
                anom_feat = self.era5_encoder(anom_2d)
                _, C_f, H_l, W_l = anom_feat.shape
                anom_feat = anom_feat.view(B_e, T_lat, C_f, H_l, W_l).permute(0, 2, 1, 3, 4)
                anom_packed = rearrange(anom_feat, 'b c t (h ph) (w pw) -> b (t h w) (c ph pw)', ph=p, pw=p)
                anom_emb = self.era5_in(anom_packed)
                clim_hidden = self.era5_clim_in(era5_clim)
                frame_ids = img_ids[..., 0].to(torch.long)
                frame_ids = frame_ids.clamp(min=0, max=clim_hidden.shape[1] - 1)
                clim_emb = torch.gather(clim_hidden,
                                        dim=1,
                                        index=frame_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size))
                img = img + dem_emb + clim_emb
                cond_components.extend([dem_emb, anom_emb])
            else:
                _, era5_feat = self.encode_eo_spatial(dem, dem_mask, era5, era5_mask, T_lat)
                era5_packed = rearrange(era5_feat, 'b c t (h ph) (w pw) -> b (t h w) (c ph pw)', ph=p, pw=p)
                era5_emb = self.era5_in(era5_packed)
                img = img + dem_emb + era5_emb
                cond_components.extend([dem_emb, era5_emb])
        if cumstress is not None:
            cs_hidden = self.cumstress_in(cumstress)
            frame_ids_cs = img_ids[..., 0].to(torch.long)
            frame_ids_cs = frame_ids_cs.clamp(min=0, max=cs_hidden.shape[1] - 1)
            cs_emb = torch.gather(cs_hidden, dim=1, index=frame_ids_cs.unsqueeze(-1).expand(-1, -1, self.hidden_size))
            cond_components.append(cs_emb)
        if eo_time_info is not None:
            time_frame_emb = self.eo_time_embedder(eo_time_info)
            frame_ids = img_ids[..., 0].to(torch.long)
            frame_ids = frame_ids.clamp(min=0, max=time_frame_emb.shape[1] - 1)
            time_token_emb = torch.gather(time_frame_emb,
                                          dim=1,
                                          index=frame_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size))
            img = img + time_token_emb
            cond_components.append(time_token_emb)
        cond_spatial = self._fuse_cond_components(cond_components, img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y_vec)
        if eo_info is not None:
            vec = vec + self.eo_embedder(eo_info)
        vec_cond = None
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        if self._input_requires_grad:
            img.requires_grad_()
            txt.requires_grad_()
        return (img, txt, vec, pe, cond_spatial, vec_cond)

    def enable_input_require_grads(self):
        """Fit peft lora. This method should not be called manually."""
        self._input_requires_grad = True

    def forward_ckpt(self,
                     img: Tensor,
                     img_ids: Tensor,
                     txt: Tensor,
                     txt_ids: Tensor,
                     timesteps: Tensor,
                     y_vec: Tensor,
                     cond: Tensor = None,
                     guidance: Tensor | None = None,
                     eo_info: Tensor | None = None,
                     eo_time_info: Tensor | None = None,
                     dem: Tensor | None = None,
                     dem_mask: Tensor | None = None,
                     era5: Tensor | None = None,
                     era5_mask: Tensor | None = None,
                     **kwargs) -> Tensor:
        img, txt, vec, pe, cond_spatial, vec_cond = self.prepare_block_inputs(img,
                                                                              img_ids,
                                                                              txt,
                                                                              txt_ids,
                                                                              timesteps,
                                                                              y_vec,
                                                                              cond,
                                                                              guidance,
                                                                              eo_info,
                                                                              eo_time_info,
                                                                              dem,
                                                                              dem_mask,
                                                                              era5,
                                                                              era5_mask,
                                                                              era5_anom=kwargs.get('era5_anom'),
                                                                              era5_clim=kwargs.get('era5_clim'),
                                                                              cumstress=kwargs.get('cumstress'))
        attn_mask = None
        reinject_n = self.config.cond_reinject_every_n
        gate_idx = 0
        for i, block in enumerate(self.double_blocks):
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe, attn_mask)
            if reinject_n > 0 and (i + 1) % reinject_n == 0 and (gate_idx < len(self.reinject_gates)):
                img = img + self.reinject_gates[gate_idx].tanh() * cond_spatial
                gate_idx += 1
        txt_len = txt.shape[1]
        img = torch.cat((txt, img), 1)
        for j, block in enumerate(self.single_blocks):
            img = auto_grad_checkpoint(block, img, vec, pe, attn_mask)
        img = img[:, txt_len:, ...]
        return self.final_layer(img, vec)

    def forward_selective_ckpt(self,
                               img: Tensor,
                               img_ids: Tensor,
                               txt: Tensor,
                               txt_ids: Tensor,
                               timesteps: Tensor,
                               y_vec: Tensor,
                               cond: Tensor = None,
                               guidance: Tensor | None = None,
                               eo_info: Tensor | None = None,
                               eo_time_info: Tensor | None = None,
                               dem: Tensor | None = None,
                               dem_mask: Tensor | None = None,
                               era5: Tensor | None = None,
                               era5_mask: Tensor | None = None,
                               **kwargs) -> Tensor:
        img, txt, vec, pe, cond_spatial, vec_cond = self.prepare_block_inputs(img,
                                                                              img_ids,
                                                                              txt,
                                                                              txt_ids,
                                                                              timesteps,
                                                                              y_vec,
                                                                              cond,
                                                                              guidance,
                                                                              eo_info,
                                                                              eo_time_info,
                                                                              dem,
                                                                              dem_mask,
                                                                              era5,
                                                                              era5_mask,
                                                                              era5_anom=kwargs.get('era5_anom'),
                                                                              era5_clim=kwargs.get('era5_clim'),
                                                                              cumstress=kwargs.get('cumstress'))
        attn_mask = None
        reinject_n = self.config.cond_reinject_every_n
        gate_idx = 0
        ckpt_depth_double = self.config.grad_ckpt_settings[0]
        for i, block in enumerate(self.double_blocks[:ckpt_depth_double]):
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe, attn_mask)
            if reinject_n > 0 and (i + 1) % reinject_n == 0 and (gate_idx < len(self.reinject_gates)):
                img = img + self.reinject_gates[gate_idx].tanh() * cond_spatial
                gate_idx += 1
        for j, block in enumerate(self.double_blocks[ckpt_depth_double:]):
            i = ckpt_depth_double + j
            img, txt = block(img, txt, vec, pe, attn_mask)
            if reinject_n > 0 and (i + 1) % reinject_n == 0 and (gate_idx < len(self.reinject_gates)):
                img = img + self.reinject_gates[gate_idx].tanh() * cond_spatial
                gate_idx += 1
        ckpt_depth_single = self.config.grad_ckpt_settings[1]
        txt_len = txt.shape[1]
        img = torch.cat((txt, img), 1)
        for j, block in enumerate(self.single_blocks[:ckpt_depth_single]):
            img = auto_grad_checkpoint(block, img, vec, pe, attn_mask)
        for j, block in enumerate(self.single_blocks[ckpt_depth_single:]):
            img = block(img, vec, pe, attn_mask)
        img = img[:, txt_len:, ...]
        return self.final_layer(img, vec)


@MODELS.register_module('flux')
def Flux(cache_dir: str = None,
         from_pretrained: str = None,
         device_map: str | torch.device = 'cuda',
         torch_dtype: torch.dtype = torch.bfloat16,
         strict_load: bool = False,
         **kwargs) -> MMDiTModel:
    config = MMDiTConfig(from_pretrained=from_pretrained, cache_dir=cache_dir, **kwargs)
    low_precision_init = from_pretrained is not None and len(from_pretrained) > 0
    if low_precision_init:
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
    with torch.device(device_map):
        model = MMDiTModel(config)
    if low_precision_init:
        torch.set_default_dtype(default_dtype)
    else:
        model = model.to(torch_dtype)
    if from_pretrained:
        model = load_checkpoint(model, from_pretrained, cache_dir=cache_dir, device_map=device_map, strict=strict_load)
    return model
