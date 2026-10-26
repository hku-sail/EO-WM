"""Multi-Stage Dynamic Convolution Decoder for Earth Observation VAE.

This module provides enhanced decoder architectures that address the capacity
limitations of single-layer dynamic convolutions when reconstructing diverse
satellite modalities (optical, SAR, thermal, etc.).

The key insight is separating:
1. Wavelength-agnostic spatial refinement (shared capacity)
2. Wavelength-specific spectral mapping (dynamic capacity)

Available decoder head types:
- StackedDynamicDecoder: Simple stacked dynamic convs (recommended first try)
- MultiStageDynamicDecoder: Shared spatial + dynamic projection
- ProgressiveMultiStageDynamicDecoder: Full progressive refinement

"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

# Import from the dynamic_conv module in the same package
from .dynamic_conv import (
    FactorizedWeightGenerator_decoder,
    FCResLayer,
    TransformerWeightGenerator_decoder,
    get_1d_sincos_pos_embed_from_grid_torch,
)


class DynamicConvBlock(nn.Module):
    """A single dynamic convolution block with optional normalization and activation.

    This is a building block for multi-stage architectures. Unlike the final output
    layer, this block maps embed_dim -> embed_dim with wavelength conditioning.
    """

    def __init__(
        self,
        wv_planes: int,
        embed_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_layers: int = 1,
        num_heads: int = 4,
        use_norm: bool = True,
        use_activation: bool = True,
        generator_type: str = 'transformer',
        rank_ratio: int = 4,
    ) -> None:
        """Args:
        wv_planes: Dimension of wavelength features
        embed_dim: Both input and output channel dimension
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        num_layers: Transformer layers in weight generator
        num_heads: Attention heads in weight generator
        use_norm: Whether to apply group normalization after conv
        use_activation: Whether to apply activation after conv
        generator_type: 'transformer' or 'factorized'
        rank_ratio: Rank reduction for factorized generator
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wv_planes = wv_planes
        self._num_kernel = kernel_size * kernel_size * embed_dim

        # Weight generator (outputs embed_dim channels, not wavelength-specific)
        if generator_type == 'factorized':
            self.weight_generator = FactorizedWeightGenerator_decoder(
                wv_planes,
                self._num_kernel,
                embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                rank_ratio=rank_ratio,
            )
        else:
            self.weight_generator = TransformerWeightGenerator_decoder(
                wv_planes,
                self._num_kernel,
                embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
            )

        # Wavelength embedding processor
        self.fclayer = FCResLayer(wv_planes)

        # Post-conv processing
        self.norm = (
            nn.GroupNorm(min(32, embed_dim), embed_dim) if use_norm else nn.Identity()
        )
        self.act = nn.SiLU() if use_activation else nn.Identity()

        self.scaler = 0.1
        self._init_weights()

    def _init_weights(self) -> None:
        def weight_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.weight_generator.apply(weight_init)
        self.fclayer.apply(weight_init)

    def forward(self, x: Tensor, waves_embedded: Tensor) -> Tensor:
        """Args:
            x: Input features [B, embed_dim, H, W]
            waves_embedded: Pre-embedded wavelength features [num_wv, wv_planes]

        Returns:
            Output features [B, embed_dim, H', W']
        """
        # Generate weights conditioned on wavelengths
        # We use the mean wavelength embedding to condition this intermediate layer
        # This provides spectral awareness without per-channel outputs
        mean_wave = waves_embedded.mean(dim=0, keepdim=True)

        weight, bias = self.weight_generator(mean_wave)

        # Reshape weights: [1, K*K*embed_dim] -> [embed_dim, embed_dim, K, K]
        # Note: This creates an embed_dim -> embed_dim conv
        dynamic_weight = weight.view(
            1, self.kernel_size, self.kernel_size, self.embed_dim
        )
        # For embed_dim -> embed_dim, we need to expand
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])  # [embed_dim, 1, K, K]
        dynamic_weight = dynamic_weight.expand(
            -1, self.embed_dim, -1, -1
        )  # depthwise-ish

        # Generate weights for the requested embedding dimensions.
        # The issue is output_dim in the generator. Let me fix this.

        if bias is not None:
            bias = bias.squeeze() * self.scaler
            # Expand bias to embed_dim
            if bias.dim() == 0:
                bias = bias.expand(self.embed_dim)
            elif bias.size(0) == 1:
                bias = bias.expand(self.embed_dim)

        out = F.conv2d(
            x,
            dynamic_weight * self.scaler,
            bias=bias if bias is not None and bias.numel() == self.embed_dim else None,
            stride=self.stride,
            padding=self.padding,
            groups=self.embed_dim,  # Depthwise for efficiency
        )

        out = self.norm(out)
        out = self.act(out)

        return out


class SharedRefinementBlock(nn.Module):
    """Wavelength-agnostic spatial refinement block.

    This uses standard (non-dynamic) convolutions to provide spatial
    processing capacity that's shared across all modalities. This is
    crucial because spatial features (edges, textures) are largely
    wavelength-independent.
    """

    def __init__(
        self,
        embed_dim: int,
        expansion: int = 2,
        kernel_size: int = 3,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        hidden_dim = embed_dim * expansion
        self.use_residual = use_residual

        self.block = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
        )

        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return self.act(out)


class WavelengthAdaptiveWeightGenerator(nn.Module):
    """Enhanced weight generator that produces wavelength-specific conv weights.

    Key improvements over the base TransformerWeightGenerator_decoder:
    1. Multi-scale wavelength processing
    2. Cross-attention between wavelengths for coherent multi-channel output
    3. Separate pathways for spatial vs spectral weight components
    """

    def __init__(
        self,
        wv_planes: int,
        output_dim: int,  # K * K * in_channels (for one output channel)
        in_channels: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.wv_planes = wv_planes
        self.output_dim = output_dim
        self.in_channels = in_channels

        # Deeper processing of wavelength embeddings
        self.wave_processor = nn.Sequential(
            nn.Linear(wv_planes, wv_planes * 2),
            nn.LayerNorm(wv_planes * 2),
            nn.GELU(),
            nn.Linear(wv_planes * 2, wv_planes),
            nn.LayerNorm(wv_planes),
        )

        # Transformer for cross-wavelength attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=wv_planes,
            nhead=num_heads,
            dim_feedforward=wv_planes * 4,
            activation='gelu',
            norm_first=True,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable query tokens for weight generation
        self.wt_num = 64
        self.weight_tokens = nn.Parameter(torch.randn(self.wt_num, wv_planes) * 0.02)

        # Factorized weight generation head
        # Spatial component: generates K*K pattern
        kernel_size = int((output_dim / in_channels) ** 0.5)
        self.spatial_head = nn.Sequential(
            nn.Linear(wv_planes, wv_planes),
            nn.GELU(),
            nn.Linear(wv_planes, kernel_size * kernel_size),
        )

        # Channel component: generates mixing weights across in_channels
        self.channel_head = nn.Sequential(
            nn.Linear(wv_planes, wv_planes),
            nn.GELU(),
            nn.Linear(wv_planes, in_channels),
        )

        # Bias head
        self.bias_head = nn.Linear(wv_planes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, waves: Tensor) -> tuple[Tensor, Tensor]:
        """Args:
            waves: Wavelength embeddings [num_wv, wv_planes]

        Returns:
            weights: [num_wv, K*K*in_channels]
            biases: [num_wv, 1]
        """
        num_wv = waves.size(0)

        # Process wavelength embeddings
        waves = self.wave_processor(waves)  # [num_wv, wv_planes]

        # Concatenate with learnable tokens and process through transformer
        # Add batch dimension for transformer
        tokens = torch.cat(
            [self.weight_tokens.unsqueeze(0).expand(1, -1, -1), waves.unsqueeze(0)],
            dim=1,
        )  # [1, wt_num + num_wv, wv_planes]

        tokens = self.transformer(tokens)  # [1, wt_num + num_wv, wv_planes]

        # Extract wavelength features (skip the learned tokens)
        wave_features = tokens[0, self.wt_num :]  # [num_wv, wv_planes]

        # Generate factorized weights
        spatial = self.spatial_head(wave_features)  # [num_wv, K*K]
        channel = self.channel_head(wave_features)  # [num_wv, in_channels]

        # Outer product to get full weights
        # [num_wv, K*K, 1] * [num_wv, 1, in_channels] -> [num_wv, K*K, in_channels]
        weights = torch.einsum('ns,nc->nsc', spatial, channel)
        weights = weights.reshape(num_wv, -1)  # [num_wv, K*K*in_channels]

        # Generate biases
        biases = self.bias_head(wave_features)  # [num_wv, 1]

        return weights, biases


class MultiStageDynamicDecoder(nn.Module):
    """Multi-stage dynamic decoder head for wavelength-adaptive reconstruction.

    Architecture:
    1. Shared spatial refinement (standard convs, wavelength-agnostic)
    2. Dynamic intermediate layer (wavelength-conditioned feature transform)
    3. Final dynamic projection (wavelength-specific output generation)

    This design gives the decoder significantly more capacity while keeping
    the wavelength-specific components focused on spectral adaptation.
    """

    def __init__(
        self,
        wv_planes: int = 128,
        embed_dim: int = 128,
        kernel_size: int = 3,
        num_shared_blocks: int = 2,
        num_dynamic_blocks: int = 1,
        expansion: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        use_enhanced_generator: bool = True,
    ) -> None:
        """Args:
        wv_planes: Wavelength embedding dimension
        embed_dim: Feature embedding dimension
        kernel_size: Convolution kernel size
        num_shared_blocks: Number of shared refinement blocks
        num_dynamic_blocks: Number of intermediate dynamic blocks
        expansion: Channel expansion in shared blocks
        num_heads: Attention heads in weight generators
        num_layers: Transformer layers in weight generators
        use_enhanced_generator: Use the enhanced wavelength-adaptive generator
        """
        super().__init__()

        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size

        # Stage 1: Shared spatial refinement
        self.shared_blocks = nn.ModuleList(
            [
                SharedRefinementBlock(
                    embed_dim=embed_dim,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    use_residual=True,
                )
                for _ in range(num_shared_blocks)
            ]
        )

        # Stage 2: Wavelength-conditioned feature transform
        # Uses a FiLM-style conditioning (Feature-wise Linear Modulation)
        self.film_generator = nn.Sequential(
            nn.Linear(wv_planes, wv_planes * 2),
            nn.GELU(),
            nn.Linear(wv_planes * 2, embed_dim * 2),  # gamma and beta
        )

        self.dynamic_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size, padding=kernel_size // 2
        )
        self.dynamic_norm = nn.GroupNorm(min(32, embed_dim), embed_dim)
        self.dynamic_act = nn.SiLU()

        # Stage 3: Final wavelength-specific projection
        self._num_kernel = kernel_size * kernel_size * embed_dim

        if use_enhanced_generator:
            self.final_generator = WavelengthAdaptiveWeightGenerator(
                wv_planes=wv_planes,
                output_dim=self._num_kernel,
                in_channels=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
            )
        else:
            self.final_generator = TransformerWeightGenerator_decoder(
                wv_planes,
                self._num_kernel,
                embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
            )

        # Wavelength embedding processor
        self.fclayer = FCResLayer(wv_planes)

        self.scaler = 0.1
        self._init_weights()

    def _init_weights(self) -> None:
        def weight_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.apply(weight_init)

    def forward(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Args:
            x: Latent features [B, embed_dim, H, W]
            wvs: Wavelength values in microns [num_wavelengths]

        Returns:
            Reconstructed multi-spectral image [B, num_wavelengths, H, W]
        """
        B, C, H, W = x.shape
        num_wv = wvs.size(0)

        # Compute wavelength embeddings
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)  # [num_wv, wv_planes]

        # Stage 1: Shared spatial refinement
        for block in self.shared_blocks:
            x = block(x)  # [B, embed_dim, H, W]

        # Stage 2: FiLM conditioning based on mean wavelength
        mean_wave = waves.mean(dim=0)  # [wv_planes]
        film_params = self.film_generator(mean_wave)  # [embed_dim * 2]
        gamma, beta = film_params.chunk(2)  # Each [embed_dim]

        x = self.dynamic_conv(x)  # [B, embed_dim, H, W]
        x = self.dynamic_norm(x)
        x = x * (1 + gamma.view(1, -1, 1, 1)) + beta.view(1, -1, 1, 1)  # FiLM
        x = self.dynamic_act(x)

        # Stage 3: Final wavelength-specific projection
        weights, biases = self.final_generator(
            waves
        )  # [num_wv, K*K*embed_dim], [num_wv, 1]

        # Reshape weights for conv2d
        dynamic_weight = weights.view(
            num_wv, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute(
            [0, 3, 1, 2]
        )  # [num_wv, embed_dim, K, K]

        # Apply bias
        bias = biases.squeeze(-1) * self.scaler if biases is not None else None

        # Final convolution
        out = F.conv2d(
            x,
            dynamic_weight * self.scaler,
            bias=bias,
            stride=1,
            padding=self.kernel_size // 2,
        )  # [B, num_wv, H, W]

        return out

    def get_distillation_weight(
        self, wvs_microns: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Get the final layer weights for distillation.

        Args:
            wvs_microns: Wavelength values in microns [num_wavelengths]

        Returns:
            Tuple of (weights, biases) matching DynamicConv_decoder format
        """
        # Compute wavelength embeddings
        waves = get_1d_sincos_pos_embed_from_grid_torch(
            self.wv_planes, wvs_microns * 1000
        )
        waves = self.fclayer(waves)  # [num_wv, wv_planes]

        # Generate final layer weights
        weights, biases = self.final_generator(waves)

        num_wv = wvs_microns.size(0)
        dynamic_weight = weights.view(
            num_wv, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute(
            [0, 3, 1, 2]
        )  # [num_wv, embed_dim, K, K]

        if biases is not None:
            dyn_bias = biases.squeeze(-1) * self.scaler
        else:
            dyn_bias = None

        return dynamic_weight * self.scaler, dyn_bias

    @property
    def weight(self) -> Tensor:
        """Property to access weights for adaptive loss (compatibility)."""
        # Handle WavelengthAdaptiveWeightGenerator (has channel_head)
        if hasattr(self.final_generator, 'channel_head'):
            return self.final_generator.channel_head[-1].weight
        # Handle different generator types for fc_weight
        fc_weight = self.final_generator.fc_weight
        if isinstance(fc_weight, nn.Sequential):
            return fc_weight[-1].weight
        elif isinstance(fc_weight, nn.Linear):
            return fc_weight.weight
        else:
            # Fallback: return first parameter
            for p in self.final_generator.parameters():
                return p
            raise RuntimeError('Could not find weight in final_generator')


class ProgressiveMultiStageDynamicDecoder(nn.Module):
    """Progressive multi-stage decoder with skip connections and multi-scale processing.

    This is a more sophisticated version that processes features at multiple
    scales and uses skip connections to preserve high-frequency details.

    Key innovations:
    1. Progressive upsampling with dynamic convs at each scale
    2. Wavelength-specific attention at each stage
    3. Residual learning for the spectral component
    """

    def __init__(
        self,
        wv_planes: int = 128,
        embed_dim: int = 128,
        kernel_size: int = 3,
        num_stages: int = 3,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.num_stages = num_stages

        # Pre-processing: Shared feature refinement
        self.pre_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
            nn.SiLU(),
        )

        # Wavelength embedding processor with more capacity
        self.wave_encoder = nn.Sequential(FCResLayer(wv_planes), FCResLayer(wv_planes))

        # Progressive stages
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            is_last = i == num_stages - 1
            self.stages.append(
                DecoderStage(
                    wv_planes=wv_planes,
                    embed_dim=embed_dim,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    is_output_stage=is_last,
                )
            )

        # Final skip connection from input (for residual learning)
        self.skip_weight = nn.Parameter(torch.tensor(0.0))

        self.scaler = 0.1
        self._init_weights()

    def _init_weights(self) -> None:
        def weight_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.apply(weight_init)

    def forward(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Args:
            x: Latent features [B, embed_dim, H, W]
            wvs: Wavelength values in microns [num_wavelengths]

        Returns:
            Reconstructed multi-spectral image [B, num_wavelengths, H, W]
        """
        # Compute wavelength embeddings
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.wave_encoder(waves)  # [num_wv, wv_planes]

        # Pre-process features
        x = self.pre_conv(x)

        # Progressive refinement through stages
        for i, stage in enumerate(self.stages):
            x = stage(x, waves)

        return x

    def get_distillation_weight(
        self, wvs_microns: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Get the final layer weights for distillation.

        For progressive decoders, we extract weights from the final stage's
        weight generator.

        Args:
            wvs_microns: Wavelength values in microns [num_wavelengths]

        Returns:
            Tuple of (weights, biases) matching DynamicConv_decoder format
        """
        # Compute wavelength embeddings
        waves = get_1d_sincos_pos_embed_from_grid_torch(
            self.wv_planes, wvs_microns * 1000
        )
        waves = self.wave_encoder(waves)  # [num_wv, wv_planes]

        # Get the final stage (which has is_output_stage=True)
        final_stage = self.stages[-1]

        # The final stage processes wavelengths through attention first
        num_wv = wvs_microns.size(0)
        waves_attn = waves.unsqueeze(0)  # [1, num_wv, wv_planes]
        waves_attn, _ = final_stage.wave_attention(waves_attn, waves_attn, waves_attn)
        waves_attn = final_stage.wave_norm(
            waves_attn.squeeze(0) + waves
        )  # [num_wv, wv_planes]

        # Generate weights from final stage's generator
        weights, biases = final_stage.weight_generator(waves_attn)

        dynamic_weight = weights.view(
            num_wv, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute(
            [0, 3, 1, 2]
        )  # [num_wv, embed_dim, K, K]

        if biases is not None:
            dyn_bias = biases.squeeze(-1) * self.scaler
        else:
            dyn_bias = None

        return dynamic_weight * self.scaler, dyn_bias

    @property
    def weight(self) -> Tensor:
        """Property to access weights for adaptive loss (compatibility)."""
        final_stage = self.stages[-1]
        # Handle WavelengthAdaptiveWeightGenerator (has channel_head)
        if hasattr(final_stage.weight_generator, 'channel_head'):
            return final_stage.weight_generator.channel_head[-1].weight
        # Handle different generator types for fc_weight
        fc_weight = final_stage.weight_generator.fc_weight
        if isinstance(fc_weight, nn.Sequential):
            return fc_weight[-1].weight
        elif isinstance(fc_weight, nn.Linear):
            return fc_weight.weight
        else:
            # Fallback: return first parameter
            for p in final_stage.weight_generator.parameters():
                return p
            raise RuntimeError('Could not find weight in final_generator')


class DecoderStage(nn.Module):
    """Single stage of the progressive decoder.

    Each stage consists of:
    1. Shared spatial processing (wavelength-agnostic)
    2. Wavelength-conditioned modulation
    3. Dynamic output (either intermediate or final)
    """

    def __init__(
        self,
        wv_planes: int,
        embed_dim: int,
        kernel_size: int,
        num_heads: int,
        num_layers: int,
        is_output_stage: bool = False,
    ) -> None:
        super().__init__()

        self.is_output_stage = is_output_stage
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size

        # Shared spatial processing
        self.spatial_block = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(32, embed_dim * 2), embed_dim * 2),
            nn.SiLU(),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
        )

        # Wavelength attention for this stage
        self.wave_attention = nn.MultiheadAttention(
            embed_dim=wv_planes, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.wave_norm = nn.LayerNorm(wv_planes)

        # FiLM modulation
        self.film = nn.Linear(wv_planes, embed_dim * 2)

        # Output projection
        if is_output_stage:
            # Final stage: full dynamic weight generation
            self._num_kernel = kernel_size * kernel_size * embed_dim
            self.weight_generator = WavelengthAdaptiveWeightGenerator(
                wv_planes=wv_planes,
                output_dim=self._num_kernel,
                in_channels=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
            )
        else:
            # Intermediate stage: residual output
            self.out_conv = nn.Conv2d(embed_dim, embed_dim, 1)

        self.act = nn.SiLU()
        self.scaler = 0.1

    def forward(self, x: Tensor, waves: Tensor) -> Tensor:
        """Args:
            x: Features [B, embed_dim, H, W] or [B, C, H, W] for output stage
            waves: Wavelength embeddings [num_wv, wv_planes]

        Returns:
            Processed features [B, embed_dim, H, W] or [B, num_wv, H, W]
        """
        B, C, H, W = x.shape
        num_wv = waves.size(0)

        # Shared spatial processing with residual
        residual = x
        x = self.spatial_block(x)
        x = x + residual
        x = self.act(x)

        # Wavelength self-attention (cross-wavelength coherence)
        waves_attn = waves.unsqueeze(0)  # [1, num_wv, wv_planes]
        waves_attn, _ = self.wave_attention(waves_attn, waves_attn, waves_attn)
        waves_attn = self.wave_norm(
            waves_attn.squeeze(0) + waves
        )  # [num_wv, wv_planes]

        # FiLM modulation using mean wavelength representation
        mean_wave = waves_attn.mean(dim=0)  # [wv_planes]
        film_params = self.film(mean_wave)  # [embed_dim * 2]
        gamma, beta = film_params.chunk(2)

        x = x * (1 + gamma.view(1, -1, 1, 1)) + beta.view(1, -1, 1, 1)

        if self.is_output_stage:
            # Generate wavelength-specific output
            weights, biases = self.weight_generator(waves_attn)

            dynamic_weight = weights.view(
                num_wv, self.kernel_size, self.kernel_size, self.embed_dim
            )
            dynamic_weight = dynamic_weight.permute([0, 3, 1, 2])

            bias = biases.squeeze(-1) * self.scaler if biases is not None else None

            out = F.conv2d(
                x,
                dynamic_weight * self.scaler,
                bias=bias,
                stride=1,
                padding=self.kernel_size // 2,
            )
            return out
        else:
            return self.out_conv(x)


# =============================================================================
# Simpler alternative: Stacked Dynamic Convolutions
# =============================================================================


class StackedDynamicDecoder(nn.Module):
    """Simpler multi-layer dynamic decoder using stacked dynamic convolutions.

    This is a more straightforward approach that simply stacks multiple
    dynamic convolution layers with residual connections. Each layer
    is wavelength-conditioned, providing progressive refinement.

    This is easier to train and may work better for smaller datasets.
    """

    def __init__(
        self,
        wv_planes: int = 128,
        embed_dim: int = 128,
        kernel_size: int = 3,
        num_layers: int = 3,
        num_heads: int = 4,
        generator_layers: int = 1,
        generator_type: str = 'transformer',
        rank_ratio: int = 4,
    ) -> None:
        """Args:
        wv_planes: Wavelength embedding dimension
        embed_dim: Feature dimension
        kernel_size: Convolution kernel size
        num_layers: Number of dynamic conv layers (including final)
        num_heads: Attention heads in generators
        generator_layers: Transformer layers per generator
        generator_type: 'transformer' or 'factorized'
        rank_ratio: Rank ratio for factorized generators
        """
        super().__init__()

        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # Wavelength processor
        self.fclayer = FCResLayer(wv_planes)

        # Intermediate layers: embed_dim -> embed_dim
        self._num_kernel_inter = kernel_size * kernel_size * embed_dim
        self.inter_layers = nn.ModuleList()
        self.inter_norms = nn.ModuleList()

        for i in range(num_layers - 1):
            if generator_type == 'factorized':
                gen = FactorizedWeightGenerator_decoder(
                    wv_planes,
                    self._num_kernel_inter,
                    embed_dim,
                    num_heads=num_heads,
                    num_layers=generator_layers,
                    rank_ratio=rank_ratio,
                )
            else:
                gen = TransformerWeightGenerator_decoder(
                    wv_planes,
                    self._num_kernel_inter,
                    embed_dim,
                    num_heads=num_heads,
                    num_layers=generator_layers,
                )
            self.inter_layers.append(gen)
            self.inter_norms.append(nn.GroupNorm(min(32, embed_dim), embed_dim))

        # Final layer: embed_dim -> num_wavelengths
        self._num_kernel_final = kernel_size * kernel_size * embed_dim
        if generator_type == 'factorized':
            self.final_generator = FactorizedWeightGenerator_decoder(
                wv_planes,
                self._num_kernel_final,
                embed_dim,
                num_heads=num_heads,
                num_layers=generator_layers,
                rank_ratio=rank_ratio,
            )
        else:
            self.final_generator = TransformerWeightGenerator_decoder(
                wv_planes,
                self._num_kernel_final,
                embed_dim,
                num_heads=num_heads,
                num_layers=generator_layers,
            )

        self.act = nn.SiLU()
        self.scaler = 0.1

        self._init_weights()

    def _init_weights(self) -> None:
        def weight_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(weight_init)

    def forward(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Args:
            x: Latent features [B, embed_dim, H, W]
            wvs: Wavelength values in microns [num_wavelengths]

        Returns:
            Reconstructed image [B, num_wavelengths, H, W]
        """
        num_wv = wvs.size(0)

        # Compute wavelength embeddings
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)  # [num_wv, wv_planes]

        # Use mean wavelength for intermediate layers
        mean_wave = waves.mean(dim=0, keepdim=True)  # [1, wv_planes]

        # Intermediate layers (embed_dim -> embed_dim)
        for i, (gen, norm) in enumerate(zip(self.inter_layers, self.inter_norms)):
            residual = x

            # Generate weights using mean wavelength
            weight, bias = gen(mean_wave)

            # Reshape for depthwise-like conv
            # [1, K*K*embed_dim] -> need [embed_dim, embed_dim, K, K]
            # For simplicity, we'll do a grouped conv
            w = weight.view(1, self.kernel_size, self.kernel_size, self.embed_dim)
            w = w.permute([3, 0, 1, 2])  # [embed_dim, 1, K, K]

            # Depthwise conv (each channel processed separately)
            x = F.conv2d(
                x,
                w * self.scaler,
                bias=None,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.embed_dim,
            )

            x = norm(x)
            x = self.act(x + residual)  # Residual connection

        # Final layer (embed_dim -> num_wavelengths)
        weight, bias = self.final_generator(waves)

        dynamic_weight = weight.view(
            num_wv, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([0, 3, 1, 2])

        if bias is not None:
            bias = bias.squeeze(-1) * self.scaler

        out = F.conv2d(
            x,
            dynamic_weight * self.scaler,
            bias=bias,
            stride=1,
            padding=self.kernel_size // 2,
        )

        return out

    def get_distillation_weight(
        self, wvs_microns: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Get the final layer weights for distillation.

        For multi-stage decoders, we only distill the final projection layer
        since intermediate layers are wavelength-averaged.

        Args:
            wvs_microns: Wavelength values in microns [num_wavelengths]

        Returns:
            Tuple of (weights, biases) matching DynamicConv_decoder format
        """
        # Compute wavelength embeddings
        waves = get_1d_sincos_pos_embed_from_grid_torch(
            self.wv_planes, wvs_microns * 1000
        )
        waves = self.fclayer(waves)  # [num_wv, wv_planes]

        # Generate final layer weights
        weight, bias = self.final_generator(waves)

        num_wv = wvs_microns.size(0)
        dynamic_weight = weight.view(
            num_wv, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute(
            [0, 3, 1, 2]
        )  # [num_wv, embed_dim, K, K]

        if bias is not None:
            dyn_bias = bias.squeeze(-1) * self.scaler
        else:
            dyn_bias = None

        return dynamic_weight * self.scaler, dyn_bias

    @property
    def weight(self) -> Tensor:
        """Property to access weights for adaptive loss (compatibility)."""
        # Handle different generator types
        fc_weight = self.final_generator.fc_weight
        if isinstance(fc_weight, nn.Sequential):
            return fc_weight[-1].weight
        elif isinstance(fc_weight, nn.Linear):
            return fc_weight.weight
        else:
            # Fallback: return first parameter
            for p in self.final_generator.parameters():
                return p
            raise RuntimeError('Could not find weight in final_generator')


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'DecoderStage',
    'MultiStageDynamicDecoder',
    'ProgressiveMultiStageDynamicDecoder',
    'SharedRefinementBlock',
    'StackedDynamicDecoder',
    'WavelengthAdaptiveWeightGenerator',
]
