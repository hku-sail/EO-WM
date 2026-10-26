from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

random_seed = 1234
torch.manual_seed(random_seed)


waves_list = {
    'COASTAL_AEROSOL': 0.44,
    'BLUE': 0.49,
    'GREEN': 0.56,
    'RED': 0.665,
    'RED_EDGE_1': 0.705,
    'RED_EDGE_2': 0.74,
    'RED_EDGE_3': 0.783,
    'NIR_BROAD': 0.832,
    'NIR_NARROW': 0.864,
    'WATER_VAPOR': 0.945,
    'CIRRUS': 1.373,
    'SWIR_1': 1.61,
    'SWIR_2': 2.20,
    'THEMRAL_INFRARED_1': 10.90,
    'THEMRAL_INFRARED_12': 12.00,
    'VV': 5.405,
    'VH': 5.405,
    'ASC_VV': 5.405,
    'ASC_VH': 5.405,
    'DSC_VV': 5.405,
    'DSC_VH': 5.405,
    'VV-VH': 5.405,
}


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor) -> Tensor:
    """Generate sinusoidal positional embeddings.

    Args:
        embed_dim: Output dimension for each position (must be even)
        pos: Positions to be encoded [M,]

    Returns:
        Positional embeddings [M, D]
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    """Transformer-based dynamic weight generator.

    Generates weights and biases for dynamic convolutions using a transformer architecture.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        """Initialize the weight generator.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output weight matrix
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation='gelu',
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Generate dynamic weights and biases.

        Args:
            x: Input features [seq_len, batch, input_dim]

        Returns:
            tuple of:
                - Generated weights [seq_len, output_dim]
                - Generated bias [embed_dim]
        """
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias


class TransformerWeightGenerator_decoder(TransformerWeightGenerator):
    """Decoder version of transformer-based weight generator with modified bias generation.

    Inherits from TransformerWeightGenerator but modifies the bias generation to output
    a single channel.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        """Initialize the decoder weight generator.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output weight matrix
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super(TransformerWeightGenerator_decoder, self).__init__(
            input_dim, output_dim, embed_dim, num_heads=num_heads, num_layers=num_layers
        )
        self.fc_bias = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Generate weights and single-channel bias.

        Args:
            x: Input features [seq_len, batch, input_dim]

        Returns:
            tuple of:
                - Generated weights [seq_len, output_dim]
                - Generated bias [1]
        """
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[self.wt_num : -1]
            + self.bias_token.repeat((pos_wave.shape[0], 1))
        )  # Using the last output to generate bias
        return weights, bias


class FactorizedWeightGenerator(nn.Module):
    """Factorized Dynamic Weight Generator.
    Uses a Low-Rank bottleneck to reduce parameter count while maintaining capacity.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        rank_ratio: int = 4,  # Reduction factor for the bottleneck
    ) -> None:
        super().__init__()

        # 1. Deeper Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            activation='gelu',
            norm_first=True,  # Pre-norm is generally more stable for deep transformers
            batch_first=False,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # 2. Factorized Projection Head (Low-Rank)
        # Instead of one massive Linear(input, output), we use:
        # Linear(input, rank) -> GELU -> Linear(rank, output)
        rank = max(32, output_dim // rank_ratio)

        self.fc_weight = nn.Sequential(
            nn.Linear(input_dim, rank), nn.GELU(), nn.Linear(rank, output_dim)
        )

        self.fc_bias = nn.Linear(input_dim, embed_dim)

        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

        # Initialize the factorized head carefully
        self._init_head()

    def _init_head(self):
        # Initialize the last layer of the head to near-zero to start with identity-like behavior
        nn.init.xavier_uniform_(self.fc_weight[0].weight)
        nn.init.zeros_(self.fc_weight[-1].weight)
        nn.init.zeros_(self.fc_weight[-1].bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)

        transformer_output = self.transformer_encoder(x)

        # Extract features corresponding to wavelengths
        # Shape: [seq_len, batch, dim]
        features = transformer_output[self.wt_num : -1] + pos_wave

        # Generate weights via factorized head
        weights = self.fc_weight(features)

        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


class FactorizedWeightGenerator_decoder(FactorizedWeightGenerator):
    """Decoder version of Factorized Dynamic Weight Generator.
    Adapts the bias generation to output a single scalar per wavelength channel.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        rank_ratio: int = 4,
    ) -> None:
        super().__init__(
            input_dim, output_dim, embed_dim, num_heads, num_layers, rank_ratio
        )
        # Decoder specific: Bias is 1 per output channel (wavelength)
        # Overwrite the fc_bias from the parent class
        self.fc_bias = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)

        transformer_output = self.transformer_encoder(x)

        # Extract features corresponding to wavelengths
        # Shape: [seq_len, batch, dim]
        features = transformer_output[self.wt_num : -1] + pos_wave

        # Generate weights via factorized head
        weights = self.fc_weight(features)

        # Decoder specific bias generation:
        # We generate a bias value for EACH wavelength token
        # Logic matches TransformerWeightGenerator_decoder: add bias token to features
        bias_features = features + self.bias_token.repeat((pos_wave.shape[0], 1))
        bias = self.fc_bias(bias_features)

        return weights, bias


class Basic1d(nn.Module):
    """Basic 1D convolutional block with optional layer norm and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        """Initialize the 1D conv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include bias in linear layer
        """
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(conv)
        if not bias:
            self.conv.add_module('ln', nn.LayerNorm(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through conv block.

        Args:
            x: Input tensor [B, in_channels]

        Returns:
            Output tensor [B, out_channels]
        """
        out = self.conv(x)
        return out


class FCResLayer(nn.Module):
    """Fully connected residual layer with ReLU activations."""

    def __init__(self, linear_size: int = 128) -> None:
        """Initialize the FC residual layer.

        Args:
            linear_size: Size of linear layers
        """
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through residual layer.

        Args:
            x: Input tensor [B, linear_size]

        Returns:
            Output tensor [B, linear_size] with residual connection
        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class DynamicConv(nn.Module):
    """Dynamic convolution layer with wavelength-dependent kernels."""

    def __init__(
        self,
        wv_planes: int,
        inter_dim: int = 128,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        embed_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        generator_type: str = 'transformer',  # 'transformer' or 'factorized'
        rank_ratio: int = 4,  # Only used for factorized generator
    ) -> None:
        """Initialize dynamic convolution.

        Args:
            wv_planes: Dimension of wavelength features (Transformer d_model)
            inter_dim: Intermediate dimension size
            kernel_size: Size of convolution kernel
            stride: Convolution stride
            padding: Convolution padding
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers in weight generator
            num_heads: Number of attention heads in weight generator
            generator_type: Type of weight generator ('transformer' or 'factorized')
            rank_ratio: Reduction ratio for factorized generator bottleneck
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1
        self.stride = stride
        self.padding = padding
        self.generator_type = generator_type

        if generator_type == 'factorized':
            self.weight_generator = FactorizedWeightGenerator(
                wv_planes,
                self._num_kernel,
                embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                rank_ratio=rank_ratio,
            )
            self.use_weight_standardization = False
        else:
            self.weight_generator = TransformerWeightGenerator(
                wv_planes,
                self._num_kernel,
                embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.use_weight_standardization = False

        self.scaler = 0.1

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m: nn.Module) -> None:
        """Initialize weights of linear layers using Xavier initialization.

        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize base weights and dynamic MLP weights."""
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def weight_standardization(self, weight: Tensor, eps: float = 1e-5) -> Tensor:
        """Apply Weight Standardization to generated weights.

        Centers the weights and scales them to unit variance.
        Crucial for HyperNetworks to prevent signal explosion.
        """
        # weight shape: [Out, In, K, K]
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        var = weight.var(dim=[1, 2, 3], keepdim=True)
        weight = (weight - mean) / (torch.sqrt(var + eps))
        return weight

    def get_distillation_weight(self, wvs_microns: Tensor):
        # 1. Positional Embedding
        waves = get_1d_sincos_pos_embed_from_grid_torch(
            self.wv_planes, wvs_microns * 1000
        )
        # ── dtype 对齐：sincos 固定输出 fp32，需与模型权重对齐 ──
        waves = waves.to(dtype=wvs_microns.dtype)
        # ─────────────────────────────────────────────────────────

        waves = self.fclayer(waves)

        # 2. Generate Raw Parameters
        weight, bias = self._get_weights(waves)

        # 3. Process Weights (Same as before)
        inplanes = wvs_microns.size(0)
        dyn_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dyn_weight = dyn_weight.permute([3, 0, 1, 2])

        if self.use_weight_standardization:
            dyn_weight = self.weight_standardization(dyn_weight)

        # 4. Process Bias (Encoder)
        if bias is not None:
            dyn_bias = bias.view([self.embed_dim]) * self.scaler
        else:
            dyn_bias = None

        return dyn_weight * self.scaler, dyn_bias

    def forward(self, img_feat: Tensor, wvs: Tensor) -> Tensor:
        """Apply dynamic convolution.

        Args:
            img_feat: Input image features [B, C, H, W]
            wvs: Wavelength values [num_wavelengths]

        Returns:
            Convolved features [B, embed_dim, H', W']
        """
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = waves.to(dtype=wvs.dtype)  # dtype对齐
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if self.use_weight_standardization:
            dynamic_weight = self.weight_standardization(dynamic_weight)

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat,
            weights,
            bias=bias,
            stride=(self.stride, self.stride),
            padding=self.padding,
        )

        return dynamic_out


class DynamicConv_decoder(nn.Module):
    """Decoder version of dynamic convolution with modified bias handling."""

    def __init__(
        self,
        wv_planes: int,
        inter_dim: int = 128,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        generator_type: str = 'transformer',  # 'transformer' or 'factorized'
        rank_ratio: int = 4,  # Only used for factorized generator
    ) -> None:
        """Initialize decoder dynamic convolution.

        Args:
            wv_planes: Dimension of wavelength features
            inter_dim: Intermediate dimension size
            kernel_size: Size of convolution kernel
            stride: Convolution stride
            padding: Convolution padding
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers in weight generator
            num_heads: Number of attention heads in weight generator
            generator_type: Type of weight generator ('transformer' or 'factorized')
            rank_ratio: Reduction ratio for factorized generator bottleneck
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1
        self.stride = stride
        self.padding = padding
        self.generator_type = generator_type

        if generator_type == 'factorized':
            self.weight_generator = FactorizedWeightGenerator_decoder(
                wv_planes,
                self._num_kernel,
                embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                rank_ratio=rank_ratio,
            )
            self.use_weight_standardization = False
        else:
            self.weight_generator = TransformerWeightGenerator_decoder(
                wv_planes,
                self._num_kernel,
                embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.use_weight_standardization = False

        self.scaler = 0.1

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m: nn.Module) -> None:
        """Initialize weights of linear layers using Xavier initialization.

        Args:
            m:
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize base weights and dynamic MLP weights."""
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def weight_standardization(self, weight: Tensor, eps: float = 1e-5) -> Tensor:
        """Apply Weight Standardization.

        For Decoder: weight shape is [Out(Wavelengths), In(Embed), K, K]
        We normalize over [In, K, K] dimensions (1, 2, 3).
        """
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        var = weight.var(dim=[1, 2, 3], keepdim=True)
        weight = (weight - mean) / (torch.sqrt(var + eps))
        return weight

    def get_distillation_weight(self, wvs_microns: Tensor):
        # 1. Positional Embedding
        waves = get_1d_sincos_pos_embed_from_grid_torch(
            self.wv_planes, wvs_microns * 1000
        )
        waves = waves.to(dtype=wvs_microns.dtype)  # dtype对齐
        waves = self.fclayer(waves)

        # 2. Generate
        weight, bias = self._get_weights(waves)

        # 3. Process Weights
        inplanes = wvs_microns.size(0)
        dyn_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dyn_weight = dyn_weight.permute([0, 3, 1, 2])

        if self.use_weight_standardization:
            dyn_weight = self.weight_standardization(dyn_weight)

        # 4. Process Bias (Decoder)
        if bias is not None:
            dyn_bias = bias.squeeze() * self.scaler
        else:
            dyn_bias = None

        return dyn_weight * self.scaler, dyn_bias

    def forward(self, img_feat: Tensor, waves: Tensor) -> Tensor:
        """Apply decoder dynamic convolution.

        Args:
            img_feat: Input image features [B, C, H, W]
            waves: Wavelength values [num_wavelengths]

        Returns:
            Convolved features [B, num_wavelengths, H', W']
        """
        inplanes = waves.size(0)
        waves_type=waves.dtype
        # wv_feats: 9,128 -> 9, 3x3x3
        self.scaler = 0.1
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, waves * 1000)
        waves = waves.to(dtype=waves_type)  # dtype对齐
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        # small bug fixed
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([0, 3, 1, 2])

        if self.use_weight_standardization:
            dynamic_weight = self.weight_standardization(dynamic_weight)

        if bias is not None:
            bias = bias.squeeze() * self.scaler

        weights = dynamic_weight * self.scaler
        if bias is not None:
            bias = bias * self.scaler

        dynamic_out = F.conv2d(
            img_feat,
            weights,
            bias=bias,
            stride=(self.stride, self.stride),
            padding=self.padding,
        )

        # temporarily set current weight so it can be accesses
        self.weight = weights

        return dynamic_out


if __name__ == '__main__':
    embed_dim = 768
    # dconv = DynamicConv(wv_planes=128, inter_dim=128, kernel_size=3, embed_dim=embed_dim).cuda()
    inp = torch.randn([1, 3, 224, 224]).cuda()
    conv_in = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1).cuda()
    print(conv_in(inp).shape)
    inp = torch.randn([1, 8, 224, 224]).cuda()
    dconv_in = DynamicConv(
        wv_planes=128, inter_dim=128, kernel_size=3, stride=1, padding=1, embed_dim=128
    ).cuda()
    wvs = torch.FloatTensor([0.665, 0.56, 0.49, 0.665, 0.56, 0.49, 0.665, 0.56]).to(
        inp.device
    )
    print(dconv_in(inp, wvs).shape)

    inp = torch.randn([1, 128, 224, 224]).cuda()
    dconv_decoder = DynamicConv_decoder(
        wv_planes=128, inter_dim=128, kernel_size=3, stride=1, padding=1, embed_dim=128
    ).cuda()
    print(dconv_decoder(inp, wvs).shape)
