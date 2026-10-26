from __future__ import annotations
# opensora/models/vae/eo_vae_wrapper.py
"""
EO-VAE 适配层：将 EOFluxVAE 包装成 Open-Sora AE 接口。

编解码接口（输入尺寸需为8的倍数）：
    encode_spatial_normalized(x, wvs) → (B, 32, H/8, W/8)
    decode_spatial_normalized(z, wvs) → (B, C, H, W)

设计原则：
    - 波长直接用微米单位传入（与评估脚本一致）
    - 不在 wrapper 内做任何 resize（由 Dataset/Dataloader 层负责）
    - no_grad 由调用者在外部控制，wrapper 内不添加装饰器
    - S2 输入需在 [0,1] 范围内（Dataset 层已完成 /10000 归一化）
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# S2 各波段中心波长（微米单位），与评估脚本保持严格一致
# EOFluxVAE accepts wavelengths in micrometers without extra normalization.
# ──────────────────────────────────────────────────────────────
S2_WAVELENGTHS_UM: Dict[str, float] = {
    'B02': 0.490,
    'B03': 0.560,
    'B04': 0.665,
    'B05': 0.705,
    'B06': 0.740,
    'B07': 0.783,
    'B08': 0.842,
    'B8A': 0.865,
    'B11': 1.610,   # 注意：非线性顺序，与评估脚本 cfg.wvs["s2"] 完全一致
    'B12': 2.190,   # 同上
}

# MMDC-EU 10波段默认顺序（与 eo_dataset.py S2_SPECTRAL_BANDS 严格对应）
DEFAULT_S2_BAND_NAMES: List[str] = [
    'B02', 'B03', 'B04', 'B05', 'B06',
    'B07', 'B08', 'B8A', 'B11', 'B12',
]

class _EOEncoderModule(nn.Module):
    """Encode images or videos through a compile-compatible nn.Module.

    The VAE and wavelength buffer are bound through ordinary attributes to
    avoid registering the same parameters twice in the parent wrapper.
    """

    def __init__(self):
        super().__init__()
        # vae 和 wavelengths 延迟绑定，避免注册为子模块导致参数重复
        self._vae = None
        self._wavelengths_ref = None  # 指向 EOVAEWrapper.wavelengths buffer
        self._vae_wrapper_target_dtype = None

    def bind(self, vae, wavelengths_buffer, target_dtype):
        """
        绑定 vae 和 wavelengths。
        使用普通赋值（不用 register_module/register_buffer），
        避免参数双重注册。
        """
        # 绕过 nn.Module.__setattr__，直接写入 __dict__
        object.__setattr__(self, '_vae', vae)
        object.__setattr__(self, '_wavelengths_ref', wavelengths_buffer)
        object.__setattr__(self, '_vae_wrapper_target_dtype', target_dtype)

    @staticmethod
    def _check_spatial_size(h: int, w: int, stride: int = 8) -> None:
        if h % stride != 0 or w % stride != 0:
            raise ValueError(
                f"[EOVAEWrapper] Input size ({h}×{w}) must be divisible by {stride}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整编码逻辑，与 EOVAEWrapper.encode 严格一致。

        Args:
            x : (B, C, H, W) 图像  或  (B, C, T, H, W) 视频
        Returns:
            z : (B, 32, H/8, W/8)  或  (B, 32, T, H/8, W/8)
        """
        # Keep VAE inputs in float32 for stable reconstruction.
        x_fp32 = x.to(torch.float32)
        is_video = (x_fp32.ndim == 5)


        if is_video:
            B, C, T, H, W = x_fp32.shape
            self._check_spatial_size(H, W)
            x_flat = x_fp32.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            B, C, H, W = x_fp32.shape
            self._check_spatial_size(H, W)
            x_flat = x_fp32
            T = 1

        # device/dtype 同步（编译后 callable 也需要）
        wv = self._wavelengths_ref.to(device=x_flat.device, dtype=x_flat.dtype)
        z_flat = self._vae.encode_spatial_normalized(x_flat, wv)

        if is_video:
            _, lC, lH, lW = z_flat.shape
            z = z_flat.reshape(B, T, lC, lH, lW).permute(0, 2, 1, 3, 4)
        else:
            z = z_flat

        # Cast encoded latents to the configured output dtype.
        return z.to(self._vae_wrapper_target_dtype) 


class EOVAEWrapper(nn.Module):
    """
    EOFluxVAE 的 Open-Sora 适配包装器。

    核心编解码接口：
        encode : encode_spatial_normalized(x, wvs) → (B, 32, H/8, W/8)
        decode : decode_spatial_normalized(z, wvs) → (B, C, H, W)

    重要约定：
        1. 输入 x 须已归一化到 [0, 1]（eo_dataset.py 已完成）
        2. 输入空间尺寸须为 8 的倍数（如 128, 256）
        3. no_grad 由调用者在外部控制
        4. resize 由 Dataset 层负责，wrapper 内不做任何插值

    Args:
        pretrained_path : 本地 checkpoint 目录（含 eo-vae.ckpt + model_config.yaml）
                          None → 从 HuggingFace 下载
        repo_id         : HuggingFace repo id
        ckpt_filename   : checkpoint 文件名
        config_filename : 配置文件名
        s2_band_names   : 使用的 S2 波段名列表，决定波长向量和输入通道数
                          须与 eo_dataset.py 中 S2_SPECTRAL_BANDS 严格对应
                          默认：MMDC-EU 的10个光谱波段
        freeze_encoder  : 是否冻结编码器（仅微调解码器时使用）
        device          : 模型加载设备
    """

    # EOFluxVAE architecture parameters.
    ENCODER_STRIDE:  int = 8   # 空间下采样倍率：H → H/8
    LATENT_CHANNELS: int = 32  # latent 通道数

    def __init__(
        self,
        pretrained_path:  Optional[str] = None,
        repo_id:          str = "nilsleh/eo-vae",
        ckpt_filename:    str = "eo-vae.ckpt",
        config_filename:  str = "model_config.yaml",
        s2_band_names:    Optional[List[str]] = None,
        freeze_encoder:   bool = False,
        device:           str = "cpu",
        **kwargs,   # Accept framework-provided device and dtype arguments.
    ):
        super().__init__()
        # 处理 device_map（train.py 传入的是实际 device 字符串）
        # device_map 优先级高于 device 参数
        if "device_map" in kwargs:
            device = kwargs.pop("device_map")
        # ── 保存 torch_dtype，用于 vae 转换 ──────────────────
        self._target_dtype = kwargs.pop("torch_dtype", None)
        if kwargs:
            logger.warning(
                f"[EOVAEWrapper] Ignoring unexpected kwargs: {list(kwargs.keys())}"
            )

        # ── 确定波段配置 ───────────────────────────────────────
        band_names = s2_band_names if s2_band_names is not None \
                     else DEFAULT_S2_BAND_NAMES

        # 校验波段名合法性
        unknown = [b for b in band_names if b not in S2_WAVELENGTHS_UM]
        if unknown:
            raise ValueError(
                f"[EOVAEWrapper] Unknown band name(s): {unknown}. "
                f"Valid names: {list(S2_WAVELENGTHS_UM.keys())}"
            )

        self.band_names: List[str] = band_names
        self.num_s2_channels: int = len(band_names)
        self.out_channels = self.num_s2_channels

        # ── 波长向量（微米，固定，不参与梯度） ───────────────
        # 与评估脚本 cfg.wvs["s2"] = [0.490, 0.560, ..., 2.190] 完全一致
        wv_um = [S2_WAVELENGTHS_UM[b] for b in band_names]
        self.register_buffer(
            "wavelengths",
            torch.tensor(wv_um, dtype=torch.float32),  # shape: (C,)
            persistent=False,
        )

        logger.info(
            f"[EOVAEWrapper] Band names : {band_names}\n"
            f"[EOVAEWrapper] Wavelengths: {wv_um} (μm)"
        )

        # ── 加载 EOFluxVAE ─────────────────────────────────────
        try:
            from eo_vae.models.new_autoencoder import EOFluxVAE
        except ImportError as e:
            raise ImportError(
                "eo_vae not found. Run: cd /path/to/eo-vae && pip install -e ."
            ) from e

        logger.info(
            f"[EOVAEWrapper] Loading EOFluxVAE | "
            f"pretrained_path={pretrained_path} | device={device}"
        )

        # Load the VAE in evaluation mode.
        self.vae = EOFluxVAE.from_pretrained(
            repo_id=repo_id,
            ckpt_filename=ckpt_filename,
            ckpt_path=pretrained_path,   # None → HF 下载；非 None → 本地目录
            config_filename=config_filename,
            device=device,
            eval_mode=True,              # Calls model.eval().
        )
        
        # Keep VAE weights in float32.
        self._target_dtype = kwargs.pop("torch_dtype", torch.float32)
        # 强制 VAE 及其权重保持 float32
        self.vae = self.vae.to(torch.float32)
        logger.info(f"[EOVAEWrapper] Forced vae to float32. DiT will use {self._target_dtype}.")

            
        # ── 可选：冻结编码器 ───────────────────────────────────
        if freeze_encoder:
            for p in self.vae.encoder.parameters():
                p.requires_grad_(False)
            logger.info("[EOVAEWrapper] Encoder frozen.")
        # ── 初始化 _EOEncoderModule ────────────────────────────
        # 注意：不能通过 nn.Module 的 register_module 注册， 否则 vae 的参数会被双重计入 EOVAEWrapper 的 parameters()
        # 改用 object.__setattr__ 绕过 nn.Module.__setattr__
        enc_module = _EOEncoderModule()
        enc_module.bind(self.vae, self.wavelengths, self._target_dtype)
        # 直接写入实例 __dict__，不触发 nn.Module.__setattr__
        object.__setattr__(self, '_eo_encoder_module', enc_module)
   
    # ──────────────────────────────────────────────────────────
    # nn.Module.__delattr__ 拦截：处理 del model_ae.decoder
    # ──────────────────────────────────────────────────────────

    def __delattr__(self, name: str) -> None:
        """
        拦截 del model_ae.decoder。

        The training script removes model_ae.decoder to save memory.

        nn.Module.__delattr__ 先查 _modules，'decoder' 不在其中
        （decoder 属于 self.vae._modules，不属于 self._modules），
        然后调用 super().__delattr__。在某些 PyTorch 版本中，
        property deleter 不能被 object.__delattr__ 可靠触发，
        因此在此处显式拦截。
        """
        if name == 'decoder':
            # 删除 vae 内部的 decoder 子模块，节省显存
            del self.vae.decoder
            logger.info("[EOVAEWrapper] vae.decoder deleted to save memory.")
        else:
            super().__delattr__(name)

    # ──────────────────────────────────────────────────────────
    # encoder property：供 train.py torch.compile 使用
    # ──────────────────────────────────────────────────────────

    @property
    def encoder(self) -> _EOEncoderModule:
        """
        返回封装了完整 encode 逻辑的 _EOEncoderModule。

        The training script compiles this module and assigns it to model_ae.encode.

        torch.compile 编译 _EOEncoderModule（一个 nn.Module），
        返回的 compiled callable 签名为 (x: Tensor) -> Tensor，
        被赋值给 model_ae.encode，覆盖原 encode 方法。

        后续 encode 调用直接进入 compiled _EOEncoderModule.forward(x)。
        """
        return self._eo_encoder_module


    @property
    def time_compression_ratio(self) -> int:
        """
        EO-VAE 是纯2D空间VAE，时序维度不压缩。
        prepare_visual_condition_uncausal 中用于计算 conditioned_t：
            conditioned_t = pixel_frames // time_compression_ratio
        ratio=1 → conditioned_t = pixel_frames（对于 2D VAE, latent帧数 = 像素帧数）
        """
        return 1

    @property  
    def cfg(self):
        """
        兼容 prepare_visual_condition_uncausal 中的
            C = model_ae.cfg.latent_channels
        """
        class _Cfg:
            latent_channels = EOVAEWrapper.LATENT_CHANNELS  # 32
        return _Cfg()

    # ──────────────────────────────────────────────────────────
    # 核心编解码接口
    # ──────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码接口（train.py 会用 compiled _EOEncoderModule 覆盖此方法）。

        直接委托给 _eo_encoder_module，逻辑与 _EOEncoderModule.forward 完全一致，
        保证覆盖前后行为相同。
        """
        return self._eo_encoder_module(x)
            # ──────────────────────────────────────────────────────────
    # 尺寸校验工具
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _check_spatial_size(h: int, w: int) -> None:
        """确保空间尺寸是 8 的倍数，否则 EO-VAE 内部卷积会产生尺寸错误。"""
        s = EOVAEWrapper.ENCODER_STRIDE
        if h % s != 0 or w % s != 0:
            raise ValueError(
                f"[EOVAEWrapper] Input spatial size ({h}x{w}) must be "
                f"divisible by {s}. Please resize in Dataset/Dataloader."
            )


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, 32, lH, lW) 或 (B, 32, T, lH, lW)
        Returns:
            x_rec : (B, C, H, W) 或 (B, C, T, H, W)
        """
        # Decode latents in float32.
        z_fp32 = z.to(torch.float32)
        is_video = (z_fp32.ndim == 5)

        if is_video:
            B, lC, T, lH, lW = z_fp32.shape
            z_flat = z_fp32.permute(0, 2, 1, 3, 4).reshape(B * T, lC, lH, lW)
        else:
            z_flat = z_fp32
            T = 1

        wv = self.wavelengths.to(device=z_flat.device, dtype=z_flat.dtype)
        x_rec_flat = self.vae.decode_spatial_normalized(z_flat, wv)

        if is_video:
            _, C, H, W = x_rec_flat.shape
            x_rec = x_rec_flat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        else:
            x_rec = x_rec_flat
        
        return x_rec.to(z.dtype)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整前向（encode → 采样 → decode + KL），训练 VAE 本身时使用。
        DiT 训练只用 encode()，不调用此方法。

        Args:
            x : (B, C, H, W) 或 (B, C, T, H, W)

        Returns:
            x_rec    : 重建结果，与 x 同形状
            kl_loss  : 标量，KL 散度均值
        """
        is_video = (x.ndim == 5)

        if is_video:
            B, C, T, H, W = x.shape
            self._check_spatial_size(H, W)
            x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            self._check_spatial_size(H, W)
            x_flat = x
            T = 1

        wv = self.wavelengths.to(device=x_flat.device, dtype=x_flat.dtype)

        # EOFluxVAE.forward() returns (reconstruction, posterior).
        # posterior 是 DiagonalGaussianDistribution，支持 .kl().mean()
        x_rec_flat, posterior = self.vae(x_flat, wv, sample_posterior=True)
        kl_loss = posterior.kl().mean()

        if is_video:
            _, C_out, H_out, W_out = x_rec_flat.shape
            x_rec = x_rec_flat.reshape(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
        else:
            x_rec = x_rec_flat

        return x_rec, kl_loss

    # ──────────────────────────────────────────────────────────
    # Open-Sora 工具接口
    # ──────────────────────────────────────────────────────────

    def get_latent_size(
        self, input_size: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """
        计算 latent 空间尺寸。
        EOFluxVAE 是纯 2D VAE，时序维 T 不压缩。

        Args:
            input_size : (T, H, W)，H/W 须为 8 的倍数

        Returns:
            (T, H // 8, W // 8)
        """
        T, H, W = input_size
        self._check_spatial_size(H, W)
        s = self.ENCODER_STRIDE  # 8
        return (T, H // s, W // s)

    @property
    def latent_channels(self) -> int:
        """latent 通道数（EOFluxVAE 固定为 32）。"""
        return self.LATENT_CHANNELS