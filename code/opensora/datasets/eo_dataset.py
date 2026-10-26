from __future__ import annotations
# opensora/datasets/eo_dataset.py
"""
EO (Earth Observation) 多模态时序数据集
支持 Sentinel-2 主模态 + Sentinel-1/ERA5/DEM 辅助模态

CSV 由 prepare_eo_meta.py 生成，每行对应一个滑动窗口：
    - in_indices  : 输入帧的原始 H5 帧索引（JSON list）
    - out_indices : 目标帧的原始 H5 帧索引（JSON list）
    - num_frames  : in_len + out_len（固定，用于 bucket 分桶）
与 Open-Sora VideoTextDataset 接口兼容，
输出 dict 供 VariableVideoBatchSampler 和 DiT 训练使用。
"""

import json
import logging
import warnings
from typing import Any, Dict, List, Optional
import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 波段配置（与HDF5实际存储对齐）
# ──────────────────────────────────────────────────────────────

# S2: h5 中 s2 最后一维 shape[-1]=12
# indices 0-9: 光谱波段
S2_SPECTRAL_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06',
                     'B07', 'B08', 'B8A', 'B11', 'B12']   # 0~9
# indices 10-11: 太阳角
S2_ANGLE_BANDS    = ['sunAzimuthAngles', 'sunZenithAngles'] # 10~11
S2_ALL_BAND_NAMES = S2_SPECTRAL_BANDS + S2_ANGLE_BANDS      # 共12个

# S1: h5 中 s1_asc 最后一维 shape[-1]=3
S1_BAND_NAMES = ['VV', 'VH', 'local_incidence_angle']   # 0~2

# ERA5 变量列表（与HDF5 key对应）
ERA5_VAR_KEYS = [
    'era5_DEW_TEMP',
    'era5_PREC',
    'era5_SOL_RAD',
    'era5_TEMP_MAX',
    'era5_TEMP_MEAN',
    'era5_TEMP_MIN',
    'era5_VAP_PRESS',
    'era5_WIND_SPEED',
]

# ──────────────────────────────────────────────────────────────
# Tile 位置索引
# ──────────────────────────────────────────────────────────────
_TILE_LOCATION_INDEX: Dict[str, dict] = {}

def load_tile_location_index(json_path: str) -> None:  # 加载那个离线制作好的 tile_id-(lat,lont)对应json
    global _TILE_LOCATION_INDEX
    if not os.path.exists(json_path):
        logger.warning(f"Tile location index not found: {json_path}")
        return
    with open(json_path, "r") as f:
        _TILE_LOCATION_INDEX = json.load(f)
    logger.info(f"Loaded tile location index: {len(_TILE_LOCATION_INDEX)} entries")

def get_tile_coords(tile_id: str) -> Dict[str, float]:
    """返回 {'lon': float, 'lat': float}，查不到则返回 nan"""
    info = _TILE_LOCATION_INDEX.get(str(tile_id), {})
    return {
        "lon": float(info["lon"]) if "lon" in info else float("nan"),
        "lat": float(info["lat"]) if "lat" in info else float("nan"),
    }

# ──────────────────────────────────────────────────────────────
# ERA5 归一化配置
# ──────────────────────────────────────────────────────────────

class ERA5NormConfig:
    """
    ERA5 各变量的物理参数与归一化范围。

    VAR_SPECS 格式:
        fill       : 原始无效值标记
        div_factor : 物理反算除数
        offset     : 物理反算偏移（物理值 = raw / div_factor + offset）
        norm_min   : 归一化下界（物理单位）
        norm_max   : 归一化上界（物理单位）
    """

    VAR_SPECS = {
        # 降水：Raw值是 mm * 100 (假设scale是0.01)
        'era5_PREC':       {'fill': 32769, 'div_factor': 100.0, 'offset': 0.0, 'norm_min': 0.0, 'norm_max': 150.0}, 
        # 水汽压：单位 Pa。日常在 1000~6000 左右
        'era5_VAP_PRESS':  {'fill': 4294967296.0, 'div_factor': 1.0, 'offset': 0.0, 'norm_min': 0.0, 'norm_max': 15000.0},
        # 太阳辐射：单位 J/m²。日累积量
        'era5_SOL_RAD':    {'fill': 4294967296.0, 'div_factor': 1.0, 'offset': 0.0, 'norm_min': 0.0, 'norm_max': 35000000.0},
        # 温度类：Raw值是 Kelvin * 100, 转换逻辑： (Raw / 100) - 273.15 = Celsius
        'era5_TEMP_MEAN':  {'fill': 32769, 'div_factor': 100.0, 'offset': -273.15, 'norm_min': -50.0,  'norm_max': 50.0},
        'era5_TEMP_MIN':   {'fill': 32769, 'div_factor': 100.0, 'offset': -273.15, 'norm_min': -50.0,  'norm_max': 50.0},
        'era5_TEMP_MAX':   {'fill': 32769, 'div_factor': 100.0, 'offset': -273.15, 'norm_min': -50.0,  'norm_max': 50.0},
        'era5_DEW_TEMP':   {'fill': 32769, 'div_factor': 100.0, 'offset': -273.15, 'norm_min': -50.0,  'norm_max': 50.0},
        # 风速：Raw值是 m/s * 100, 转换逻辑： Raw / 100 = m/s
        'era5_WIND_SPEED': {'fill': 32769, 'div_factor': 100.0, 'offset': 0.0, 'norm_min': 0.0, 'norm_max': 50.0},
    }

# ──────────────────────────────────────────────────────────────
# 归一化工具函数
# ──────────────────────────────────────────────────────────────

def _norm_s2(data: np.ndarray, normalize: bool) -> np.ndarray:
    """
    S2数据归一化。
    前10通道为光谱波段（DN值0-10000），后2通道为太阳角度（度）。
    data shape: (T, H, W, C)，C 可为10或12。
    """
    S2_CLIP_MAX = 10000.0
    data = data.astype(np.float32)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=S2_CLIP_MAX, neginf=0.0)
    if not normalize:
        return data

    n_spectral = min(10, data.shape[-1])

    # 光谱波段：clip 到 [0, 10000] 然后 /10000
    spec = data[..., :n_spectral]
    np.clip(spec, 0.0, S2_CLIP_MAX, out=spec)
    spec /= S2_CLIP_MAX

    if data.shape[-1] > 10:
        # sunAzimuthAngles: [0, 360] -> /180  （与你原始实现保持一致）
        data[..., 10] = np.clip(data[..., 10], 0.0, 180.0) / 180.0
        # sunZenithAngles:  [0, 90]  -> /90
        data[..., 11] = np.clip(data[..., 11], 0.0, 90.0)  / 90.0

    return data


def _norm_s1(data: np.ndarray, normalize: bool) -> np.ndarray:
    """
    S1 linear-scale sigma0 归一化。
    实测 MMDC-EU 数据范围（基于p99截断，去除异常值）：
        VV    : p99 ≈ 0.60  → clip(0, 1.0) 覆盖99%正常值
        VH    : p99 ≈ 0.13  → clip(0, 0.5) 覆盖99%正常值，留余量
        Angle : [10, 78]    → clip(0, 90) / 90
    data shape: (T, H, W, 3)。
    """
    data = data.astype(np.float32)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    if not normalize:
        return data
    # VV: linear sigma0, clip at 1.0 (covers >99% of valid data)
    if data.shape[-1] >= 1:
        np.clip(data[..., 0], 0.0, 1.0, out=data[..., 0])
    # VH: linear sigma0, clip at 0.5
    if data.shape[-1] >= 2:
        np.clip(data[..., 1], 0.0, 0.5, out=data[..., 1])
        data[..., 1] /= 0.5   # → [0, 1]
    if data.shape[-1] == 3:
        np.clip(data[..., 2], 0.0, 90.0, out=data[..., 2])
        data[..., 2] /= 90.0
    return data


def _norm_dem(data: np.ndarray, normalize: bool) -> np.ndarray:
    """DEM 归一化 [0, 3000m]。data shape: (H, W)。"""
    data = data.astype(np.float32)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=3000.0, neginf=0.0)
    if not normalize:
        return data
    np.clip(data, 0.0, 3000.0, out=data)
    data /= 3000.0
    return data


def _norm_era5_channel(
    raw:      np.ndarray,
    var_name: str,         # ERA5_VAR_KEYS 中的 key，如 'era5_TEMP_MEAN'
    normalize: bool,
) -> np.ndarray:
    """
    对单个 ERA5 变量做物理反算 + 归一化。
    raw shape: (T, H, W)，dtype 任意，输出 float32。
    """
    spec = ERA5NormConfig.VAR_SPECS[var_name]

    fill = spec['fill']
    div = spec['div_factor']
    offset = spec['offset']
    n_min = spec['norm_min']
    n_max = spec['norm_max']

    raw = raw.astype(np.float32)

    # 标记无效值
    mask = (raw >= 1e9) if fill > 1e9 else (raw == fill)
    # 物理反算
    phys = raw / div + offset

    # 填充无效值 (这里实际不会触发,因为align+crop时已经把无效值变为0)
    fill_phys = 0.0
    # (
    #     0.0
    #     if var_name in ('era5_PREC', 'era5_SOL_RAD')
    #     else (n_min + n_max) / 2.0
    # )
    phys[mask] = fill_phys

    if not normalize:
        return phys

    np.clip(phys, n_min, n_max, out=phys)  # 这里对于raw=0的比如温度值,会从-273.15 clip到-50
    phys = (phys - n_min) / (n_max - n_min + 1e-8)
    return phys

import threading
_thread_local = threading.local()
def _get_h5_handle(h5_path: str) -> h5py.File:
    """每个 worker 线程复用同一个文件句柄，避免重复 open/close。"""
    cache = getattr(_thread_local, 'h5_cache', {})
    if h5_path not in cache:
        cache[h5_path] = h5py.File(h5_path, "r", swmr=True)
        _thread_local.h5_cache = cache
    return cache[h5_path]


# ──────────────────────────────────────────────────────────────
# 主 Dataset 类
# ──────────────────────────────────────────────────────────────

class EOVideoDataset(Dataset):
    """
    EO 多模态时序数据集，兼容 Open-Sora VideoTextDataset 接口。

    CSV 由 prepare_eo_meta.py（滑动窗口版）生成，每行对应一个窗口：
        in_indices  : 输入帧原始索引（JSON），质量过滤已在离线预处理完成
        out_indices : 目标帧原始索引（JSON），同上
        num_frames  : in_len + out_len（固定，供 bucket 分桶使用）

    Args:
        data_path          : CSV 元数据文件路径
        s2_bands           : 要使用的 S2 波段索引（相对于 12 维最后轴）
                             默认 None → 使用全部 10 个光谱波段 [0..9]
        include_s2_angles  : 是否额外附加 2 个角度通道（indices 10, 11）
        load_s1            : 是否加载 S1 辅助模态
        load_era5          : 是否加载 ERA5 条件数据
        load_dem           : 是否加载 DEM 静态数据
        normalize_s2       : 是否归一化 S2
        normalize_s1       : 是否归一化 S1
        normalize_era5     : 是否归一化 ERA5
        normalize_dem      : 是否归一化 DEM
        min_frames         : 过滤 num_frames 低于此值的行（粗过滤，离线已保证）
    """

    # ── Open-Sora VariableVideoBatchSampler 必需的类属性 ──────
    bucket_class: str = "Bucket"   # Required by VariableVideoBatchSampler.

    def __init__(
        self,
        data_path:         str,
        tile_location_json:    Optional[str] = None,
        s2_bands:          Optional[List[int]] = None,
        include_s2_angles: bool  = False,
        load_s1:           bool  = False,
        load_era5:         bool  = True,
        load_dem:          bool  = True,
        normalize_s2:      bool  = True,
        normalize_s1:      bool  = True,
        normalize_era5:    bool  = True,
        normalize_dem:     bool  = True,
        min_frames:        int   = 1,
        fps_max:           float = 1.0,   # Fixed sampling rate for EO sequences.
        memory_efficient:  bool  = False,   
        **kwargs,  
    ):
        super().__init__()

        # ── 保存配置 ──────────────────────────────────
        self.data_path         = data_path
        self.include_s2_angles = include_s2_angles
        self.load_s1           = load_s1
        self.load_era5         = load_era5
        self.load_dem          = load_dem
        self.normalize_s2      = normalize_s2
        self.normalize_s1      = normalize_s1
        self.normalize_era5    = normalize_era5
        self.normalize_dem     = normalize_dem
        self.min_frames        = min_frames
        self.fps_max = fps_max    # Used by sampler.py group_by_bucket.
        self.memory_efficient  = memory_efficient   
        if tile_location_json is not None:
            print(f'-----tile_location_json:{tile_location_json}------')
            load_tile_location_index(tile_location_json)

        # ── 确定 S2 波段索引 ──────────────────────────
        # 默认：全部10个光谱波段（不含角度）
        self.s2_spectral_indices: List[int] = (
            s2_bands if s2_bands is not None else list(range(10))
        )
        self.s2_angle_indices: List[int] = [10, 11] if include_s2_angles else []
        # 最终送入 EO-VAE 的波段索引（按顺序）
        self.s2_final_indices: List[int] = (
            self.s2_spectral_indices + self.s2_angle_indices
        )
        self.num_s2_channels: int = len(self.s2_final_indices)

        # ── 加载元数据 CSV ────────────────────────────
        self.meta_df = self._load_meta(data_path)
        # Expose metadata through the sampler interface.
        # sampler.py 中访问 self.dataset.data，必须是这个名字
        self.data = self.meta_df   # alias，供 sampler 访问

        logger.info(
            f"EOVideoDataset: {len(self.meta_df)} windows | "
            f"S2 channels={self.num_s2_channels} "
            f"({'spectral+angles' if include_s2_angles else 'spectral only'}) | "
            f"S1={load_s1}, ERA5={load_era5}, DEM={load_dem}"
        )

    # ──────────────────────────────────────────────
    # Open-Sora 接口
    # ──────────────────────────────────────────────

    # TextDataset-compatible metadata interface.
    def to_efficient(self):
        """
        内存高效模式占位实现，与 TextDataset.to_efficient() 接口对齐。
        EOVideoDataset 的元数据是纯 CSV（pandas DataFrame），
        不依赖 EfficientParquet 的分片 parquet 机制，
        故仅在 memory_efficient=True 时打印提示，不做实际转换。
        如果未来需要支持分片 parquet，可参考 TextDataset 的实现扩展。
        """
        if self.memory_efficient:
            logger.info(
                "[EOVideoDataset] memory_efficient=True, but EOVideoDataset "
                "uses plain CSV/DataFrame and does not support EfficientParquet. "
                "Skipping to_efficient() conversion."
            )
        # else: memory_efficient=False，直接跳过，无需任何操作

    def __len__(self) -> int:
        return len(self.meta_df)

    def get_data_info(self, idx: int) -> dict:
        """
        供 VariableVideoBatchSampler 调用，返回 bucket 分桶所需信息。
        num_frames = in_len + out_len，固定值，无需运行时计算。
        """
        row = self.meta_df.iloc[idx]
        return {
            "num_frames":   int(row["num_frames"]),
            "height":       int(row["height"]),
            "width":        int(row["width"]),
            "aspect_ratio": float(row["aspect_ratio"]),
            "resolution":   int(row["resolution"]),
            "fps":          float(row["fps"]),
        }

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        # ── 解析 bucket sampler 传入的复合 index ──────────────────
        # 格式："{sample_idx}-{num_frames}-{height}-{width}"
        # 例如："889-30-128-128"
        if isinstance(idx, str):
            parts      = idx.split("-")
            sample_idx = int(parts[0])
            num_frames = int(parts[1]) if len(parts) > 1 else None
            height     = int(parts[2]) if len(parts) > 2 else None
            width      = int(parts[3]) if len(parts) > 3 else None
        else:
            sample_idx = int(idx)
            num_frames = None
            height     = None
            width      = None
        # ──────────────────────────────────────────────────────────
        row = self.meta_df.iloc[sample_idx]
        try:
            return self._load_window(row, num_frames=num_frames,
                                    height=height, width=width)
        except Exception as e:
            logger.warning(
                f"[EOVideoDataset] Failed idx={idx} "
                f"sample={row.get('sample_id', '?')} "
                f"window={row.get('window_idx', '?')}: {e}",
                exc_info=True,
            )
            return None
    # ──────────────────────────────────────────────
    # 核心加载逻辑
    # ──────────────────────────────────────────────

    def _load_window(
        self,
        row:        pd.Series,
        num_frames: Optional[int] = None,   # bucket 指定帧数，None→用 CSV 值
        height:     Optional[int] = None,   # bucket 指定高度，None→用原始尺寸
        width:      Optional[int] = None,   # bucket 指定宽度， None→用原始尺寸
    ) -> Optional[Dict[str, Any]]:
        """
        根据 CSV 一行（一个滑动窗口）加载所有模态数据。

        in_indices / out_indices 已在离线预处理阶段完成质量过滤，
        此处直接使用，无需运行时二次过滤。
        """
        h5_path   = str(row["path"])
        sample_id = str(row["sample_id"])

        # ── 解析窗口索引（离线预处理已保证质量） ────────
        in_indices  = np.array(json.loads(row["in_indices"]),  dtype=np.int32)
        out_indices = np.array(json.loads(row["out_indices"]), dtype=np.int32)

        # 合并为完整窗口帧索引，保持时序顺序
        all_indices = np.concatenate([in_indices, out_indices])  # (in_len + out_len,)

        # 完整性校验
        expected_num_frames = int(row["num_frames"])  # in_len + out_len
        target_frames  = num_frames if num_frames is not None else expected_num_frames
        if len(all_indices) < target_frames:
            logger.warning(
                f"[SKIP] {sample_id}: need {target_frames} frames "
                f"but only {len(all_indices)} available."
            )
            return None
            

        # ── 打开 H5，读取各模态 ──────────────────────────
        # 每次 __getitem__ 新开文件句柄，与 DataLoader 多进程兼容
        f = _get_h5_handle(h5_path)          # 替换为这行
        grp  = f[f"samples/{sample_id}"]
        data = self._read_modalities(grp, all_indices)

        # ── 空间 resize（bucket 指定了与原始不同的 H/W） ─────────
        csv_h = int(row["height"])
        csv_w = int(row["width"])
        target_h = height if height is not None else csv_h
        target_w = width  if width  is not None else csv_w
        if target_h != csv_h or target_w != csv_w:
            data = self._resize_modalities(data, target_h, target_w)
        return self._build_output(data, row, in_indices, out_indices,
                                target_h, target_w, target_frames)

    def _resize_modalities(
        self,
        data:     Dict[str, Any],
        height:   int,
        width:    int,
    ) -> Dict[str, Any]:
        """
        对所有空间模态做双线性插值 resize。
        只处理 numpy array，torch resize 在 collate 后做代价更高。
        """
        import torch.nn.functional as F_

        def _resize_4d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
            """(T, C, H, W) numpy → resize → (T, C, h, w) numpy"""
            t = torch.from_numpy(arr).float()          # (T, C, H, W)
            t = F_.interpolate(t, size=(h, w),
                            mode='bilinear',
                            align_corners=False)    # (T, C, h, w)
            return t.numpy()

        def _resize_3d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
            """(T, H, W) numpy → resize → (T, h, w) numpy"""
            t = torch.from_numpy(arr).float().unsqueeze(1)  # (T, 1, H, W)
            t = F_.interpolate(t, size=(h, w),
                            mode='bilinear',
                            align_corners=False)         # (T, 1, h, w)
            return t.squeeze(1).numpy()

        def _resize_2d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
            """(1, H, W) 或 (H, W) numpy → resize → same leading dims"""
            if arr.ndim == 2:
                t = torch.from_numpy(arr).float()[None, None]  # (1,1,H,W)
                t = F_.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
                return t[0, 0].numpy()
            else:  # (1, H, W)
                t = torch.from_numpy(arr).float()[None]        # (1,1,H,W)
                t = F_.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
                return t[0].numpy()

        # s2: (T, C, H, W)
        if 's2' in data:
            data['s2'] = _resize_4d(data['s2'], height, width)

        # s2_clm / s2_data_mask: (T, H, W)
        for key in ('s2_clm', 's2_data_mask'):
            if key in data:
                data[key] = _resize_3d(data[key], height, width)

        # s1: (T, 3, H, W)
        if 's1' in data:
            data['s1'] = _resize_4d(data['s1'], height, width)

        # era5: (T, V, H, W)
        if 'era5' in data:
            data['era5'] = _resize_4d(data['era5'], height, width)

        # dem: (1, H, W)
        if 'dem' in data:
            data['dem'] = _resize_2d(data['dem'], height, width)

        return data

    def _read_modalities(
        self,
        grp:         h5py.Group,
        all_indices: np.ndarray,   # shape=(T_win,)，已排序的帧索引
    ) -> Dict[str, Any]:
        """从 H5 group 中读取并归一化所有模态。"""
        data: Dict[str, Any] = {}

        # ── S2 主模态 ─────────────────────────────────────
        # H5 存储: (T_total, H, W, 12)  float16
        s2_raw  = grp['s2'][all_indices].astype(np.float32)      # (T, H, W, 12)
        s2_norm = _norm_s2(s2_raw, self.normalize_s2)            # (T, H, W, 12)
        s2_sel  = s2_norm[:, :, :, self.s2_final_indices]        # (T, H, W, C_sel)
        data['s2'] = s2_sel.transpose(0, 3, 1, 2)               # (T, C_sel, H, W)

        # ── S2 质量掩膜（原始，供外部损失函数使用） ──────
        # CLM      : 1=有云, 0=晴空（uint16，值域0或1）
        # dataMask : 1=有效, 0=黑边（uint16，值域0或1）
        if 's2_CLM' in grp:
            # uint16 → float32，直接使用（值域已是0/1）
            data['s2_clm'] = grp['s2_CLM'][all_indices].astype(np.float32)
        if 's2_dataMask' in grp:
            data['s2_data_mask'] = grp['s2_dataMask'][all_indices].astype(np.float32)

        # ── S2 时间戳 ──────────────────────────────────────
        data['s2_time'] = grp['s2_time'][all_indices].astype(np.float64)  # (T,) Unix 时间戳

        # ── S1 辅助模态（可选） ────────────────────────────
        if self.load_s1:
            if 's1_asc' in grp:
                # H5 存储: (T_total, H, W, 3)  float32
                s1_raw  = grp['s1_asc'][all_indices].astype(np.float32)
                s1_norm = _norm_s1(s1_raw, self.normalize_s1)
                data['s1'] = s1_norm.transpose(0, 3, 1, 2)             # (T, 3, H, W)
                data['s1_time'] = grp['s1_asc_time'][all_indices].astype(np.float64)
            else:
                logger.debug(f"[{grp.name}] s1_asc not found, skipping S1.")

        # ── ERA5 气候条件（可选） ──────────────────────────
        if self.load_era5:
            era5_channels:  List[np.ndarray] = []
            era5_names_out: List[str] = []

            for var_key in ERA5_VAR_KEYS:
                if var_key not in grp:
                    print(f'Warning: can not found {var_key} in {grp}!')
                    continue
                # H5 存储: (T_total, H, W)  float16 / float32
                raw_var = grp[var_key][all_indices].astype(np.float32)  # (T, H, W)
                normed  = _norm_era5_channel(raw_var, var_key, self.normalize_era5)
                era5_channels.append(normed)
                era5_names_out.append(var_key)

            if era5_channels:
                data['era5']       = np.stack(era5_channels, axis=1)         # (T, V, H, W)
                data['era5_names'] = era5_names_out
                data['era5_time']  = grp['era5_time'][all_indices].astype(np.float64)

        # ── DEM 静态数据（可选） ──────────────────────────
        if self.load_dem:
            if 'dem' in grp:
                # H5 存储: (H, W)  float16
                dem_raw  = grp['dem'][:].astype(np.float32)           # (H, W)
                dem_norm = _norm_dem(dem_raw, self.normalize_dem)      # (H, W)
                data['dem'] = dem_norm[np.newaxis]                     # (1, H, W)
            else:
                logger.debug(f"[{grp.name}] dem not found, skipping DEM.")

        return data

    def _build_output(
        self,
        data:        Dict[str, Any],
        row:         pd.Series,
        in_indices:  np.ndarray,   # shape=(in_len,)，输入帧原始索引
        out_indices: np.ndarray,   # shape=(out_len,)，目标帧原始索引
        height:       Optional[int] = None,   
        width:        Optional[int] = None,   
        num_frames:   Optional[int] = None,   
    ) -> Dict[str, Any]:
        """
        将归一化后的 numpy 数组转换为 Tensor dict。

        核心字段（与 Open-Sora VideoTextDataset 对齐）：
            video      : Tensor (T, C, H, W)   完整窗口（in + out）
            text       : str
            fps        : float
            height     : int
            width      : int
            num_frames : int                   = in_len + out_len

        EO 专属字段：
            in_video   : Tensor (in_len,  C, H, W)  输入段
            out_video  : Tensor (out_len, C, H, W)  预测目标段
            in_len     : int
            out_len    : int
        """
        s2_arr = data['s2']               # (T, C, H, W)，T = in_len + out_len
        T, C, H, W = s2_arr.shape

        # 优先用动态值，fallback 到实际 shape
        H = height     if height     is not None else H
        W = width      if width      is not None else W
        T = num_frames if num_frames is not None else T


        in_len  = len(in_indices)
        out_len = len(out_indices)

        output: Dict[str, Any] = {
            # ── Open-Sora 核心字段 ──
            "video":      torch.from_numpy(s2_arr),
            "text":       str(row["text"]),
            "fps":        float(row["fps"]),
            "height":     H,
            "width":      W,
            "num_frames": T,
            # ── EO 窗口拆分 ──────────────────────────────
            # in_video / out_video 方便训练时直接取用，
            # 与 video 共享内存（view），无额外拷贝
            "in_video":   torch.from_numpy(s2_arr[:in_len]),   # (in_len,  C, H, W)
            "out_video":  torch.from_numpy(s2_arr[in_len:]),   # (out_len, C, H, W)
            "in_len":     in_len,
            "out_len":    out_len,
            # ── EO 元数据 ────────────────────────────────
            "sample_id":  str(row["sample_id"]),
            "window_idx": int(row["window_idx"]),
            "year":       str(row["year"]),
            "s2_time":    torch.from_numpy(data["s2_time"]),   # (T,)
            # ── 窗口质量统计（来自 CSV，无需运行时重算） ──
            "in_valid_frac":  float(row["in_cloud_frac"]),
            "out_valid_frac": float(row["out_cloud_frac"]),
        }

        # ── quality_mask: 1=有效, 0=无效，shape=(T, H, W) ──
        # 合并 CLM（云掩膜）和 dataMask（数据有效性）
        if 's2_clm' in data and 's2_data_mask' in data:
            # CLM: 1=云, 0=晴 → 取反；dataMask: 1=有效, 0=黑边
            clm_valid  = 1.0 - np.clip(data['s2_clm'], 0.0, 1.0)  # (T,H,W)
            data_valid = np.clip(data['s2_data_mask'], 0.0, 1.0)  # (T,H,W)
            qmask      = (clm_valid * data_valid).astype(np.float32)     # (T,H,W)
        elif 's2_clm' in data:
            qmask = (1.0 - np.clip(data['s2_clm'], 0.0, 1.0)).astype(np.float32)
        elif 's2_data_mask' in data:
            qmask = np.clip(data['s2_data_mask'], 0.0, 1.0).astype(np.float32)
        else:
            qmask = np.ones((T, H, W), dtype=np.float32)

        output["quality_mask"] = torch.from_numpy(qmask)   # (T, H, W)

        # ── S1 辅助模态 ───────────────────────────────
        if 's1' in data:
            output['s1']      = torch.from_numpy(data['s1'])       # (T, 3, H, W)
            output['s1_time'] = torch.from_numpy(data['s1_time'])  # (T,)

        # ── ERA5 条件数据 ─────────────────────────────
        if 'era5' in data:
            output['era5']       = torch.from_numpy(data['era5'])      # (T, V, H, W)
            output['era5_names'] = data['era5_names']                  # List[str]
            output['era5_time']  = torch.from_numpy(data['era5_time']) # (T,)

        # ── DEM 静态数据 ──────────────────────────────
        if 'dem' in data:
            output['dem'] = torch.from_numpy(data['dem'])  # (1, H, W)
        
        # —— 经纬度 ——
        tile_id_str = str(row.get("tile_id", ""))
        coords = get_tile_coords(tile_id_str)
        # 健壮性检查：如果拿到 nan 且 tile_id 并不为空，打印警告
        if tile_id_str != "" and (np.isnan(coords["lon"]) or np.isnan(coords["lat"])):
            # 只在 rank 0 打印一次或者限制打印频率
            if not hasattr(self, "_warned_tiles"): self._warned_tiles = set()
            if tile_id_str not in self._warned_tiles:
                print(f"[WARNING] Tile ID '{tile_id_str}' not found in location index! Using NaNs.")
                self._warned_tiles.add(tile_id_str)

        output["lon"] = coords["lon"]   # float，可能是 nan
        output["lat"] = coords["lat"]   # float，可能是 nan

        # —— 生成 DEM 和 ERA5 的有效性 Mask ——
        if 'dem' in data:
            dem_t = torch.from_numpy(data['dem'])
            # 绝对值求和>0认为是有效数据，[1, H, W]
            output['dem_mask'] = (torch.abs(dem_t).sum(dim=0, keepdim=True) > 1e-5).to(torch.float32)

        if 'era5' in data:
            era5_t = torch.from_numpy(data['era5'])
            # ERA5 在变量维度(dim=1)上求和，判断是否有效，[1, T, H, W]
            output['era5_mask'] = (torch.abs(era5_t).sum(dim=1, keepdim=True) > 1e-5).to(torch.float32)

        return output

    # ──────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────

    def _load_meta(self, data_path: str) -> pd.DataFrame:
        """加载并验证元数据 CSV。"""
        df = pd.read_csv(data_path)
        if "tile_id" in df.columns:
            # 先转 int 再转 str 比较稳妥，避免出现 .0
            df["tile_id"] = df["tile_id"].fillna(0).astype(np.int64).astype(str)

        # 必要列检查
        required_cols = [
            "path", "sample_id", "text",
            "num_frames", "height", "width", "aspect_ratio", "resolution", "fps",
            "in_indices", "out_indices",          # 新格式：窗口索引
            "window_idx",
            "in_len", "out_len",
            "in_cloud_frac", "out_cloud_frac",
            "year",
            "tile_id",   
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[EOVideoDataset] CSV missing required columns: {missing}\n"
                f"Available columns: {df.columns.tolist()}"
            )


        # 粗过滤：num_frames < min_frames 的行（离线处理已保证，此处作为安全网）
        before = len(df)
        df = df[df["num_frames"] >= self.min_frames].reset_index(drop=True)
        filtered = before - len(df)
        if filtered > 0:
            logger.info(
                f"[EOVideoDataset] Filtered {filtered} rows "
                f"with num_frames < {self.min_frames}"
            )

        logger.info(
            f"[EOVideoDataset] Loaded {len(df)} windows "
            f"from {df['sample_id'].nunique()} unique samples. "
            f"num_frames range: [{df['num_frames'].min()}, {df['num_frames'].max()}]"
        )
        return df

# Registry 注册（如果 DATASETS 在这里还未初始化，改到 eo_dataset.py 末尾）
from opensora.registry import DATASETS
DATASETS.register_module(module=EOVideoDataset)