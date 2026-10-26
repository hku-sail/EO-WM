from __future__ import annotations
'\nEarthNet2021 时序数据集，兼容 Open-Sora VideoTextDataset 接口。\n\nCSV 由 prepare_earthnet_meta.py 生成，每行对应一个滑动窗口。\n\nChannels:\nTrain (Blue, Green, Red, Near-Infrared, Sen2Cor Cloud Mask, ESA Scene Classification, EarthNet2021 Data Quality Mask)\nTest (Blue, Green, Red, Near-Infrared, EarthNet2021 Data Quality Mask)\n\nNPZ 结构（train集）:\n    highresdynamic : (128, 128, 7, 30)  float16\n        ch0: blue, ch1: green, ch2: red, ch3: nir   → 送入 EO-VAE\n        ch4: cloud (百分比 0~100)                   → 质量掩膜\n        ch5: scene (场景分类)                        → 可选条件\n        ch6: mask  (1=NaN帧无效, 0=有效帧)          → 质量掩膜\n    highresstatic  : (128, 128, 1)      float16     DEM，已归一化[0,1]\n    mesodynamic    : (80, 80, 5, 150)   float16     气象，已归一化[0,1]\n        ch0: precipitation, ch1: pressure,\n        ch2: temp_mean, ch3: temp_min, ch4: temp_max\n    mesostatic     : (80, 80, 1)        float16     meso DEM，已归一化[0,1]\n    \nNPZ 结构（test集, 分为 context 和 target）:\n# 📦 context_29SND_2019-02-20_2019-07-19_3129_3257_1081_1209_48_128_16_96.npz\n├── 📄 highresdynamic: shape=(128, 128, 5, 10), dtype=float16 | compression=gzip\n├── 📄 highresstatic: shape=(128, 128, 1), dtype=float16 | compression=gzip\n├── 📄 mesodynamic: shape=(80, 80, 5, 150), dtype=float16 | compression=gzip\n└── 📄 mesostatic: shape=(80, 80, 1), dtype=float16 | compression=gzip\n\n# 📦 target_29SND_2019-02-20_2019-07-19_3129_3257_1081_1209_48_128_16_96.npz\n├── 📁 contents/\n└── 📄 highresdynamic: shape=(128, 128, 5, 20), dtype=float16 | compression=gzip\n\n\n与 EOVideoDataset (eo_dataset.py) 接口完全对齐:\n    - 输出字段名、shape、dtype 完全一致\n    - E-OBS mesodynamic（5通道气象；沿用 era5_* 兼容字段名）\n    - DEM  → highresstatic（1通道高程）\n    - S2   → highresdynamic[:,:,:4,:]（4波段光学）\n'
import json
import logging
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .cumulative_stress import CUMSTRESS_VARIANTS, build_cumulative_stress_features

logger = logging.getLogger(__name__)
EN_BAND_NAMES = ['blue', 'green', 'red', 'nir']
EN_CLOUD_CH = 4
EN_SCENE_CH = 5
EN_MASK_CH = 6
EN_TEST_MASK_CH = 4
MESO_BAND_NAMES = ['precipitation', 'pressure', 'temp_mean', 'temp_min', 'temp_max']
EN_WAVELENGTHS = [0.49, 0.56, 0.665, 0.842]
_thread_local = threading.local()


def _get_npz_data(npz_path: str) -> Dict[str, np.ndarray]:
    """
    每个 DataLoader worker 线程独立缓存 npz 数据，
    避免重复 IO（同一 worker 可能多次访问同一文件）。
    注意：npz 文件较小(~几MB)，全量缓存可行；
    若内存紧张可去掉缓存，直接 np.load。
    """
    cache = getattr(_thread_local, 'npz_cache', {})
    if npz_path not in cache:
        if len(cache) > 32:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[npz_path] = dict(np.load(npz_path, allow_pickle=False))
        _thread_local.npz_cache = cache
    return cache[npz_path]


def _normalize_optional_path(path_like: Any) -> Optional[str]:
    """将CSV中的可选路径字段规范化为字符串；空值返回None。"""
    if path_like is None:
        return None
    if isinstance(path_like, float) and np.isnan(path_like):
        return None
    s = str(path_like).strip()
    if not s or s.lower() in {'nan', 'none'}:
        return None
    return s


def _infer_target_path_from_context(context_path: str) -> Optional[str]:
    """
    若 CSV 未给出 path_target，尝试从 context 路径推断对应 target 路径：
      .../context/.../context_xxx.npz -> .../target/.../target_xxx.npz
    """
    p = Path(context_path)
    name = p.name
    if not name.startswith('context_'):
        return None
    parts = list(p.parts)
    context_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == 'context':
            context_idx = i
            break
    if context_idx is None:
        return None
    parts[context_idx] = 'target'
    target_name = 'target_' + name[len('context_'):]
    target_path = Path(*parts[:-1]) / target_name
    if target_path.exists():
        return str(target_path)
    return None


def _merge_context_target_npz(context_npz: Dict[str, np.ndarray], target_npz: Dict[str, np.ndarray], context_path: str,
                              target_path: str) -> Dict[str, np.ndarray]:
    """
    将 test split 的 context/target 合并为一个“train-like”字典：
      - highresdynamic: 按时间维拼接
      - highresstatic/mesodynamic/mesostatic: 优先用 context
    """
    if 'highresdynamic' not in context_npz or 'highresdynamic' not in target_npz:
        raise KeyError(
            f"Missing 'highresdynamic' in context/target pair: context={context_path}, target={target_path}")
    ctx_hd = context_npz['highresdynamic']
    tgt_hd = target_npz['highresdynamic']
    if ctx_hd.ndim != 4 or tgt_hd.ndim != 4:
        raise ValueError(f'highresdynamic ndim must be 4, got ctx={ctx_hd.ndim}, tgt={tgt_hd.ndim}')
    if ctx_hd.shape[:3] != tgt_hd.shape[:3]:
        raise ValueError(
            f'Context/target highresdynamic shape mismatch at H/W/C dims: ctx={ctx_hd.shape}, tgt={tgt_hd.shape} (context={context_path}, target={target_path})'
        )
    merged: Dict[str, np.ndarray] = {'highresdynamic': np.concatenate([ctx_hd, tgt_hd], axis=-1)}
    for key in ('highresstatic', 'mesodynamic', 'mesostatic'):
        if key in context_npz:
            merged[key] = context_npz[key]
        elif key in target_npz:
            merged[key] = target_npz[key]
    return merged


def _get_npz_pair_data(context_path: str, target_path: str) -> Dict[str, np.ndarray]:
    """线程内缓存 context/target 合并结果，减少重复IO与拼接。"""
    pair_cache = getattr(_thread_local, 'npz_pair_cache', {})
    cache_key = f'{context_path}||{target_path}'
    if cache_key not in pair_cache:
        if len(pair_cache) > 32:
            oldest_key = next(iter(pair_cache))
            del pair_cache[oldest_key]
        context_npz = _get_npz_data(context_path)
        target_npz = _get_npz_data(target_path)
        pair_cache[cache_key] = _merge_context_target_npz(context_npz=context_npz,
                                                          target_npz=target_npz,
                                                          context_path=context_path,
                                                          target_path=target_path)
        _thread_local.npz_pair_cache = pair_cache
    return pair_cache[cache_key]


def _norm_optical(data: np.ndarray, normalize: bool) -> np.ndarray:
    """
    EarthNet 光学波段归一化。
    输入 data: (T, H, W, 4) float32，原始值域约[0,1]（已预归一化）。
    
    注意：
    - 原始数据已是反射率量级，直接 clip 到 [0,1] 即可
    - NaN 用 0 填充（对应无效像素，配合 quality_mask 使用）
    - 不同于 MMDC-EU 的 S2 需要 /10000
    """
    data = data.astype(np.float32)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    if normalize:
        np.clip(data, 0.0, 1.0, out=data)
    return data


def _norm_meso(data: np.ndarray, normalize: bool) -> np.ndarray:
    """
    mesodynamic 气象数据处理。
    输入 data: (T, H, W, 5) float32，原始已归一化到[0,1]。
    NaN 填 0，clip 到 [0,1]。
    说明：EarthNet 中 PP=0 常作为缺测提示，缺测掩膜在 _read_modalities 中单独处理。
    """
    data = data.astype(np.float32)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    if normalize:
        np.clip(data, 0.0, 1.0, out=data)
    return data


def _norm_dem(data: np.ndarray, normalize: bool) -> np.ndarray:
    """
    highresstatic DEM 处理。
    输入 data: (H, W, 1) 或 (H, W) float32，原始已归一化到[0,1]。
    """
    data = data.astype(np.float32)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    if normalize:
        np.clip(data, 0.0, 1.0, out=data)
    return data


def _parse_iso_date(value: Any) -> Optional[datetime]:
    """解析 YYYY-MM-DD 为 UTC datetime；失败返回 None。"""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return None
    s = s[:10]
    try:
        d = date.fromisoformat(s)
    except Exception:
        return None
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _extract_start_end_datetime(row: pd.Series) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    优先使用 CSV 的 start_date/end_date；
    若缺失则回退从 sample_id 或 path 文件名解析。
    """
    start_dt = _parse_iso_date(row.get('start_date'))
    end_dt = _parse_iso_date(row.get('end_date'))
    if start_dt is not None and end_dt is not None:
        return (start_dt, end_dt)
    sample_like = str(row.get('sample_id', '')).strip()
    if not sample_like:
        sample_like = Path(str(row.get('path', ''))).stem
    parts = sample_like.split('_')
    if len(parts) >= 3:
        s0 = _parse_iso_date(parts[1])
        s1 = _parse_iso_date(parts[2])
        if s0 is not None and s1 is not None:
            return (s0, s1)
    return (None, None)


def _build_uniform_time_axis(start_dt: Optional[datetime], end_dt: Optional[datetime], num_steps: int) -> np.ndarray:
    """
    构造长度为 num_steps 的统一时间轴（单位：天）。
    返回值与训练代码约定保持一致：自 1990-01-01 UTC 起的天数。
    - 有日期时：在 [start, end] 间线性均匀采样
    - 无日期时：退化为 0..num_steps-1
    """
    if num_steps <= 0:
        return np.zeros((0, ), dtype=np.float64)
    if start_dt is None or end_dt is None:
        return np.arange(num_steps, dtype=np.float64)
    ref = datetime(1990, 1, 1, tzinfo=timezone.utc)
    start_ts = float((start_dt - ref).total_seconds() / 86400.0)
    end_ts = float((end_dt - ref).total_seconds() / 86400.0)
    if num_steps == 1:
        return np.array([start_ts], dtype=np.float64)
    if end_ts < start_ts:
        start_ts, end_ts = (end_ts, start_ts)
    return np.linspace(start_ts, end_ts, num=num_steps, dtype=np.float64)


class EarthNetVideoDataset(Dataset):
    """EarthNet2021 windows with optical, terrain, location, and weather conditions."""
    bucket_class: str = 'Bucket'

    def __init__(self,
                 data_path: str,
                 load_meso: bool = True,
                 load_dem: bool = True,
                 normalize_optical: bool = True,
                 normalize_meso: bool = True,
                 normalize_dem: bool = True,
                 cloud_prob_thresh: float = 0.05,
                 era5_out_channels: int = 8,
                 resize_meso_to_highres: bool = True,
                 meso_ratio: int = 5,
                 min_frames: int = 1,
                 fps_max: float = 1.0,
                 tile_location_json: Optional[str] = None,
                 memory_efficient: bool = False,
                 use_cumstress: bool = False,
                 stress_variant: str = 'directional_product',
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.load_meso = load_meso
        self.load_dem = load_dem
        self.normalize_optical = normalize_optical
        self.normalize_meso = normalize_meso
        self.normalize_dem = normalize_dem
        self.cloud_prob_thresh = float(cloud_prob_thresh)
        if not 0.0 <= self.cloud_prob_thresh <= 1.0:
            raise ValueError(f'cloud_prob_thresh must be in [0, 1], got {self.cloud_prob_thresh}')
        self.era5_out_channels = int(era5_out_channels)
        if self.era5_out_channels <= 0:
            raise ValueError(f'era5_out_channels must be positive, got {self.era5_out_channels}')
        self.resize_meso_to_highres = bool(resize_meso_to_highres)
        self.meso_ratio = meso_ratio
        self.min_frames = min_frames
        self.fps_max = fps_max
        self.memory_efficient = memory_efficient
        self.use_cumstress = bool(use_cumstress)
        self.stress_variant = str(stress_variant)
        if self.stress_variant not in CUMSTRESS_VARIANTS:
            raise ValueError(
                f'Unknown stress_variant={self.stress_variant!r}; expected one of {sorted(CUMSTRESS_VARIANTS)}')
        self._tile_loc: Dict[str, Dict] = {}
        if tile_location_json is not None:
            self._load_tile_location(tile_location_json)
        self.meta_df = self._load_meta(data_path)
        self.data = self.meta_df
        self.era5_climatology_mean: Optional[torch.Tensor] = None
        self.era5_climatology_std: Optional[torch.Tensor] = None
        self.era5_climatology_global_mean: Optional[torch.Tensor] = None
        self.era5_climatology_global_std: Optional[torch.Tensor] = None
        self.climatology_statistics_split: Optional[str] = None
        self._climatology_path: Optional[str] = kwargs.get('climatology_path', None)
        if self._climatology_path is not None:
            self._load_era5_climatology(self._climatology_path)
        if self.use_cumstress and (not self.load_meso):
            raise ValueError('use_cumstress=True requires load_meso=True')
        if self.use_cumstress and self._climatology_path is None:
            raise ValueError('use_cumstress=True requires climatology_path')
        if self.use_cumstress and self.era5_climatology_mean is None:
            raise RuntimeError(
                f'use_cumstress=True but E-OBS climatology failed to load from {self._climatology_path}')
        logger.info(
            f"EarthNetVideoDataset: {len(self.meta_df)} windows | meso={load_meso}, dem={load_dem} | cumstress={self.use_cumstress}({(self.stress_variant if self.use_cumstress else 'off')}) | num_frames range: [{self.meta_df['num_frames'].min()}, {self.meta_df['num_frames'].max()}]"
        )
        if self.era5_climatology_mean is not None:
            logger.info(f'  E-OBS climatology loaded: {self.era5_climatology_mean.shape}')

    def _load_tile_location(self, json_path: str) -> None:
        import os, json as _json
        if not os.path.exists(json_path):
            logger.warning(f'Tile location json not found: {json_path}')
            return
        with open(json_path, 'r') as f:
            self._tile_loc = _json.load(f)
        logger.info(f'Loaded tile location index: {len(self._tile_loc)} entries')

    def _load_era5_climatology(self, climatology_path: str) -> None:
        """Load monthly training-split weather means and standard deviations."""
        clim_path = Path(climatology_path)
        if clim_path.is_dir():
            pt_file = clim_path / 'era5_climatology.pt'
        else:
            pt_file = clim_path if str(clim_path).endswith('.pt') else None
        if pt_file is not None and pt_file.exists():
            try:
                import torch
                clim_data = torch.load(pt_file, map_location='cpu')
                self.era5_tile_names = clim_data['tile_names']
                self.era5_tile_to_idx = clim_data['tile_to_idx']
                self.era5_climatology_mean = clim_data['climatology_mean']
                self.era5_climatology_std = clim_data['climatology_std']
                self.era5_climatology_global_mean = clim_data.get('climatology_mean_global', None)
                self.era5_climatology_global_std = clim_data.get('climatology_std_global', None)
                statistics_split = clim_data.get('statistics_split', None)
                if statistics_split is not None:
                    self.climatology_statistics_split = str(statistics_split)
                logger.info(f'Loaded per-tile E-OBS climatology from {pt_file}')
                logger.info(
                    f'  Tiles: {len(self.era5_tile_names)}, global fallback: {self.era5_climatology_global_mean is not None}'
                )
                return
            except Exception as e:
                logger.warning(f'Failed to load .pt climatology: {e}')
        npz_file = clim_path / 'era5_monthly_climatology.npz' if clim_path.is_dir() else clim_path
        if str(npz_file).endswith('.npz') and Path(npz_file).exists():
            try:
                data = np.load(npz_file, allow_pickle=False)
                mean = np.asarray(data['climatology_mean'], dtype=np.float32)
                std = np.asarray(data['climatology_std'], dtype=np.float32)
                if mean.ndim == 3 and mean.shape[1:] == (12, 5):
                    tile_names = [str(tile) for tile in data['tile_names'].tolist()]
                    self.era5_tile_names = tile_names
                    self.era5_tile_to_idx = {tile: index for index, tile in enumerate(tile_names)}
                    self.era5_climatology_mean = torch.from_numpy(mean)
                    self.era5_climatology_std = torch.from_numpy(std)
                    self.era5_climatology_global_mean = torch.from_numpy(
                        np.asarray(data['climatology_mean_global'], dtype=np.float32))
                    self.era5_climatology_global_std = torch.from_numpy(
                        np.asarray(data['climatology_std_global'], dtype=np.float32))
                    for field in ():
                        if field in data:
                            setattr(self, field, torch.from_numpy(np.asarray(data[field], dtype=np.float32)))
                    if 'statistics_split' in data:
                        self.climatology_statistics_split = str(data['statistics_split'].item())
                    logger.info(f'Loaded train-only per-tile E-OBS climatology from {npz_file}')
                elif mean.ndim == 4 and mean.shape[:2] == (12, 5):
                    self.era5_climatology_mean = torch.from_numpy(mean[None, ...])
                    self.era5_climatology_std = torch.from_numpy(std[None, ...])
                    self.era5_tile_names = ['global']
                    self.era5_tile_to_idx = {'global': 0}
                    logger.info(f'Loaded legacy E-OBS climatology (global only) from {npz_file}')
                else:
                    raise ValueError(f'Unsupported climatology_mean shape {mean.shape} in {npz_file}')
            except Exception as e:
                logger.warning(f'Failed to load .npz climatology: {e}')
        else:
            logger.warning(f'Climatology file not found: {clim_path}')

    @staticmethod
    def _tensor_to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, 'detach') and hasattr(value, 'cpu'):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _get_tile_coords(self, tile_id: str) -> Tuple[float, float]:
        """返回 (lon, lat)，查不到则返回 (nan, nan)"""
        info = self._tile_loc.get(str(tile_id), {})
        lon = float(info['lon']) if 'lon' in info else 1e309 - 1e309
        lat = float(info['lat']) if 'lat' in info else 1e309 - 1e309
        return (lon, lat)

    @staticmethod
    def _compute_frame_months(start_dt: Optional[datetime], end_dt: Optional[datetime],
                              meso_indices: Optional[np.ndarray], t_meso_total: int, T: int) -> List[int]:
        """
        Compute calendar month (1-12) for each of T ERA5 frames.

        Uses meso_indices to map each selected frame to its exact date within
        [start_dt, end_dt], then extracts the month.  This ensures that frames
        spanning Oct → Mar get correct per-frame climatology instead of all
        using the start month.

        Args:
            start_dt:      sample start date (from CSV / cube name)
            end_dt:        sample end date
            meso_indices:  selected meso frame indices, shape (T,)
            t_meso_total:  total number of meso timesteps in the NPZ
            T:             number of output frames (= len(meso_indices))

        Returns:
            List of T month values (1-12).
        """
        if start_dt is None or end_dt is None:
            fallback = start_dt.month if start_dt else 1
            return [fallback] * T
        total_days = (end_dt - start_dt).days
        if total_days <= 0:
            return [start_dt.month] * T
        if meso_indices is not None and t_meso_total > 1:
            months = []
            for mi in meso_indices:
                day_off = round(int(mi) * total_days / (t_meso_total - 1))
                frame_date = start_dt + timedelta(days=day_off)
                months.append(frame_date.month)
            return months
        else:
            return [(start_dt + timedelta(days=round(i * total_days / max(T - 1, 1)))).month for i in range(T)]

    def _compute_cumulative_stress(self,
                                   md_raw: np.ndarray,
                                   meso_indices: np.ndarray,
                                   tile_id: str,
                                   start_dt: Optional[datetime],
                                   end_dt: Optional[datetime],
                                   t_highres_total: Optional[int] = None,
                                   first_highres_index: Optional[int] = None) -> np.ndarray:
        """Accumulate daily heat and precipitation-deficit anomalies from the selected window boundary."""
        RR_IDX, TG_IDX = (0, 2)
        T_meso = md_raw.shape[-1]
        if t_highres_total is None or t_highres_total <= 0 or first_highres_index is None:
            raise ValueError(
                'CumSt requires t_highres_total and first_highres_index for an explicit highres-to-mesodynamic window boundary'
            )
        accumulation_start_index = int(round(int(first_highres_index) * T_meso / t_highres_total))
        accumulation_start_index = int(np.clip(accumulation_start_index, 0, max(T_meso - 1, 0)))
        meso_full = md_raw.transpose(3, 2, 0, 1).astype(np.float32)
        pressure = meso_full[:, 1, :, :]
        valid = (np.isfinite(pressure) & (pressure > 0.0)).astype(np.float32)
        np.nan_to_num(meso_full, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(meso_full, 0.0, 1.0, out=meso_full)
        if hasattr(self, 'era5_tile_to_idx') and tile_id in self.era5_tile_to_idx:
            tidx = self.era5_tile_to_idx[tile_id]
            clim_all = self._tensor_to_numpy(self.era5_climatology_mean[tidx])
            std_all = self._tensor_to_numpy(self.era5_climatology_std[tidx])
        elif self.era5_climatology_global_mean is not None:
            clim_all = self._tensor_to_numpy(self.era5_climatology_global_mean)
            std_all = self._tensor_to_numpy(self.era5_climatology_global_std)
        else:
            T_win = len(meso_indices)
            return np.zeros((T_win, 3), dtype=np.float32)
        frame_months = self._compute_frame_months(start_dt, end_dt, np.arange(T_meso), T_meso, T_meso)
        clim_dense = np.stack([clim_all[m - 1] for m in frame_months])
        std_dense = np.stack([std_all[m - 1] for m in frame_months])
        clim_4d = clim_dense[:, :, None, None]
        std_4d = std_dense[:, :, None, None]
        anom = (meso_full - clim_4d) * valid[:, None, :, :]
        anom_normed = anom / (std_4d + 1e-06)
        return build_cumulative_stress_features(meso_full=meso_full,
                                                meso_indices=meso_indices,
                                                valid_mask=valid,
                                                temp_mean_anomaly=anom_normed[:, TG_IDX, :, :],
                                                precip_anomaly=anom_normed[:, RR_IDX, :, :],
                                                accumulation_start_index=accumulation_start_index)

    def _select_timestamps_from_indices(self, row: pd.Series, indices: np.ndarray, total_steps: int) -> np.ndarray:
        """
        根据 row 的起止日期与全长 total_steps 构造统一时间轴，
        再按 indices 取对应时间戳（自 1990-01-01 起的天数）。
        """
        start_dt, end_dt = _extract_start_end_datetime(row)
        axis = _build_uniform_time_axis(start_dt, end_dt, total_steps)
        if axis.size == 0:
            return np.zeros((0, ), dtype=np.float64)
        idx = np.clip(indices.astype(np.int64), 0, axis.size - 1)
        return axis[idx]

    def to_efficient(self) -> None:
        """内存高效模式占位，与 EOVideoDataset.to_efficient() 接口对齐。"""
        if self.memory_efficient:
            logger.info(
                '[EarthNetVideoDataset] memory_efficient=True, but EarthNetVideoDataset uses plain CSV and does not support EfficientParquet. Skipping to_efficient().'
            )

    def __len__(self) -> int:
        return len(self.meta_df)

    def get_data_info(self, idx: int) -> Dict:
        """供 VariableVideoBatchSampler 调用，返回 bucket 分桶所需信息。"""
        row = self.meta_df.iloc[idx]
        return {
            'num_frames': int(row['num_frames']),
            'height': int(row['height']),
            'width': int(row['width']),
            'aspect_ratio': float(row['aspect_ratio']),
            'resolution': int(row['resolution']),
            'fps': float(row['fps'])
        }

    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        """
        支持 int 索引和 bucket sampler 的复合字符串索引
        格式: "{sample_idx}-{num_frames}-{height}-{width}"
        """
        if isinstance(idx, str):
            parts = idx.split('-')
            sample_idx = int(parts[0])
            num_frames = int(parts[1]) if len(parts) > 1 else None
            height = int(parts[2]) if len(parts) > 2 else None
            width = int(parts[3]) if len(parts) > 3 else None
        else:
            sample_idx = int(idx)
            num_frames = None
            height = None
            width = None
        row = self.meta_df.iloc[sample_idx]
        try:
            return self._load_window(row, num_frames=num_frames, height=height, width=width)
        except Exception as e:
            logger.warning(
                f"[EarthNetVideoDataset] Failed idx={idx} sample={row.get('sample_id', '?')} window={row.get('window_idx', '?')}: {e}",
                exc_info=True)
            return None

    def _load_window(self,
                     row: pd.Series,
                     num_frames: Optional[int] = None,
                     height: Optional[int] = None,
                     width: Optional[int] = None) -> Optional[Dict[str, Any]]:
        npz_path = str(row['path'])
        sample_id = str(row['sample_id'])
        target_path = _normalize_optional_path(row.get('path_target', None))
        if target_path is None:
            target_path = _infer_target_path_from_context(npz_path)
        in_indices = np.array(json.loads(row['in_indices']), dtype=np.int32)
        out_indices = np.array(json.loads(row['out_indices']), dtype=np.int32)
        all_indices = np.concatenate([in_indices, out_indices])
        expected_num_frames = int(row['num_frames'])
        target_frames = num_frames if num_frames is not None else expected_num_frames
        if len(all_indices) < target_frames:
            logger.warning(f'[SKIP] {sample_id}: need {target_frames} frames but only {len(all_indices)} available.')
            return None
        if target_path is not None:
            npz_data = _get_npz_pair_data(npz_path, target_path)
        else:
            npz_data = _get_npz_data(npz_path)
        tile_id = str(row.get('tile_id', ''))
        start_dt, end_dt = _extract_start_end_datetime(row)
        modalities = self._read_modalities(npz_data, all_indices, tile_id=tile_id, start_dt=start_dt, end_dt=end_dt)
        csv_h = int(row['height'])
        csv_w = int(row['width'])
        target_h = height if height is not None else csv_h
        target_w = width if width is not None else csv_w
        if target_h != csv_h or target_w != csv_w:
            modalities = self._resize_modalities(modalities, target_h, target_w)
        return self._build_output(modalities, row, in_indices, out_indices, target_h, target_w, target_frames)

    def _read_modalities(self,
                         npz_data: Dict[str, np.ndarray],
                         all_indices: np.ndarray,
                         tile_id: Optional[str] = None,
                         start_dt: Optional[datetime] = None,
                         end_dt: Optional[datetime] = None) -> Dict[str, Any]:
        """
        从 npz dict 中读取并归一化所有模态。
        all_indices 是在 T_highres(=30) 维度上的下标。
        """
        result: Dict[str, Any] = {}
        hd = npz_data['highresdynamic']
        result['t_highres_total'] = int(hd.shape[-1])
        n_ch = int(hd.shape[2])
        optical_raw = hd[:, :, :4, :][:, :, :, all_indices]
        optical_raw = optical_raw.transpose(3, 0, 1, 2)
        optical = _norm_optical(optical_raw, self.normalize_optical)
        result['optical'] = optical.transpose(0, 3, 1, 2)
        if n_ch >= 7:
            cloud_raw = hd[:, :, EN_CLOUD_CH, :][:, :, all_indices]
            cloud_raw = cloud_raw.astype(np.float32)
            cloud_frac = np.where(np.isnan(cloud_raw), 1.0, np.clip(cloud_raw / 100.0, 0.0, 1.0))
            result['cloud_frac'] = cloud_frac.transpose(2, 0, 1)
            mask_raw = hd[:, :, EN_MASK_CH, :][:, :, all_indices].astype(np.float32)
            mask_raw = np.where(np.isfinite(mask_raw), mask_raw, 1.0)
            quality = np.clip(1.0 - mask_raw, 0.0, 1.0)
            cloud_binary = (cloud_frac < self.cloud_prob_thresh).astype(np.float32)
            quality_combined = quality * cloud_binary
            result['quality_mask'] = quality_combined.transpose(2, 0, 1)
        elif n_ch >= 5:
            mask_raw = hd[:, :, EN_TEST_MASK_CH, :][:, :, all_indices].astype(np.float32)
            mask_raw = np.where(np.isfinite(mask_raw), mask_raw, 1.0)
            quality = np.clip(1.0 - mask_raw, 0.0, 1.0)
            result['cloud_frac'] = (1.0 - quality).transpose(2, 0, 1)
            result['quality_mask'] = quality.transpose(2, 0, 1)
        else:
            raise ValueError(f'Unsupported highresdynamic channel count: {n_ch}. Expected at least 5 channels.')
        if self.load_dem and 'highresstatic' in npz_data:
            dem_raw = npz_data['highresstatic']
            dem = _norm_dem(dem_raw, self.normalize_dem)
            result['dem'] = dem.transpose(2, 0, 1)
        if self.load_meso and 'mesodynamic' in npz_data:
            md = npz_data['mesodynamic']
            T_meso = md.shape[-1]
            T_highres = hd.shape[-1]

            def _map_highres_to_meso(indices: np.ndarray) -> np.ndarray:
                if T_highres > 0 and T_meso > 0 and (T_meso % T_highres == 0):
                    step = T_meso // T_highres
                    mapped = (indices.astype(np.int32) + 1) * step - 1
                elif T_highres > 1 and T_meso > 1:
                    scale = (T_meso - 1) / float(T_highres - 1)
                    mapped = np.rint(indices.astype(np.float32) * scale).astype(np.int32)
                else:
                    mapped = np.clip(indices * self.meso_ratio, 0, max(T_meso - 1, 0)).astype(np.int32)
                return np.clip(mapped, 0, T_meso - 1)

            meso_indices = _map_highres_to_meso(all_indices)
            weather_highres_indices = all_indices
            meso_source_indices = _map_highres_to_meso(weather_highres_indices)
            meso_raw = md[:, :, :, meso_source_indices]
            meso_raw = meso_raw.transpose(3, 0, 1, 2)
            pp_raw = meso_raw[..., 1]
            meso_valid_mask = (np.isfinite(pp_raw) & (pp_raw > 0.0)).astype(np.float32)
            meso = _norm_meso(meso_raw, self.normalize_meso)
            result['meso'] = meso.transpose(0, 3, 1, 2)
            result['meso_valid_mask'] = meso_valid_mask
            result['meso_indices'] = meso_indices
            result['t_meso_total'] = int(T_meso)
            if self.use_cumstress and self.era5_climatology_mean is not None and (tile_id is not None):
                result['cumstress'] = self._compute_cumulative_stress(md_raw=md,
                                                                      meso_indices=meso_indices,
                                                                      tile_id=tile_id,
                                                                      start_dt=start_dt,
                                                                      end_dt=end_dt,
                                                                      t_highres_total=T_highres,
                                                                      first_highres_index=int(all_indices[0]))
            if self.resize_meso_to_highres:
                h_hr, w_hr = (int(hd.shape[0]), int(hd.shape[1]))
                h_m, w_m = (int(result['meso'].shape[-2]), int(result['meso'].shape[-1]))
                if (h_m, w_m) != (h_hr, w_hr):
                    meso_t = torch.from_numpy(result['meso']).float()
                    meso_t = F.interpolate(meso_t, size=(h_hr, w_hr), mode='bilinear', align_corners=False)
                    result['meso'] = meso_t.numpy()
                    if 'meso_valid_mask' in result:
                        mask_t = torch.from_numpy(result['meso_valid_mask']).float().unsqueeze(1)
                        mask_t = F.interpolate(mask_t, size=(h_hr, w_hr), mode='nearest')
                        result['meso_valid_mask'] = mask_t.squeeze(1).numpy()
        return result

    def _resize_modalities(self, data: Dict[str, Any], height: int, width: int) -> Dict[str, Any]:

        def _resize_4d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
            """(T, C, H, W) → (T, C, h, w)"""
            t = torch.from_numpy(arr).float()
            t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
            return t.numpy()

        def _resize_3d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
            """(T, H, W) → (T, h, w)"""
            t = torch.from_numpy(arr).float().unsqueeze(1)
            t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
            return t.squeeze(1).numpy()

        def _resize_3d_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
            """(T, H, W) → (T, h, w)，最近邻（适合离散/掩膜）"""
            t = torch.from_numpy(arr).float().unsqueeze(1)
            t = F.interpolate(t, size=(h, w), mode='nearest')
            return t.squeeze(1).numpy()

        def _resize_2d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
            """(1, H, W) → (1, h, w)"""
            t = torch.from_numpy(arr).float().unsqueeze(0)
            t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
            return t.squeeze(0).numpy()

        if 'optical' in data:
            data['optical'] = _resize_4d(data['optical'], height, width)
        if 'cloud_frac' in data:
            data['cloud_frac'] = _resize_3d(data['cloud_frac'], height, width)
        if 'quality_mask' in data:
            data['quality_mask'] = _resize_3d_nearest(data['quality_mask'], height, width)
        if 'dem' in data:
            data['dem'] = _resize_2d(data['dem'], height, width)
        if 'meso' in data:
            data['meso'] = _resize_4d(data['meso'], height, width)
        if 'meso_valid_mask' in data:
            data['meso_valid_mask'] = _resize_3d_nearest(data['meso_valid_mask'], height, width)
        return data

    def _build_output(self, data: Dict[str, Any], row: pd.Series, in_indices: np.ndarray, out_indices: np.ndarray,
                      height: int, width: int, num_frames: int) -> Dict[str, Any]:
        optical = data['optical']
        T, C, H, W = optical.shape
        in_len = len(in_indices)
        out_len = len(out_indices)
        in_valid_frac = np.nan
        out_valid_frac = np.nan
        qmask_np = data.get('quality_mask', None)
        if isinstance(qmask_np, np.ndarray) and qmask_np.ndim == 3 and (qmask_np.shape[0] >= in_len + out_len):
            q_in = qmask_np[:in_len]
            q_out = qmask_np[in_len:in_len + out_len]
            if q_in.size > 0:
                in_valid_frac = float(np.nanmean(q_in))
            if q_out.size > 0:
                out_valid_frac = float(np.nanmean(q_out))
        if pd.isna(in_valid_frac):
            csv_in_valid = row.get('in_valid_frac', np.nan)
            if not pd.isna(csv_in_valid):
                in_valid_frac = float(csv_in_valid)
            else:
                in_cloud_frac = row.get('in_cloud_frac', np.nan)
                in_valid_frac = 1.0 - float(in_cloud_frac) if not pd.isna(in_cloud_frac) else 1.0
        if pd.isna(out_valid_frac):
            csv_out_valid = row.get('out_valid_frac', np.nan)
            if not pd.isna(csv_out_valid):
                out_valid_frac = float(csv_out_valid)
            else:
                out_cloud_frac = row.get('out_cloud_frac', np.nan)
                out_valid_frac = 1.0 - float(out_cloud_frac) if not pd.isna(out_cloud_frac) else 1.0
        in_valid_frac = float(np.clip(in_valid_frac, 0.0, 1.0))
        out_valid_frac = float(np.clip(out_valid_frac, 0.0, 1.0))
        output: Dict[str, Any] = {
            'video': torch.from_numpy(optical),
            'text': str(row['text']),
            'fps': float(row['fps']),
            'height': height,
            'width': width,
            'num_frames': num_frames,
            'in_video': torch.from_numpy(optical[:in_len]),
            'out_video': torch.from_numpy(optical[in_len:]),
            'in_len': in_len,
            'out_len': out_len,
            'sample_id': str(row['sample_id']),
            'window_idx': int(row['window_idx']),
            'year': str(row['year']),
            'in_valid_frac': in_valid_frac,
            'out_valid_frac': out_valid_frac
        }
        if 'benchmark_meta_json' in row.index:
            meta_json = row.get('benchmark_meta_json', None)
            if meta_json is not None and (not pd.isna(meta_json)):
                output['benchmark_meta_json'] = str(meta_json)
        for passthrough_key in ('noise_group_id', 'context_side', 'forcing_side', 'context_window_uid',
                                'forcing_window_uid'):
            if passthrough_key in row.index:
                value = row.get(passthrough_key, None)
                if value is not None and (not pd.isna(value)):
                    output[passthrough_key] = str(value)
        all_indices = np.concatenate([in_indices, out_indices]).astype(np.int32)
        t_highres_total = int(data.get('t_highres_total', max(int(all_indices.max()) + 1, 1)))
        s2_time = self._select_timestamps_from_indices(row, all_indices, t_highres_total)
        output['s2_time'] = torch.from_numpy(s2_time)
        output['quality_mask'] = torch.from_numpy(data['quality_mask'].astype(np.float32))
        if 'dem' in data:
            dem_t = torch.from_numpy(data['dem'])
            output['dem'] = dem_t
            output['dem_mask'] = (torch.abs(dem_t).sum(dim=0, keepdim=True) > 1e-05).float()
        if 'meso' in data:
            meso_t = torch.from_numpy(data['meso'])
            n_src = int(meso_t.shape[1])
            n_tgt = int(self.era5_out_channels)
            if n_src < n_tgt:
                pad_c = n_tgt - n_src
                pad = torch.zeros((meso_t.shape[0], pad_c, meso_t.shape[2], meso_t.shape[3]), dtype=meso_t.dtype)
                era5_t = torch.cat([meso_t, pad], dim=1)
                era5_names = MESO_BAND_NAMES + [f'pad_{i}' for i in range(pad_c)]
            else:
                era5_t = meso_t[:, :n_tgt]
                era5_names = MESO_BAND_NAMES[:n_tgt]
            if self.era5_climatology_mean is not None:
                tile_id = str(row.get('tile_id', ''))
                T_era5 = era5_t.shape[0]
                if hasattr(self, 'era5_tile_to_idx') and tile_id in self.era5_tile_to_idx:
                    tile_idx = self.era5_tile_to_idx[tile_id]
                    clim_all = self.era5_climatology_mean[tile_idx]
                    std_all = self.era5_climatology_std[tile_idx]
                elif self.era5_climatology_global_mean is not None:
                    clim_all = self.era5_climatology_global_mean
                    std_all = self.era5_climatology_global_std
                else:
                    clim_all = torch.zeros(12, 5, dtype=meso_t.dtype)
                    std_all = torch.ones(12, 5, dtype=meso_t.dtype)
                start_dt, end_dt = _extract_start_end_datetime(row)
                meso_idx_arr = data.get('meso_indices', None)
                t_meso_total = int(data.get('t_meso_total', T_era5))
                frame_months = self._compute_frame_months(start_dt, end_dt, meso_idx_arr, t_meso_total, T_era5)
                clim_per_frame = torch.stack([clim_all[m - 1] for m in frame_months])
                std_per_frame = torch.stack([std_all[m - 1] for m in frame_months])
                clim_4d = clim_per_frame.unsqueeze(-1).unsqueeze(-1)
                era5_anom = era5_t[:, :5, :, :] - clim_4d
                if 'meso_valid_mask' in data:
                    valid_mask = torch.from_numpy(data['meso_valid_mask'].astype(np.float32)).unsqueeze(1)
                    era5_anom = era5_anom * valid_mask
                output['era5_anom'] = era5_anom
                output['era5_clim'] = clim_per_frame
                output['era5_clim_std'] = std_per_frame
            if 'cumstress' in data:
                output['cumstress'] = torch.from_numpy(data['cumstress'].astype(np.float32))
            output['era5'] = era5_t
            output['era5_names'] = era5_names
            if 'meso_indices' in data:
                meso_indices = data['meso_indices'].astype(np.int32)
                t_meso_total = int(data.get('t_meso_total', max(int(meso_indices.max()) + 1, 1)))
                era5_time = self._select_timestamps_from_indices(row, meso_indices, t_meso_total)
                output['era5_time'] = torch.from_numpy(era5_time)
            if 'meso_valid_mask' in data:
                output['era5_mask'] = torch.from_numpy(data['meso_valid_mask'].astype(np.float32)).unsqueeze(1)
            else:
                output['era5_mask'] = (torch.abs(era5_t).sum(dim=1, keepdim=True) > 1e-05).float()
        lon, lat = self._get_tile_coords(str(row.get('tile_id', '')))
        output['lon'] = lon
        output['lat'] = lat
        return output

    def _load_meta(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        if 'tile_id' in df.columns:
            df['tile_id'] = df['tile_id'].fillna('').astype(str)
        required_cols = [
            'path', 'sample_id', 'text', 'num_frames', 'height', 'width', 'aspect_ratio', 'resolution', 'fps',
            'in_indices', 'out_indices', 'in_len', 'out_len', 'window_idx', 'in_cloud_frac', 'out_cloud_frac', 'year',
            'tile_id'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f'[EarthNetVideoDataset] CSV missing columns: {missing}\nAvailable: {df.columns.tolist()}')
        before = len(df)
        df = df[df['num_frames'] >= self.min_frames].reset_index(drop=True)
        filtered = before - len(df)
        if filtered > 0:
            logger.info(f'Filtered {filtered} rows with num_frames < {self.min_frames}')
        logger.info(
            f"[EarthNetVideoDataset] Loaded {len(df)} windows from {df['sample_id'].nunique()} unique samples.")
        return df


from opensora.registry import DATASETS

DATASETS.register_module(module=EarthNetVideoDataset)
