from __future__ import annotations
'EO-WM forecasting and benchmark evaluation.\n\nUse eval_earthnet.sh for release settings. See ../../../README.md and --help.\n'
import argparse
import dataclasses
import json
import math
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.booster import Booster
from colossalai.utils import set_seed
from einops import rearrange
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
OPEN_SORA_ROOT = SCRIPT_PATH.parents[2]
if str(OPEN_SORA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_SORA_ROOT))
for parent in SCRIPT_PATH.parents:
    if (parent / 'utils' / 'earthnet_scores.py').exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
EARTHFORMER_SRC = SCRIPT_PATH.parents[3] / 'earth-forecasting-transformer' / 'src'
if EARTHFORMER_SRC.exists() and str(EARTHFORMER_SRC) not in sys.path:
    sys.path.insert(0, str(EARTHFORMER_SRC))
from opensora.registry import MODELS, build_module
from opensora.utils.config import read_config
from opensora.utils.misc import to_torch_dtype
from opensora.utils.optimizer import create_optimizer
from opensora.utils.sampling import pack, prepare
from opensora.utils.train import create_colossalai_plugin
from opensora.datasets.eo_collate_earthnet import collate_fn_eo_earthnet
from opensora.datasets.eo_dataset_earthnet import EarthNetVideoDataset

os.environ['AE_SPATIAL_COMPRESSION'] = '8'
VIS_SAMPLES = 20


class _NaNSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to None and numpy types to Python types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, )):
            return int(obj)
        if isinstance(obj, (np.floating, )):
            v = float(obj)
            return None if not math.isfinite(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, obj):
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj


try:
    from earthformer.datasets.earthnet.earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
    METRICS_AVAILABLE = True
    METRICS_IMPL = 'earthformer'
    print('[eval] EarthNet2021 metrics loaded from Earthformer')
except Exception:
    try:
        from utils.earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
        METRICS_AVAILABLE = True
        METRICS_IMPL = 'opensora-fallback'
    except Exception:
        METRICS_AVAILABLE = False
        METRICS_IMPL = 'none'


def compute_ndvi(x: torch.Tensor, eps: float = 1e-08) -> torch.Tensor:
    """Compute NDVI from 4-channel EarthNet data [B, G, R, NIR].

    Args:
        x: (C, T, H, W) or (N, C, T, H, W) with C >= 4.
    Returns:
        NDVI: (T, H, W) or (N, T, H, W), values in [-1, 1].
    """
    if x.ndim == 4:
        nir, red = (x[3], x[2])
    elif x.ndim == 5:
        nir, red = (x[:, 3], x[:, 2])
    else:
        raise ValueError(f'Expected 4D or 5D tensor, got {x.ndim}D')
    return (nir - red) / (nir + red + eps)


def compute_vegetation_mask(ndvi: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Boolean mask where NDVI >= threshold (vegetation pixels)."""
    return ndvi >= threshold


def load_master_weights(model, ckpt_dir: str, dtype: torch.dtype):
    """Load master.safetensors weights into model."""
    master_path = os.path.join(ckpt_dir, 'master.safetensors')
    assert os.path.exists(master_path), f'Cannot find {master_path}'
    print(f'[eval] Loading weights: {master_path}')
    state_dict = load_file(master_path, device='cpu')
    unwrapped = model.unwrap() if hasattr(model, 'unwrap') else model
    model_keys = set(unwrapped.state_dict().keys())
    match_rate = len(set(state_dict.keys()) & model_keys) / max(len(model_keys), 1)
    if match_rate < 0.5:
        for prefix in ('module.', 'model.', '_orig_mod.'):
            stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            rate = len(set(stripped.keys()) & model_keys) / max(len(model_keys), 1)
            if rate > match_rate:
                state_dict = stripped
                match_rate = rate
                print(f"[eval] Stripped prefix '{prefix}', match rate: {match_rate:.2%}")
                break
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[eval] missing ({len(missing)}): {missing[:3]}')
    if unexpected:
        print(f'[eval] unexpected ({len(unexpected)}): {unexpected[:3]}')
    print('[eval] Weights loaded')
    return unwrapped


def prepare_inputs_for_inference(batch: dict,
                                 cond_latents: list,
                                 model_t5,
                                 model_clip,
                                 cfg,
                                 total_t: int,
                                 lH: int,
                                 lW: int,
                                 device: torch.device,
                                 dtype: torch.dtype,
                                 seq_align: int = 1) -> Tuple[dict, torch.Tensor]:
    """Prepare conditioning inputs for V2V inference (identical to extreme_summer_bench)."""
    patch_size = cfg.get('patch_size', 2)
    in_len_cfg = cfg.get('condition_config', {}).get('in_len', None)
    bsz = len(cond_latents)
    lC = cond_latents[0].shape[0]
    inp: dict = {}
    lat = batch.get('lat', None)
    lon = batch.get('lon', None)
    s2_time = batch.get('s2_time', None)
    frame_mask = batch.get('frame_mask', None)
    if True and lat is not None and (lon is not None) and (s2_time is not None):
        lat = torch.nan_to_num(lat.to(device=device, dtype=torch.float32), nan=0.0)
        lon = torch.nan_to_num(lon.to(device=device, dtype=torch.float32), nan=0.0)
        s2_time = torch.nan_to_num(s2_time.to(device=device, dtype=torch.float32), nan=0.0)
        lat_rad = lat * math.pi / 180.0
        lon_rad = lon * math.pi / 180.0
        x_coord = torch.cos(lat_rad) * torch.cos(lon_rad)
        y_coord = torch.cos(lat_rad) * torch.sin(lon_rad)
        z_coord = torch.sin(lat_rad)
        t0 = s2_time[:, 0]
        doy = torch.remainder(t0, 365.25)
        sin_doy = torch.sin(2 * math.pi * doy / 365.25)
        cos_doy = torch.cos(2 * math.pi * doy / 365.25)
        year = 1990.0 + t0 / 365.25
        year_norm = 2.0 * ((year - 1950.0) / (2050.0 - 1950.0)) - 1.0
        eo_info = torch.stack([x_coord, y_coord, z_coord, sin_doy, cos_doy, year_norm], dim=-1)
        inp['eo_info'] = eo_info.to(device=device, dtype=dtype)
        doy_all = torch.remainder(s2_time, 365.25)
        sin_doy_all = torch.sin(2 * math.pi * doy_all / 365.25)
        cos_doy_all = torch.cos(2 * math.pi * doy_all / 365.25)
        year_all = 1990.0 + s2_time / 365.25
        year_norm_all = 2.0 * ((year_all - 1950.0) / (2050.0 - 1950.0)) - 1.0
        eo_time_info = torch.stack([sin_doy_all, cos_doy_all, year_norm_all], dim=-1)
        if frame_mask is not None:
            valid = frame_mask.to(device=device, dtype=eo_time_info.dtype).unsqueeze(-1)
            eo_time_info = eo_time_info * valid
        inp['eo_time_info'] = eo_time_info.to(device=device, dtype=dtype)
    dem = batch.get('dem', None)
    if dem is not None:
        inp['dem'] = dem.to(device, dtype)
        inp['dem_mask'] = batch['dem_mask'].to(device, dtype)
    era5 = batch.get('era5', None)
    if era5 is not None:
        inp['era5'] = era5.to(device, dtype)
        inp['era5_mask'] = batch['era5_mask'].to(device, dtype)
    era5_anom = batch.get('era5_anom', None)
    if era5_anom is not None:
        inp['era5_anom'] = era5_anom.to(device, dtype)
    era5_clim = batch.get('era5_clim', None)
    if era5_clim is not None:
        inp['era5_clim'] = era5_clim.to(device, dtype)
    cumstress = batch.get('cumstress', None)
    if cumstress is not None:
        inp['cumstress'] = cumstress.to(device, dtype)
    masks = torch.zeros(bsz, 1, total_t, lH, lW, device=device, dtype=dtype)
    latent = torch.zeros(bsz, lC, total_t, lH, lW, device=device, dtype=dtype)
    in_lens = batch.get('in_len', None)
    for i, cl in enumerate(cond_latents):
        if in_lens is not None:
            ti = int(in_lens[i].item())
        else:
            ti = int(in_len_cfg) if in_len_cfg is not None else int(cl.shape[1])
        ti = min(ti, cl.shape[1], total_t)
        latent[i, :, :ti] = cl[:, :ti].to(device, dtype)
        masks[i, :, :ti] = 1.0
    cond = torch.cat([masks, latent], dim=1)
    inp['cond'] = pack(cond, patch_size=patch_size)
    if in_lens is not None:
        inp['n_cond_frames'] = int(in_lens[0].item())
    elif in_len_cfg is not None:
        inp['n_cond_frames'] = int(in_len_cfg)
    x_1_spatial = torch.randn(bsz, lC, total_t, lH, lW, device=device, dtype=dtype)
    texts = batch['text']
    inp_text = prepare(model_t5, model_clip, x_1_spatial, prompt=texts, seq_align=seq_align, patch_size=patch_size)
    inp_text.pop('img', None)
    inp.update(inp_text)
    return (inp, x_1_spatial)


@torch.no_grad()
def euler_sample_v2v(model, inp_base, x_1_spatial, num_steps, guidance, patch_size, dtype, device):
    """Euler sampling for weather-conditioned Earth observation forecasts."""
    x_t_packed = pack(x_1_spatial, patch_size=patch_size)
    bsz = x_t_packed.shape[0]
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    for i in range(num_steps):
        t_cur = timesteps[i]
        dt = timesteps[i + 1] - t_cur
        inp_step = dict(inp_base)
        inp_step['img'] = x_t_packed
        inp_step['timesteps'] = t_cur.expand(bsz).to(dtype)
        inp_step['guidance'] = torch.full((bsz, ), guidance, device=device, dtype=dtype)
        with torch.autocast('cuda', dtype=dtype):
            v_pred = model(**inp_step)
        x_t_packed = x_t_packed + dt * v_pred
    return x_t_packed


def unpack_latent(x_packed, t, lH, lW, patch_size=2):
    """Unpack latent from sequence form to spatial form."""
    p = patch_size
    gH, gW = (lH // p, lW // p)
    return rearrange(x_packed, 'b (t h w) (c ph pw) -> b c t (h ph) (w pw)', t=t, h=gH, w=gW, ph=p, pw=p)


def s2_to_rgb(x: torch.Tensor, gamma: float = 0.7) -> np.ndarray:
    """Convert S2 tensor (C, H, W) to RGB uint8."""
    x_np = x.float().cpu().numpy()
    rgb = np.stack([x_np[2], x_np[1], x_np[0]], axis=-1)
    rgb = np.clip(rgb / 0.3, 0, 1)
    rgb = np.power(rgb, gamma)
    return (rgb * 255).astype(np.uint8)


def _sample_indices(t: int, n: int = 10) -> np.ndarray:
    if t <= 0:
        return np.zeros((0, ), dtype=int)
    return np.linspace(0, t - 1, min(n, t), dtype=int)


def _concat_frames(frames: list, gap: int = 2, gap_color: int = 22) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError('frames must be non-empty')
    if len(frames) == 1 or gap <= 0:
        return frames[0]
    h = frames[0].shape[0]
    sep = np.full((h, gap, 3), gap_color, dtype=np.uint8)
    items = [frames[0]]
    for frm in frames[1:]:
        items.extend([sep, frm])
    return np.concatenate(items, axis=1)


def _annotate_frame(rgb: np.ndarray, tag: str) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((2, 2, 48, 18), fill=(0, 0, 0))
    draw.text((5, 4), tag, fill=(255, 255, 0))
    return np.array(img)


def _render_strip(tensor: torch.Tensor, idx: np.ndarray, tag_prefix: str) -> np.ndarray:
    frames = []
    for tt in idx:
        rgb = s2_to_rgb(tensor[:, int(tt)])
        frames.append(_annotate_frame(rgb, f'{tag_prefix}{int(tt):02d}'))
    if len(frames) == 0:
        h, w = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        return np.zeros((h, w, 3), dtype=np.uint8)
    return _concat_frames(frames, gap=2, gap_color=24)


def _pad_width(img: np.ndarray, w: int, fill: int = 0) -> np.ndarray:
    if img.shape[1] >= w:
        return img
    pad = np.full((img.shape[0], w - img.shape[1], 3), fill, dtype=np.uint8)
    return np.concatenate([img, pad], axis=1)


def _with_row_label(row_img: np.ndarray, label: str, label_w: int = 240) -> np.ndarray:
    h, w = row_img.shape[:2]
    canvas = np.zeros((h, label_w + w, 3), dtype=np.uint8)
    canvas[:, :label_w] = np.array([18, 18, 18], dtype=np.uint8)
    canvas[:, label_w:] = row_img
    out = Image.fromarray(canvas)
    draw = ImageDraw.Draw(out)
    txt_bbox = draw.textbbox((0, 0), label)
    txt_h = max(1, txt_bbox[3] - txt_bbox[1])
    y = max(2, (h - txt_h) // 2)
    draw.text((10, y), label, fill=(235, 235, 235))
    return np.array(out)


def save_vis(sample_dir, in_s2, gt_s2, gen_s2, inv_mask, text, ensemble_preds=None):
    """Save visualization strips."""
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, 'prompt.txt'), 'w') as f:
        f.write(text)
    idx_in = _sample_indices(int(in_s2.shape[1]), n=10)
    idx_out = _sample_indices(int(gt_s2.shape[1]), n=10)
    row_in = _render_strip(in_s2, idx_in, 'in')
    row_gt = _render_strip(gt_s2, idx_out, 'f')
    row_gen = _render_strip(gen_s2, idx_out, 'f')
    Image.fromarray(row_in).save(os.path.join(sample_dir, 'input_strip.png'))
    Image.fromarray(row_gt).save(os.path.join(sample_dir, 'gt_strip.png'))
    Image.fromarray(row_gen).save(os.path.join(sample_dir, 'pred_mean_strip.png'))
    rows = [('Input context', row_in), ('GT future', row_gt), ('Pred (ensemble mean)', row_gen)]
    if ensemble_preds is not None:
        for k in range(min(3, len(ensemble_preds))):
            row_ens = _render_strip(ensemble_preds[k], idx_out, 'f')
            Image.fromarray(row_ens).save(os.path.join(sample_dir, f'pred_ens{k}_strip.png'))
            rows.append((f'Ensemble member {k}', row_ens))
    w_max = max((img.shape[1] for _, img in rows))
    labeled_rows = [_with_row_label(_pad_width(img, w_max), label) for label, img in rows]
    divider = np.full((4, labeled_rows[0].shape[1], 3), 95, dtype=np.uint8)
    canvas = [divider]
    for ridx, row in enumerate(labeled_rows):
        canvas.append(row)
        if ridx != len(labeled_rows) - 1:
            canvas.append(divider)
    Image.fromarray(np.concatenate(canvas, axis=0)).save(os.path.join(sample_dir, 'overview.png'))


def _normalize_optional_path(path_like):
    if path_like is None:
        return None
    if isinstance(path_like, float) and np.isnan(path_like):
        return None
    s = str(path_like).strip()
    if not s or s.lower() in {'nan', 'none'}:
        return None
    return s


def _infer_target_path_from_context(context_path: str):
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


def _cache_put(cache: dict, key: str, value, max_items: int):
    if key not in cache and len(cache) >= max_items:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _build_row_lookup_from_dataset(dataset: EarthNetVideoDataset) -> dict:
    """Build (sample_id, window_idx) -> entry lookup from dataset meta_df."""
    meta_df = getattr(dataset, 'meta_df', None)
    if meta_df is None:
        return {}
    lookup = {}
    for _, row in meta_df.iterrows():
        sample_id = str(row['sample_id'])
        window_idx = int(row['window_idx'])
        path = str(row['path'])
        path_target = _normalize_optional_path(row.get('path_target', None))
        if path_target is None:
            path_target = _infer_target_path_from_context(path)
        key = (sample_id, window_idx)
        lookup[key] = {
            'path': path,
            'path_target': path_target,
            'cache_key': f"{path}||{path_target or ''}",
            'in_indices': np.array(json.loads(row['in_indices']), dtype=np.int64),
            'out_indices': np.array(json.loads(row['out_indices']), dtype=np.int64)
        }
    return lookup


def _load_npz_cached(npz_path: str, npz_cache: dict, max_items: int = 64) -> dict:
    if npz_path not in npz_cache:
        data = dict(np.load(npz_path, allow_pickle=False))
        _cache_put(npz_cache, npz_path, data, max_items=max_items)
    return npz_cache[npz_path]


def _get_earthformer_mask_all_cached(entry: dict,
                                     npz_cache: dict,
                                     mask_cache: dict,
                                     max_items: int = 64) -> np.ndarray:
    cache_key = entry['cache_key']
    if cache_key in mask_cache:
        return mask_cache[cache_key]
    path = entry['path']
    path_target = entry['path_target']
    if path_target is not None:
        context_npz = _load_npz_cached(path, npz_cache)
        target_npz = _load_npz_cached(path_target, npz_cache)
        highresdynamic = np.concatenate([context_npz['highresdynamic'], target_npz['highresdynamic']], axis=-1)
    else:
        highresdynamic = _load_npz_cached(path, npz_cache)['highresdynamic']
    highresdynamic = highresdynamic.astype(np.float32, copy=False)
    highresdynamic = np.nan_to_num(highresdynamic, copy=True, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(highresdynamic, 0.0, 1.0, out=highresdynamic)
    n_ch = int(highresdynamic.shape[2])
    if n_ch >= 7:
        mask_ch = 6
    elif n_ch >= 5:
        mask_ch = 4
    else:
        raise ValueError(f'Unsupported highresdynamic channel count: {n_ch}')
    mask_all = highresdynamic[:, :, mask_ch, :]
    _cache_put(mask_cache, cache_key, mask_all, max_items=max_items)
    return mask_all


def _resize_mask_thw(mask_thw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if int(mask_thw.shape[-2]) == h and int(mask_thw.shape[-1]) == w:
        return mask_thw
    resized = F.interpolate(mask_thw.unsqueeze(1), size=(h, w), mode='nearest')
    return resized.squeeze(1)


def _fit_time(mask_thw: torch.Tensor, t: int, fill_value: float) -> torch.Tensor:
    cur_t = int(mask_thw.shape[0])
    if cur_t == t:
        return mask_thw
    if cur_t > t:
        return mask_thw[:t]
    pad = torch.full((t - cur_t, int(mask_thw.shape[1]), int(mask_thw.shape[2])), fill_value, dtype=mask_thw.dtype)
    return torch.cat([mask_thw, pad], dim=0)


def _build_valid_masks_from_earthformer_raw(entry: dict, npz_cache: dict, mask_cache: dict, in_len: int, out_len: int,
                                            h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (in_valid, out_valid) masks from raw earthformer quality channel."""
    invalid_all = _get_earthformer_mask_all_cached(entry, npz_cache, mask_cache)
    in_idx = entry['in_indices']
    out_idx = entry['out_indices']
    invalid_in = torch.from_numpy(invalid_all[:, :, in_idx].transpose(2, 0, 1).astype(np.float32))
    invalid_out = torch.from_numpy(invalid_all[:, :, out_idx].transpose(2, 0, 1).astype(np.float32))
    invalid_in = _fit_time(_resize_mask_thw(invalid_in, h, w), in_len, fill_value=1.0)
    invalid_out = _fit_time(_resize_mask_thw(invalid_out, h, w), out_len, fill_value=1.0)
    in_valid = (1.0 - invalid_in).unsqueeze(0).to(device=device, dtype=torch.float32)
    out_valid = (1.0 - invalid_out).unsqueeze(0).to(device=device, dtype=torch.float32)
    return (in_valid, out_valid)


def parse_seasonal_benchmark_meta(meta_json_str: str) -> dict:
    """Parse benchmark_meta_json for seasonal paired benchmark.

    Returns dict with:
        pair_id, sample_side ("a"/"b"), paired_window_uid,
        year_index, window_start_frame_in_year, final_rank,
        n_tracks, track_labels,
        D_init, D_weather_future, D_heat_future,
        D_outcome_future, D_outcome_future_shift_tolerant, D_pixel_divergence
    """
    raw = json.loads(meta_json_str)

    def _float(key, default=float('nan')):
        v = raw.get(key, default)
        if v is None:
            return float('nan')
        try:
            f = float(v)
            return f if np.isfinite(f) else float('nan')
        except (ValueError, TypeError):
            return float('nan')

    def _int(key, default=0):
        v = raw.get(key, default)
        if v is None:
            return default
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default

    return {
        'pair_id': str(raw.get('pair_id', '')),
        'sample_side': str(raw.get('sample_side', '')),
        'paired_window_uid': str(raw.get('paired_window_uid', '')),
        'year_index': _int('year_index'),
        'window_start_frame_in_year': _int('window_start_frame_in_year'),
        'final_rank': _int('final_rank'),
        'n_tracks': _int('n_tracks'),
        'track_labels': str(raw.get('track_labels', '')),
        'D_init': _float('D_init'),
        'D_weather_future': _float('D_weather_future'),
        'D_heat_future': _float('D_heat_future'),
        'D_outcome_future': _float('D_outcome_future'),
        'D_outcome_future_shift_tolerant': _float('D_outcome_future_shift_tolerant'),
        'D_pixel_divergence': _float('D_pixel_divergence')
    }


def _nanmean_axis0(arr: np.ndarray) -> np.ndarray:
    """np.nanmean along axis=0, suppressing RuntimeWarning for all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(arr, axis=0)


@dataclasses.dataclass
class WindowSummary:
    """Per-window prediction summary computed during inference.

    NOTE: pred_spatial is NOT stored here to avoid OOM during all_gather_object.
    Spatial predictions are saved per-rank to disk separately.
    """
    window_uid: str
    ndvi_pred_trajectory: np.ndarray
    ndvi_gt_trajectory: np.ndarray
    pixel_pred_mean: np.ndarray
    pixel_gt_mean: np.ndarray
    mask_valid_ratio: np.ndarray
    pixel_mae: float
    ndvi_mae: float


def compute_window_summary(pred_ensemble: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor,
                           window_uid: str) -> Tuple[WindowSummary, torch.Tensor]:
    """Compute WindowSummary from ensemble predictions and GT.

    Args:
        pred_ensemble: (N, C, T_out, H, W) ensemble predictions (CPU tensors).
        gt: (C, T_out, H, W) ground truth (CPU tensor).
        valid_mask: (1, T_out, H, W) validity mask (1=valid, CPU tensor).
        window_uid: unique window identifier (= sample_id from inference CSV).

    Returns:
        (WindowSummary, pred_mean) where pred_mean is (C, T_out, H, W) for
        optional spatial saving by the caller (kept out of WindowSummary to
        avoid OOM during all_gather_object).
    """
    N, C, T_out, H, W = pred_ensemble.shape
    mask = valid_mask.squeeze(0)
    pred_mean = pred_ensemble.mean(dim=0)
    gt_ndvi = compute_ndvi(gt)
    veg_mask = compute_vegetation_mask(gt_ndvi)
    combined_mask = mask * veg_mask.float()
    n_veg_per_frame = combined_mask.sum(dim=(-2, -1))
    n_veg_safe = n_veg_per_frame.clamp(min=1)
    gt_ndvi_masked_sum = (gt_ndvi * combined_mask).sum(dim=(-2, -1))
    ndvi_gt_traj = (gt_ndvi_masked_sum / n_veg_safe).numpy().astype(np.float64)
    ndvi_gt_traj[n_veg_per_frame.numpy() == 0] = np.nan
    pred_ndvi_all = compute_ndvi(pred_ensemble)
    pred_ndvi_masked_sum = (pred_ndvi_all * combined_mask.unsqueeze(0)).sum(dim=(-2, -1))
    ndvi_pred_traj = (pred_ndvi_masked_sum / n_veg_safe.unsqueeze(0)).numpy().astype(np.float64)
    ndvi_pred_traj[:, n_veg_per_frame.numpy() == 0] = np.nan
    n_pix_per_frame = mask.sum(dim=(-2, -1))
    n_pix_safe = n_pix_per_frame.clamp(min=1)
    gt_masked_sum = (gt * mask.unsqueeze(0)).sum(dim=(-2, -1))
    pixel_gt_mean = (gt_masked_sum / n_pix_safe.unsqueeze(0)).permute(1, 0).numpy().astype(np.float64)
    pixel_gt_mean[n_pix_per_frame.numpy() == 0, :] = np.nan
    pred_masked_sum = (pred_ensemble * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))
    pixel_pred_mean = (pred_masked_sum / n_pix_safe.unsqueeze(0).unsqueeze(0)).permute(0, 2,
                                                                                       1).numpy().astype(np.float64)
    pixel_pred_mean[:, n_pix_per_frame.numpy() == 0, :] = np.nan
    mask_valid_ratio = mask.float().mean(dim=(-2, -1)).numpy().astype(np.float64)
    diff = (pred_mean - gt) * mask.unsqueeze(0)
    n_valid = mask.sum() * C
    pixel_mae = diff.abs().sum().item() / n_valid.item() if n_valid > 0 else float('nan')
    ndvi_pred_mean_map = compute_ndvi(pred_mean)
    ndvi_diff = (ndvi_pred_mean_map - gt_ndvi) * combined_mask
    n_ndvi_valid = combined_mask.sum()
    ndvi_mae = ndvi_diff.abs().sum().item() / n_ndvi_valid.item() if n_ndvi_valid > 0 else float('nan')
    summary = WindowSummary(window_uid=window_uid,
                            ndvi_pred_trajectory=ndvi_pred_traj,
                            ndvi_gt_trajectory=ndvi_gt_traj,
                            pixel_pred_mean=pixel_pred_mean,
                            pixel_gt_mean=pixel_gt_mean,
                            mask_valid_ratio=mask_valid_ratio,
                            pixel_mae=pixel_mae,
                            ndvi_mae=ndvi_mae)
    return (summary, pred_mean)


class PairMetricComputer:
    """Computes paired comparison metrics across all benchmark pairs.

    Metrics:
        DRR  — Divergence Reproduction Ratio (magnitude fidelity)
        DHR  — Directional Hit Rate (sign fidelity)
        PDC  — Paired Divergence Correlation (ranking fidelity, Spearman)
    """
    DRR_NOISE_THR = 0.02
    DHR_NOISE_THR = 0.02
    TRACKS = ['weather', 'heat', 'outcome_shift', 'pixel']

    def __init__(self, benchmark_pairs_csv: str, window_summaries: Dict[str, WindowSummary]):
        self.pairs_df = pd.read_csv(benchmark_pairs_csv)
        self.summaries = window_summaries

    def compute_all(self) -> Tuple[dict, List[dict]]:
        """Compute all metrics. Returns (overall_metrics_dict, per_pair_records_list)."""
        per_pair_records: List[dict] = []
        n_skipped = 0
        for _, row in self.pairs_df.iterrows():
            pair_id = str(row['pair_id'])
            uid_a = str(row['window_uid_a'])
            uid_b = str(row['window_uid_b'])
            sa = self.summaries.get(uid_a)
            sb = self.summaries.get(uid_b)
            if sa is None or sb is None:
                n_skipped += 1
                continue
            rec = self._compute_pair_metrics(sa, sb, row)
            rec['pair_id'] = pair_id
            rec['cube_id'] = str(row.get('cube_id', ''))
            rec['track_labels'] = str(row.get('track_labels', ''))
            rec['D_weather_future'] = float(row.get('D_weather_future', np.nan))
            rec['D_heat_future'] = float(row.get('D_heat_future', np.nan))
            rec['D_init'] = float(row.get('D_init', np.nan))
            per_pair_records.append(rec)
        if n_skipped > 0:
            print(f'[PairMetricComputer] skipped {n_skipped} pairs (missing window summaries)')
        overall = self._compute_overall(per_pair_records)
        overall['n_pairs_evaluated'] = len(per_pair_records)
        overall['n_pairs_skipped'] = n_skipped
        return (overall, per_pair_records)

    def _compute_overall(self, per_pair_records):
        if not per_pair_records:
            return {}
        drr_values = [r['DRR'] for r in per_pair_records if np.isfinite(r.get('DRR', np.nan))]
        hits = sum((r.get('dhr_hits', 0) for r in per_pair_records))
        count = sum((r.get('dhr_valid_steps', 0) for r in per_pair_records))
        return {
            'DRR_mean': float(np.mean(drr_values)) if drr_values else float('nan'),
            'DHR': hits / count if count > 0 else float('nan'),
            'PDC_spearman': self._compute_pdc(per_pair_records),
            'DRR_count': len(drr_values),
            'DHR_total_steps': count
        }

    def _compute_pair_metrics(self, summary_a: WindowSummary, summary_b: WindowSummary, pair_row: pd.Series) -> dict:
        """Compute DRR, DHR, and divergence totals for one pair."""
        ndvi_pred_a = _nanmean_axis0(summary_a.ndvi_pred_trajectory)
        ndvi_pred_b = _nanmean_axis0(summary_b.ndvi_pred_trajectory)
        ndvi_gt_a = summary_a.ndvi_gt_trajectory
        ndvi_gt_b = summary_b.ndvi_gt_trajectory
        drr = self._compute_drr(ndvi_pred_a, ndvi_pred_b, ndvi_gt_a, ndvi_gt_b)
        dhr_hits, dhr_valid = self._compute_dhr(ndvi_pred_a, ndvi_pred_b, ndvi_gt_a, ndvi_gt_b)
        dhr = dhr_hits / dhr_valid if dhr_valid > 0 else float('nan')
        real_div = np.abs(ndvi_gt_a - ndvi_gt_b)
        pred_div = np.abs(ndvi_pred_a - ndvi_pred_b)
        finite_mask = np.isfinite(real_div) & np.isfinite(pred_div)
        real_total = float(np.sum(real_div[finite_mask])) if finite_mask.any() else float('nan')
        pred_total = float(np.sum(pred_div[finite_mask])) if finite_mask.any() else float('nan')
        return {
            'DRR': drr,
            'DHR': dhr,
            'dhr_hits': dhr_hits,
            'dhr_valid_steps': dhr_valid,
            'real_ndvi_div_total': real_total,
            'pred_ndvi_div_total': pred_total,
            'mae_window_a': summary_a.pixel_mae,
            'mae_window_b': summary_b.pixel_mae,
            'ndvi_mae_a': summary_a.ndvi_mae,
            'ndvi_mae_b': summary_b.ndvi_mae
        }

    def _compute_drr(self, ndvi_pred_a: np.ndarray, ndvi_pred_b: np.ndarray, ndvi_gt_a: np.ndarray,
                     ndvi_gt_b: np.ndarray) -> float:
        """Divergence Reproduction Ratio for one pair.

        DRR = mean(pred_div at valid t) / mean(real_div at valid t)
        Only at timesteps where real_div_t > DRR_NOISE_THR and both are finite.
        Ideal = 1.0.
        """
        real_div = np.abs(ndvi_gt_a - ndvi_gt_b)
        pred_div = np.abs(ndvi_pred_a - ndvi_pred_b)
        valid = np.isfinite(real_div) & np.isfinite(pred_div) & (real_div > self.DRR_NOISE_THR)
        if not valid.any():
            return float('nan')
        mean_real = float(np.mean(real_div[valid]))
        mean_pred = float(np.mean(pred_div[valid]))
        if mean_real < 1e-10:
            return float('nan')
        return mean_pred / mean_real

    def _compute_dhr(self, ndvi_pred_a: np.ndarray, ndvi_pred_b: np.ndarray, ndvi_gt_a: np.ndarray,
                     ndvi_gt_b: np.ndarray) -> Tuple[int, int]:
        """Directional Hit Rate for one pair.

        Returns (n_hits, n_valid_timesteps) as integers to avoid float round-trip.
        """
        gt_diff = ndvi_gt_a - ndvi_gt_b
        pred_diff = ndvi_pred_a - ndvi_pred_b
        valid = np.isfinite(gt_diff) & np.isfinite(pred_diff) & (np.abs(gt_diff) > self.DHR_NOISE_THR)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return (0, 0)
        hits = int(np.sum(np.sign(gt_diff[valid]) == np.sign(pred_diff[valid])))
        return (hits, n_valid)

    def _compute_pdc(self, per_pair_records: List[dict]) -> float:
        """Paired Divergence Correlation (Spearman) across all pairs."""
        real_divs = []
        pred_divs = []
        for r in per_pair_records:
            rv = r.get('real_ndvi_div_total', np.nan)
            pv = r.get('pred_ndvi_div_total', np.nan)
            if np.isfinite(rv) and np.isfinite(pv):
                real_divs.append(rv)
                pred_divs.append(pv)
        if len(real_divs) < 3:
            return float('nan')
        rho, _ = scipy_stats.spearmanr(real_divs, pred_divs)
        return float(rho)


class SeasonalMetricAccumulator:
    """Accumulates per-window summaries and standard quality metrics.

    After inference on all ranks, call reduce_across_ranks() to gather
    all WindowSummary objects into a single dict on rank 0.
    """

    def __init__(self):
        self.window_summaries: Dict[str, WindowSummary] = {}
        self.per_window_records: List[dict] = []
        self.sum_pixel_mae = 0.0
        self.sum_ndvi_mae = 0.0
        self.pixel_mae_count = 0
        self.ndvi_mae_count = 0
        self.n_windows = 0

    def add_window(self, summary: WindowSummary, benchmark_meta: Optional[dict] = None):
        """Register one window's summary after inference."""
        uid = summary.window_uid
        self.window_summaries[uid] = summary
        self.n_windows += 1
        if np.isfinite(summary.pixel_mae):
            self.sum_pixel_mae += summary.pixel_mae
            self.pixel_mae_count += 1
        if np.isfinite(summary.ndvi_mae):
            self.sum_ndvi_mae += summary.ndvi_mae
            self.ndvi_mae_count += 1
        rec = {
            'window_uid': uid,
            'pixel_mae': summary.pixel_mae,
            'ndvi_mae': summary.ndvi_mae,
            'ndvi_gt_trajectory': json.dumps(summary.ndvi_gt_trajectory.tolist()),
            'ndvi_pred_trajectory_mean': json.dumps(_nanmean_axis0(summary.ndvi_pred_trajectory).tolist())
        }
        if benchmark_meta is not None:
            rec['pair_id'] = benchmark_meta.get('pair_id', '')
            rec['sample_side'] = benchmark_meta.get('sample_side', '')
            rec['track_labels'] = benchmark_meta.get('track_labels', '')
        self.per_window_records.append(rec)

    def reduce_across_ranks(self, device: torch.device):
        """All-gather window summaries from all GPUs to rank 0.

        After this call, rank 0 has the complete window_summaries dict.
        Uses dist.all_gather_object for the summary dicts.
        Running sums are reduced via all_reduce for the quality summary.
        """
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        all_summaries_list = [None] * dist.get_world_size()
        dist.all_gather_object(all_summaries_list, self.window_summaries)
        merged = {}
        for rank_dict in all_summaries_list:
            if rank_dict is not None:
                merged.update(rank_dict)
        self.window_summaries = merged
        all_records_list = [None] * dist.get_world_size()
        dist.all_gather_object(all_records_list, self.per_window_records)
        merged_records = []
        for rank_records in all_records_list:
            if rank_records is not None:
                merged_records.extend(rank_records)
        self.per_window_records = merged_records
        vals = torch.tensor([
            self.sum_pixel_mae, self.sum_ndvi_mae,
            float(self.pixel_mae_count),
            float(self.ndvi_mae_count),
            float(self.n_windows)
        ],
                            device=device,
                            dtype=torch.float64)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        vals = vals.cpu().tolist()
        self.sum_pixel_mae = vals[0]
        self.sum_ndvi_mae = vals[1]
        self.pixel_mae_count = int(vals[2])
        self.ndvi_mae_count = int(vals[3])
        self.n_windows = int(vals[4])

    def compute_quality_summary(self) -> dict:
        """Compute aggregated per-window quality metrics."""
        result = {'n_windows': self.n_windows}
        if self.pixel_mae_count > 0:
            result['pixel_mae'] = self.sum_pixel_mae / self.pixel_mae_count
        if self.ndvi_mae_count > 0:
            result['ndvi_mae'] = self.sum_ndvi_mae / self.ndvi_mae_count
        return result


def main():
    parser = argparse.ArgumentParser('Seasonal Matched-Pair Benchmark V2V Eval')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--data_csv', required=True, help='Step4 inference CSV (seasonal_benchmark_infer.csv)')
    parser.add_argument('--benchmark_pairs_csv', required=True, help='Step3 final_benchmark_pairs.csv for re-pairing')
    parser.add_argument('--tile_json', default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_ensemble',
                        type=int,
                        default=5,
                        help='Ensemble members per window (different noise seeds)')
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--guidance', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_frames', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--vis_samples', type=int, default=VIS_SAMPLES)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--mask_source',
                        type=str,
                        default='earthformer_raw',
                        choices=['earthformer_raw', 'dataset_quality'])
    parser.add_argument('--save_spatial',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Store ensemble-mean spatial predictions to disk (--no-save_spatial to disable)')
    args = parser.parse_args()
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=4))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_master = rank == 0
    set_seed(args.seed)
    if is_master:
        print(f'[eval] world_size={world_size}, rank={rank}, device={device}')
        print(f'[eval] num_ensemble={args.num_ensemble}')
        print(f'[eval] benchmark_pairs_csv={args.benchmark_pairs_csv}')
    cfg = read_config(args.config)
    dtype = to_torch_dtype(cfg.get('dtype', 'bf16'))
    patch_size = int(cfg.get('patch_size', 2))
    seq_align = int(cfg.get('plugin_config', {}).get('sp_size', 1))
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).eval()
    model_ae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
    model_t5 = build_module(cfg.t5, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
    model_clip = build_module(cfg.clip, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
    plugin = create_colossalai_plugin(plugin=cfg.get('plugin', 'zero2'),
                                      dtype=cfg.get('dtype', 'bf16'),
                                      grad_clip=cfg.get('grad_clip', 0),
                                      **cfg.get('plugin_config', {}))
    booster = Booster(plugin=plugin)
    optimizer = create_optimizer(model, cfg.optim)
    torch.set_default_dtype(dtype)
    model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)
    torch.set_default_dtype(torch.float)
    model_unwrapped = load_master_weights(model, args.ckpt_path, dtype).eval()
    model_ae.to(torch.float32)
    dataset_cfg = cfg.dataset.to_dict() if hasattr(cfg.dataset, 'to_dict') else dict(cfg.dataset)
    dataset_cfg.pop('type', None)
    dataset_cfg['data_path'] = args.data_csv
    if args.tile_json:
        dataset_cfg['tile_location_json'] = args.tile_json
    dataset = EarthNetVideoDataset(**dataset_cfg)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            collate_fn=collate_fn_eo_earthnet,
                            num_workers=args.num_workers,
                            pin_memory=True)
    if is_master:
        print(f'[eval] Dataset windows: {len(dataset)}, batches: {len(dataloader)}')
    row_lookup = {}
    npz_cache = {}
    mask_cache = {}
    if args.mask_source == 'earthformer_raw':
        row_lookup = _build_row_lookup_from_dataset(dataset)
    accumulator = SeasonalMetricAccumulator()
    ens_metric = None
    if METRICS_AVAILABLE:
        ens_metric = EarthNet2021ScoreUpdateWithoutCompute(layout='NTHWC').to(device)
    output_dir = os.path.join(args.output_dir, os.path.basename(args.ckpt_path.rstrip('/')))
    os.makedirs(output_dir, exist_ok=True)
    vis_count = 0
    total_count = 0
    skipped_dup_count = 0
    processed_uids: set = set()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='evaluating', disable=not is_master)):
        bsz = batch['video'].shape[0]
        total_t = int(batch['video'].shape[2]) if args.num_frames <= 0 else int(args.num_frames)
        h = int(batch['video'].shape[-2])
        w = int(batch['video'].shape[-1])
        stride = int(model_ae.ENCODER_STRIDE)
        lH, lW = (h // stride, w // stride)
        in_video = batch['in_video'].to(device, torch.float32)
        with torch.no_grad():
            cond_lat_full = model_ae.encode(in_video)
        cond_latents = [cond_lat_full[i] for i in range(bsz)]
        with torch.no_grad():
            inp_base, x_1_spatial = prepare_inputs_for_inference(batch=batch,
                                                                 cond_latents=cond_latents,
                                                                 model_t5=model_t5,
                                                                 model_clip=model_clip,
                                                                 cfg=cfg,
                                                                 total_t=total_t,
                                                                 lH=lH,
                                                                 lW=lW,
                                                                 device=device,
                                                                 dtype=dtype,
                                                                 seq_align=seq_align)
        ensemble_decoded = []
        for ens_k in range(args.num_ensemble):
            x_1_k = torch.randn(bsz, cond_latents[0].shape[0], total_t, lH, lW, device=device, dtype=dtype)
            x_0_packed = euler_sample_v2v(model=model_unwrapped,
                                          inp_base=inp_base,
                                          x_1_spatial=x_1_k,
                                          num_steps=args.num_steps,
                                          guidance=args.guidance,
                                          patch_size=patch_size,
                                          dtype=dtype,
                                          device=device)
            x_0_spatial = unpack_latent(x_0_packed, total_t, lH, lW, patch_size)
            with torch.no_grad():
                decoded_k = model_ae.decode(x_0_spatial.to(torch.float32)).float().clamp(0.0, 1.0)
            ensemble_decoded.append(decoded_k.cpu())
            del x_1_k, x_0_packed, x_0_spatial, decoded_k
            torch.cuda.empty_cache()
        ensemble_stack = torch.stack(ensemble_decoded, dim=0)
        del ensemble_decoded
        quality_mask_all = batch.get('quality_mask', None)
        texts = batch['text']
        benchmark_meta_jsons = batch.get('benchmark_meta_json', None)
        for i in range(bsz):
            in_len_i = int(batch['in_len'][i].item())
            out_len_i = int(batch['out_len'][i].item())
            if in_len_i <= 0 or out_len_i <= 0:
                continue
            window_uid_i = str(batch['sample_id'][i])
            if window_uid_i in processed_uids:
                skipped_dup_count += 1
                total_count += 1
                continue
            processed_uids.add(window_uid_i)
            pred_ensemble_i = ensemble_stack[:, i, :, in_len_i:in_len_i + out_len_i]
            gt_i = batch['video'][i, :, in_len_i:in_len_i + out_len_i].float()
            in_s2_i = batch['video'][i, :, :in_len_i].float()
            if quality_mask_all is not None:
                q_mask_out = quality_mask_all[i, in_len_i:in_len_i + out_len_i].float()
                valid_i = q_mask_out.unsqueeze(0)
                q_mask_in = quality_mask_all[i, :in_len_i].float()
                in_valid_i = q_mask_in.unsqueeze(0)
            else:
                valid_i = torch.ones(1, out_len_i, h, w, dtype=torch.float32)
                in_valid_i = torch.ones(1, in_len_i, h, w, dtype=torch.float32)
            if args.mask_source == 'earthformer_raw':
                sample_id = str(batch['sample_id'][i])
                window_idx = int(batch['window_idx'][i].item())
                row_entry = row_lookup.get((sample_id, window_idx), None)
                if row_entry is not None:
                    try:
                        in_valid_ef, out_valid_ef = _build_valid_masks_from_earthformer_raw(entry=row_entry,
                                                                                            npz_cache=npz_cache,
                                                                                            mask_cache=mask_cache,
                                                                                            in_len=in_len_i,
                                                                                            out_len=out_len_i,
                                                                                            h=h,
                                                                                            w=w,
                                                                                            device=torch.device('cpu'))
                        in_valid_i = in_valid_ef
                        valid_i = out_valid_ef
                    except Exception:
                        pass
            if 'out_mask' in batch:
                frame_valid = batch['out_mask'][i, :out_len_i].float().view(1, out_len_i, 1, 1)
                valid_i = valid_i * frame_valid
            if 'in_mask' in batch:
                in_frame_valid = batch['in_mask'][i, :in_len_i].float().view(1, in_len_i, 1, 1)
                in_valid_i = in_valid_i * in_frame_valid
            valid_i = valid_i.clamp(0.0, 1.0)
            in_valid_i = in_valid_i.clamp(0.0, 1.0)
            bmeta = None
            if benchmark_meta_jsons is not None and i < len(benchmark_meta_jsons):
                try:
                    bmeta = parse_seasonal_benchmark_meta(benchmark_meta_jsons[i])
                except Exception:
                    pass
            window_uid = str(batch['sample_id'][i])
            ws, pred_mean_i = compute_window_summary(pred_ensemble=pred_ensemble_i,
                                                     gt=gt_i,
                                                     valid_mask=valid_i,
                                                     window_uid=window_uid)
            accumulator.add_window(ws, benchmark_meta=bmeta)
            if args.save_spatial:
                spatial_dir = os.path.join(output_dir, 'spatial_preds', f'rank{rank}')
                os.makedirs(spatial_dir, exist_ok=True)
                safe_uid = window_uid.replace('|', '_').replace('/', '_')
                spatial_path = os.path.join(spatial_dir, f'{safe_uid}.npy')
                np.save(spatial_path, pred_mean_i.permute(1, 2, 3, 0).numpy().astype(np.float16))
            if ens_metric is not None:
                inv_mask = 1.0 - valid_i
                pred_nthwc = pred_mean_i.permute(1, 2, 3, 0).unsqueeze(0).to(device)
                gt_nthwc = gt_i.permute(1, 2, 3, 0).unsqueeze(0).to(device)
                mask_nthwc = inv_mask.permute(1, 2, 3, 0).unsqueeze(0).to(device)
                ens_metric.update(pred_nthwc, gt_nthwc, mask_nthwc)
            total_count += 1
            if vis_count < args.vis_samples and is_master:
                sample_dir = os.path.join(output_dir, f'vis_{total_count - 1:04d}')
                pred_mean_vis = pred_ensemble_i.mean(dim=0).cpu()
                ens_vis = [pred_ensemble_i[k].cpu() for k in range(pred_ensemble_i.shape[0])]
                save_vis(sample_dir,
                         in_s2_i.cpu(),
                         gt_i.cpu(),
                         pred_mean_vis, (1.0 - valid_i).cpu(),
                         texts[i],
                         ensemble_preds=ens_vis if args.num_ensemble > 1 else None)
                vis_count += 1
        del in_video, cond_lat_full, cond_latents, inp_base, x_1_spatial, ensemble_stack
        torch.cuda.empty_cache()
    if skipped_dup_count > 0:
        print(
            f'[eval][rank={rank}] skipped {skipped_dup_count} duplicate window(s) (already inferred on this rank); processed {len(processed_uids)} unique'
        )
    torch.cuda.synchronize(device)
    dist.barrier()
    count_tensor = torch.tensor([float(total_count), float(skipped_dup_count)], device=device, dtype=torch.float64)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    total_count_all = int(count_tensor[0].item())
    skipped_dup_all = int(count_tensor[1].item())
    accumulator.reduce_across_ranks(device)
    ens_dict = None
    if ens_metric is not None:
        ens_dict = ens_metric.compute()
        dist.barrier()
    if is_master:
        print('\n[eval] === Seasonal Matched-Pair Benchmark ===')
        print(f'[eval] total windows inferred: {total_count_all}')
        print(
            f'[eval] duplicate windows skipped: {skipped_dup_all} (same window in multiple pairs, ~{100 * skipped_dup_all / max(total_count_all, 1):.1f}%)'
        )
        print(f'[eval] unique window summaries: {len(accumulator.window_summaries)}')
        quality = accumulator.compute_quality_summary()
        print(f'[eval] Per-window quality: {quality}')
        pair_computer = PairMetricComputer(benchmark_pairs_csv=args.benchmark_pairs_csv,
                                           window_summaries=accumulator.window_summaries)
        overall_metrics, per_pair_records = pair_computer.compute_all()
        print('\n[eval] === Paired Metrics (Overall) ===')
        for k in ['DRR_mean', 'DHR', 'PDC_spearman', 'per_window_pixel_mae', 'per_window_ndvi_mae']:
            v = overall_metrics.get(k, 'N/A')
            if isinstance(v, float):
                print(f'  {k}: {v:.4f}')
            else:
                print(f'  {k}: {v}')
        results = {
            'num_windows': total_count_all,
            'num_ensemble': args.num_ensemble,
            'mask_source': args.mask_source,
            'metric_impl': METRICS_IMPL,
            'benchmark_pairs_csv': args.benchmark_pairs_csv,
            'per_window_quality': quality,
            'paired_metrics': overall_metrics
        }
        if ens_dict is not None:
            results['earthnet_score'] = {
                k: v.item() if isinstance(v, torch.Tensor) else float(v)
                for k, v in ens_dict.items()
            }
        result_path = os.path.join(output_dir, 'metrics.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, cls=_NaNSafeEncoder)
        print(f'\n[eval] Metrics saved: {result_path}')
        if per_pair_records:
            pair_csv_path = os.path.join(output_dir, 'per_pair_metrics.csv')
            pd.DataFrame(per_pair_records).to_csv(pair_csv_path, index=False)
            print(f'[eval] Per-pair CSV saved: {pair_csv_path}')
        if accumulator.per_window_records:
            window_csv_path = os.path.join(output_dir, 'per_window_metrics.csv')
            wdf = pd.DataFrame(accumulator.per_window_records)
            wdf = wdf.drop_duplicates(subset='window_uid', keep='first')
            for column in wdf.columns:
                if wdf[column].map(lambda value: isinstance(value, (list, dict))).any():
                    wdf[column] = wdf[column].map(lambda value: json.dumps(value, cls=_NaNSafeEncoder)
                                                  if isinstance(value, (list, dict)) else value)
            wdf.to_csv(window_csv_path, index=False)
            print(f'[eval] Per-window CSV saved: {window_csv_path} ({len(wdf)} unique windows)')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
