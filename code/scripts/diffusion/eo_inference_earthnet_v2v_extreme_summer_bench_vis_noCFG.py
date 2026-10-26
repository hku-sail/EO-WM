from __future__ import annotations
'EO-WM forecasting and benchmark evaluation.\n\nUse eval_earthnet.sh for release settings. See ../../../README.md and --help.\n'
import argparse
import json
import math
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path

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
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from safetensors.torch import load_file
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
    """Compute NDVI from 4-channel EarthNet data.

    EarthNet channels: [B, G, R, NIR] (indices 0,1,2,3)
    NDVI = (NIR - R) / (NIR + R + eps) = (band3 - band2) / (band3 + band2 + eps)

    Args:
        x: (C, T, H, W) or (N, C, T, H, W) with C >= 4.
        eps: small value to avoid division by zero.

    Returns:
        NDVI tensor: (T, H, W) or (N, T, H, W), values in [-1, 1].
    """
    if x.ndim == 4:
        nir = x[3]
        red = x[2]
    elif x.ndim == 5:
        nir = x[:, 3]
        red = x[:, 2]
    else:
        raise ValueError(f'Expected 4D or 5D tensor, got {x.ndim}D')
    return (nir - red) / (nir + red + eps)


def compute_vegetation_mask(ndvi: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Create vegetation mask where NDVI >= threshold.

    Args:
        ndvi: (T, H, W) or (N, T, H, W).
        threshold: minimum NDVI to be considered vegetation.

    Returns:
        Boolean mask, same shape as ndvi, True = vegetation pixel.
    """
    return ndvi >= threshold


def load_master_weights(model, ckpt_dir: str, dtype: torch.dtype):
    """Load master.safetensors weights into model (identical to base script)."""
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
                                 cond_latents: list[torch.Tensor],
                                 model_t5,
                                 model_clip,
                                 cfg,
                                 total_t: int,
                                 lH: int,
                                 lW: int,
                                 device: torch.device,
                                 dtype: torch.dtype,
                                 seq_align: int = 1) -> tuple[dict, torch.Tensor]:
    """Prepare conditioning inputs for V2V inference (identical to base script)."""
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
    p = patch_size
    gH, gW = (lH // p, lW // p)
    return rearrange(x_packed, 'b (t h w) (c ph pw) -> b c t (h ph) (w pw)', t=t, h=gH, w=gW, ph=p, pw=p)


def s2_to_rgb(x: torch.Tensor, gamma: float = 0.7) -> np.ndarray:
    x_np = x.float().cpu().numpy()
    rgb = np.stack([x_np[2], x_np[1], x_np[0]], axis=-1)
    rgb = np.clip(rgb / 0.3, 0, 1)
    rgb = np.power(rgb, gamma)
    return (rgb * 255).astype(np.uint8)


def _sample_indices(t: int, n: int = 10) -> np.ndarray:
    if t <= 0:
        return np.zeros((0, ), dtype=int)
    return np.linspace(0, t - 1, min(n, t), dtype=int)


def _concat_frames(frames: list[np.ndarray], gap: int = 2, gap_color: int = 22) -> np.ndarray:
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


def _fig_to_numpy(fig) -> np.ndarray:
    """Convert a matplotlib figure to an RGB numpy array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return arr


def _plot_ndvi_trajectory(sample_dir, in_s2, gt_s2, gen_s2, inv_mask, in_valid=None, ensemble_preds=None):
    """Plot NDVI time-series: input context + GT vs pred (with ensemble spread).

    Args:
        in_s2: (C, T_in, H, W)
        gt_s2: (C, T_out, H, W)
        gen_s2: (C, T_out, H, W) — ensemble mean prediction
        inv_mask: (1, T_out, H, W) — 1=invalid, 0=valid
        in_valid: (1, T_in, H, W) — 1=valid, 0=invalid (optional)
        ensemble_preds: list of (C, T_out, H, W) tensors (optional)

    Returns:
        numpy RGB image of the plot, or None on failure.
    """
    try:
        t_in = int(in_s2.shape[1])
        t_out = int(gt_s2.shape[1])
        ndvi_in = compute_ndvi(in_s2)
        ndvi_gt = compute_ndvi(gt_s2)
        ndvi_pred = compute_ndvi(gen_s2)
        veg_mask_in = compute_vegetation_mask(ndvi_in)
        veg_mask_out = compute_vegetation_mask(ndvi_gt)
        ndvi_in_vals = []
        for t in range(t_in):
            mask_t = veg_mask_in[t]
            if in_valid is not None:
                mask_t = mask_t & (in_valid[0, t] > 0.5)
            pixels = ndvi_in[t][mask_t]
            ndvi_in_vals.append(float(pixels.mean()) if pixels.numel() > 0 else float('nan'))
        valid_out = 1.0 - inv_mask[0] > 0.5
        ndvi_gt_vals, ndvi_pred_vals = ([], [])
        for t in range(t_out):
            mask_t = veg_mask_out[t] & valid_out[t]
            gt_pixels = ndvi_gt[t][mask_t]
            pred_pixels = ndvi_pred[t][mask_t]
            ndvi_gt_vals.append(float(gt_pixels.mean()) if gt_pixels.numel() > 0 else float('nan'))
            ndvi_pred_vals.append(float(pred_pixels.mean()) if pred_pixels.numel() > 0 else float('nan'))
        ens_trajectories = []
        if ensemble_preds is not None:
            for ens_pred in ensemble_preds:
                ndvi_ens = compute_ndvi(ens_pred)
                vals = []
                for t in range(t_out):
                    mask_t = veg_mask_out[t] & valid_out[t]
                    px = ndvi_ens[t][mask_t]
                    vals.append(float(px.mean()) if px.numel() > 0 else float('nan'))
                ens_trajectories.append(vals)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=120)
        x_in = np.arange(t_in)
        x_out = np.arange(t_in, t_in + t_out)
        ax.plot(x_in, ndvi_in_vals, 'b-o', markersize=3, linewidth=1.5, label='Input (obs)')
        ax.plot(x_out, ndvi_gt_vals, 'g-o', markersize=3, linewidth=1.5, label='GT future')
        ax.plot(x_out, ndvi_pred_vals, 'r-o', markersize=3, linewidth=1.5, label='Pred (mean)')
        if ens_trajectories:
            for vals in ens_trajectories:
                ax.plot(x_out, vals, color='red', alpha=0.15, linewidth=0.7)
        ax.axvline(x=t_in - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Frame index (5-day cadence)')
        ax.set_ylabel('Spatial-mean NDVI (vegetation pixels)')
        ax.set_title('NDVI Trajectory')
        ax.legend(loc='lower left', fontsize=8)
        ax.set_xlim(-0.5, t_in + t_out - 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(sample_dir, 'ndvi_trajectory.png'), dpi=120, bbox_inches='tight')
        img = _fig_to_numpy(fig)
        return img
    except Exception:
        return None


def _plot_weather_curves(sample_dir, era5, era5_mask=None, era5_names=None, t_in=10):
    """Plot temperature and precipitation curves from ERA5 weather data.

    Args:
        era5: (T_total, C, H_meso, W_meso) — ERA5 weather, normalized [0,1]
        era5_mask: (T_total, 1, H_meso, W_meso) or None
        era5_names: list of channel names or None
        t_in: number of input frames

    Returns:
        numpy RGB image of the plot, or None.
    """
    if era5 is None:
        return None
    try:
        era5_np = era5.numpy() if isinstance(era5, torch.Tensor) else era5
        t_total = era5_np.shape[0]
        ch_names = era5_names if era5_names is not None else [
            'precipitation', 'pressure', 'temp_mean', 'temp_min', 'temp_max'
        ]
        ch_idx = {name: i for i, name in enumerate(ch_names)}
        if era5_mask is not None:
            mask_np = era5_mask.numpy() if isinstance(era5_mask, torch.Tensor) else era5_mask
        else:
            mask_np = None
        means = {}
        for name, ci in ch_idx.items():
            vals = []
            for t in range(t_total):
                ch_data = era5_np[t, ci]
                if mask_np is not None:
                    m = mask_np[t, 0] > 0.5 if mask_np.ndim == 4 else mask_np[t] > 0.5
                    ch_data = ch_data[m]
                v = float(np.mean(ch_data)) if ch_data.size > 0 else float('nan')
                vals.append(v)
            means[name] = np.array(vals)
        x = np.arange(t_total)
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=120, sharex=True)
        ax_temp = axes[0]
        if 'temp_mean' in means:
            ax_temp.plot(x, means['temp_mean'], 'r-', linewidth=1.5, label='Temp mean')
        if 'temp_min' in means:
            ax_temp.plot(x, means['temp_min'], 'b--', linewidth=1, label='Temp min')
        if 'temp_max' in means:
            ax_temp.plot(x, means['temp_max'], 'r--', linewidth=1, label='Temp max')
        if 'temp_min' in means and 'temp_max' in means:
            ax_temp.fill_between(x, means['temp_min'], means['temp_max'], alpha=0.1, color='red')
        ax_temp.axvline(x=t_in - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax_temp.set_ylabel('Temperature (normalized)')
        ax_temp.set_title('Weather Variables (ERA5)')
        ax_temp.legend(loc='upper right', fontsize=7)
        ax_temp.grid(True, alpha=0.3)
        ax_precip = axes[1]
        if 'precipitation' in means:
            ax_precip.bar(x, means['precipitation'], width=0.8, color='steelblue', alpha=0.7, label='Precipitation')
        ax_precip.axvline(x=t_in - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax_precip.set_xlabel('Frame index (5-day cadence)')
        ax_precip.set_ylabel('Precipitation (normalized)')
        ax_precip.legend(loc='upper right', fontsize=7)
        ax_precip.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(sample_dir, 'weather_curves.png'), dpi=120, bbox_inches='tight')
        img = _fig_to_numpy(fig)
        return img
    except Exception:
        return None


def save_vis(sample_dir,
             in_s2,
             gt_s2,
             gen_s2,
             inv_mask,
             text,
             ensemble_preds=None,
             in_valid=None,
             era5=None,
             era5_mask=None,
             era5_names=None):
    """Save visualization strips. Optionally include ensemble member predictions."""
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, 'prompt.txt'), 'w') as f:
        f.write(text)
    idx_in = _sample_indices(int(in_s2.shape[1]), n=10)
    idx_out = _sample_indices(int(gt_s2.shape[1]), n=20)
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
    overview_img = np.concatenate(canvas, axis=0)
    Image.fromarray(overview_img).save(os.path.join(sample_dir, 'overview.png'))
    ndvi_img = _plot_ndvi_trajectory(sample_dir,
                                     in_s2,
                                     gt_s2,
                                     gen_s2,
                                     inv_mask,
                                     in_valid=in_valid,
                                     ensemble_preds=ensemble_preds)
    weather_img = _plot_weather_curves(sample_dir,
                                       era5,
                                       era5_mask=era5_mask,
                                       era5_names=era5_names,
                                       t_in=int(in_s2.shape[1]))
    parts = [overview_img]
    overview_w = overview_img.shape[1]
    for plot_img in [ndvi_img, weather_img]:
        if plot_img is not None:
            ph, pw = plot_img.shape[:2]
            new_h = int(ph * overview_w / pw)
            plot_resized = np.array(Image.fromarray(plot_img).resize((overview_w, new_h), Image.LANCZOS))
            parts.append(np.full((6, overview_w, 3), 95, dtype=np.uint8))
            parts.append(plot_resized)
    if len(parts) > 1:
        Image.fromarray(np.concatenate(parts, axis=0)).save(os.path.join(sample_dir, 'overview_combined.png'))


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
                                            h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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


def parse_benchmark_meta(meta_json_str: str) -> dict:
    """Parse benchmark_meta_json string into a dict with typed fields.

    Returns dict with keys:
        extreme_score, drop_amplitude, peak_ndvi, trough_ndvi,
        baseline_thr, baseline_ndvi, min_target_ndvi,
        t_trough_absolute, t_peak_absolute, t_stable_absolute,
        slice_start_idx, in_len (=10), sample_id, etc.
    """
    raw = json.loads(meta_json_str)

    def _float(key, default=None):
        v = raw.get(key, default)
        if v is None:
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _int(key, default=None):
        v = raw.get(key, default)
        if v is None:
            return default
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default

    return {
        'sample_id': raw.get('sample_id', 'unknown'),
        'extreme_score': _float('extreme_score', 0.0),
        'drop_amplitude': _float('drop_amplitude', 0.0),
        'peak_ndvi': _float('peak_ndvi', 0.0),
        'trough_ndvi': _float('trough_ndvi', 0.0),
        'baseline_ndvi': _float('baseline_ndvi', 0.0),
        'baseline_thr': _float('baseline_thr', 0.0),
        'min_target_ndvi': _float('min_target_ndvi', 0.0),
        't_trough_absolute': _int('t_trough_absolute'),
        't_peak_absolute': _int('t_peak_absolute'),
        't_stable_absolute': _int('t_stable_absolute'),
        'slice_start_idx': _int('slice_start_idx', 0),
        'in_len': 10
    }


class ExtremeMetricAccumulator:
    """Accumulate the Extreme Summer metrics reported in the paper."""

    def __init__(self):
        self.per_sample_records = []
        self.sum_pixel_mae = 0.0
        self.pixel_count = 0.0
        self.sum_ndvi_mae = 0.0
        self.ndvi_pixel_count = 0.0
        self.sum_trough_ndvi_mae = 0.0
        self.trough_ndvi_count = 0.0
        self.sum_drop_amp_error = 0.0
        self.drop_amp_count = 0
        self.n_samples = 0

    def update_sample(self, pred_ensemble, gt, valid_mask, benchmark_meta, in_s2=None, in_valid=None):
        self.n_samples += 1
        record = {
            'sample_id':
            benchmark_meta.get('sample_id', 'unknown') if benchmark_meta else f'sample_{self.n_samples - 1}',
            'extreme_score': benchmark_meta.get('extreme_score') if benchmark_meta else None
        }
        record.update(self._compute_group_a(pred_ensemble.mean(dim=0), gt, valid_mask, benchmark_meta))
        self.per_sample_records.append(record)

    def _compute_group_a(self, pred_mean: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor,
                         benchmark_meta: dict | None) -> dict:
        """Compute deterministic metrics on ensemble mean.

        Returns per-sample metric dict.
        """
        record = {}
        mask = valid_mask
        diff = (pred_mean - gt) * mask
        n_valid = mask.sum() * pred_mean.shape[0]
        if n_valid > 0:
            sample_mae = diff.abs().sum().item() / n_valid.item()
        else:
            sample_mae = float('nan')
        record['pixel_mae'] = sample_mae
        if not math.isnan(sample_mae):
            self.sum_pixel_mae += sample_mae * n_valid.item()
            self.pixel_count += n_valid.item()
        pred_ndvi = compute_ndvi(pred_mean)
        gt_ndvi = compute_ndvi(gt)
        mask_squeezed = mask.squeeze(0)
        gt_veg_mask = compute_vegetation_mask(gt_ndvi)
        combined_mask = mask_squeezed * gt_veg_mask.float()
        ndvi_diff = (pred_ndvi - gt_ndvi) * combined_mask
        n_ndvi_valid = combined_mask.sum()
        if n_ndvi_valid > 0:
            sample_ndvi_mae = ndvi_diff.abs().sum().item() / n_ndvi_valid.item()
        else:
            sample_ndvi_mae = float('nan')
        record['ndvi_mae'] = sample_ndvi_mae
        if not math.isnan(sample_ndvi_mae):
            self.sum_ndvi_mae += sample_ndvi_mae * n_ndvi_valid.item()
            self.ndvi_pixel_count += n_ndvi_valid.item()
        trough_ndvi_mae = float('nan')
        if benchmark_meta is not None:
            t_trough_abs = benchmark_meta.get('t_trough_absolute')
            slice_start = benchmark_meta.get('slice_start_idx', 0)
            in_len = benchmark_meta.get('in_len', 10)
            if t_trough_abs is not None:
                t_trough_rel = int(t_trough_abs) - int(slice_start) - int(in_len)
                T_out = pred_ndvi.shape[0]
                if 0 <= t_trough_rel < T_out:
                    trough_mask = combined_mask[t_trough_rel]
                    trough_diff = (pred_ndvi[t_trough_rel] - gt_ndvi[t_trough_rel]) * trough_mask
                    n_trough = trough_mask.sum()
                    if n_trough > 0:
                        trough_ndvi_mae = trough_diff.abs().sum().item() / n_trough.item()
                        self.sum_trough_ndvi_mae += trough_ndvi_mae * n_trough.item()
                        self.trough_ndvi_count += n_trough.item()
        record['trough_ndvi_mae'] = trough_ndvi_mae
        drop_amp_error = float('nan')
        if benchmark_meta is not None:
            gt_drop = benchmark_meta.get('drop_amplitude', None)
            baseline_ndvi = benchmark_meta.get('baseline_ndvi', None)
            if gt_drop is not None and baseline_ndvi is not None:
                per_frame_ndvi_means = []
                for t_idx in range(pred_ndvi.shape[0]):
                    frame_mask = combined_mask[t_idx]
                    n_pix = frame_mask.sum()
                    if n_pix > 0:
                        mean_ndvi = (pred_ndvi[t_idx] * frame_mask).sum().item() / n_pix.item()
                        per_frame_ndvi_means.append(mean_ndvi)
                if len(per_frame_ndvi_means) > 0:
                    pred_min_target_ndvi = min(per_frame_ndvi_means)
                    pred_drop = baseline_ndvi - pred_min_target_ndvi
                    drop_amp_error = abs(pred_drop - gt_drop)
                    self.sum_drop_amp_error += drop_amp_error
                    self.drop_amp_count += 1
        record['drop_amplitude_error'] = drop_amp_error
        return record

    def compute_summary(self, all_records=None):
        summary = {'n_samples': self.n_samples}
        for key, total, count in (('pixel_mae', self.sum_pixel_mae, self.pixel_count), ('ndvi_mae', self.sum_ndvi_mae,
                                                                                        self.ndvi_pixel_count),
                                  ('trough_ndvi_mae', self.sum_trough_ndvi_mae, self.trough_ndvi_count),
                                  ('drop_amplitude_error', self.sum_drop_amp_error, self.drop_amp_count)):
            summary[key] = total / count if count > 0 else float('nan')
        records = all_records if all_records is not None else self.per_sample_records
        summary['by_extreme_bin'] = self._compute_bin_metrics(records)
        return summary

    def _compute_bin_metrics(self, records: list[dict] | None = None) -> dict:
        """Split samples into low/mid/high extreme_score bins, report metrics per bin."""
        records = records if records is not None else self.per_sample_records
        if not records:
            return {}
        scored = [r for r in records if r.get('extreme_score') is not None]
        if not scored:
            return {}
        scores = [r['extreme_score'] for r in scored]
        low_thr = np.percentile(scores, 33.3)
        high_thr = np.percentile(scores, 66.7)
        bins = {'low': [], 'mid': [], 'high': []}
        for r in scored:
            s = r['extreme_score']
            if s <= low_thr:
                bins['low'].append(r)
            elif s <= high_thr:
                bins['mid'].append(r)
            else:
                bins['high'].append(r)
        result = {}
        for bin_name, records in bins.items():
            if not records:
                continue
            bin_summary = {'n_samples': len(records)}
            for metric_key in ['trough_ndvi_mae', 'drop_amplitude_error']:
                vals = [
                    r[metric_key] for r in records
                    if metric_key in r and (not (isinstance(r[metric_key], float) and math.isnan(r[metric_key])))
                ]
                if vals:
                    bin_summary[metric_key] = sum(vals) / len(vals)
            result[bin_name] = bin_summary
        return result

    def get_per_sample_records(self) -> list[dict]:
        """Return list of per-sample metric records for CSV export."""
        return self.per_sample_records

    def reduce_across_ranks(self, device):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        keys = [
            'sum_pixel_mae', 'pixel_count', 'sum_ndvi_mae', 'ndvi_pixel_count', 'sum_trough_ndvi_mae',
            'trough_ndvi_count', 'sum_drop_amp_error'
        ]
        int_keys = ['drop_amp_count', 'n_samples']
        values = torch.tensor([getattr(self, k) for k in keys] + [float(getattr(self, k)) for k in int_keys],
                              device=device,
                              dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        values = values.cpu().tolist()
        for i, key in enumerate(keys):
            setattr(self, key, values[i])
        for i, key in enumerate(int_keys):
            setattr(self, key, int(values[len(keys) + i]))


def main():
    parser = argparse.ArgumentParser('Extreme Summer Benchmark V2V Eval')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--data_csv', required=True)
    parser.add_argument('--tile_json', default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_ensemble',
                        type=int,
                        default=1,
                        help='Number of ensemble members per sample (different noise seeds)')
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
    accumulator = ExtremeMetricAccumulator()
    ens_metric = None
    if METRICS_AVAILABLE:
        ens_metric = EarthNet2021ScoreUpdateWithoutCompute(layout='NTHWC').to(device)
    output_dir = os.path.join(args.output_dir, os.path.basename(args.ckpt_path.rstrip('/')))
    os.makedirs(output_dir, exist_ok=True)
    vis_count = 0
    total_count = 0
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
                bmeta = parse_benchmark_meta(benchmark_meta_jsons[i])
            accumulator.update_sample(pred_ensemble=pred_ensemble_i,
                                      gt=gt_i,
                                      valid_mask=valid_i,
                                      benchmark_meta=bmeta,
                                      in_s2=in_s2_i,
                                      in_valid=in_valid_i)
            if ens_metric is not None:
                pred_mean = pred_ensemble_i.mean(dim=0)
                inv_mask = 1.0 - valid_i
                pred_nthwc = pred_mean.permute(1, 2, 3, 0).unsqueeze(0).to(device)
                gt_nthwc = gt_i.permute(1, 2, 3, 0).unsqueeze(0).to(device)
                mask_nthwc = inv_mask.permute(1, 2, 3, 0).unsqueeze(0).to(device)
                ens_metric.update(pred_nthwc, gt_nthwc, mask_nthwc)
            total_count += 1
            if vis_count < args.vis_samples and is_master:
                sample_dir = os.path.join(output_dir, f'vis_{total_count - 1:04d}')
                pred_mean_vis = pred_ensemble_i.mean(dim=0).cpu()
                ens_vis = [pred_ensemble_i[k].cpu() for k in range(pred_ensemble_i.shape[0])]
                era5_i = None
                era5_mask_i = None
                era5_names_i = None
                if batch.get('era5', None) is not None:
                    t_total = in_len_i + out_len_i
                    era5_i = batch['era5'][i, :t_total].cpu().float()
                    if batch.get('era5_mask', None) is not None:
                        era5_mask_i = batch['era5_mask'][i, :t_total].cpu().float()
                    era5_names_i = batch.get('era5_names', None)
                save_vis(sample_dir,
                         in_s2_i.cpu(),
                         gt_i.cpu(),
                         pred_mean_vis, (1.0 - valid_i).cpu(),
                         texts[i],
                         ensemble_preds=ens_vis if args.num_ensemble > 1 else None,
                         in_valid=in_valid_i.cpu(),
                         era5=era5_i,
                         era5_mask=era5_mask_i,
                         era5_names=era5_names_i)
                vis_count += 1
        del in_video, cond_lat_full, cond_latents, inp_base, x_1_spatial, ensemble_stack
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    dist.barrier()
    count_tensor = torch.tensor([float(total_count)], device=device, dtype=torch.float64)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    total_count_all = int(count_tensor.item())
    accumulator.reduce_across_ranks(device)
    all_records_nested = [None] * world_size
    dist.all_gather_object(all_records_nested, accumulator.get_per_sample_records())
    all_per_sample_records = [r for rank_records in all_records_nested for r in rank_records]
    ens_dict = None
    if ens_metric is not None:
        ens_dict = ens_metric.compute()
        dist.barrier()
    if is_master:
        extreme_summary = accumulator.compute_summary(all_records=all_per_sample_records)
        results = {
            'num_samples': total_count_all,
            'num_ensemble': args.num_ensemble,
            'mask_source': args.mask_source,
            'metric_impl': METRICS_IMPL
        }
        if ens_dict is not None:
            results.update({k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in ens_dict.items()})
        results['extreme_metrics'] = extreme_summary
        print('\n[eval] === Extreme Summer Benchmark Results ===')
        print(f'[eval] samples={total_count_all}, num_ensemble={args.num_ensemble}')
        if ens_dict is not None:
            print('[eval] EarthNetScore metrics:')
            for k, v in ens_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                print(f'  {k}: {val:.6f}')
        print('[eval] Extreme benchmark metrics:')
        for k, v in extreme_summary.items():
            if isinstance(v, dict):
                print(f'  {k}:')
                for kk, vv in v.items():
                    print(f'    {kk}: {vv}')
            else:
                print(f'  {k}: {v}')
        result_path = os.path.join(output_dir, 'metrics.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'[eval] Metrics saved: {result_path}')
        if all_per_sample_records:
            csv_path = os.path.join(output_dir, 'per_sample_metrics.csv')
            df = pd.DataFrame(all_per_sample_records)
            for column in df.columns:
                if df[column].map(lambda value: isinstance(value, (list, dict))).any():
                    df[column] = df[column].map(lambda value: json.dumps(value)
                                                if isinstance(value, (list, dict)) else value)
            df.to_csv(csv_path, index=False)
            print(f'[eval] Per-sample CSV saved: {csv_path}')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
