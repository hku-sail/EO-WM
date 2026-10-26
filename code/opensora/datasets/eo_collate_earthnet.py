"""
EarthNet 数据集专用 collate function。

与 Open-Sora 训练输入格式对齐：
    - video/in_video/out_video: (B, C, T, H, W)
    - 其他标量信息按 batch 维堆叠

相比 collate_fn_eo，本实现对 EarthNet 的可选键更宽松：
    - s2_time / era5_time 可缺省
    - era5_mask 若缺省则由 era5 自动生成
"""
import warnings
from typing import Any, Dict, List, Optional
import torch


def collate_fn_eo_earthnet(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        raise ValueError('[collate_fn_eo_earthnet] All samples in batch are None!')
    B = len(batch)
    total_lens = [x['num_frames'] for x in batch]
    in_lens = [x['in_len'] for x in batch]
    out_lens = [x['out_len'] for x in batch]
    T_max = max(total_lens)
    T_in_max = max(in_lens)
    T_out_max = max(out_lens)
    frame_mask = torch.zeros(B, T_max, dtype=torch.bool)
    in_mask = torch.zeros(B, T_in_max, dtype=torch.bool)
    out_mask = torch.zeros(B, T_out_max, dtype=torch.bool)
    for i, (tt, ti, to) in enumerate(zip(total_lens, in_lens, out_lens)):
        frame_mask[i, :tt] = True
        in_mask[i, :ti] = True
        out_mask[i, :to] = True
    ret: Dict[str, Any] = {
        'text': [x['text'] for x in batch],
        'fps': torch.tensor([x['fps'] for x in batch], dtype=torch.float32),
        'height': torch.tensor([x['height'] for x in batch], dtype=torch.long),
        'width': torch.tensor([x['width'] for x in batch], dtype=torch.long),
        'num_frames': torch.tensor(total_lens, dtype=torch.long),
        'frame_mask': frame_mask,
        'in_len': torch.tensor(in_lens, dtype=torch.long),
        'out_len': torch.tensor(out_lens, dtype=torch.long),
        'in_mask': in_mask,
        'out_mask': out_mask,
        'sample_id': [x['sample_id'] for x in batch],
        'window_idx': torch.tensor([x['window_idx'] for x in batch], dtype=torch.long),
        'year': [x['year'] for x in batch],
        'in_valid_frac': torch.tensor([x['in_valid_frac'] for x in batch], dtype=torch.float32),
        'out_valid_frac': torch.tensor([x['out_valid_frac'] for x in batch], dtype=torch.float32)
    }
    if _all_have(batch, 'lon'):
        ret['lon'] = torch.tensor([x['lon'] for x in batch], dtype=torch.float32)
    if _all_have(batch, 'lat'):
        ret['lat'] = torch.tensor([x['lat'] for x in batch], dtype=torch.float32)
    if _all_have(batch, 'benchmark_meta_json'):
        ret['benchmark_meta_json'] = [x['benchmark_meta_json'] for x in batch]
    for passthrough_key in ('noise_group_id', 'context_side', 'forcing_side', 'context_window_uid',
                            'forcing_window_uid'):
        if _all_have(batch, passthrough_key):
            ret[passthrough_key] = [x[passthrough_key] for x in batch]
    ret['video'] = _pad_video([x['video'] for x in batch], T_max)
    ret['in_video'] = _pad_video([x['in_video'] for x in batch], T_in_max)
    ret['out_video'] = _pad_video([x['out_video'] for x in batch], T_out_max)
    if _all_have(batch, 'quality_mask'):
        ret['quality_mask'] = _pad_temporal([x['quality_mask'] for x in batch], T_max)
    if _all_have(batch, 's2_time'):
        ret['s2_time'] = _pad_1d([x['s2_time'] for x in batch], T_max)
    if _all_have(batch, 'era5'):
        T_era5_max = max((x['era5'].shape[0] for x in batch))
        ret['era5'] = _pad_video([x['era5'] for x in batch], T_era5_max)
        if _all_have(batch, 'era5_time'):
            ret['era5_time'] = _pad_1d([x['era5_time'] for x in batch], T_era5_max)
        if _all_have(batch, 'era5_mask'):
            ret['era5_mask'] = _pad_video([x['era5_mask'] for x in batch], T_era5_max)
        else:
            era5_mask_list = []
            for x in batch:
                mask = (torch.abs(x['era5']).sum(dim=1, keepdim=True) > 1e-05).to(torch.float32)
                era5_mask_list.append(mask)
            ret['era5_mask'] = _pad_video(era5_mask_list, T_era5_max)
        if _all_have(batch, 'era5_names'):
            names_0 = batch[0]['era5_names']
            if not all((x['era5_names'] == names_0 for x in batch)):
                warnings.warn('[collate_fn_eo_earthnet] era5_names inconsistent across batch, using batch[0].')
            ret['era5_names'] = names_0
        if _all_have(batch, 'era5_anom'):
            ret['era5_anom'] = _pad_video([x['era5_anom'] for x in batch], T_era5_max)
        if _all_have(batch, 'era5_clim'):
            ret['era5_clim'] = _pad_2d([x['era5_clim'] for x in batch], T_era5_max)
        if _all_have(batch, 'era5_clim_std'):
            ret['era5_clim_std'] = _pad_2d([x['era5_clim_std'] for x in batch], T_era5_max)
        if _all_have(batch, 'cumstress'):
            ret['cumstress'] = _pad_2d([x['cumstress'] for x in batch], T_era5_max)
    if _all_have(batch, 'dem'):
        _assert_same_shape([x['dem'] for x in batch], 'dem')
        ret['dem'] = torch.stack([x['dem'] for x in batch], dim=0)
        if _all_have(batch, 'dem_mask'):
            ret['dem_mask'] = torch.stack([x['dem_mask'] for x in batch], dim=0)
    return ret


def _pad_video(tensors: List[torch.Tensor], T_max: int, pad_value: float = 0.0) -> torch.Tensor:
    """(T_i, C, H, W) list -> (B, C, T_max, H, W)"""
    B = len(tensors)
    t0 = tensors[0]
    _, C, H, W = t0.shape
    _assert_same_chw(tensors, 'video-like')
    out = torch.full((B, T_max, C, H, W), pad_value, dtype=t0.dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out.permute(0, 2, 1, 3, 4).contiguous()


def _pad_temporal(tensors: List[torch.Tensor], T_max: int, pad_value: float = 0.0) -> torch.Tensor:
    """(T_i, H, W) list -> (B, T_max, H, W)"""
    B = len(tensors)
    _, H, W = tensors[0].shape
    out = torch.full((B, T_max, H, W), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


def _pad_1d(tensors: List[torch.Tensor], T_max: int, pad_value: float = 0.0) -> torch.Tensor:
    """(T_i,) list -> (B, T_max)"""
    B = len(tensors)
    out = torch.full((B, T_max), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


def _pad_2d(tensors: List[torch.Tensor], T_max: int, pad_value: float = 0.0) -> torch.Tensor:
    """(T_i, D) list -> (B, T_max, D)"""
    B = len(tensors)
    D = tensors[0].shape[1]
    out = torch.full((B, T_max, D), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


def _assert_same_chw(tensors: List[torch.Tensor], name: str) -> None:
    ref = tensors[0].shape[1:]
    for i, t in enumerate(tensors[1:], 1):
        if t.shape[1:] != ref:
            raise ValueError(
                f"[collate_fn_eo_earthnet] '{name}' shape mismatch: sample 0={ref}, sample {i}={t.shape[1:]}")


def _assert_same_shape(tensors: List[torch.Tensor], name: str) -> None:
    ref = tensors[0].shape
    for i, t in enumerate(tensors[1:], 1):
        if t.shape != ref:
            raise ValueError(f"[collate_fn_eo_earthnet] '{name}' shape mismatch: sample 0={ref}, sample {i}={t.shape}")


def _all_have(batch: List[Dict[str, Any]], key: str) -> bool:
    return all((key in x for x in batch))
