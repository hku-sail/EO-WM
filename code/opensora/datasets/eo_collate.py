# opensora/datasets/eo_collate.py
"""
EO 数据集专用 collate function。

video / in_video / out_video 输出格式: (B, C, T, H, W)
与 Open-Sora DiT 输入格式对齐。
"""

import warnings
from typing import Any, Dict, List, Optional

import torch


def collate_fn_eo(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    EO 时序滑动窗口数据 collate function。

    输出 Tensor shapes（B=batch_size）：
    ── Open-Sora 核心 ──────────────────────────────────────────
        video         : (B, C_s2, T,     H, W)  完整窗口 (C,T,H,W 格式)
        num_frames    : (B,)                     每样本实际帧数
        frame_mask    : (B, T_max)               bool，True=有效帧
        fps           : (B,)                     float32
        height        : (B,)                     long
        width         : (B,)                     long
        text          : List[str]                长度=B

    ── EO 窗口拆分 ─────────────────────────────────────────────
        in_video      : (B, C_s2, T_in_max,  H, W)
        out_video     : (B, C_s2, T_out_max, H, W)
        in_len        : (B,)   long
        out_len       : (B,)   long
        in_mask       : (B, T_in_max)   bool
        out_mask      : (B, T_out_max)  bool

    ── EO 质量与时序 ────────────────────────────────────────────
        quality_mask  : (B, T_max, H, W)   像素级质量掩膜，pad=0
        s2_time       : (B, T_max)         Unix时间戳，pad=0

    ── EO 辅助模态（全部样本均有时才存在） ─────────────────────
        s1            : (B, 3,  T_max, H, W)
        s1_time       : (B, T_max)
        era5          : (B, V,  T_max, H, W)
        era5_time     : (B, T_max)
        era5_names    : List[str]
        dem           : (B, 1, H, W)

    ── EO 元数据 ────────────────────────────────────────────────
        sample_id      : List[str]
        window_idx     : (B,)   long
        year           : List[str]
        in_valid_frac  : (B,)   float32
        out_valid_frac : (B,)   float32
    """
    # ── Step 1: 过滤 None ──────────────────────────────────────
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        raise ValueError("[collate_fn_eo] All samples in batch are None!")
    B = len(batch)

    # ── Step 2: 计算各维度最大长度 ────────────────────────────
    total_lens = [x["num_frames"] for x in batch]
    in_lens    = [x["in_len"]     for x in batch]
    out_lens   = [x["out_len"]    for x in batch]
    T_max      = max(total_lens)
    T_in_max   = max(in_lens)
    T_out_max  = max(out_lens)

    # ── Step 3: 帧掩膜 ────────────────────────────────────────
    frame_mask = torch.zeros(B, T_max, dtype=torch.bool)
    in_mask    = torch.zeros(B, T_in_max,  dtype=torch.bool)
    out_mask   = torch.zeros(B, T_out_max, dtype=torch.bool)
    for i, (tt, ti, to) in enumerate(zip(total_lens, in_lens, out_lens)):
        frame_mask[i, :tt] = True
        in_mask[i,    :ti] = True
        out_mask[i,   :to] = True

    # ── Step 4: 标量与列表字段 ────────────────────────────────
    ret: Dict[str, Any] = {
        # Open-Sora 核心
        "text": [x["text"] for x in batch],
        "fps": torch.tensor([x["fps"] for x in batch], dtype=torch.float32),
        "height": torch.tensor([x["height"] for x in batch], dtype=torch.long),
        "width": torch.tensor([x["width"]  for x in batch], dtype=torch.long),
        "num_frames": torch.tensor(total_lens, dtype=torch.long),
        "frame_mask": frame_mask,
        # EO 窗口信息
        "in_len":  torch.tensor(in_lens,  dtype=torch.long),
        "out_len": torch.tensor(out_lens, dtype=torch.long),
        "in_mask":  in_mask,
        "out_mask": out_mask,
        # EO 元数据
        "sample_id":      [x["sample_id"] for x in batch],
        "window_idx":     torch.tensor([x["window_idx"]     for x in batch], dtype=torch.long),
        "year":           [x["year"]       for x in batch],
        "in_valid_frac":  torch.tensor([x["in_valid_frac"]  for x in batch], dtype=torch.float32),
        "out_valid_frac": torch.tensor([x["out_valid_frac"] for x in batch], dtype=torch.float32),
        "lon":            torch.tensor([x["lon"] for x in batch], dtype=torch.float32),  # (B,)
        "lat":            torch.tensor([x["lat"] for x in batch], dtype=torch.float32),  # (B,)
    }

    # ── Step 5: 视频主模态 ────────────────────────────────────
    # 单样本: (T, C, H, W) → pad → (B, T_max, C, H, W) → permute → (B, C, T_max, H, W)
    ret["video"] = _pad_video(
        [x["video"] for x in batch], T_max
    )  # (B, C, T_max, H, W)

    ret["in_video"] = _pad_video(
        [x["in_video"] for x in batch], T_in_max
    )  # (B, C, T_in_max, H, W)

    ret["out_video"] = _pad_video(
        [x["out_video"] for x in batch], T_out_max
    )  # (B, C, T_out_max, H, W)

    # ── Step 6: quality_mask ──────────────────────────────────
    # 单样本: (T, H, W) → (B, T_max, H, W)，保持 T 在前（掩膜不需要 C 维）
    if "quality_mask" in batch[0]:
        ret["quality_mask"] = _pad_temporal(
            [x["quality_mask"] for x in batch], T_max
        )  # (B, T_max, H, W)

    # ── Step 7: S2 时间戳 ─────────────────────────────────────
    if "s2_time" in batch[0]:
        ret["s2_time"] = _pad_1d(
            [x["s2_time"] for x in batch], T_max
        )  # (B, T_max)

    # ── Step 8: S1 辅助模态 ───────────────────────────────────
    if _all_have(batch, "s1"):
        # 单样本: (T, 3, H, W) → (B, 3, T_max, H, W)
        ret["s1"]      = _pad_video([x["s1"]      for x in batch], T_max)
        ret["s1_time"] = _pad_1d(  [x["s1_time"]  for x in batch], T_max)

    # ── Step 9: ERA5 ──────────────────────────────────────────
    if _all_have(batch, "era5"):
        # 单样本: (T, V, H, W)，V=变量数，与 C_s2 无关
        # → (B, V, T_max, H, W)
        ret["era5"]      = _pad_video([x["era5"] for x in batch], T_max)
        ret["era5_time"] = _pad_1d([x["era5_time"] for x in batch], T_max)
        # Collate the ERA5 validity mask.
        # 单样本 mask: (T_i, 1, H, W) → (B, 1, T_max, H, W)
        ret["era5_mask"] = _pad_video([x["era5_mask"] for x in batch], T_max)

        # era5_names 同一数据集内固定，取第一个样本
        names_0 = batch[0]["era5_names"]
        if not all(x["era5_names"] == names_0 for x in batch):
            warnings.warn("[collate_fn_eo] era5_names inconsistent across batch, using batch[0].")
        ret["era5_names"] = names_0  # List[str]

    # ── Step 10: DEM 静态数据 ─────────────────────────────────
    # 单样本: (1, H, W) → stack → (B, 1, H, W)，无时序维
    if _all_have(batch, "dem"):
        _assert_same_shape([x["dem"] for x in batch], "dem")
        ret["dem"] = torch.stack([x["dem"] for x in batch], dim=0)  # (B, 1, H, W)
        ret["dem_mask"] = torch.stack([x["dem_mask"] for x in batch], dim=0) 

    return ret


# ──────────────────────────────────────────────────────────────
# Padding 工具函数
# ──────────────────────────────────────────────────────────────

def _pad_video(
    tensors: List[torch.Tensor],  # 每个 shape: (T_i, C, H, W)
    T_max:   int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    (T_i, C, H, W) list → (B, C, T_max, H, W)
    先 pad T 维到 T_max，再 permute 到 Open-Sora 期望格式。
    """
    B = len(tensors)
    t0 = tensors[0]
    _, C, H, W = t0.shape
    _assert_same_chw(tensors, "video-like")

    # (B, T_max, C, H, W)
    out = torch.zeros(B, T_max, C, H, W, dtype=t0.dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t

    # → (B, C, T_max, H, W)
    return out.permute(0, 2, 1, 3, 4).contiguous()


def _pad_temporal(
    tensors:   List[torch.Tensor],  # 每个 shape: (T_i, H, W)
    T_max:     int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """(T_i, H, W) list → (B, T_max, H, W)，保持 T 在第1维（掩膜惯例）"""
    B = len(tensors)
    _, H, W = tensors[0].shape
    out = torch.full((B, T_max, H, W), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


def _pad_1d(
    tensors:   List[torch.Tensor],  # 每个 shape: (T_i,)
    T_max:     int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """(T_i,) list → (B, T_max)"""
    B   = len(tensors)
    out = torch.full((B, T_max), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


# ──────────────────────────────────────────────────────────────
# 校验工具
# ──────────────────────────────────────────────────────────────

def _assert_same_chw(tensors: List[torch.Tensor], name: str) -> None:
    """断言 batch 内所有 tensor 的 (C, H, W) 维度相同。"""
    ref = tensors[0].shape[1:]   # (C, H, W)
    for i, t in enumerate(tensors[1:], 1):
        if t.shape[1:] != ref:
            raise ValueError(
                f"[collate_fn_eo] '{name}' shape mismatch: "
                f"sample 0={ref}, sample {i}={t.shape[1:]}"
            )


def _assert_same_shape(tensors: List[torch.Tensor], name: str) -> None:
    """断言 batch 内所有 tensor 形状完全相同（用于无时序维的字段）。"""
    ref = tensors[0].shape
    for i, t in enumerate(tensors[1:], 1):
        if t.shape != ref:
            raise ValueError(
                f"[collate_fn_eo] '{name}' shape mismatch: "
                f"sample 0={ref}, sample {i}={t.shape}"
            )


def _all_have(batch: List[Dict], key: str) -> bool:
    """检查 batch 中所有样本是否都含有指定 key。"""
    return all(key in x for x in batch)