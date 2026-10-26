from __future__ import annotations
# eo_vae/datasets/earthnet2021_datamodule.py

import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _safe_mean_per_frame(x_hw_t: np.ndarray, default: float) -> np.ndarray:
    """
    计算每帧像素均值，自动处理 NaN/Inf。
    x_hw_t: (H, W, T)
    """
    t = x_hw_t.shape[-1]
    flat = x_hw_t.reshape(-1, t).astype(np.float32, copy=False)
    valid = np.isfinite(flat)
    summ = np.where(valid, flat, 0.0).sum(axis=0, dtype=np.float64)
    cnt = valid.sum(axis=0)
    return np.divide(
        summ,
        cnt,
        out=np.full((t,), float(default), dtype=np.float64),
        where=cnt > 0,
    ).astype(np.float32)


def _compute_frame_good_ratio(hd: np.ndarray) -> np.ndarray:
    """
    EarthNet highresdynamic 帧级有效像素比例:
      - train-like (C>=7): good = (1-mask_ch6) * (1-cloud_ch4/100)
      - test-like  (C>=5): good = (1-mask_ch4)
    返回 good_ratio: (T,), 范围[0,1]
    """
    c = int(hd.shape[2])
    if c >= 7:
        cloud_raw = hd[:, :, 4, :].astype(np.float32, copy=False)  # 0~100
        mask_raw = hd[:, :, 6, :].astype(np.float32, copy=False)   # 1=bad, 0=good
        # 与 eo_dataset_earthnet.py 对齐：
        # - cloud 非有限值按 100% 云量处理（最差）
        # - mask 非有限值按无效处理（最差）
        cloud_raw = np.where(np.isfinite(cloud_raw), cloud_raw, 100.0)
        mask_raw = np.where(np.isfinite(mask_raw), mask_raw, 1.0)

        cloud = np.clip(cloud_raw / 100.0, 0.0, 1.0)
        valid = 1.0 - np.clip(mask_raw, 0.0, 1.0)
        good = valid * (1.0 - cloud)  # 逐像素“清晰且有效”分数
        return np.clip(_safe_mean_per_frame(good, default=0.0), 0.0, 1.0).astype(np.float32)

    if c >= 5:
        mask_raw = hd[:, :, 4, :].astype(np.float32, copy=False)  # test-like: 1=bad, 0=good
        mask_raw = np.where(np.isfinite(mask_raw), mask_raw, 1.0)
        valid = 1.0 - np.clip(mask_raw, 0.0, 1.0)
        return np.clip(_safe_mean_per_frame(valid, default=0.0), 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unsupported EarthNet highresdynamic channel count: C={c}")


class EarthNet2021FrameDataset(Dataset):
    """
    以“单帧”为样本单位的 EarthNet 数据集（用于 EO-VAE 微调）。
    每个样本返回:
      {
        "s2_image": Tensor(4,H,W),
        "has_s1": False,
        "s1_image": empty tensor
      }
    """

    def __init__(
        self,
        npz_paths: List[Path],
        data_qua_thresh: float = 0.3,
        mode: str = "train",
        seed: int = 42,
        max_frames_per_file: int = -1,
        cache_size: int = 4,
    ):
        self.npz_paths = [Path(p) for p in npz_paths]
        self.data_qua_thresh = float(data_qua_thresh)
        self.mode = mode
        self.seed = int(seed)
        self.max_frames_per_file = int(max_frames_per_file)
        self.cache_size = max(int(cache_size), 0)

        if not (0.0 <= self.data_qua_thresh <= 1.0):
            raise ValueError(f"data_qua_thresh must be in [0,1], got {self.data_qua_thresh}")

        self._rng = random.Random(self.seed)
        self._npz_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self.valid_frames: List[Tuple[str, int]] = self._build_index()

    def _build_index(self) -> List[Tuple[str, int]]:
        target_ratio = 1.0 - self.data_qua_thresh
        valid_frames: List[Tuple[str, int]] = []

        for npz_path in tqdm(self.npz_paths, desc=f"Scanning EarthNet [{self.mode}] frames"):
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "highresdynamic" not in data:
                        print(f"[WARN] missing 'highresdynamic': {npz_path}")
                        continue

                    hd = data["highresdynamic"]  # (H,W,C,T)
            except Exception as e:
                print(f"[WARN] failed load {npz_path}: {e}")
                continue

            if hd.ndim != 4 or hd.shape[2] < 5:
                print(f"[WARN] invalid highresdynamic shape {hd.shape}: {npz_path}")
                continue

            try:
                good_ratio = _compute_frame_good_ratio(hd)  # (T,)
            except Exception as e:
                print(f"[WARN] failed quality parsing {npz_path}: {e}")
                continue

            valid_t = np.where(good_ratio >= target_ratio)[0].tolist()
            if not valid_t:
                best_t = int(np.argmax(good_ratio))
                valid_t = [best_t]

            if self.max_frames_per_file > 0 and len(valid_t) > self.max_frames_per_file:
                valid_t = self._rng.sample(valid_t, k=self.max_frames_per_file)
                valid_t.sort()

            valid_frames.extend((str(npz_path), int(t)) for t in valid_t)

        print(
            f"[{self.mode.upper()}] found {len(valid_frames)} valid frames "
            f"from {len(self.npz_paths)} files."
        )
        return valid_frames

    def __len__(self) -> int:
        return len(self.valid_frames)

    def _get_npz(self, npz_path: str) -> Dict[str, np.ndarray]:
        if self.cache_size <= 0:
            with np.load(npz_path, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}

        if npz_path in self._npz_cache:
            self._npz_cache.move_to_end(npz_path)
            return self._npz_cache[npz_path]

        with np.load(npz_path, allow_pickle=False) as raw:
            data = {k: raw[k] for k in raw.files}
        self._npz_cache[npz_path] = data
        self._npz_cache.move_to_end(npz_path)
        while len(self._npz_cache) > self.cache_size:
            self._npz_cache.popitem(last=False)
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        npz_path, t = self.valid_frames[idx]
        data = self._get_npz(npz_path)
        hd = data["highresdynamic"]  # (H,W,C,T)

        # optical: B/G/R/NIR = ch0..3
        s2 = hd[:, :, :4, int(t)].astype(np.float32)  # (H,W,4)
        np.nan_to_num(s2, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(s2, 0.0, 1.0, out=s2)

        s2_tensor = torch.from_numpy(s2).permute(2, 0, 1).contiguous()  # (4,H,W)
        return {
            "s2_image": s2_tensor,
            "has_s1": False,
            "s1_image": torch.empty(0, dtype=torch.float32),
        }


class EarthNet2021TemporalDataset(Dataset):
    """Temporal (multi-frame) EarthNet dataset for the temporal EO-VAE finetune.

    Each sample = one npz cube, from which we sample ``num_timesteps`` valid
    frames (reusing the same per-frame quality gating as
    ``EarthNet2021FrameDataset``) and return them as a [T, 4, H, W] tensor with a
    relative time index ``0..T-1``. Time is intentionally NOT folded into the
    wavelength vector (the dynamic conv keys its kernel off len(wvs)==C); the
    time index drives the temporal RoPE inside the attention blocks.

    If a cube has fewer than ``num_timesteps`` valid frames, valid frames are
    sampled with replacement (the repeated frames keep their original relative
    index, so no spurious temporal offset is introduced). Cubes with zero valid
    frames are dropped at index-build time.
    """

    def __init__(
        self,
        npz_paths: List[Path],
        data_qua_thresh: float = 0.3,
        mode: str = "train",
        seed: int = 42,
        max_frames_per_file: int = -1,
        cache_size: int = 4,
        num_timesteps: int = 4,
        temporal_stride: int = 1,
        use_real_dates: bool = False,
    ):
        self.npz_paths = [Path(p) for p in npz_paths]
        self.data_qua_thresh = float(data_qua_thresh)
        self.mode = mode
        self.seed = int(seed)
        self.max_frames_per_file = int(max_frames_per_file)
        self.cache_size = max(int(cache_size), 0)
        self.num_timesteps = int(num_timesteps)
        self.temporal_stride = int(temporal_stride)
        self.use_real_dates = bool(use_real_dates)

        if self.num_timesteps < 1:
            raise ValueError("num_timesteps must be >= 1")
        if not (0.0 <= self.data_qua_thresh <= 1.0):
            raise ValueError(f"data_qua_thresh must be in [0,1], got {self.data_qua_thresh}")

        self._rng = random.Random(self.seed)
        self._npz_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        # Per-entry: (npz_path, [valid frame indices into the cube's T axis])
        self.samples: List[Tuple[str, List[int], Dict[str, np.ndarray]]] = self._build_index()

    def _build_index(self):
        target_ratio = 1.0 - self.data_qua_thresh
        samples: List[Tuple[str, List[int], Dict[str, np.ndarray]]] = []

        for npz_path in tqdm(
            self.npz_paths, desc=f"Scanning EarthNet [{self.mode}] temporal cubes"
        ):
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "highresdynamic" not in data:
                        print(f"[WARN] missing 'highresdynamic': {npz_path}")
                        continue
                    hd = data["highresdynamic"]  # (H,W,C,T)
                    # Only carry date arrays forward when real-date indexing is
                    # enabled. Keeping ALL npz arrays per cube (across 23425
                    # cubes x4 DDP ranks) had bloated RAM and triggered the
                    # container memory-cgroup OOM during the good_ratio scan.
                    if self.use_real_dates:
                        date_keys = (
                            "highresdates", "dates", "highresyear", "highresyears",
                        )
                        meta = {k: data[k] for k in date_keys if k in data.files}
                    else:
                        meta = {}
            except Exception as e:
                print(f"[WARN] failed load {npz_path}: {e}")
                continue

            if hd.ndim != 4 or hd.shape[2] < 5:
                print(f"[WARN] invalid highresdynamic shape {hd.shape}: {npz_path}")
                continue

            try:
                good_ratio = _compute_frame_good_ratio(hd)  # (T,)
            except Exception as e:
                print(f"[WARN] failed quality parsing {npz_path}: {e}")
                continue

            valid_t = np.where(good_ratio >= target_ratio)[0].tolist()
            if not valid_t:
                best_t = int(np.argmax(good_ratio))
                valid_t = [best_t]

            samples.append((str(npz_path), valid_t, meta))

        # Each cube contributes exactly one multi-frame sample; we keep the
        # Dataset length = number of usable cubes.
        print(
            f"[{self.mode.upper()}] temporal: {len(samples)} usable cubes, "
            f"num_timesteps={self.num_timesteps}, stride={self.temporal_stride}"
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_npz(self, npz_path: str) -> Dict[str, np.ndarray]:
        if self.cache_size <= 0:
            with np.load(npz_path, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}

        if npz_path in self._npz_cache:
            self._npz_cache.move_to_end(npz_path)
            return self._npz_cache[npz_path]

        with np.load(npz_path, allow_pickle=False) as raw:
            data = {k: raw[k] for k in raw.files}
        self._npz_cache[npz_path] = data
        self._npz_cache.move_to_end(npz_path)
        while len(self._npz_cache) > self.cache_size:
            self._npz_cache.popitem(last=False)
        return data

    def _sample_frame_indices(self, valid_t: List[int]) -> List[int]:
        """Pick num_timesteps frame indices from valid_t, optionally strided."""
        T = self.num_timesteps
        if len(valid_t) >= T:
            # Random start along the strided candidate grid; wrap with modulo so
            # we always produce exactly T indices.
            candidates = list(range(0, len(valid_t), max(1, self.temporal_stride)))
            if len(candidates) >= T:
                start = self._rng.randint(0, len(candidates) - T)
                chosen = [valid_t[candidates[start + j]] for j in range(T)]
            else:
                # Stride too coarse: fall back to a plain even-ish selection.
                idxs = [valid_t[int(round(j * (len(valid_t) - 1) / max(1, T - 1)))] for j in range(T)]
                chosen = idxs if len(set(idxs)) == len(idxs) else self._rng.sample(valid_t, T)
                # ensure sorted by cube-time
                chosen = sorted(chosen)
            return chosen
        # Fewer valid frames than requested: sample with replacement to pad.
        chosen = [self._rng.choice(valid_t) for _ in range(T)]
        return chosen

    def _maybe_real_time_index(self, npz_meta: Dict[str, np.ndarray], frame_idx: int) -> int:
        """Return days-since-2000 for a frame if a date field is available,
        else None (caller falls back to relative ordering)."""
        if not self.use_real_dates:
            return None
        for key in ("highresdates", "dates", "highresyear", "highresyears"):
            arr = npz_meta.get(key)
            if arr is None:
                continue
            try:
                arr = np.asarray(arr).ravel()
                return int(arr[frame_idx])
            except Exception:
                continue
        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        npz_path, valid_t, meta = self.samples[idx]
        data = self._get_npz(npz_path)
        hd = data["highresdynamic"]  # (H,W,C,T)

        chosen = self._sample_frame_indices(valid_t)  # len == num_timesteps
        frames = []
        t_index = []
        for k, t in enumerate(chosen):
            s2 = hd[:, :, :4, int(t)].astype(np.float32)  # (H,W,4)
            np.nan_to_num(s2, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            np.clip(s2, 0.0, 1.0, out=s2)
            frames.append(torch.from_numpy(s2).permute(2, 0, 1).contiguous())  # (4,H,W)
            # Relative index by default; absolute days-since-2000 if available.
            real = self._maybe_real_time_index(meta, int(t))
            t_index.append(real if real is not None else k)

        s2_seq = torch.stack(frames, dim=0)  # (T,4,H,W)
        t_index_t = torch.tensor(t_index, dtype=torch.long)  # (T,)

        return {
            "s2_image": s2_seq,
            "t_index": t_index_t,
            "has_s1": False,
            "s1_image": torch.empty(0, dtype=torch.float32),
        }


class EarthNet2021DataModule(LightningDataModule):
    def __init__(
        self,
        earthnet_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        data_qua_thresh: float = 0.3,
        seed: int = 42,
        max_files: int = -1,
        max_frames_per_file: int = -1,
        cache_size: int = 4,
        temporal: bool = False,
        num_timesteps: int = 4,
        temporal_stride: int = 1,
        use_real_dates: bool = False,
    ):
        super().__init__()
        self.earthnet_dir = earthnet_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_split = float(val_split)
        self.data_qua_thresh = float(data_qua_thresh)
        self.seed = int(seed)
        self.max_files = int(max_files)
        self.max_frames_per_file = int(max_frames_per_file)
        self.cache_size = int(cache_size)
        self.temporal = bool(temporal)
        self.num_timesteps = int(num_timesteps)
        self.temporal_stride = int(temporal_stride)
        self.use_real_dates = bool(use_real_dates)

        self.wvs_dict = {
            # EarthNet2021 optical bands: B, G, R, NIR
            "s2": torch.tensor([0.490, 0.560, 0.665, 0.842], dtype=torch.float32),
        }

        self.train_dataset: Optional[EarthNet2021FrameDataset] = None
        self.val_dataset: Optional[EarthNet2021FrameDataset] = None

    def _collect_npz_paths(self) -> List[Path]:
        root = Path(self.earthnet_dir)
        if not root.exists():
            raise FileNotFoundError(f"earthnet_dir not found: {root}")
        paths = sorted(root.rglob("*.npz"))
        if self.max_files > 0:
            paths = paths[: self.max_files]
        if len(paths) == 0:
            raise RuntimeError(f"No npz found under: {root}")
        return paths

    def _split_paths(self, paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        rng = random.Random(self.seed)
        shuffled = paths.copy()
        rng.shuffle(shuffled)

        if self.val_split <= 0.0:
            return shuffled, []

        split_idx = int(len(shuffled) * (1.0 - self.val_split))
        split_idx = min(max(split_idx, 1), len(shuffled) - 1)  # train/val 都至少 1 个文件
        return shuffled[:split_idx], shuffled[split_idx:]

    def setup(self, stage: str = None):
        npz_paths = self._collect_npz_paths()
        train_paths, val_paths = self._split_paths(npz_paths)
        print(f"[EarthNet2021] files: train={len(train_paths)}, val={len(val_paths)}")

        if self.temporal:
            self.train_dataset = EarthNet2021TemporalDataset(
                npz_paths=train_paths,
                data_qua_thresh=self.data_qua_thresh,
                mode="train",
                seed=self.seed,
                max_frames_per_file=self.max_frames_per_file,
                cache_size=self.cache_size,
                num_timesteps=self.num_timesteps,
                temporal_stride=self.temporal_stride,
                use_real_dates=self.use_real_dates,
            )
            self.val_dataset = EarthNet2021TemporalDataset(
                npz_paths=val_paths if len(val_paths) > 0 else train_paths,
                data_qua_thresh=self.data_qua_thresh,
                mode="val",
                seed=self.seed + 1,
                max_frames_per_file=self.max_frames_per_file,
                cache_size=self.cache_size,
                num_timesteps=self.num_timesteps,
                temporal_stride=self.temporal_stride,
                use_real_dates=self.use_real_dates,
            )
            self._collate_fn = self._collate_temporal
        else:
            self.train_dataset = EarthNet2021FrameDataset(
                npz_paths=train_paths,
                data_qua_thresh=self.data_qua_thresh,
                mode="train",
                seed=self.seed,
                max_frames_per_file=self.max_frames_per_file,
                cache_size=self.cache_size,
            )
            self.val_dataset = EarthNet2021FrameDataset(
                npz_paths=val_paths if len(val_paths) > 0 else train_paths,
                data_qua_thresh=self.data_qua_thresh,
                mode="val",
                seed=self.seed + 1,
                max_frames_per_file=self.max_frames_per_file,
                cache_size=self.cache_size,
            )
            self._collate_fn = None  # use _collate_train / _collate_val below

        print(
            f"[EarthNet2021] total {'cubes' if self.temporal else 'frames'}: "
            f"train={len(self.train_dataset)}, val={len(self.val_dataset)}"
            + (" [TEMPORAL]" if self.temporal else "")
        )

    def _collate_train(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        images = [b["s2_image"] for b in batch]
        return {
            "image": torch.stack(images, dim=0),
            "wvs": self.wvs_dict["s2"],
            "modality": "s2",
        }

    def _collate_val(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        images = [b["s2_image"] for b in batch]
        return {
            "image": torch.stack(images, dim=0),
            "wvs": self.wvs_dict["s2"],
            "modality": "s2",
        }

    def _collate_temporal(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Stack [T,4,H,W] samples into [B,T,4,H,W] + per-sample t_index [B,T]."""
        images = [b["s2_image"] for b in batch]  # each (T,4,H,W)
        t_indices = [b["t_index"] for b in batch]  # each (T,)
        return {
            "image": torch.stack(images, dim=0),           # [B,T,C,H,W]
            "t_index": torch.stack(t_indices, dim=0),      # [B,T]
            "wvs": self.wvs_dict["s2"],
            "modality": "s2",
        }

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_temporal if self.temporal else self._collate_train,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader().")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_temporal if self.temporal else self._collate_val,
            persistent_workers=self.num_workers > 0,
        )
