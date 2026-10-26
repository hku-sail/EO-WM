from __future__ import annotations
# eo_vae/datasets/mmdc_datamodule.py
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightning import LightningDataModule
import random
from tqdm import tqdm

_S2_BAND_CLIP_MAX = np.array(
    [3000., 3000., 3000., 3000., 6000., 6000., 6000., 6000., 4500., 4000.],
    dtype=np.float32
)


class MMDCDataset(Dataset):
    def __init__(self, h5_path: str, keys: list, data_qua_thresh: float = 0.3, mode: str = "train"):
        self.h5_path = h5_path
        self.keys = keys
        self.data_qua_thresh = data_qua_thresh
        self.mode = mode
        self._file_handle = None
        self.valid_frames = self._build_index()

    def _build_index(self):
        valid_frames = []
        with h5py.File(self.h5_path, 'r', libver='latest', swmr=True) as f:
            for key in tqdm(self.keys, desc=f"Scanning [{self.h5_path}] {self.mode} frames"):
                grp = f[f'samples/{key}']
                clm = grp['s2_CLM'][:].astype(np.float32)
                T = clm.shape[0]

                if 's2_dataMask' in grp:
                    datamask = grp['s2_dataMask'][:].astype(np.float32)
                    if datamask.ndim == 4:
                        datamask = datamask[:, 0] if datamask.shape[1] == 1 else datamask[..., 0]
                else:
                    datamask = np.ones((T, 128, 128), dtype=np.float32)

                cloud_clear = 1.0 - np.clip(clm, 0.0, 1.0)
                good_pixels = datamask * cloud_clear
                good_ratio = good_pixels.reshape(T, -1).mean(axis=1)

                target_ratio = 1.0 - self.data_qua_thresh
                valid_t = np.where(good_ratio >= target_ratio)[0]

                if len(valid_t) > 0:
                    for t in valid_t:
                        valid_frames.append((key, t))
                else:
                    best_t = np.argsort(good_ratio)[-1]
                    valid_frames.append((key, best_t))

        print(f"[{self.mode.upper()}] [{os.path.basename(self.h5_path)}] "
              f"Found {len(valid_frames)} valid frames from {len(self.keys)} locations.")
        return valid_frames

    def __len__(self) -> int:
        return len(self.valid_frames)

    def _get_file(self) -> h5py.File:
        if self._file_handle is None or not self._file_handle.id.valid:
            self._file_handle = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        return self._file_handle

    def _s1_normalize(self, s1_raw: np.ndarray) -> np.ndarray:
        vv = np.clip(s1_raw[..., 0], 0.0, 1.0)
        vh = np.clip(s1_raw[..., 1], 0.0, 0.5) / 0.5
        angle = np.clip(s1_raw[..., 2], 0.0, 90.0) / 90.0
        return np.stack([vv, vh, angle], axis=-1).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        key, t = self.valid_frames[idx]
        f = self._get_file()
        grp = f[f'samples/{key}']

        s2_np = grp['s2'][t, :, :, :10].astype(np.float32) / 10000.0
        s2_np = np.clip(s2_np, 0.0, 1.0)
        s2_tensor = torch.from_numpy(s2_np).permute(2, 0, 1).contiguous()


        has_s1 = 's1_asc' in grp
        s1_tensor = torch.empty(0)
        if has_s1:
            s1_np = self._s1_normalize(grp['s1_asc'][t])
            s1_tensor = torch.from_numpy(s1_np).permute(2, 0, 1).contiguous()

        return {
            "s2_image": s2_tensor,
            "has_s1": has_s1,
            "s1_image": s1_tensor
        }

    def __del__(self):
        if self._file_handle is not None and self._file_handle.id.valid:
            self._file_handle.close()


class MMDCDataModule(LightningDataModule):
    def __init__(
        self,
        h5_path: str,                    # 主数据集路径（保持向后兼容）
        batch_size: int = 16,
        num_workers: int = 4,
        val_split: float = 0.1,
        data_qua_thresh: float = 0.3,
        extra_h5_paths: list = None,     # Collect paths for all dataset splits.
    ):
        super().__init__()
        self.h5_path = h5_path
        self.extra_h5_paths = extra_h5_paths or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.data_qua_thresh = data_qua_thresh

        self.wvs_dict = {
            "s2": torch.tensor(
                [0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190],
                dtype=torch.float32
            ),
            "s1": torch.tensor([5.4, 5.6, 5.4], dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # 内部工具：从单个 H5 文件读取所有 keys，并按 val_split 分割
    # ------------------------------------------------------------------
    def _split_keys(self, h5_path: str, seed_offset: int = 0):
        """返回 (train_keys, val_keys)"""
        with h5py.File(h5_path, 'r') as f:
            all_keys = list(f['samples'].keys())
        rng = random.Random(42 + seed_offset)
        rng.shuffle(all_keys)
        split_idx = int(len(all_keys) * (1 - self.val_split))
        return all_keys[:split_idx], all_keys[split_idx:]

    # ------------------------------------------------------------------
    def setup(self, stage: str = None):
        if isinstance(self.extra_h5_paths, str):
            self.extra_h5_paths = [self.extra_h5_paths]
        elif self.extra_h5_paths is None:
            self.extra_h5_paths = []
        all_h5_paths = [self.h5_path] + self.extra_h5_paths

        train_datasets = []
        val_datasets = []

        for i, h5_path in enumerate(all_h5_paths):
            train_keys, val_keys = self._split_keys(h5_path, seed_offset=i)
            print(f"[H5 #{i}] {os.path.basename(h5_path)} → "
                  f"train={len(train_keys)}, val={len(val_keys)}")

            train_datasets.append(
                MMDCDataset(h5_path, train_keys,
                            data_qua_thresh=self.data_qua_thresh, mode="train")
            )
            val_datasets.append(
                MMDCDataset(h5_path, val_keys,
                            data_qua_thresh=self.data_qua_thresh, mode="val")
            )

        # ConcatDataset 无缝拼接，训练/验证 DataLoader 行为完全不变
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset   = ConcatDataset(val_datasets)

        print(f"[TOTAL] train={len(self.train_dataset)}, val={len(self.val_dataset)} frames")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=True, collate_fn=self._collate_train,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=self._collate_val,
            persistent_workers=self.num_workers > 0,
        )

    def _collate_train(self, batch):
        target_modality = random.choice(["s2", "s1"])
        target_modality = "s2"  # 当前固定 s2

        if target_modality == "s1":
            all_have_s1 = all(b["has_s1"] for b in batch)
            if not all_have_s1:
                target_modality = "s2"

        images = []
        for b in batch:
            images.append(b["s2_image"] if target_modality == "s2" else b["s1_image"])

        return {
            "image": torch.stack(images),
            "wvs": self.wvs_dict[target_modality],
            "modality": target_modality,
        }

    def _collate_val(self, batch):
        images = [b["s2_image"] for b in batch]
        return {
            "image": torch.stack(images),
            "wvs": self.wvs_dict["s2"],
            "modality": "s2",
        }