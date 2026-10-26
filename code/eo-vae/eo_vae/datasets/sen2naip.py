"""Sen2Naip Dataset for Super Resolution."""
from __future__ import annotations

import logging

logging.getLogger('rasterio._env').setLevel(logging.ERROR)

import json
import os
from collections.abc import Callable, Sequence
from glob import glob
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import torch
import torch.nn.functional as F
from shapely.wkt import loads as wkt_loads
from torch import Tensor
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import Path, array_to_tensor


def assign_spatial_split(
    df: pd.DataFrame,
    lon_col: str = 'lon',
    lat_col: str = 'lat',
    n_blocks_x: int = 12,
    n_blocks_y: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Split dataset into spatial blocks (grid) and randomly assign blocks to splits.
    Implementing the 'random' pattern strategy.
    """
    if df.empty:
        return df

    # 1. Calculate Bounds and Grid
    minx, maxx = df[lon_col].min(), df[lon_col].max()
    miny, maxy = df[lat_col].min(), df[lat_col].max()

    # Buffer to avoiding edge issues
    buffer_x = (maxx - minx) * 0.001
    buffer_y = (maxy - miny) * 0.001
    minx -= buffer_x
    maxx += buffer_x
    miny -= buffer_y
    maxy += buffer_y

    x_step = (maxx - minx) / n_blocks_x
    y_step = (maxy - miny) / n_blocks_y

    # Assign Grid Indices
    block_x = ((df[lon_col] - minx) / x_step).astype(int).clip(0, n_blocks_x - 1)
    block_y = ((df[lat_col] - miny) / y_step).astype(int).clip(0, n_blocks_y - 1)
    df['block_id'] = block_y * n_blocks_x + block_x

    # 2. Block-wise Splitting (Random)
    # Generate all possible block IDs for the grid
    total_grid_blocks = n_blocks_x * n_blocks_y
    all_blocks = np.arange(total_grid_blocks)

    rs = np.random.RandomState(random_state)
    rs.shuffle(all_blocks)

    # Determine number of blocks for each split (e.g. 10% val, 10% test)
    n_test_blocks = max(1, int(total_grid_blocks * 0.1))
    n_val_blocks = max(1, int(total_grid_blocks * 0.1))

    # Assign sets of blocks
    test_blocks = set(all_blocks[:n_test_blocks])
    val_blocks = set(all_blocks[n_test_blocks : n_test_blocks + n_val_blocks])
    # Remaining are train

    def get_split_label(bid):
        if bid in test_blocks:
            return 'test'
        if bid in val_blocks:
            return 'val'
        return 'train'

    df['split'] = df['block_id'].map(get_split_label)
    return df


class Sen2NaipCrossSensor(NonGeoDataset):
    """Sen2Naip Cross Sensor dataset for super resolution."""

    # filename = "sen2naipv2-histmatch.0000.part.taco"
    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Sen2Naip dataset instance.

        Args:
            root: Root directory where the dataset should be stored.
            split: Dataset split to load. Must be one of 'train' or 'val'.
            transforms: A function/transform that takes input sample and its target as entry
                and returns a transformed version.
            download: Whether to download the dataset if it is not found on disk.
            checksum: Whether to verify the integrity of the dataset after download.
        """
        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self.aois = sorted(
            glob(os.path.join(self.root, '**', 'metadata.json'), recursive=True)
        )

        self.metadata_df = self._create_spatial_split(self.aois)

    def _create_spatial_split(self, aoi_paths: list[str]) -> pd.DataFrame:
        """Parse metadata and create a deterministic spatial split."""
        data = []
        transformers = {}
        target_crs = pyproj.CRS('EPSG:4326')

        for path in aoi_paths:
            with open(path) as f:
                meta = json.load(f)

            # Extract geometry info
            wkt = meta.get('proj:geometry')
            epsg = meta.get('proj:epsg')
            if not wkt or not epsg:
                continue

            # Get or create transformer for this EPSG
            if epsg not in transformers:
                source_crs = pyproj.CRS(f'EPSG:{epsg}')
                transformers[epsg] = pyproj.Transformer.from_crs(
                    source_crs, target_crs, always_xy=True
                ).transform

            # Parse geometry and reproject centroid to WGS84
            geom = wkt_loads(wkt)
            minx, miny, maxx, maxy = geom.bounds
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            lon, lat = transformers[epsg](cx, cy)

            data.append(
                {
                    'aoi_id': meta.get('id', os.path.basename(os.path.dirname(path))),
                    'lr_path': os.path.join(os.path.dirname(path), 'lr.tif'),
                    'hr_path': os.path.join(os.path.dirname(path), 'hr.tif'),
                    'lon': lon,
                    'lat': lat,
                }
            )

        df = pd.DataFrame(data)

        df = assign_spatial_split(df, lon_col='lon', lat_col='lat')


        df = df[df['split'] == self.split].reset_index(drop=True)
        return df

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return dataset sample at this index."""
        row = self.metadata_df.iloc[idx]

        lr_path: str = row['lr_path']
        hr_path: str = row['hr_path']

        with rasterio.open(lr_path) as lr_src, rasterio.open(hr_path) as hr_src:
            lr_data: np.ndarray = lr_src.read()
            hr_data: np.ndarray = hr_src.read()

        # resize hr image to 512x512 from 520x520
        # resize lr image to 128x128 from 121x121
        hq_image = F.interpolate(
            array_to_tensor(hr_data).float().unsqueeze(0),
            size=512,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        lq_image = F.interpolate(
            array_to_tensor(lr_data).float().unsqueeze(0),
            size=128,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        sample = {'image_lr': lq_image, 'image_hr': hq_image, 'aoi': row['aoi_id']}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class RunningStatsButFast(torch.nn.Module):
    def __init__(self, shape, dims):
        """Initializes the RunningStatsButFast method.

        A PyTorch module that can be put on the GPU and calculate the multidimensional
        mean and variance of inputs online in a numerically stable way. This is useful
        for calculating the channel-wise mean and variance of a big dataset because you
        don't have to load the entire dataset into memory.

        Uses the "Parallel algorithm" from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        Similar implementation here: https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py#L5

        Access the mean, variance, and standard deviation of the inputs with the
        `mean`, `var`, and `std` attributes.

        Example:
        ```
        rs = RunningStatsButFast((12,), [0, 2, 3])
        for inputs, _ in dataloader:
            rs(inputs)
        print(rs.mean)
        print(rs.var)
        print(rs.std)
        ```

        Args:
            shape: The shape of resulting mean and variance. For example, if you
                are calculating the mean and variance over the 0th, 2nd, and 3rd
                dimensions of inputs of size (64, 12, 256, 256), this should be 12.
            dims: The dimensions of your input to calculate the mean and variance
                over. In the above example, this should be [0, 2, 3].
        """
        super(RunningStatsButFast, self).__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('std', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1))
        self.register_buffer('min', torch.zeros(shape))
        self.register_buffer('max', torch.zeros(shape))
        self.dims = dims

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=self.dims)
            batch_var = torch.var(x, dim=self.dims)
            batch_min = torch.amin(x, dim=self.dims)
            batch_max = torch.amax(x, dim=self.dims)
            batch_count = torch.tensor(x.shape[self.dims[0]], dtype=torch.float)

            n_ab = self.count + batch_count
            m_a = self.mean * self.count
            m_b = batch_mean * batch_count
            M2_a = self.var * self.count
            M2_b = batch_var * batch_count

            delta = batch_mean - self.mean

            self.mean = (m_a + m_b) / (n_ab)
            # we don't subtract -1 from the denominator to match the standard Numpy/PyTorch variances
            self.var = (M2_a + M2_b + delta**2 * self.count * batch_count / (n_ab)) / (
                n_ab
            )
            self.count += batch_count
            self.std = torch.sqrt(self.var + 1e-8)

            self.min = torch.min(self.min, batch_min)
            self.max = torch.max(self.max, batch_max)

    def forward(self, x):
        self.update(x)
        return x



LATENT_STATS = {
    'eo-vae': {
        'mean': torch.tensor(
            [
                -1.7888,
                0.2182,
                2.2564,
                1.4281,
                -1.0304,
                -0.3264,
                1.5834,
                -2.0467,
                -0.1252,
                -0.2653,
                3.3076,
                -2.5082,
                -0.2019,
                -0.2360,
                1.9879,
                -0.7527,
                -2.2597,
                1.2077,
                1.3112,
                -1.9992,
                -0.5512,
                -0.9615,
                -0.8980,
                0.8066,
                -0.9225,
                -0.7091,
                2.0922,
                0.7928,
                1.4194,
                -0.8662,
                0.7048,
                -0.2155,
            ]
        ),
        'std': torch.tensor(
            [
                1.8264,
                1.4056,
                1.8661,
                1.8705,
                1.3326,
                1.2475,
                2.4357,
                1.5394,
                1.9876,
                1.3553,
                1.4558,
                1.8897,
                1.5064,
                1.3615,
                1.4830,
                1.8453,
                1.5027,
                1.4612,
                1.3570,
                2.1994,
                1.3983,
                1.3112,
                1.4463,
                1.2903,
                1.5551,
                1.5043,
                1.7396,
                1.3403,
                1.5106,
                1.8003,
                1.2992,
                1.4572,
            ]
        ),
    },
    'flux-vae': {
        'mean': torch.tensor(
            [
                -0.3110,
                -0.2957,
                -0.1417,
                0.2420,
                -0.4210,
                0.5541,
                -0.2168,
                -0.5609,
                -0.0898,
                0.0221,
                0.6188,
                -0.2450,
                -0.0713,
                0.1702,
                -0.1953,
                0.9041,
                1.3106,
                -0.0538,
                -0.0452,
                0.7375,
                -0.2078,
                -0.5744,
                -0.1226,
                0.1522,
                0.2211,
                0.5034,
                0.6076,
                0.6781,
                -0.3545,
                -0.0317,
                -0.6657,
                0.1903,
            ]
        ),
        'std': torch.tensor(
            [
                1.6392,
                1.4433,
                1.6919,
                2.1979,
                1.4754,
                1.9841,
                1.7678,
                1.6417,
                2.0310,
                1.4272,
                1.6523,
                1.3537,
                1.5575,
                1.6536,
                1.6211,
                2.2615,
                1.5211,
                1.4070,
                1.5067,
                2.2955,
                1.5587,
                1.5841,
                1.4642,
                1.6591,
                1.5140,
                1.4887,
                1.6871,
                1.6049,
                1.5996,
                1.6341,
                1.5165,
                1.7143,
            ]
        ),
    },
    'flux-vae-01': {
        'mean': torch.tensor(
            [
                0.9525,
                -0.2780,
                -1.6124,
                -0.8007,
                0.5785,
                0.7348,
                -0.6915,
                0.8150,
                -0.0080,
                0.3117,
                -1.8728,
                1.3549,
                0.4879,
                0.0548,
                -1.0679,
                0.1232,
                2.1280,
                -0.3882,
                -0.7236,
                0.5356,
                0.6244,
                0.4626,
                0.3694,
                -0.6455,
                0.7851,
                0.7942,
                -0.9845,
                -0.1993,
                -0.4270,
                1.0656,
                -0.6977,
                -0.3048,
            ]
        ),
        'std': torch.tensor(
            [
                0.8103,
                1.0975,
                0.7813,
                0.6653,
                1.0548,
                1.2067,
                0.6721,
                0.9574,
                0.6883,
                0.9746,
                1.0592,
                0.6727,
                1.0272,
                1.0635,
                1.1388,
                0.8830,
                1.0074,
                1.0010,
                1.1671,
                0.7029,
                1.1314,
                1.2662,
                1.0016,
                1.1524,
                1.0172,
                0.9952,
                0.8982,
                1.2664,
                0.9789,
                0.8747,
                1.1687,
                1.0384,
            ]
        ),
    },
}


class Sen2NaipCrossSensorLatent(NonGeoDataset):
    """Sen2Naip latent encodings dataset."""

    valid_splits = ['train', 'val', 'test']

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        latent_scale_factor: float = 1.0,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        normalize: bool = True,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Sen2Naip dataset instance.

        Args:
            root: Root directory where the dataset should be stored.
            split: Dataset split to load. Must be one of 'train' or 'val'.
            latent_scale_factor: Scale factor to apply to the latent encodings.
            transforms: A function/transform that takes input sample and its target as entry
                and returns a transformed version.
            download: Whether to download the dataset if it is not found on disk.
            checksum: Whether to verify the integrity of the dataset after download.
        """
        assert split in self.valid_splits, f'Split must be one of {self.valid_splits}'

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.normalize = normalize
        self.latent_scale_factor = latent_scale_factor

        self.aois = glob(os.path.join(self.root, split, '*.npz'))

        self.metadata_df = pd.DataFrame(self.aois, columns=['path'])

        # --- Load Normalization Stats from JSON ---
        stats_path = os.path.join(self.root, 'latent_stats.json')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f'Latent stats file not found at {stats_path}')

        with open(stats_path) as f:
            stats_data = json.load(f)

        # Pre-load statistics as tensors to avoid overhead in __getitem__
        # We assume the file contains 'lr_latent' and 'hr_latent' keys
        self.lr_mean = torch.tensor(
            stats_data['lr_latent']['mean'], dtype=torch.float32
        ).view(-1, 1, 1)
        self.lr_std = torch.tensor(
            stats_data['lr_latent']['std'], dtype=torch.float32
        ).view(-1, 1, 1)

        self.hr_mean = torch.tensor(
            stats_data['hr_latent']['mean'], dtype=torch.float32
        ).view(-1, 1, 1)
        self.hr_std = torch.tensor(
            stats_data['hr_latent']['std'], dtype=torch.float32
        ).view(-1, 1, 1)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return dataset sample at this index.

        Args:
            idx: Index of the dataset sample.

        Returns:
            A dataset sample containing the lr and hr image
        """
        row = self.metadata_df.iloc[idx]
        path = row['path']

        with np.load(path) as data:
            hr_latent = torch.from_numpy(data['hr_latent'])
            lr_latent = torch.from_numpy(data['lr_latent'])
            orig_image_hr = torch.from_numpy(data['hr_image'])
            orig_image_lr = torch.from_numpy(data['lr_image'])

        # Normalize latents using the loaded statistics
        # IMPORTANT: We normalise BOTH LR and HR latents using the HR statistics.
        # This preserves the signal magnitude difference (blurriness) between LR and HR.
        # If we used lr_std, we would amplify the noise in the smooth LR latents.
        if self.normalize:
            hr_latent = (hr_latent - self.hr_mean) / self.hr_std
            lr_latent = (lr_latent - self.hr_mean) / self.hr_std

        # apply latent scale factor
        hr_latent = hr_latent * self.latent_scale_factor
        lr_latent = lr_latent * self.latent_scale_factor

        sample = {
            'image_hr': hr_latent,
            'image_lr': lr_latent,
            'orig_image_hr': orig_image_hr,
            'orig_image_lr': orig_image_lr,
            'wvs': torch.tensor([0.665, 0.56, 0.49, 0.842]),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


tm_indices = [3, 2, 1, 7]

tm_global_mean = torch.tensor(
    [
        1718.9949,
        1825.5669,
        2043.5834,
        2175.4543,
        2522.9522,
        3114.2216,
        3323.3469,
        3417.3660,
        3470.9655,
        3489.4869,
        2725.9735,
        2152.0551,
    ]
)
tm_global_std = torch.tensor(
    [
        2126.3409,
        2140.1035,
        2044.6618,
        2125.3351,
        2065.3251,
        1874.4652,
        1808.0426,
        1839.0210,
        1737.9521,
        1738.5136,
        1456.5919,
        1365.1743,
    ]
)


def sen2naip_cross_sensor_collate_fn(
    batch: Sequence[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Collate function for the Sen2NaipCrossSensor dataset."""
    # LR Stats
    lr_mean = torch.tensor([1302.9685, 1085.2820, 764.7739, 2769.4824]).view(1, 4, 1, 1)
    lr_std = torch.tensor([780.8768, 513.2825, 414.3385, 793.6396]).view(1, 4, 1, 1)

    # HR Stats
    hr_mean = torch.tensor([125.1176, 121.9117, 100.0240, 143.8500]).view(1, 4, 1, 1)
    hr_std = torch.tensor([39.8066, 30.3501, 28.9109, 28.8952]).view(1, 4, 1, 1)


    images_hr = torch.stack([sample['image_hr'] for sample in batch])

    # Z-score normalization for HR
    new_images_hr = (images_hr - hr_mean) / hr_std

    images_lr = torch.stack([sample['image_lr'] for sample in batch])

    # Z-score normalization for LR
    images_lr = (images_lr - lr_mean) / lr_std

    # Interpolate low res image to high res image size
    new_images_lr = F.interpolate(
        images_lr, size=images_hr.shape[-2:], mode='bicubic', align_corners=False
    )

    return {
        'image_lr': new_images_lr,
        'image_hr': new_images_hr,
        'aoi': [sample['aoi'] for sample in batch],
    }


def new_sen2naip_cross_sensor_collate_fn(
    batch: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for the Sen2NaipCrossSensor dataset.
    Performs Domain Adaptation to match Terramesh training statistics.
    """
    # --- 1. Define Normalization Constants ---

    # LR: Terramesh Global Stats (Source of Truth for the Model)
    # Select channels [4, 3, 2, 8] for cross-sensor normalization.
    # that successfully aligned the mean to -0.4.
    tm_lr_mean = torch.tensor([2199.116, 1853.926, 1718.211, 3132.235]).view(1, 4, 1, 1)
    tm_lr_std = torch.tensor([2105.179, 2152.477, 2059.311, 1775.656]).view(1, 4, 1, 1)

    # HR: Sen2NAIP Local Stats (To standardize input to N(0,1))
    naip_mean = torch.tensor([125.1176, 121.9118, 100.0240, 143.8501]).view(1, 4, 1, 1)
    naip_std = torch.tensor([39.8066, 30.3501, 28.9109, 28.8951]).view(1, 4, 1, 1)

    # HR: Target Domain Shift (To look like "Clean Land" in Terramesh)
    # Target mean for normalized training inputs: -0.4.
    # Target Scale 0.6 (conservative variance for land-only patches)
    target_loc = -0.4
    target_scale = 0.6

    # --- 2. Process High Res (NAIP) ---
    images_hr = torch.stack([sample['image_hr'] for sample in batch])

    # A. Standardize NAIP to N(0, 1) using its own stats
    z_hr = (images_hr - naip_mean) / naip_std

    # B. Shift to Training Domain (Domain Adaptation)
    new_images_hr = z_hr * target_scale + target_loc

    # --- 3. Process Low Res (Sentinel-2) ---
    images_lr = torch.stack([sample['image_lr'] for sample in batch])

    # A. Safety: Clamp negative values (nodata handling)
    images_lr = torch.clamp(images_lr, min=0.0)

    # B. Normalize using TERRAMESH stats (Force alignment)
    # Note: We do NOT use the local Sen2Naip mean/std here.
    images_lr_norm = (images_lr - tm_lr_mean) / tm_lr_std

    # C. Interpolate to HR size
    # We interpolate the *normalized* features to avoid scaling artifacts
    new_images_lr = F.interpolate(
        images_lr_norm, size=images_hr.shape[-2:], mode='bicubic', align_corners=False
    )

    return {
        'image_lr': new_images_lr,
        'image_hr': new_images_hr,
        'aoi': [sample['aoi'] for sample in batch],
    }


class Sen2NaipCrossSensorDataModule(NonGeoDataModule):
    std = torch.tensor([1])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new Original WorldStrat DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Sen2NaipCrossSensor`.
        """
        super().__init__(Sen2NaipCrossSensor, batch_size, num_workers, **kwargs)

        self.collate_fn = sen2naip_cross_sensor_collate_fn

    def setup(self, stage: str) -> None:
        """Set up datasets."""
        self.train_dataset = self.dataset_class(**self.kwargs, split='train')
        self.val_dataset = self.dataset_class(**self.kwargs, split='val')
        self.test_dataset = self.dataset_class(**self.kwargs, split='test')

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Hook to modify batch after transfer to device."""
        # IMPORTANT to not use torchgeo default for now
        return batch


class Sen2NaipLatentCrossSensorDataModule(NonGeoDataModule):
    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new Original WorldStrat DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Sen2NaipCrossSensor`.
        """
        super().__init__(Sen2NaipCrossSensorLatent, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets."""
        self.train_dataset = self.dataset_class(split='train', **self.kwargs)
        self.val_dataset = self.dataset_class(split='val', **self.kwargs)
        self.test_dataset = self.dataset_class(split='test', **self.kwargs)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Hook to modify batch after transfer to device."""
        # IMPORTANT to not use torchgeo default for now
        return batch


import torch
import torch.nn as nn


class StreamingHistogram(nn.Module):
    def __init__(self, num_channels, min_val=-1200, max_val=1000, bins=2200):
        """Accumulates histogram counts for a specific value range.
        Default range [-1200, 1000] with 2200 bins gives us ~1 unit precision per bin.
        """
        super().__init__()
        self.num_channels = num_channels
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins

        # Store counts: Shape (Channels, Bins)
        self.register_buffer('hist_counts', torch.zeros(num_channels, bins))

        # Create bin edges for plotting later
        self.bin_edges = torch.linspace(min_val, max_val, steps=bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def update(self, x):
        """Updates the histogram with a new batch of data."""
        with torch.no_grad():
            # Flatten: (B, C, H, W) -> (C, -1)
            if x.ndim == 4:
                x_flat = x.permute(1, 0, 2, 3).reshape(self.num_channels, -1)
            else:
                x_flat = x.reshape(self.num_channels, -1)

            for c in range(self.num_channels):
                # torch.histc only works on 1D tensors and doesn't do batching well
                # So we loop over channels (fast enough since C is small, e.g. 12)
                batch_hist = torch.histc(
                    x_flat[c].float(),
                    bins=self.bins,
                    min=self.min_val,
                    max=self.max_val,
                )
                self.hist_counts[c] += batch_hist

    def plot(self, channel_names=None):
        """Plots the accumulated histograms."""
        counts_np = self.hist_counts.cpu().numpy()
        centers_np = self.bin_centers.cpu().numpy()

        plt.figure(figsize=(15, 8))

        for c in range(self.num_channels):
            label = channel_names[c] if channel_names else f'Ch {c}'

            # Use log scale for Y because NoData spikes are usually huge
            # compared to normal data
            plt.plot(centers_np, counts_np[c], label=label, alpha=0.7)

        plt.yscale('log')
        plt.title("Data Distribution: The 'Problem Zone' (Negative Values)")
        plt.xlabel('Pixel Value')
        plt.ylabel('Count (Log Scale)')
        plt.grid(True, which='both', ls='-', alpha=0.2)
        plt.legend()

        # Draw lines to help analysis
        plt.axvline(
            x=-999, color='r', linestyle='--', alpha=0.5, label='Expected NoData (-999)'
        )
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Zero')

        plt.savefig('streaming_histogram_sen2naip.png')
        plt.show()


if __name__ == '__main__':
    from tqdm import tqdm

    dm = Sen2NaipCrossSensorDataModule(
        root='/path/to/sen2naip/cross-sensor/cross-sensor',
        batch_size=16,
        num_workers=4,
    )
    dm.setup('fit')

    train_loader = dm.train_dataloader()

    hist = StreamingHistogram(num_channels=4, min_val=-100, max_val=10000, bins=2000)

    for batch in tqdm(train_loader):
        hist.update(batch['image_lr'])

    hist.plot(channel_names=['Red', 'Green', 'Blue', 'NIR'])
