import os
import random

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningDataModule
from omegaconf import ListConfig
from torch.utils.data import DataLoader

from .terramesh import build_terramesh_dataset

# =============================================================================
# Constants
# =============================================================================

# Wavelength definitions (unchanged)
WAVELENGTHS = {
    'S2RGB': [0.665, 0.56, 0.49],
    'S1RTC': [5.4, 5.6],
    'S2L2A': [
        0.443,
        0.490,
        0.560,
        0.665,
        0.705,
        0.740,
        0.783,
        0.842,
        0.865,
        1.610,
        2.190,
        0.945,
    ],
    'S2L1C': [
        0.443,
        0.490,
        0.560,
        0.665,
        0.705,
        0.740,
        0.783,
        0.842,
        0.865,
        0.945,
        1.375,
        1.610,
        2.190,
    ],
}

# Original z-score normalization stats (from TerraMesh)
NORM_STATS_LEGACY = {
    'S2L2A': {
        'mean': [
            1375.648,
            1489.600,
            1709.087,
            1831.752,
            2186.075,
            2794.358,
            3008.528,
            3096.780,
            3155.180,
            3169.651,
            2415.761,
            1838.622,
        ],
        'std': [
            2101.107,
            2138.673,
            2033.628,
            2118.186,
            2061.646,
            1869.234,
            1801.386,
            1841.173,
            1734.404,
            1751.174,
            1375.131,
            1284.165,
        ],
    },
    'S1RTC': {'mean': [-10.793, -17.198], 'std': [4.278, 4.346]},
    'S2L1C': {
        'mean': [
            2475.625,
            2260.839,
            2143.561,
            2230.225,
            2445.427,
            2992.950,
            3257.843,
            3171.695,
            3440.958,
            1567.433,
            561.076,
            2562.809,
            1924.178,
        ],
        'std': [
            1761.905,
            1804.267,
            1661.263,
            1932.020,
            1918.007,
            1812.421,
            1795.179,
            1734.280,
            1780.039,
            1082.531,
            512.077,
            1350.580,
            1177.511,
        ],
    },
    'S2RGB': {'mean': [110.349, 99.507, 75.843], 'std': [69.905, 53.708, 53.378]},
    'DEM': {'mean': [651.663], 'std': [928.168]},
}

# Alias for backward compatibility
NORM_STATS = NORM_STATS_LEGACY


# =============================================================================
# Custom Normalization Classes (New Scheme) - Time-Aware
# =============================================================================


class Sentinel2L2ANorm(nn.Module):
    """Custom normalization for Sentinel-2 L2A with clipping.

    Processing:
      1. Clip to [0, 10000]
      2. Apply z-score normalization with custom statistics

    Note: The +1000 time-aware shift for post-baseline images should be applied
    at the dataset/transform level for efficiency, not here.
    """

    def __init__(self):
        super().__init__()
        # Updated statistics computed with time-aware harmonization and 10k clipping
        self.register_buffer(
            'mean',
            torch.tensor(
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
            ).view(12, 1, 1),
        )

        self.register_buffer(
            'std',
            torch.tensor(
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
            ).view(12, 1, 1),
        )

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        """Normalize Sentinel-2 L2A images.

        Args:
            x: Image tensor (B, C, H, W) - should already have +1000 shift applied if needed
            timestamps: Ignored - shift should be done at dataset level

        Returns:
            Normalized tensor (B, C, H, W)
        """
        # 1. Clip to [0, 10000]
        x = torch.clamp(x, min=0.0, max=10000.0)

        # 2. Standardize (Z-Score)
        x = (x - self.mean) / self.std

        return x


class Sentinel2L1CNorm(nn.Module):
    """Custom normalization for Sentinel-2 L1C with clipping.

    Processing:
      1. Clip to [0, 10000]
      2. Apply z-score normalization with custom statistics

    Note: L1C (Top-of-Atmosphere) was not affected by the baseline change.
    """

    def __init__(self):
        super().__init__()
        # Computed L1C Stats (fixed dataset statistics)
        # 13 Channels: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
        self.register_buffer(
            'mean',
            torch.tensor(
                [
                    2424.2556,
                    2207.7019,
                    2098.2302,
                    2167.1584,
                    2382.3115,
                    2938.8499,
                    3204.8447,
                    3126.6599,
                    3389.0706,
                    1580.1287,
                    572.5726,
                    2552.1208,
                    1917.9390,
                ]
            ).view(13, 1, 1),
        )

        self.register_buffer(
            'std',
            torch.tensor(
                [
                    1700.3824,
                    1731.5450,
                    1610.9904,
                    1833.5536,
                    1808.5067,
                    1694.4427,
                    1678.2327,
                    1625.7446,
                    1659.3112,
                    1093.5255,
                    515.6395,
                    1300.8892,
                    1151.6169,
                ]
            ).view(13, 1, 1),
        )

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        """Normalize Sentinel-2 L1C images.

        Args:
            x: Image tensor (B, C, H, W)
            timestamps: Ignored - kept for API consistency

        Returns:
            Normalized tensor (B, C, H, W)
        """
        # 1. Clip to [0, 10000]
        x = torch.clamp(x, min=0.0, max=10000.0)

        # 2. Standardize (Z-Score)
        x = (x - self.mean) / self.std

        return x


class LegacyZScoreNorm(nn.Module):
    """Legacy z-score normalization using original TerraMesh stats."""

    def __init__(self, modality: str):
        super().__init__()
        if modality not in NORM_STATS_LEGACY:
            raise ValueError(f'Unknown modality {modality} for normalization.')

        stats = NORM_STATS_LEGACY[modality]
        self.register_buffer('mean', torch.tensor(stats['mean']).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor(stats['std']).view(-1, 1, 1))

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        """Apply legacy z-score normalization.

        Args:
            x: Image tensor (B, C, H, W)
            timestamps: Ignored - kept for API consistency
        """
        return (x - self.mean) / (self.std + 1e-8)


# =============================================================================
# Normalizer Factory
# =============================================================================


class NormalizerFactory:
    """Factory for creating normalizers based on scheme and modality."""

    # Mapping of (scheme, modality) -> normalizer class
    CUSTOM_NORMALIZERS = {
        ('custom', 'S2L2A'): Sentinel2L2ANorm,
        ('custom', 'S2L1C'): Sentinel2L1CNorm,
    }

    @classmethod
    def create(cls, modality: str, scheme: str = 'legacy') -> nn.Module:
        """Create a normalizer for the given modality and scheme.

        Args:
            modality: Modality name (e.g., 'S2L2A', 'S2L1C', 'S1RTC')
            scheme: Normalization scheme - 'legacy' (original z-score) or 'custom' (new scheme)

        Returns:
            nn.Module that normalizes tensors
        """
        if scheme == 'custom' and (scheme, modality) in cls.CUSTOM_NORMALIZERS:
            return cls.CUSTOM_NORMALIZERS[(scheme, modality)]()
        else:
            # Fall back to legacy z-score for all other cases
            return LegacyZScoreNorm(modality)

    @classmethod
    def get_available_schemes(cls) -> list:
        """Return list of available normalization schemes."""
        return ['legacy', 'custom']

    @classmethod
    def get_custom_modalities(cls) -> list:
        """Return list of modalities with custom normalization available."""
        return list(set(mod for (scheme, mod) in cls.CUSTOM_NORMALIZERS.keys()))


# =============================================================================
# Augmentation Functions
# =============================================================================


def apply_batch_augmentations(images: torch.Tensor) -> torch.Tensor:
    """Applies random geometric augmentations (D4 Symmetry Group) to a batch of EO images.
    Input: (B, C, H, W)

    Includes:
    - Random Horizontal Flip
    - Random Vertical Flip
    - Random 90-degree Rotation (0, 90, 180, 270)
    """
    # 1. Random Horizontal Flip
    if random.random() > 0.5:
        images = torch.flip(images, dims=[-1])

    # 2. Random Vertical Flip
    if random.random() > 0.5:
        images = torch.flip(images, dims=[-2])

    # 3. Random 90-degree Rotation
    k = random.randint(0, 3)
    if k > 0:
        images = torch.rot90(images, k, dims=[-2, -1])

    return images


# =============================================================================
# Legacy Normalization Functions (for backward compatibility)
# =============================================================================


def normalize_image(
    image: torch.Tensor, modality: str, method='zscore'
) -> torch.Tensor:
    """Normalize image tensor using legacy z-score method."""
    if modality not in NORM_STATS:
        raise ValueError(f'Unknown modality {modality} for normalization.')

    stats = NORM_STATS[modality]
    device = image.device

    if method == 'zscore':
        mean = torch.tensor(stats['mean'], device=device).view(-1, 1, 1)
        std = torch.tensor(stats['std'], device=device).view(-1, 1, 1)
        return (image - mean) / (std + 1e-8)
    else:
        raise ValueError(f'Normalization method {method} not supported.')


def unnormalize_image(
    image: torch.Tensor, modality: str, method='zscore'
) -> torch.Tensor:
    """Inverse normalization to recover original units (DN or dB)."""
    if modality not in NORM_STATS:
        return image

    stats = NORM_STATS[modality]
    device = image.device

    if method == 'zscore':
        mean = torch.tensor(stats['mean'], device=device).view(-1, 1, 1)
        std = torch.tensor(stats['std'], device=device).view(-1, 1, 1)
        return (image * std) + mean

    return image


# =============================================================================
# Collate Functions
# =============================================================================


def single_modality_collate_fn(
    modalities,
    normalize=True,
    norm_scheme='legacy',
    target_size=(224, 224),
    mode='train',
    return_metadata=False,
):
    """Collate function that randomly selects ONE modality per batch.

    Args:
        modalities: List of modalities to randomly select from (e.g., ['S2RGB', 'S2L2A'])
        normalize: Whether to apply normalization
        norm_scheme: 'legacy' for original z-score, 'custom' for new normalization
        target_size: Target image size for interpolation
        mode: 'train' or 'eval' - controls augmentation
        return_metadata: Whether to include metadata (time, etc.) in output

    Returns:
        Callable collate function
    """
    # Pre-create normalizers for all modalities
    normalizers = {
        mod: NormalizerFactory.create(mod, norm_scheme) for mod in modalities
    }

    def collate(batch):
        """Process batch by selecting a single random modality."""
        # Handle single modality case where key is "image"
        if len(modalities) == 1:
            selected_modality = modalities[0]
            if 'image' in batch:
                images = batch['image']
            else:
                raise ValueError(
                    f"Expected 'image' key for single modality {selected_modality}, "
                    f'but found: {batch.keys()}'
                )
        else:
            # Randomly select one modality for this batch
            available_modalities = [mod for mod in modalities if mod in batch]
            if not available_modalities:
                raise ValueError(
                    f'None of the specified modalities {modalities} found in '
                    f'batch keys {batch.keys()}'
                )
            selected_modality = random.choice(available_modalities)
            images = batch[selected_modality]

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        else:
            images = images.float()

        if normalize:
            normalizer = normalizers[selected_modality]
            images = normalizer(images)

        if target_size is not None and images.shape[-2:] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=target_size, mode='bilinear', align_corners=False
            )

        if mode == 'train':
            images = apply_batch_augmentations(images)

        # Get wavelengths for the selected modality
        wvs = torch.FloatTensor(WAVELENGTHS[selected_modality])

        result = {'image': images, 'wvs': wvs, 'modality': selected_modality}

        # Optionally include metadata
        if return_metadata:
            if 'time' in batch:
                timestamps = batch['time']
                if isinstance(timestamps, np.ndarray):
                    timestamps = torch.from_numpy(timestamps).long()
                result['time'] = timestamps
            # Add any other metadata fields from batch
            for key in ['lat', 'lon', 'crs', 'grid_id', 'center_lat', 'center_lon']:
                if key in batch:
                    result[key] = batch[key]

        return result

    return collate


def deterministic_modality_collate_fn(
    modality,
    normalize=True,
    norm_scheme='legacy',
    target_size=(224, 224),
    mode='train',
    return_metadata=False,
):
    """Collate function that always uses the SAME modality.

    Useful for validation where you want consistent modality.

    Args:
        modality: Single modality name (e.g., 'S2L2A')
        normalize: Whether to apply normalization
        norm_scheme: 'legacy' for original z-score, 'custom' for new normalization
        target_size: Target image size for interpolation
        mode: 'train' or 'eval' - controls augmentation
        return_metadata: Whether to include metadata (time, etc.) in output

    Returns:
        Callable collate function
    """
    # Pre-create normalizer for this modality
    normalizer = NormalizerFactory.create(modality, norm_scheme)

    def collate(batch):
        # Handle case where single modality uses "image" key
        if modality in batch:
            images = batch[modality]
        elif 'image' in batch:
            images = batch['image']
        else:
            raise ValueError(
                f'Modality {modality} not found in batch. Available: {batch.keys()}'
            )

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        else:
            images = images.float()

        if normalize:
            images = normalizer(images)

        if target_size is not None and images.shape[-2:] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=target_size, mode='bilinear', align_corners=False
            )

        if mode == 'train':
            images = apply_batch_augmentations(images)

        wvs = torch.FloatTensor(WAVELENGTHS[modality])

        result = {'image': images, 'wvs': wvs, 'modality': modality}

        # Optionally include metadata
        if return_metadata:
            if 'time' in batch:
                timestamps = batch['time']
                if isinstance(timestamps, np.ndarray):
                    timestamps = torch.from_numpy(timestamps).long()
                result['time'] = timestamps
            # Add any other metadata fields from batch
            for key in ['lat', 'lon', 'crs', 'grid_id', 'center_lat', 'center_lon']:
                if key in batch:
                    result[key] = batch[key]

        return result

    return collate


# =============================================================================
# DataModule
# =============================================================================


class TerraMeshDataModule(LightningDataModule):
    """Lightning DataModule for TerraMesh dataset with configurable normalization.

    Args:
        data_path: Path to TerraMesh data
        modalities: List of modalities to use
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_workers: Number of dataloader workers
        train_collate_mode: 'random' for random modality selection, or specific modality name
        val_collate_mode: Modality to use for validation
        normalize: Whether to apply normalization
        norm_scheme: Normalization scheme - 'legacy' or 'custom'
                    - 'legacy': Original z-score normalization from TerraMesh
                    - 'custom': New normalization with time-aware harmonization for S2L2A,
                               clipping for S2L1C
        target_size: Target image size (H, W)
        return_metadata: Whether to return metadata (time, lat, lon, etc.) in batches
        **kwargs: Additional arguments passed to build_terramesh_dataset

    Note on time-aware harmonization:
        When using norm_scheme='custom' with S2L2A modality, the normalizer will
        automatically apply the +1000 harmonization offset only to images captured
        ON OR AFTER January 24, 2022 (Sentinel-2 processing baseline change date).
        Images captured BEFORE this date do not receive the offset.

        This requires return_metadata=True in the dataset builder (set automatically
        when norm_scheme='custom' and S2L2A is in modalities).
    """

    def __init__(
        self,
        data_path,
        modalities,
        batch_size=8,
        eval_batch_size=16,
        num_workers=4,
        train_collate_mode='random',
        val_collate_mode='S2L2A',
        normalize=True,
        norm_scheme='legacy',
        norm_method='zscore',
        target_size=(224, 224),
        return_metadata=False,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.modalities = modalities
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.norm_scheme = norm_scheme
        self.norm_method = norm_method
        self.return_metadata = return_metadata
        self.kwargs = kwargs

        if isinstance(target_size, ListConfig):
            target_size = tuple(target_size)
        self.target_size = target_size

        # Make norm_stats accessible (legacy stats for reference)
        self.norm_stats = NORM_STATS_LEGACY

        # Validate modalities
        for mod in modalities:
            if mod not in WAVELENGTHS:
                raise ValueError(
                    f'Modality {mod} not supported. Available: {list(WAVELENGTHS.keys())}'
                )

        # Validate norm_scheme
        valid_schemes = NormalizerFactory.get_available_schemes()
        if norm_scheme not in valid_schemes:
            raise ValueError(
                f'norm_scheme must be one of {valid_schemes}, got {norm_scheme}'
            )

        # Check if we need the harmonization transform for S2L2A custom normalization
        self._needs_s2l2a_harmonization = (
            norm_scheme == 'custom' and 'S2L2A' in modalities
        )

        # Log which normalization is being used
        custom_mods = NormalizerFactory.get_custom_modalities()
        if norm_scheme == 'custom':
            affected = [m for m in modalities if m in custom_mods]
            if affected:
                print(f'Using CUSTOM normalization for: {affected}')
                if 'S2L2A' in affected:
                    print(
                        '  -> S2L2A: Time-aware harmonization (+1000 for images >= 2022-01-24)'
                    )
                    print('            Clipping to [0, 10000]')
                if 'S2L1C' in affected:
                    print('  -> S2L1C: Clipping to [0, 10000]')
            legacy_mods = [m for m in modalities if m not in custom_mods]
            if legacy_mods:
                print(f'Using LEGACY normalization for: {legacy_mods}')
        else:
            print('Using LEGACY normalization for all modalities')

        # Create collate functions
        if train_collate_mode == 'random':
            self.train_collate_fn = single_modality_collate_fn(
                modalities=modalities,
                normalize=normalize,
                norm_scheme=norm_scheme,
                target_size=target_size,
                mode='train',
                return_metadata=return_metadata,
            )
        else:
            self.train_collate_fn = deterministic_modality_collate_fn(
                modality=train_collate_mode,
                normalize=normalize,
                norm_scheme=norm_scheme,
                target_size=target_size,
                mode='train',
                return_metadata=return_metadata,
            )

        self.val_collate_fn = deterministic_modality_collate_fn(
            modality=val_collate_mode,
            normalize=normalize,
            norm_scheme=norm_scheme,
            target_size=target_size,
            mode='eval',
            return_metadata=return_metadata,
        )

    def setup(self, stage=None):
        """Set up the train and validation datasets."""
        if len(self.modalities) > 1:
            mod_path_segment = f'[{",".join(self.modalities)}]'
        else:
            mod_path_segment = self.modalities[0]

        train_urls = os.path.join(
            self.data_path,
            'train',
            mod_path_segment,
            'majortom_shard_{000001..000025}.tar',
        )

        val_urls = os.path.join(
            self.data_path,
            'val',
            mod_path_segment,
            'majortom_shard_{000001..000005}.tar',
        )

        test_urls = os.path.join(
            self.data_path,
            'val',
            mod_path_segment,
            'majortom_shard_{000006..000008}.tar',
        )

        self.train_dataset = build_terramesh_dataset(
            path=self.data_path,
            urls=train_urls,
            modalities=self.modalities,
            split='train',
            batch_size=self.batch_size,
            shuffle=True,
            probs=[1.0, 0.0],
            shardshuffle=1000,
            return_metadata=self.return_metadata,
            harmonize_s2l2a=self._needs_s2l2a_harmonization,
            **self.kwargs,
        )

        self.val_dataset = build_terramesh_dataset(
            path=self.data_path,
            modalities=self.modalities,
            split='val',
            urls=val_urls,
            batch_size=self.eval_batch_size,
            shuffle=False,
            return_metadata=self.return_metadata,
            harmonize_s2l2a=self._needs_s2l2a_harmonization,
            **self.kwargs,
        )

        self.test_dataset = build_terramesh_dataset(
            path=self.data_path,
            modalities=self.modalities,
            split='val',
            urls=test_urls,
            batch_size=self.eval_batch_size,
            shuffle=False,
            return_metadata=self.return_metadata,
            harmonize_s2l2a=self._needs_s2l2a_harmonization,
            **self.kwargs,
        )

    def train_dataloader(self):
        """Return the training DataLoader with random modality selection."""
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return the validation DataLoader with deterministic modality."""
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return the test DataLoader with deterministic modality."""
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
            pin_memory=True,
        )

    def get_normalizer(self, modality: str) -> nn.Module:
        """Get the normalizer module for a specific modality.
        Useful for inference or visualization.
        """
        return NormalizerFactory.create(modality, self.norm_scheme)


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm

    dm = TerraMeshDataModule(
        data_path='/path/to/terramesh',
        modalities=['S2L2A'],  # , "S1RTC", "S2L1C", "S2RGB"],
        batch_size=16,
        num_workers=0,
        return_metadata=True,
        norm_method='zscore',
        norm_scheme='custom',
        normalize=True,
    )

    stats = RunningStatsButFast((12,), [0, 2, 3])
    dm.setup('fit')
    train_loader = dm.train_dataloader()

    batch_stats_history = []

    i = 0
    for batch in tqdm(train_loader):
        img = batch['image']
        # Calculate batch-wise stats (not running stats)
        # Dimensions [0, 2, 3] correspond to Batch, Height, Width
        # Resulting shape: (12,)
        b_mean = img.mean(dim=[0, 2, 3]).cpu().numpy()
        b_std = img.std(dim=[0, 2, 3]).cpu().numpy()

        # Store for every channel
        for channel_idx in range(len(b_mean)):
            batch_stats_history.append(
                {
                    'Batch_Index': i,
                    'Channel': f'Ch{channel_idx:02d}',  # e.g., Ch01, Ch02...
                    'Batch_Mean': b_mean[channel_idx],
                    'Batch_Std': b_std[channel_idx],
                }
            )

        if i >= 200:
            break
        i += 1

    # Convert to DataFrame for easy plotting
    df = pd.DataFrame(batch_stats_history)

    # --- 2. Plotting ---
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Plot A: Mean Traces (Stability over time)
    sns.lineplot(
        data=df,
        x='Batch_Index',
        y='Batch_Mean',
        hue='Channel',
        ax=axes[0, 0],
        alpha=0.7,
    )
    axes[0, 0].set_title('Batch-wise Mean Stability (Trace)', fontsize=14)
    axes[0, 0].set_ylabel('Mean Value')

    # Plot B: Std Traces (Stability over time)
    sns.lineplot(
        data=df, x='Batch_Index', y='Batch_Std', hue='Channel', ax=axes[0, 1], alpha=0.7
    )
    axes[0, 1].set_title('Batch-wise Std Stability (Trace)', fontsize=14)
    axes[0, 1].set_ylabel('Standard Deviation')

    # Plot C: Boxplot of Means (Variance distribution)
    sns.boxplot(data=df, x='Channel', y='Batch_Mean', ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Distribution of Batch Means (Variance Check)', fontsize=14)

    # Plot D: Boxplot of Stds (Variance distribution)
    sns.boxplot(data=df, x='Channel', y='Batch_Std', ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Distribution of Batch Stds (Variance Check)', fontsize=14)

    plt.tight_layout()
    plt.savefig('terramesh_normalization_diagnostics.png', dpi=300)
    plt.show()
