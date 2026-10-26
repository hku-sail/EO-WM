from __future__ import annotations
# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file includes code adapted from the original work by EPFL and Apple Inc.,
# licensed under the Apache License, Version 2.0.
# Source: https://github.com/apple/ml-4m/

import io
import os
import re
import warnings
from collections.abc import Callable, Iterable

import braceexpand
import fsspec

# import albumentations
import numpy as np
import torch
import webdataset as wds
import zarr
from torch.utils.data._utils.collate import default_collate
from webdataset.handlers import warn_and_continue

# January 24, 2022 - Sentinel-2 L2A processing baseline change date (nanoseconds)
# Images ON OR AFTER this date need +1000 offset correction
S2L2A_BASELINE_CUTOFF_NS = 1642982400000000000  # 2022-01-24 00:00:00 UTC

# Definition of all shard files in TerraMesh
split_files = {
    'ssl4eos12': {
        'train': ['ssl4eos12_shard_{000794..000889}.tar'],
        'val': ['ssl4eos12_shard_000009.tar'],
    },
    'majortom': {
        'train': ['majortom_shard_{000001..000793}.tar'],
        'val': ['majortom_shard_{000001..000008}.tar'],
    },
    'combined': {
        'train': [
            'majortom_shard_{000001..000793}.tar',
            'ssl4eos12_shard_{000794..000889}.tar',
        ],
        'val': ['majortom_shard_{000001..000008}.tar', 'ssl4eos12_shard_000009.tar'],
    },
}

statistics = {
    'mean': {
        'S2L1C': [
            2357.090,
            2137.398,
            2018.799,
            2082.998,
            2295.663,
            2854.548,
            3122.860,
            3040.571,
            3306.491,
            1473.849,
            506.072,
            2472.840,
            1838.943,
        ],
        'S2L2A': [
            1390.461,
            1503.332,
            1718.211,
            1853.926,
            2199.116,
            2779.989,
            2987.025,
            3083.248,
            3132.235,
            3162.989,
            2424.902,
            1857.665,
        ],
        'S2RGB': [110.349, 99.507, 75.843],
        'S1GRD': [-12.577, -20.265],
        'S1RTC': [-10.93, -17.329],
        'NDVI': [0.327],
        'DEM': [651.663],
    },
    'std': {
        'S2L1C': [
            1673.639,
            1722.641,
            1602.205,
            1873.138,
            1866.055,
            1779.839,
            1776.496,
            1724.114,
            1771.041,
            1079.786,
            512.404,
            1340.879,
            1172.435,
        ],
        'S2L2A': [
            2131.157,
            2163.666,
            2059.311,
            2152.477,
            2105.179,
            1912.773,
            1842.326,
            1893.568,
            1775.656,
            1814.907,
            1436.282,
            1336.155,
        ],
        'S2RGB': [69.905, 53.708, 53.378],
        'S1GRD': [5.179, 5.872],
        'S1RTC': [4.391, 4.459],
        'NDVI': [0.322],
        'DEM': [928.168],
    },
}


def build_terramesh_dataset(
    path: str = 'https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/',
    modalities: list[str] | str = None,
    split: str = 'val',
    urls: str | None = None,
    transform: Callable = None,
    batch_size: int = 8,
    return_metadata: bool = False,
    shuffle: bool = None,
    shardshuffle: int = 100,
    deterministic: bool = False,
    seed: int = None,
    time_dim: bool = False,
    partial: bool = None,
    probs: list[int] = None,
    harmonize_s2l2a: bool = False,
    **kwargs,
):
    """Builds a dataset for TerraMesh, see https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh.

    :param path: URL or local path to dataset root that with data structure ./{split}/{modality}/shard_{id}.tar
    :param modalities: List of modalities or a single modality name
    :param split: Split name ("train", "val"). Default to "val".
    :param urls: Specify custom shard urls instead of providing the path, modalities, and split.
    :param batch_size: Specify batch size to load batches instead of samples via webdataset (Recommended).
        It requires batch_size=None in the data loader constructor.
    :param transform: Transform function to apply to the data, use MultimodalTransforms.
    :param return_metadata: Load center coordinates, timestamp (ns as int) and cloud mask (if available).
    :param shuffle: Shuffle samples and shards. Default to True for train and False for val.
    :param shardshuffle: The number of shards to shuffle, or None. Defaults to 100.
    :param deterministic: Whether to use deterministic shuffling. Defaults to False.
    :param seed: Random seed for shuffling. Defaults to None which uses random seeds.
    :param kwargs: Optional keyword arguments for single-modality which are passed to WebDataset constructor.
    :param empty_check: Check if shards are empty. Defaults to False.
    :param time_dim: If True, keeps time dimension. Defaults to False.
    :param partial: Load partial batch at the end. Defaults to False for train and True for val.
    :param probs: List of probabilities for each subset (majortom and ssl4eos12). Defaults to [0.8, 0.2].
    :param harmonize_s2l2a: If True, applies +1000 offset to S2L2A images captured on/after Jan 24, 2022.
    :return: WebDataset (single modality) or DataPipeline (multiple modalities)
    """
    if len(modalities) == 1:
        # Single modality
        modalities = modalities[0]

    # No shuffle and partial load for val
    shuffle = shuffle
    partial = partial if partial is not None else split == 'val'
    shardshuffle = shardshuffle * shuffle

    if isinstance(modalities, str):
        # Build standard WebDataset for single modality
        dataset = build_wds_dataset(
            path=path,
            modality=modalities,
            split=split,
            urls=urls,
            batch_size=batch_size,
            transform=transform,
            shuffle=shuffle,
            return_metadata=return_metadata,
            shardshuffle=shardshuffle,
            deterministic=deterministic,
            seed=seed,
            time_dim=time_dim,
            partial=partial,
            harmonize_s2l2a=harmonize_s2l2a,
            **kwargs,
        )
        return dataset

    else:
        if len(kwargs):
            warnings.warn(
                f'keyword arguments ({kwargs}) are ignored for multiple modalities.'
            )
        # Build custom multi-modal dataset
        dataset = build_multimodal_dataset(
            path=path,
            modalities=modalities,
            split=split,
            urls=urls,
            batch_size=batch_size,
            transform=transform,
            return_metadata=return_metadata,
            shardshuffle=shardshuffle,
            deterministic=deterministic,
            seed=seed,
            time_dim=time_dim,
            partial=partial,
            probs=probs,
            harmonize_s2l2a=harmonize_s2l2a,
        )
        return dataset


def zarr_decoding(key, value):
    if key == 'zarr.zip' or key.endswith('.zarr.zip'):
        mapper = fsspec.filesystem(
            'zip', fo=io.BytesIO(value), block_size=None
        ).get_mapper('')
        return zarr.open_consolidated(mapper, mode='r')['bands'][...].astype(np.float32)


def zarr_decoding_harmonized(key, value):
    """Fast zarr decoder with S2L2A harmonization - only reads bands and time."""
    if key == 'zarr.zip' or key.endswith('.zarr.zip'):
        mapper = fsspec.filesystem(
            'zip', fo=io.BytesIO(value), block_size=None
        ).get_mapper('')
        data = zarr.open_consolidated(mapper, mode='r')
        bands = data['bands'][...].astype(np.float32)
        timestamp = data['time'][...]

        # Apply +1000 shift for post-baseline images
        if timestamp >= S2L2A_BASELINE_CUTOFF_NS:
            bands = bands + 1000.0

        return bands


def zarr_metadata_decoding(sample, harmonize_s2l2a: bool = False):
    for key, value in list(sample.items()):
        if key == 'zarr.zip' or key.endswith('.zarr.zip'):
            mapper = fsspec.filesystem(
                'zip', fo=io.BytesIO(value), block_size=None
            ).get_mapper('')
            data = zarr.open_consolidated(mapper, mode='r')
            bands = data['bands'][...].astype(np.float32)
            timestamp = data['time'][...]

            # In multimodal, keys are 'S2L2A.zarr.zip'.
            # In single modality, key is 'zarr.zip' (and the caller ensures harmonize_s2l2a is only True for S2L2A).
            is_s2l2a_key = 'S2L2A' in key or key == 'zarr.zip'

            # Apply S2L2A harmonization ONLY if it's the correct modality
            if (
                harmonize_s2l2a
                and is_s2l2a_key
                and timestamp >= S2L2A_BASELINE_CUTOFF_NS
            ):
                bands = bands + 1000.0

            sample[key] = bands

            # Add metadata
            if (
                'center_lon' not in sample.keys()
            ):  # Same center point for all modalities
                sample['center_lon'] = data['center_lon'][...].copy()
                sample['center_lat'] = data['center_lat'][...].copy()
            if (
                'cloud_mask' in data and 'cloud_mask' not in sample.keys()
            ):  # Same S2 mask in all optical modalities
                sample['cloud_mask'] = data['cloud_mask'][...][
                    np.newaxis, ...
                ].copy()  # Add channel dim to mask
            if timestamp > 1e6:  # DEM has no valid timestamp (value = 0)
                time_key = 'time' if key == 'zarr.zip' else 'time_' + key
                sample[time_key] = timestamp  # Integer values of type "datetime64[ns]"
        elif isinstance(value, str):
            # Skip str data
            pass
        else:
            # Fallback to webdataset autodecoder
            sample[key] = next(wds.decode()([{key: value}]))[key]

    return sample


def make_zarr_metadata_decoder(harmonize_s2l2a: bool = False):
    """Factory function to create zarr_metadata_decoding with harmonization option."""

    def decoder(sample):
        return zarr_metadata_decoding(sample, harmonize_s2l2a=harmonize_s2l2a)

    return decoder


def identity(sample):
    """Identity function that does nothing."""
    return sample


def drop_time_dim(value, dim: int = 0):
    """Remove time dimension from data tensors."""
    if (
        isinstance(value, np.ndarray) or isinstance(value, torch.Tensor)
    ) and value.shape[dim] == 1:
        return value.squeeze(dim)

    elif isinstance(value, dict):
        if 'image' in value:
            value['image'] = value['image'].squeeze(dim)
            return value
        else:
            # multimodal case
            for k, v in value.items():
                if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                    # if v.shape[dim] == 1:
                    value[k] = v.squeeze(dim)
            return value
    else:
        return value


def build_wds_dataset(
    path: str = 'https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/',
    modality: str = 'S2L2A',
    split: str = 'val',
    urls: str | None = None,
    batch_size: int = 8,
    transform: Callable = None,
    return_metadata: bool = False,
    shardshuffle: int = 100,
    deterministic: bool = False,
    seed: int = None,
    empty_check: bool = False,
    time_dim: bool = False,
    partial: bool = False,
    shuffle: bool = False,
    harmonize_s2l2a: bool = False,
    *args,
    **kwargs,
):
    if urls is None:
        # Select split files
        if modality == 'S1GRD':
            files = split_files['ssl4eos12'][split]
        elif modality == 'S1RTC':
            files = split_files['majortom'][split]
        else:
            files = split_files['combined'][split]

        # Joins majortom and ssl4eos12 shard files with "::" (except for S-1 modalities)
        urls = '::'.join([os.path.join(path, split, modality, f) for f in files])

    if modality == 'S1GRD' and split == 'val' and empty_check:
        # Setting empty_check to True to avoid errors because of a single shard file in SSL4EOS12 S1GRD val split
        empty_check = False

    # Build dataset
    dataset = wds.WebDataset(
        urls,
        *args,
        shardshuffle=shardshuffle,
        detshuffle=deterministic,
        seed=seed,
        handler=warn_and_continue,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        empty_check=empty_check,
        **kwargs,
    )

    if harmonize_s2l2a and modality == 'S2L2A':
        # Fast path: use decode() with harmonized decoder (same speed as legacy)
        dataset = dataset.decode(zarr_decoding_harmonized)
    else:
        dataset = dataset.decode(zarr_decoding)

    # Rename modality to "image" and remove temporal dimension
    dataset = dataset.rename(image='zarr.zip')

    if not time_dim:
        dataset = dataset.map(drop_time_dim)

    if transform is not None:
        dataset = dataset.map(transform)

    # Add sample-level shuffling if shuffle is True
    if shuffle:
        dataset = dataset.shuffle(1000, seed=seed)

    # Create batches
    if batch_size is not None:
        dataset = dataset.batched(batch_size, partial=partial)

    return dataset


def _subset_pipeline(
    urls,
    *,
    batch_size,
    shardshuffle,
    deterministic,
    seed,
    empty_check,
    return_metadata,
    transform,
    time_dim,
    partial,
    harmonize_s2l2a=False,
):
    # Determine decoder based on metadata and harmonization needs
    if return_metadata or harmonize_s2l2a:
        decoder = wds.map(make_zarr_metadata_decoder(harmonize_s2l2a=harmonize_s2l2a))
    else:
        decoder = wds.decode(zarr_decoding)

    return wds.DataPipeline(
        wds.ResampledShards(
            urls, deterministic=deterministic, seed=seed, empty_check=empty_check
        )
        if shardshuffle
        else wds.SimpleShardList(urls),
        wds.split_by_node,
        wds.split_by_worker,
        # Extract individual samples from multi-modal tar files
        multi_tarfile_samples,
        wds.shuffle(shardshuffle, seed=seed),
        # Decode from bytes to numpy arrays, etc.
        decoder,
        # Remove time dimension from tensors
        wds.map(drop_time_dim) if not time_dim else wds.map(identity),
        wds.map(remove_extensions),
        wds.map(transform) if transform is not None else wds.map(identity),
        wds.batched(batch_size, collation_fn=collate_fn, partial=partial),
    )


def build_multimodal_dataset(
    path: str = 'https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/',
    modalities: list = None,
    split: str = 'val',
    urls: str | None = None,
    batch_size: int = 8,
    transform: Callable = None,
    return_metadata: bool = False,
    shardshuffle: int = 100,
    deterministic: bool = False,
    seed: int = None,
    empty_check: bool = False,
    time_dim: bool = True,
    partial: bool = False,
    probs: list[int] = None,
    harmonize_s2l2a: bool = False,
):
    if modalities is None:
        modalities = [
            'S2L2A',
            'S2L1C',
            'S2RGB',
            'S1GRD',
            'S1RTC',
            'DEM',
            'NDVI',
            'LULC',
        ]  # Default

    # Only apply harmonization if S2L2A is in the modalities
    should_harmonize = harmonize_s2l2a and 'S2L2A' in modalities

    if urls is None:
        # Filter modalities based availability (S1GRD and S1RTC not present in all subsets)
        def filter_list(lst, value):
            lst = lst.copy()
            # helper function to filter modalities
            if value in lst:
                lst.remove(value)
            return lst

        urls_majortom = os.path.join(
            path,
            split,
            f'[{",".join(filter_list(modalities, "S1GRD"))}]',
            split_files['majortom'][split][0],
        )
        urls_ssl4eos12 = os.path.join(
            path,
            split,
            f'[{",".join(filter_list(modalities, "S1RTC"))}]',
            split_files['ssl4eos12'][split][0],
        )
    else:
        if '::' in urls:
            urls_majortom, urls_ssl4eos12 = urls.split('::')
        else:
            urls_majortom = urls_ssl4eos12 = urls

    ds_mt = _subset_pipeline(
        urls_majortom,
        batch_size=batch_size,
        shardshuffle=shardshuffle,
        deterministic=deterministic,
        seed=seed,
        empty_check=empty_check,
        return_metadata=return_metadata,
        transform=transform,
        time_dim=time_dim,
        partial=partial,
        harmonize_s2l2a=should_harmonize,
    )

    ds_ssl = _subset_pipeline(
        urls_ssl4eos12,
        batch_size=batch_size,
        shardshuffle=shardshuffle,
        deterministic=deterministic,
        seed=seed,
        empty_check=empty_check,
        return_metadata=return_metadata,
        transform=transform,
        time_dim=time_dim,
        partial=partial,
        harmonize_s2l2a=should_harmonize,
    )

    # mix batches (never mixes samples)
    dataset = wds.RandomMix(
        [ds_mt, ds_ssl],
        probs=probs or [0.8, 0.2],
        longest=not shardshuffle,  # Load all samples if shuffle is false
    )

    return dataset


def collate_fn(batch):
    # Wrapper for debugging
    try:
        return default_collate(batch)
    except Exception as e:
        for s in batch:
            print(s['__key__'])
            print(s['__url__'])
            print(s.keys())
        raise e


def extract_modality_names(s):
    """Function from https://github.com/apple/ml-4m/blob/main/fourm/data/unified_datasets.py."""
    # Regular expression pattern to match anything enclosed in "{" and "}", and comma separated
    pattern = r'\{([^}]*)\}'
    match = re.search(pattern, s)
    return match.group(1).split(',') if match else []


def remove_ext_with_gz(s):
    """Function from https://github.com/apple/ml-4m/blob/main/fourm/data/unified_datasets.py."""
    if s.endswith('.gz'):
        s = s.replace('.gz', '')
    if s.endswith('.zip'):
        s = s.replace('.zip', '')
    return os.path.splitext(s)[0]


def remove_extensions(sample):
    """Function from https://github.com/apple/ml-4m/blob/main/fourm/data/unified_datasets.py.

    In webdatasets, we identify the type of a given modality by adding an extension
    in the form f"{modality_name}.{modality_extension}", e.g. "rgb.jpg" or "caption.json".
    This function removes them and returns a dictionary of {f"{modality_name}": modality}.
    """
    return {remove_ext_with_gz(k): v for k, v in sample.items()}


def multi_tarfile_samples(src_iter: Iterable[dict]):
    """This function is adapted from https://github.com/apple/ml-4m/blob/main/fourm/data/unified_datasets.py.

    Webdataset does not support splitting up shards by modality, so we need to do this manually.
    Usually, we would need to save all modalities in the same tar file, e.g. shard_root_train/{00000..12345}.tar,
    where each shard contains 1000 samples and each sample contains all modalities.
    This is not flexible when adding new modalities, so we instead save each modality in a separate tar file,
    e.g. shard_root_train_rgb/{00000..12345}.tar, shard_root_train_caption/{00000..12345}.tar, etc., where each shard contains
    again 1000 samples, but each sample contains only one modality. All samples in all shards have to be aligned.

    This function takes an iterator over shard URLs, where we use brace expansion to specify multiple tar files per modality.
    E.g. shard_root_train_[rgb,caption]/00123.tar will be expanded to shard_root_train_rgb/00123.tar and shard_root_train_caption/00123.tar,
    and the samples from these two tar files will be combined into a single sample.

    Args:
        src_iter: Iterator over shards that *already brace expanded the shard numbers*,
            e.g. {"url": "shard_root_train_[rgb,caption]/00000.tar"}, {"url": "shard_root_train_[rgb,caption]/00001.tar"}, ...
            This function will also work when no square braces for multiple modalities are used, e.g. {"url": "shard_root_train/00000.tar"}, ...
            It can be a drop-in replacement for wds.tarfile_samples.

    Yields:
        Dictionary of aligned samples from all modalities.
    """
    for src in src_iter:
        # Multi tar file URLs use brace expansion with square braces
        multi_tar_urls = src['url'].translate(str.maketrans('[]', '{}'))
        modality_names = extract_modality_names(multi_tar_urls)
        multi_tar_urls = list(braceexpand.braceexpand(multi_tar_urls))

        # Create tar iterators for shards of all modalities
        tar_iters = [
            wds.tarfile_samples([{'url': tar_url}]) for tar_url in multi_tar_urls
        ]

        # try:
        #     # Loop over these iterators in parallel and combine the tar files from different modalities
        for multi_tar_files in zip(*tar_iters):
            merged_dict = {}
            merged_dict['__key__'] = multi_tar_files[0]['__key__']
            merged_dict['__url__'] = src['url']

            for modality_name, modality_dict in zip(modality_names, multi_tar_files):
                _key = modality_dict.pop('__key__')
                _url = modality_dict.pop('__url__')

                if _key != merged_dict['__key__']:
                    raise ValueError(
                        f'Divergence detected! Trying to merge keys {_key} of {modality_name} and {merged_dict["__key__"]} of merged_dict with modalities {merged_dict.keys()}.'
                    )

                for k, v in modality_dict.items():
                    if modality_name is None:
                        merged_dict[k] = v
                    else:
                        merged_dict[f'{modality_name}.{k}'] = v

            yield merged_dict



def default_non_image_transform(array):
    if hasattr(array, 'dtype') and (array.dtype == float or array.dtype == int):
        return torch.from_numpy(array.copy())
    else:
        return array




class MultimodalNormalize(Callable):
    def __init__(self, mean: dict[str, list[float]], std: dict[str, list[float]]):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, **batch):
        for m in self.mean.keys():
            if m not in batch.keys():
                continue
            batch[m] = (batch[m] - self.mean[m]) / self.std[m]
        return batch

    def add_targets(self, targets):
        """Required by albumentations"""
        pass
