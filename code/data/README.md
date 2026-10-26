# EarthNet2021 data assets

Portable metadata and small preprocessing assets are included here. Download raw EarthNet2021 NPZ files and checkpoints separately.

| File | Rows | Purpose |
| --- | ---: | --- |
| `metadata/earthnet2021_train_in10out20_refine-caption_multi-p_corrected-temp50.csv` | 23,904 | Training windows and corrected captions |
| `benchmarks/extreme_summer/extreme_summer_benchmark_stage2_infer.csv` | 1,440 | Extreme summer inference |
| `benchmarks/seasonal/seasonal_benchmark_infer_subset.csv` | 844 | Seasonal inference |
| `benchmarks/seasonal/final_benchmark_pairs.csv` | 422 | Seasonal matched pairs |
| `tiles/earthnet2021_tile_location_index.json` | — | MGRS tile-center coordinates |
| `climatology/era5_climatology_train.pt` | — | Train-only monthly climatology and stress statistics |

## Prepare local paths

Download the official [EarthNet2021 data](https://www.earthnet.tech/resources/datasets/earthnet2021). Keep `train/<tile>/` and test `context/<tile>/`, `target/<tile>/` layouts.

Released CSV paths are relative to the dataset root. From `EO-WM/code`, run:

```bash
python scripts/data_tools/prepare_release_data.py \
  --earthnet-root /path/to/EarthNet2021 \
  --output-dir data/local \
  --check
```

The launchers read the generated `data/local/metadata/` and `data/local/benchmarks/` directories. The preparation script changes path columns and matching path fields inside `benchmark_meta_json`; it preserves captions, sample order, frame indices, and membership. Generated files are ignored by Git.

Add `--split train|extreme|seasonal` when preparing only one track. Seasonal manifests point to `seasonal_test_split/`; window selection is encoded in CSV indices.

## Preprocessing conventions

- Optical channels are blue, green, red, and near infrared (B02/B03/B04/B08), with wavelengths `[0.490, 0.560, 0.665, 0.842]` micrometers.
- The main model observes 10 frames and predicts 20. Test loaders combine context and target NPZ files before selecting the manifest window.
- Meteorological inputs are E-OBS variables. Some `era5_*` identifiers remain for compatibility.
- The included climatology is estimated from the training split only. It provides monthly means, standard deviations, and global fallback statistics.
- Training captions are included in the supplied manifest.
