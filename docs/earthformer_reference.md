# Earthformer reference evaluation

### 1. Download EarthNet2021

Download the `extreme` and `seasonal` splits from the official [EarthNet2021 page](https://www.earthnet.tech/resources/datasets/earthnet2021).

Using the official toolkit:

```bash
pip install earthnet
python -c "import earthnet as en; en.Downloader.get('/path/to/EarthNet2021', ['extreme', 'seasonal'])"
```

Expected layout:

```text
/path/to/EarthNet2021/
|-- extreme_test_split/
|   |-- context/
|   `-- target/
`-- seasonal_test_split/
    |-- context/
    `-- target/
```

### 2. Prepare Earthformer

The provided scripts use Earthformer as a reference model. Install Earthformer and expose its repo path:

```bash
git clone https://github.com/amazon-science/earth-forecasting-transformer.git
cd earth-forecasting-transformer
pip install -e .
export EARTHFORMER_REPO=/path/to/earth-forecasting-transformer
```

Install script dependencies:

```bash
pip install numpy pandas scipy einops omegaconf tqdm pillow earthnet
```

Install PyTorch separately according to your CUDA version.

### 3. Inference

#### Extreme Summer

```bash
cd /path/to/EO-WM
export EARTHFORMER_REPO=/path/to/earth-forecasting-transformer

python script/earthformer_eval_extreme_summer_bench.py \
  --cfg /path/to/earthformer/cfg.yaml \
  --ckpt_path /path/to/earthformer/checkpoint.ckpt \
  --earthnet_root /path/to/EarthNet2021 \
  --data_csv benchmark_csv/extreme_summer_benchmark.csv \
  --output_dir outputs/earthformer_extreme \
  --batch_size 4 \
  --device cuda:0
```

Main outputs: `metrics.json`, `per_sample_metrics.csv`, and optional visualizations under `vis/`.

#### Seasonal Matched-Pair

```bash
cd /path/to/EO-WM
export EARTHFORMER_REPO=/path/to/earth-forecasting-transformer

python script/earthformer_eval_seasonal_bench.py \
  --cfg /path/to/earthformer/cfg.yaml \
  --ckpt_path /path/to/earthformer/checkpoint.ckpt \
  --earthnet_root /path/to/EarthNet2021 \
  --benchmark_pairs_csv benchmark_csv/seasonal_pairs_benchmark.csv \
  --output_dir outputs/earthformer_seasonal \
  --batch_size 4 \
  --device cuda:0
```

The seasonal script expands the pair CSV into per-window inference samples internally. Main outputs: `metrics.json`, `per_pair_metrics.csv`, and `per_window_metrics.csv`.

#### Metrics

- **Standard metrics**: EarthNetScore (ENS), Pixel-MAE (P-MAE), and NDVI-MAE (N-MAE).
- **Extreme Summer**: Trough NDVI-MAE (TN-MAE) and Drop Amplitude Error (DAE), including low/mid/high severity-bin summaries.
- **Seasonal Matched-Pair**: Divergence Reproduction Ratio (DRR), Directional Hit Rate (DHR), and Paired Divergence Correlation (PDC).

The model-forward code is Earthformer-specific, while the metric logic can be adapted to any model that produces aligned EarthNet2021 target-window predictions.

