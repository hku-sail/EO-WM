<div align="center">

# EO-WM: Toward Physically Informed World Modeling for Probabilistic Earth Observation Forecasting

<p>
  <strong><a href="https://scholar.google.com/citations?user=6XibZaYAAAAJ&hl=en">Junwei Luo</a><sup>1,2*</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://scholar.google.com/citations?hl=zh-CN&user=gu1tkjYAAAAJ">Shuai Yuan</a><sup>1*</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://huster-yzy.github.io/">Zhenya Yang</a><sup>1</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://scholar.google.com/citations?user=wn9hc6UAAAAJ&hl=zh-CN&oi=sra">Yansheng Li</a><sup>2</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://happinesslz.github.io/">Zhe Liu</a><sup>1</sup></strong>
  &nbsp;&nbsp;
  <strong><a href="https://i.cs.hku.hk/~hszhao/">Hengshuang Zhao</a><sup>1†</sup></strong>
</p>

<p>
  <sup>1</sup> The University of Hong Kong &nbsp;&nbsp;
  <sup>2</sup> Wuhan University
  <br>
  <sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author
</p>

<p>
  <b>NeurIPS 2026</b>
</p>

<p>
  <a href="https://arxiv.org/abs/2606.27277"><img alt="arXiv: 2606.27277" src="https://img.shields.io/badge/arXiv-2606.27277-b31b1b.svg?style=flat-square"></a>
  <a href="https://github.com/hku-sail/EO-WM/tree/main/code"><img alt="Training and inference code" src="https://img.shields.io/badge/Code-Training_%26_Inference-2563eb?style=flat-square&logo=github"></a>
  <a href="https://huggingface.co/ll-13/eo-wm_weight"><img alt="Hugging Face model weights" src="https://img.shields.io/badge/Hugging_Face-Model_Weights-ffca28?style=flat-square&logo=huggingface"></a>
</p>
<p>
  <strong>Note:</strong> The current arXiv version is the preprint version. An updated version incorporating the camera-ready revisions will be posted on arXiv after the camera-ready manuscript is finalized.
</p>

<p>
  <a href="https://arxiv.org/pdf/2606.27277">Preprint</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#model-checkpoints">Models</a> ·
  <a href="#training">Training</a> ·
  <a href="#inference-and-evaluation">Inference</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#citation">Citation</a>
</p>

Official repository of EO-WM, a weather-conditioned latent video diffusion model for probabilistic multispectral Earth Observation forecasting.

</div>

<a id="news"></a>

## 📰 News

- **2026-10-26**: We release model weights and code.
- **2026-09-26**: EO-WM is accepted by NeurIPS 2026.
- **2026-06-26**: We release the preprint, benchmark CSVs, and Earthformer reference evaluation scripts.

<a id="contents"></a>

## 🧭 Contents

[Overview](#highlights) · [Installation](#getting-started) · [Checkpoints](#model-checkpoints) · [Data](#data-preparation) · [Training](#training) · [Inference & Evaluation](#inference-and-evaluation) · [Benchmarks](#benchmarks) · [Citation](#citation)

<a id="highlights"></a>

## ✨ Highlights

<p align="center">
  <img src="figures/Intro.png" width="95%" alt="EO-WM overview">
</p>

- **Toward weather-driven EO world modeling**: we formulate partially observed EO forecasting as a weather-conditioned prediction problem, using sparse satellite observations and meteorological forcing to model plausible future observations.
- **Physically motivated weather conditioning**: EO-WM organizes meteorological inputs into climatological baselines, weather anomalies, and cumulative stress signals to provide structured conditioning for future EO forecasting.
- **Forecasting-oriented evaluation**: two [EarthNet2021](https://www.earthnet.tech/resources/datasets/earthnet2021)-based benchmarks evaluate forecasting under extreme summer conditions and matched-pair prediction divergence under differing weather conditions.

<a id="network-architecture"></a>

## Network Architecture

<p align="center">
  <img src="figures/Pipeline.png" width="95%" alt="EO-WM network architecture">
</p>

EO-WM is a weather-conditioned latent video diffusion model for probabilistic multispectral EO forecasting. An EO-specific VAE encodes sparse satellite observations into latent video tokens, while meteorological inputs are organized into climatological baselines, weather anomalies, and cumulative stress signals. The diffusion transformer processes these heterogeneous conditions through separate conditioning paths and periodically reinjects spatial context into the video-token stream to generate stochastic future EO observations conditioned on meteorological forcing.

<a id="getting-started"></a>

## ⚙️ Getting Started

The implementation is based on Open-Sora 2.0 and includes the EO-VAE package used by EO-WM. Our reference setup uses Python 3.10, and CUDA 11.8.

```bash
git clone https://github.com/hku-sail/EO-WM.git
cd EO-WM/code
bash install_env.sh
conda activate eo-wm
```

The environment uses the following core versions:

| Component | Version |
| --- | --- |
| Python / CUDA toolkit | 3.10 / 11.8 |
| PyTorch / torchvision | 2.4.0 / 0.19.0 |
| FlashAttention / xFormers | 2.6.3 / 0.0.27.post2 |
| ColossalAI / Transformers | 0.5.0 / 4.41.2 |
| NumPy / torchmetrics | 1.26.4 / 1.8.2 |
| scikit-image for reference ENS | 0.18.3 |

See [`code/requirements.txt`](code/requirements.txt) for the complete dependency list. CUDA extensions may need to be installed for your local toolkit.

For reference EarthNetScore evaluation, also make Earthformer available:

```bash
git clone https://github.com/amazon-science/earth-forecasting-transformer.git ../earth-forecasting-transformer
export EARTHFORMER_REPO="$(cd ../earth-forecasting-transformer && pwd)"
```

Earthformer provides the reference EarthNetScore implementation used for evaluation.

<a id="model-checkpoints"></a>

## 🤗 Model Checkpoints

| Component | Download | Local location under `code/` |
| --- | --- | --- |
| **EO-WM** | [Hugging Face](https://huggingface.co/ll-13/eo-wm_weight/tree/main/eo-wm) | `weights/eo-wm/master.safetensors` |
| **EO-VAE, EarthNet-finetuned** | [Hugging Face](https://huggingface.co/ll-13/eo-wm_weight/tree/main/eo-vae) | `weights/eo-vae/eo-vae.ckpt` |
| T5-v1.1-XXL | [Hugging Face](https://huggingface.co/google/t5-v1_1-xxl) | Hugging Face cache, or `T5_PATH` |
| CLIP ViT-L/14 | [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14) | Hugging Face cache, or `CLIP_PATH` |

The upstream EO-VAE checkpoint differs from the EarthNet-finetuned checkpoint used in our forecasting experiments.

Download the model files into `code/weights/` (run from `EO-WM/code`):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ll-13/eo-wm_weight",
    local_dir="weights",
    allow_patterns=["eo-wm/*", "eo-vae/*"],
)
```

The downloaded files are arranged as follows:

```text
code/weights/
├── eo-wm/
│   └── master.safetensors
└── eo-vae/
    ├── eo-vae.ckpt
    └── model_config.yaml
```

The matching VAE architecture configuration is included in the download and at [`code/configs/vae/earthnet/model_config.yaml`](code/configs/vae/earthnet/model_config.yaml).

T5 and CLIP are downloaded automatically on first use. Set `AE_PATH`, `T5_PATH`, and `CLIP_PATH` to use your own local weight directories; see [configuration notes](docs/configuration.md#local-paths).

<a id="data-preparation"></a>

## 🗂️ Data Preparation

Download and unpack the required splits from the official [EarthNet2021 dataset page](https://www.earthnet.tech/resources/datasets/earthnet2021). The repository includes metadata, tile coordinates, and train-only climatology statistics; obtain the raw NPZ data separately.

```text
EarthNet2021/
├── train/<tile>/*.npz
├── iid_test_split/{context,target}/<tile>/*.npz
├── ood_test_split/{context,target}/<tile>/*.npz
├── extreme_test_split/{context,target}/<tile>/*.npz
└── seasonal_test_split/{context,target}/<tile>/*.npz
```

Use the original full context/target sequences for benchmark evaluation. The CSVs specify the temporal windows.

From `EO-WM/code`, generate local manifests with your own dataset root:

```bash
python scripts/data_tools/prepare_release_data.py \
  --earthnet-root /path/to/EarthNet2021 \
  --output-dir data/local \
  --check
```

Local manifests are written under `data/local/`, ready for the training and evaluation launchers. Add `--split train|extreme|seasonal` to prepare a single split.

| Manifest | Size | Purpose |
| --- | --- | --- |
| Training | 23,904 windows | 10 observed + 20 future frames |
| Extreme Summer | 1,440 windows | Event-focused forecasting |
| Seasonal Matched-Pair | 844 windows / 422 pairs | Weather-conditioned divergence |

The supplied captions, window indices, tile index, and train-only climatology are described in [`code/data/README.md`](code/data/README.md).

<a id="training"></a>

## 🚀 Training

All commands below run from **`EO-WM/code`**. Prepare the finetuned EO-VAE and local training manifest first.

```bash
AE_PATH=weights/eo-vae \
GPU_IDS=0,1,2,3 NPROC=4 \
OUTPUT_DIR=outputs/earthnet \
bash train_earthnet.sh
```

The launcher uses [`the main EO-WM configuration`](code/configs/diffusion/train/eo_earthnet_stage1_cond_in10out20_neweovae.py): a randomly initialized diffusion transformer, frozen EO-VAE and text encoders, seed 42, and a per-GPU batch size of 64 on 4 GPUs.


<a id="inference-and-evaluation"></a>

## 🔍 Inference and Evaluation

Prepare `weights/eo-wm/master.safetensors`, the finetuned VAE, local evaluation manifests, and `EARTHFORMER_REPO`. The launcher performs forecasting and computes metrics for each track:

```bash
# Event and paired-weather benchmarks
TRACK=extreme CKPT=weights/eo-wm bash eval_earthnet.sh
TRACK=seasonal CKPT=weights/eo-wm bash eval_earthnet.sh
```

`CKPT` may point to any checkpoint directory containing `master.safetensors`.

<a id="benchmarks"></a>

## 🌍 Benchmarks

<p align="center">
  <img src="figures/benchmark.png" width="95%" alt="EO-WM benchmark overview">
</p>

| Benchmark | What it tests | Public metadata |
| --- | --- | --- |
| **Extreme Summer** | Severity-aware prediction of vegetation degradation under heat and drought stress. | [Window manifest](benchmark_csv/extreme_summer_benchmark.csv) |
| **Seasonal Matched-Pair** | Whether forecasts change in the correct direction and magnitude when weather forcing changes. | [Pair manifest](benchmark_csv/seasonal_pairs_benchmark.csv) |

The original public CSV paths remain relative to an EarthNet2021 root. EO-WM-ready inference manifests are included under `code/data/benchmarks/`. The original [Earthformer evaluation examples](docs/earthformer_reference.md) remain available under `script/`.

<a id="repository-layout"></a>

## 📁 Repository Layout

```text
EO-WM/
├── README.md                    # Setup, model assets, and usage
├── benchmark_csv/               # Original public benchmark manifests
├── figures/                     # Method and benchmark illustrations
├── script/                      # Original Earthformer reference examples
├── docs/                        # Configuration and reference evaluation
└── code/
    ├── configs/                 # Model and VAE configuration
    ├── opensora/                # EO-WM model, conditioning, and datasets
    ├── eo-vae/                  # Bundled EO-VAE package
    ├── scripts/                 # Training, inference, metrics, and data tools
    ├── data/                    # Portable manifests, tile index, climatology
    ├── install_env.sh
    ├── train_earthnet.sh
    └── eval_earthnet.sh
```

<a id="citation"></a>

## 📖 Citation

If you find EO-WM useful in your research, please cite:

```bibtex
@article{luo2026eo,
  title={EO-WM: A Physically Informed World Model for Probabilistic Earth Observation Forecasting},
  author={Luo, Junwei and Yuan, Shuai and Yang, Zhenya and Li, Yansheng and Liu, Zhe and Zhao, Hengshuang},
  journal={arXiv preprint arXiv:2606.27277},
  year={2026}
}
```

For questions, please contact `junwei.luo@connect.hku.hk` or `luojunwei@whu.edu.cn`.

<a id="acknowledgements"></a>

## 🤝 Acknowledgements

We thank the creators and maintainers of [EarthNet2021](https://www.earthnet.tech/resources/datasets/earthnet2021), [EO-VAE](https://github.com/nilsleh/eo-vae), [Open-Sora](https://github.com/hpcaitech/Open-Sora), and [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer) for making their resources publicly available. Upstream licenses and attribution are retained in [LICENSE](LICENSE) and [NOTICE](NOTICE).
