#!/usr/bin/env bash
# TRACK=extreme|seasonal.
# CKPT points to a directory containing master.safetensors.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
PYTHON="${PYTHON:-python}"
TRACK="${TRACK:-extreme}"
CONFIG="${CONFIG:-configs/diffusion/train/eo_earthnet_stage1_cond_in10out20_neweovae.py}"
BENCH="${BENCH:-${REPO_DIR}/data/local/benchmarks}"
TILE_JSON="${TILE_JSON:-${REPO_DIR}/data/tiles/earthnet2021_tile_location_index.json}"
export AE_PATH="${AE_PATH:-${REPO_DIR}/weights/eo-vae}"
export CLIM_PATH="${CLIM_PATH:-${REPO_DIR}/data/climatology/era5_climatology_train.pt}"
CKPT="${CKPT:-${REPO_DIR}/weights/eo-wm}"
OUT_DIR="${OUT_DIR:-outputs/eval_${TRACK}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29610}"
NUM_STEPS="${NUM_STEPS:-30}"
GUIDANCE="${GUIDANCE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VIS_SAMPLES="${VIS_SAMPLES:-20}"
case "${TRACK}" in
  seasonal)
    SCRIPT=scripts/diffusion/eo_inference_earthnet_v2v_seasonal_bench_noCFG.py
    DATA_CSV="${BENCH}/seasonal/seasonal_benchmark_infer_subset.csv" ;;
  extreme)
    SCRIPT=scripts/diffusion/eo_inference_earthnet_v2v_extreme_summer_bench_vis_noCFG.py
    DATA_CSV="${BENCH}/extreme_summer/extreme_summer_benchmark_stage2_infer.csv" ;;
  *) echo "Unknown TRACK=${TRACK}" >&2; exit 1 ;;
esac
EXTRA_ARGS=(--mask_source earthformer_raw --vis_samples "${VIS_SAMPLES}")
EXTRA_ARGS+=(--height 128 --width 128 --num_ensemble "${NUM_ENSEMBLE:-5}")
if [ "${TRACK}" = seasonal ]; then
  EXTRA_ARGS+=(--benchmark_pairs_csv "${BENCH}/seasonal/final_benchmark_pairs.csv")
  [ -f "${BENCH}/seasonal/final_benchmark_pairs.csv" ] || { echo "Prepare seasonal metadata first." >&2; exit 1; }
fi
for f in "${CONFIG}" "${DATA_CSV}" "${TILE_JSON}" "${CLIM_PATH}" "${CKPT}/master.safetensors" "${AE_PATH}/eo-vae.ckpt" "${AE_PATH}/model_config.yaml"; do
  [ -f "${f}" ] || { echo "Missing file: ${f}. See README.md." >&2; exit 1; }
done
# Use the reference ENS implementation consistently, never a silent fallback.
if [ -n "${EARTHFORMER_REPO:-}" ]; then
  export PYTHONPATH="${EARTHFORMER_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi
"${PYTHON}" -c 'from earthformer.datasets.earthnet.earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute; print("ENS implementation: earthformer")'
"${PYTHON}" - <<'PY_CHECK'
import numpy, skimage
from packaging.version import Version
if Version(numpy.__version__) >= Version("2.0") or Version(skimage.__version__) >= Version("0.19"):
    raise SystemExit('Reference ENS requires numpy<2.0 and scikit-image<0.19.0; install the listed dependency versions.')
PY_CHECK
export AE_SPATIAL_COMPRESSION=8 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" -m torch.distributed.run --nnodes=1 \
  --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" "${SCRIPT}" \
  --config "${CONFIG}" --ckpt_path "${CKPT}" --data_csv "${DATA_CSV}" \
  --tile_json "${TILE_JSON}" --output_dir "${OUT_DIR}" \
  --num_steps "${NUM_STEPS}" --guidance "${GUIDANCE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" "${EXTRA_ARGS[@]}"
