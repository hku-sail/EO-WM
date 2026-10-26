#!/usr/bin/env bash
# Train EO-WM: random DiT initialization, seed 42, 18,000 updates.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/diffusion/train/eo_earthnet_stage1_cond_in10out20_neweovae.py}"
DATA_CSV="${DATA_CSV:-${REPO_DIR}/data/local/metadata/earthnet2021_train_in10out20_refine-caption_multi-p_corrected-temp50.csv}"
TILE_JSON="${TILE_JSON:-${REPO_DIR}/data/tiles/earthnet2021_tile_location_index.json}"
export AE_PATH="${AE_PATH:-${REPO_DIR}/weights/eo-vae}"
export CLIM_PATH="${CLIM_PATH:-${REPO_DIR}/data/climatology/era5_climatology_train.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/earthnet}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29520}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-18000}"
for f in "${CONFIG}" "${DATA_CSV}" "${TILE_JSON}" "${CLIM_PATH}" "${AE_PATH}/eo-vae.ckpt" "${AE_PATH}/model_config.yaml"; do
  [ -f "${f}" ] || { echo "Missing file: ${f}. See README.md for data/weight preparation." >&2; exit 1; }
done
"${PYTHON}" - "${DATA_CSV}" <<'PY_CHECK'
import csv, pathlib, sys
with open(sys.argv[1], encoding="utf-8", newline="") as f:
    row = next(csv.DictReader(f))
if not pathlib.Path(row["path"]).is_file():
    raise SystemExit("Dataset path does not exist; run prepare_release_data.py with --check.")
PY_CHECK
export AE_SPATIAL_COMPRESSION=8 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" -m torch.distributed.run --nnodes=1 \
  --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  scripts/diffusion/train.py "${CONFIG}" \
  --dataset.data_path "${DATA_CSV}" \
  --dataset.cloud_prob_thresh 0.2 \
  --dataset.tile_location_json "${TILE_JSON}" \
  --dataset.climatology_path "${CLIM_PATH}" \
  --quality_mask_area_thresh 0.5 \
  --ae.pretrained_path "${AE_PATH}" \
  --outputs "${OUTPUT_DIR}" --seed "${SEED}" \
  --epochs 1000 --max_steps "${MAX_STEPS}" \
  --ckpt_every 2000 --log_every 10 --wandb False \
  --no_i2v_ref_loss True --num_workers 4
