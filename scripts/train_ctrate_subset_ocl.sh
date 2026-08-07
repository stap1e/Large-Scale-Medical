#!/usr/bin/env bash
set -euo pipefail

# Production CT-RATE OCL launcher.  The smoke script intentionally uses tiny,
# synchronous settings and must not be reused for utilization measurements.
: "${DATA_ROOT:?set DATA_ROOT to CT_RATE/train}"
: "${DATALIST_JSON:?set DATALIST_JSON to the fixed subset manifest}"
: "${CACHE_DIR:?set CACHE_DIR to a fully prewarmed persistent cache}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GPU_IDS="${GPU_IDS:-0}"
MASTER_PORT="${MASTER_PORT:-28814}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
NUM_STEPS="${NUM_STEPS:-2000000}"
LOG_EVERY="${LOG_EVERY:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20000}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/Self-supervised/runs/ctrate_ocl}"

DATA_ROOT="$(realpath -m -- "${DATA_ROOT}")"
DATALIST_JSON="$(realpath -m -- "${DATALIST_JSON}")"
CACHE_DIR="$(realpath -m -- "${CACHE_DIR}")"
mkdir -p "${LOGDIR}"
LOGDIR="$(cd "${LOGDIR}" && pwd)"
cd "${REPO_ROOT}/Self-supervised"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    ocl_train.py \
    --dataset_mode ctrate_subset \
    --data_root "${DATA_ROOT}" \
    --datalist_json "${DATALIST_JSON}" \
    --cache_dir "${CACHE_DIR}" \
    --cache \
    --batch_size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
    --persistent_workers \
    --pin_memory \
    --continuous_dataloader \
    --num_steps "${NUM_STEPS}" \
    --log_every "${LOG_EVERY}" \
    --eval_num "${CHECKPOINT_EVERY}" \
    --logdir "${LOGDIR}"
