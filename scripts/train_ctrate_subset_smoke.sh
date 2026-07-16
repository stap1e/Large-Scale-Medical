#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

GPU_ID="${GPU_ID:-0}"
NUM_STEPS="${NUM_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-0}"
NUM_PATIENTS="${NUM_PATIENTS:-256}"
FEATURE_SIZE="${FEATURE_SIZE:-48}"
PRETRAIN_METHOD="${PRETRAIN_METHOD:-ocl}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/Self-supervised/runs/ctrate_subset_${PRETRAIN_METHOD}_smoke}"
DATA_ROOT="${DATA_ROOT:-/home/bld/data/dataset/CT_RATE/train}"
DATALIST_JSON="${DATALIST_JSON:-$(dirname "${DATA_ROOT}")/ct_rate_subset_${NUM_PATIENTS}.json}"
CACHE_DIR="${CACHE_DIR:-/home/bld/data/cache/ctrate_subset_${NUM_PATIENTS}}"
CACHE="${CACHE:-0}"
RUN_DATA_CHECK="${RUN_DATA_CHECK:-1}"
DATA_CHECK_ONLY="${DATA_CHECK_ONLY:-0}"
RESUME="${RESUME:-0}"
LOG_EVERY="${LOG_EVERY:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
SYNC_TIMING="${SYNC_TIMING:-1}"

# Resolve user-provided relative paths before changing into Self-supervised.
DATA_ROOT="$(realpath -m -- "${DATA_ROOT}")"
DATALIST_JSON="$(realpath -m -- "${DATALIST_JSON}")"
CACHE_DIR="$(realpath -m -- "${CACHE_DIR}")"
LOGDIR="$(realpath -m -- "${LOGDIR}")"
case "${RESUME,,}" in
    0|1|false|true|no|yes|off|on|none|auto|"") ;;
    *) RESUME="$(realpath -m -- "${RESUME}")" ;;
esac

if [[ -z "${SAVE_EVERY:-}" ]]; then
    if (( NUM_STEPS <= 10 )); then
        SAVE_EVERY="${NUM_STEPS}"
    else
        SAVE_EVERY=100
    fi
fi

case "${PRETRAIN_METHOD}" in
    ocl) ENTRYPOINT="ocl_train.py" ;;
    voco) ENTRYPOINT="voco_train.py" ;;
    *) echo "PRETRAIN_METHOD must be 'ocl' or 'voco'" >&2; exit 2 ;;
esac

if [[ ! -f "${DATALIST_JSON}" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/build_ctrate_subset.py" \
        --data_root "${DATA_ROOT}" \
        --output_json "${DATALIST_JSON}" \
        --num_patients "${NUM_PATIENTS}" \
        --seed 2026 \
        --selection_mode one_volume_per_patient \
        --validate_header
fi

COMMON_ARGS=(
    --dataset_mode ctrate_subset
    --data_root "${DATA_ROOT}"
    --datalist_json "${DATALIST_JSON}"
    --cache_dir "${CACHE_DIR}"
    --batch_size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --feature_size "${FEATURE_SIZE}"
    --logdir "${LOGDIR}"
)

if [[ "${CACHE}" == "1" || "${CACHE,,}" == "true" ]]; then
    CACHE_ARGS=(--cache)
else
    CACHE_ARGS=(--no-cache)
fi

cd "${REPO_ROOT}/Self-supervised"

if [[ "${DATA_CHECK_ONLY}" == "1" || "${DATA_CHECK_ONLY,,}" == "true" ]]; then
    RUN_DATA_CHECK=1
fi

DID_DATA_CHECK=0
if [[ "${RUN_DATA_CHECK}" == "1" || "${RUN_DATA_CHECK,,}" == "true" ]]; then
    "${PYTHON_BIN}" "${ENTRYPOINT}" \
        "${COMMON_ARGS[@]}" \
        "${CACHE_ARGS[@]}" \
        --data_check_only
    DID_DATA_CHECK=1
fi

if [[ "${DATA_CHECK_ONLY}" == "1" || "${DATA_CHECK_ONLY,,}" == "true" ]]; then
    exit 0
fi

if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    if [[ "${DID_DATA_CHECK}" == "1" ]]; then
        echo "CUDA is unavailable; CPU data_check_only completed, GPU smoke training was skipped." >&2
    else
        echo "CUDA is unavailable and RUN_DATA_CHECK=0; GPU smoke training was skipped." >&2
    fi
    exit 0
fi

RESUME_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME,,}" == "true" || "${RESUME,,}" == "yes" || "${RESUME,,}" == "on" || "${RESUME,,}" == "auto" ]]; then
    RESUME_ARGS=(--resume)
elif [[ "${RESUME}" != "0" && "${RESUME,,}" != "false" && "${RESUME,,}" != "no" && "${RESUME,,}" != "off" && "${RESUME,,}" != "none" ]]; then
    RESUME_ARGS=(--resume "${RESUME}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ENTRYPOINT}" \
    "${COMMON_ARGS[@]}" \
    "${CACHE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --num_steps "${NUM_STEPS}" \
    --eval_num "${SAVE_EVERY}" \
    --log_every "${LOG_EVERY}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --sync_timing "${SYNC_TIMING}"
