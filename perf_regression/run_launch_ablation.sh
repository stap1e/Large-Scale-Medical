#!/usr/bin/env bash
set -euo pipefail

# This matrix isolates the harmful long-run defaults from the smoke launcher.
# Batch size stays fixed at 1 throughout; L4 changes only the input pipeline.
: "${DATA_ROOT:?set DATA_ROOT to the CT-RATE train directory}"
: "${DATALIST_JSON:?set DATALIST_JSON to the fixed datalist JSON}"
: "${CACHE_DIR:?set CACHE_DIR to a fully prewarmed cache for L4}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
MEASURE_STEPS="${MEASURE_STEPS:-300}"
TOTAL_STEPS=$((WARMUP_STEPS + MEASURE_STEPS))
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/perf_regression/launch_ablation}"

DATA_ROOT="$(realpath -m -- "${DATA_ROOT}")"
DATALIST_JSON="$(realpath -m -- "${DATALIST_JSON}")"
CACHE_DIR="$(realpath -m -- "${CACHE_DIR}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

run_case() {
    local case_name="$1"
    shift
    local case_dir="${OUTPUT_ROOT}/${case_name}"
    mkdir -p "${case_dir}"
    local dmon_pid=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi dmon -i "${GPU_ID}" -s pucvmet -d 1 -o DT \
            >"${case_dir}/nvidia_dmon.txt" &
        dmon_pid=$!
    fi
    set +e
    (
        cd "${REPO_ROOT}/Self-supervised"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ocl_train.py \
            --dataset_mode ctrate_subset \
            --data_root "${DATA_ROOT}" \
            --datalist_json "${DATALIST_JSON}" \
            --cache_dir "${CACHE_DIR}" \
            --batch_size 1 \
            --num_steps "${TOTAL_STEPS}" \
            --throughput_warmup_steps "${WARMUP_STEPS}" \
            --throughput_measure_steps "${MEASURE_STEPS}" \
            --throughput_output "${case_dir}/throughput.json" \
            --disable_encoder_export \
            --disable_final_artifacts \
            --logdir "${case_dir}/checkpoints" \
            "$@"
    ) 2>&1 | tee "${case_dir}/train.log"
    local status=${PIPESTATUS[0]}
    set -e
    if [[ -n "${dmon_pid}" ]]; then
        kill "${dmon_pid}" 2>/dev/null || true
        wait "${dmon_pid}" 2>/dev/null || true
    fi
    return "${status}"
}

# Harmful training flags from the smoke launcher (apart from requested total
# steps); its one-time RUN_DATA_CHECK startup is intentionally excluded.
run_case L0_smoke_defaults \
    --workers 0 --no-cache --log_every 1 --eval_num 100 --sync_timing

# One-factor removals keep the same workers=0/no-cache path.
run_case L1_no_sync \
    --workers 0 --no-cache --log_every 1 --eval_num 100 --no-sync-timing
run_case L2_no_sync_no_logging \
    --workers 0 --no-cache --log_every 1 --eval_num 100 --no-sync-timing \
    --disable_training_logging
run_case L3_no_sync_no_logging_no_checkpoint \
    --workers 0 --no-cache --log_every 1 --eval_num 100 --no-sync-timing \
    --disable_training_logging --disable_checkpoint

# Same batch/model, production input settings; this isolates main-thread input.
run_case L4_production_input_pipeline \
    --workers 16 --cache --prefetch_factor 2 --persistent_workers --pin_memory \
    --log_every "${TOTAL_STEPS}" --eval_num "${TOTAL_STEPS}" --no-sync-timing \
    --disable_training_logging --disable_checkpoint --disable_cache_write

echo "Compare each throughput.json and timestamped nvidia_dmon.txt; do not merge this"
echo "batch_size=1 launch study with the fixed-batch A0-A6 regression table."
