#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT}"
: "${DATALIST_JSON:?set DATALIST_JSON}"
: "${CACHE_DIR:?set CACHE_DIR to a prewarmed cache}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
MEASURE_STEPS="${MEASURE_STEPS:-500}"
TOTAL_STEPS=$((WARMUP_STEPS + MEASURE_STEPS))
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/perf_regression/dataloader_sweep}"

DATA_ROOT="$(realpath -m -- "${DATA_ROOT}")"
DATALIST_JSON="$(realpath -m -- "${DATALIST_JSON}")"
CACHE_DIR="$(realpath -m -- "${CACHE_DIR}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

for workers in 4 8 12 16; do
    for prefetch in 2 4; do
        run_name="workers${workers}_prefetch${prefetch}"
        run_dir="${OUTPUT_ROOT}/${run_name}"
        mkdir -p "${run_dir}"

        # Intrusive timing pass.
        (
            cd "${REPO_ROOT}/Self-supervised"
            CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ocl_train.py \
                --dataset_mode ctrate_subset \
                --data_root "${DATA_ROOT}" \
                --datalist_json "${DATALIST_JSON}" \
                --cache_dir "${CACHE_DIR}" \
                --cache \
                --batch_size "${BATCH_SIZE}" \
                --workers "${workers}" \
                --prefetch_factor "${prefetch}" \
                --persistent_workers \
                --pin_memory \
                --num_steps "${TOTAL_STEPS}" \
                --eval_num "${TOTAL_STEPS}" \
                --log_every "${TOTAL_STEPS}" \
                --diagnose_gpu_gaps \
                --gap_profile_steps "${TOTAL_STEPS}" \
                --gap_timing_csv "${run_dir}/step_timing.csv" \
                --disable_training_logging \
                --disable_checkpoint \
                --disable_validation \
                --disable_cache_write \
                --disable_encoder_export \
                --disable_final_artifacts \
                --logdir "${run_dir}/timing_checkpoints"
        ) 2>&1 | tee "${run_dir}/timing_train.log"
        "${PYTHON_BIN}" "${REPO_ROOT}/perf_regression/analyze_step_timing.py" \
            "${run_dir}/step_timing.csv" \
            --warmup "${WARMUP_STEPS}" \
            --measure "${MEASURE_STEPS}" \
            --output "${run_dir}/summary.json" \
            --require-no-cache-miss

        # Non-intrusive utilization/throughput pass.
        dmon_pid=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi dmon -i "${GPU_ID}" -s pucvmet -d 1 -o DT >"${run_dir}/nvidia_dmon.txt" &
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
                --cache \
                --batch_size "${BATCH_SIZE}" \
                --workers "${workers}" \
                --prefetch_factor "${prefetch}" \
                --persistent_workers \
                --pin_memory \
                --num_steps "${TOTAL_STEPS}" \
                --eval_num "${TOTAL_STEPS}" \
                --log_every "${TOTAL_STEPS}" \
                --disable_training_logging \
                --disable_checkpoint \
                --disable_validation \
                --disable_cache_write \
                --disable_encoder_export \
                --disable_final_artifacts \
                --throughput_warmup_steps "${WARMUP_STEPS}" \
                --throughput_measure_steps "${MEASURE_STEPS}" \
                --throughput_output "${run_dir}/throughput.json" \
                --logdir "${run_dir}/util_checkpoints"
        ) 2>&1 | tee "${run_dir}/util_train.log"
        util_status=${PIPESTATUS[0]}
        set -e
        if [[ -n "${dmon_pid}" ]]; then
            kill "${dmon_pid}" 2>/dev/null || true
            wait "${dmon_pid}" 2>/dev/null || true
        fi
        if (( util_status != 0 )); then
            exit "${util_status}"
        fi
    done
done

echo "Choose by low iterator_wait_p95 + low idle gaps + high samples/s, not worker count."
