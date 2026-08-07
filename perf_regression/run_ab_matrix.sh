#!/usr/bin/env bash
set -euo pipefail

# Required paths must identify one fixed, already-prewarmed CT-RATE subset.
: "${DATA_ROOT:?set DATA_ROOT to the CT-RATE train directory}"
: "${DATALIST_JSON:?set DATALIST_JSON to the fixed datalist JSON}"
: "${CACHE_DIR:?set CACHE_DIR to the fixed, prewarmed PersistentDataset cache}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
MEASURE_STEPS="${MEASURE_STEPS:-500}"
TOTAL_STEPS=$((WARMUP_STEPS + MEASURE_STEPS))
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/perf_regression/runs}"
TRACE_CASE="${TRACE_CASE:-A1_current}"
CASES="${CASES:-A1_current A2_no_logging A3_no_checkpoint A4_no_cache_write A5_old_loader A6_fixed}"

DATA_ROOT="$(realpath -m -- "${DATA_ROOT}")"
DATALIST_JSON="$(realpath -m -- "${DATALIST_JSON}")"
CACHE_DIR="$(realpath -m -- "${CACHE_DIR}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

COMMON_ARGS=(
    --dataset_mode ctrate_subset
    --data_root "${DATA_ROOT}"
    --datalist_json "${DATALIST_JSON}"
    --cache_dir "${CACHE_DIR}"
    --cache
    --batch_size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --pin_memory
    --persistent_workers
    --num_steps "${TOTAL_STEPS}"
    --eval_num "${CHECKPOINT_INTERVAL}"
    --log_every "${LOG_INTERVAL}"
    --disable_encoder_export
    --disable_final_artifacts
)

# These flags reproduce the tracked pre-fix runtime behavior inside the
# instrumented entrypoint.  They are intentionally harmful and A/B-only.
REGRESSION_FLAGS=(
    --legacy_amp_scale_polling
    --legacy_loss_item_each_step
    --cpu_concat_before_h2d
    --duplicate_checkpoint_serialization
    --no-continuous-dataloader
    --no-skip-unused-ocl-crops
)

run_case() {
    local case_name="$1"
    shift
    local case_dir="${OUTPUT_ROOT}/${case_name}"
    mkdir -p "${case_dir}"

    # Pass 1: intrusive CUDA-event timing.  It synchronizes once per profiled
    # step by design and must not be used to claim production utilization.
    set +e
    (
        cd "${REPO_ROOT}/Self-supervised"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ocl_train.py \
            "${COMMON_ARGS[@]}" \
            --diagnose_gpu_gaps \
            --gap_profile_steps "${TOTAL_STEPS}" \
            --gap_timing_csv "${case_dir}/step_timing.csv" \
            --logdir "${case_dir}/timing_checkpoints" \
            "$@"
    ) 2>&1 | tee "${case_dir}/timing_train.log"
    local timing_status=${PIPESTATUS[0]}
    set -e

    if (( timing_status != 0 )); then
        return "${timing_status}"
    fi
    "${PYTHON_BIN}" "${REPO_ROOT}/perf_regression/analyze_step_timing.py" \
        "${case_dir}/step_timing.csv" \
        --warmup "${WARMUP_STEPS}" \
        --measure "${MEASURE_STEPS}" \
        --output "${case_dir}/summary.json" \
        --require-no-cache-miss

    # Pass 2: production-like utilization/throughput.  Only the two measurement
    # boundaries synchronize; there is no per-step diagnostic sync.
    local dmon_pid=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi dmon -i "${GPU_ID}" -s pucvmet -d 1 -o DT >"${case_dir}/nvidia_dmon.txt" &
        dmon_pid=$!
    fi
    set +e
    (
        cd "${REPO_ROOT}/Self-supervised"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ocl_train.py \
            "${COMMON_ARGS[@]}" \
            --throughput_warmup_steps "${WARMUP_STEPS}" \
            --throughput_measure_steps "${MEASURE_STEPS}" \
            --throughput_output "${case_dir}/throughput.json" \
            --logdir "${case_dir}/util_checkpoints" \
            "$@"
    ) 2>&1 | tee "${case_dir}/util_train.log"
    local util_status=${PIPESTATUS[0]}
    set -e
    if [[ -n "${dmon_pid}" ]]; then
        kill "${dmon_pid}" 2>/dev/null || true
        wait "${dmon_pid}" 2>/dev/null || true
    fi
    if (( util_status != 0 )); then
        return "${util_status}"
    fi

    # Run the profiler only after the production measurement so this
    # TRACE_CASE-specific extra pass cannot warm its OS page cache beforehand.
    # It has NVTX/record_function ranges but no per-step diagnostic sync.
    if [[ "${case_name}" == "${TRACE_CASE}" ]]; then
        set +e
        (
            cd "${REPO_ROOT}/Self-supervised"
            CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ocl_train.py \
                "${COMMON_ARGS[@]}" \
                --save_gap_trace \
                --gap_trace_dir "${case_dir}/traces" \
                --logdir "${case_dir}/trace_checkpoints" \
                "$@"
        ) 2>&1 | tee "${case_dir}/trace_train.log"
        local trace_status=${PIPESTATUS[0]}
        set -e
        if (( trace_status != 0 )); then
            return "${trace_status}"
        fi
    fi
    return 0
}

for case_name in ${CASES}; do
    case "${case_name}" in
        A1_current)
            run_case "${case_name}" "${REGRESSION_FLAGS[@]}"
            ;;
        A2_no_logging)
            run_case "${case_name}" "${REGRESSION_FLAGS[@]}" --disable_training_logging
            ;;
        A3_no_checkpoint)
            run_case "${case_name}" "${REGRESSION_FLAGS[@]}" --disable_checkpoint
            ;;
        A4_no_cache_write)
            run_case "${case_name}" "${REGRESSION_FLAGS[@]}" --disable_cache_write
            ;;
        A5_old_loader)
            run_case "${case_name}" "${REGRESSION_FLAGS[@]}" \
                --no-persistent-workers \
                --no-ocl-drop-last
            ;;
        A6_fixed)
            run_case "${case_name}"
            ;;
        *)
            echo "unknown case: ${case_name}" >&2
            exit 2
            ;;
    esac
done

"${PYTHON_BIN}" "${REPO_ROOT}/perf_regression/aggregate_ab_results.py" \
    "${OUTPUT_ROOT}"

cat <<'EOF'
A0 is intentionally not fabricated here. Run the user's known-good old
entrypoint with the identical DATA_ROOT/DATALIST_JSON/cache snapshot and save
its nvidia-smi/Nsight series under perf_regression/runs/A0_known_good.
The tracked pre-484c8ee commit has no CT-RATE loader, so it cannot provide a
same-dataset A0 without cherry-picking code and invalidating the commit claim.
EOF
