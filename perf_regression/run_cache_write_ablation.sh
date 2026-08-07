#!/usr/bin/env bash
set -euo pipefail

# Controlled cold-cache diagnostic. These directories must be dedicated and
# empty; the script never deletes cache data on the user's behalf.
: "${DATA_ROOT:?set DATA_ROOT}"
: "${DATALIST_JSON:?set DATALIST_JSON}"
: "${WRITE_CACHE_DIR:?set WRITE_CACHE_DIR to a dedicated empty directory}"
: "${READ_ONLY_CACHE_DIR:?set READ_ONLY_CACHE_DIR to another empty directory}"
: "${SUBSET_SIZE:?set SUBSET_SIZE to numTraining in DATALIST_JSON}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STABLE_STEPS="${STABLE_STEPS:-300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/perf_regression/cache_write_ablation}"
CASE_ORDER="${CASE_ORDER:-CW0_CW1}"

case "${CASE_ORDER}" in
    CW0_CW1)
        COLD_CASES=(CW0 CW1)
        ;;
    CW1_CW0)
        COLD_CASES=(CW1 CW0)
        ;;
    *)
        echo "CASE_ORDER must be CW0_CW1 or CW1_CW0" >&2
        exit 2
        ;;
esac

if (( SUBSET_SIZE % BATCH_SIZE != 0 )); then
    echo "SUBSET_SIZE must be divisible by BATCH_SIZE for a clean one-epoch boundary" >&2
    exit 2
fi
EPOCH_STEPS=$((SUBSET_SIZE / BATCH_SIZE))
TOTAL_STEPS=$((EPOCH_STEPS + STABLE_STEPS))

ensure_empty_directory() {
    local directory="$1"
    mkdir -p "${directory}"
    if find "${directory}" -mindepth 1 -print -quit | grep -q .; then
        echo "cache directory must be empty: ${directory}" >&2
        exit 2
    fi
}
ensure_empty_directory "${WRITE_CACHE_DIR}"
ensure_empty_directory "${READ_ONLY_CACHE_DIR}"
DATA_ROOT="$(realpath -m -- "${DATA_ROOT}")"
DATALIST_JSON="$(realpath -m -- "${DATALIST_JSON}")"
WRITE_CACHE_DIR="$(cd "${WRITE_CACHE_DIR}" && pwd)"
READ_ONLY_CACHE_DIR="$(cd "${READ_ONLY_CACHE_DIR}" && pwd)"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

# Keep intrusive timing and production-like utilization passes on independent
# cold PersistentDataset caches. The production passes run first; otherwise
# timing would also prewarm the shared OS page cache before utilization capture.
# The OS page cache is still shared between cases, which is why both case
# orders are required.
WRITE_TIMING_CACHE_DIR="${WRITE_CACHE_DIR}/timing"
WRITE_PRODUCTION_CACHE_DIR="${WRITE_CACHE_DIR}/production"
READ_ONLY_TIMING_CACHE_DIR="${READ_ONLY_CACHE_DIR}/timing"
READ_ONLY_PRODUCTION_CACHE_DIR="${READ_ONLY_CACHE_DIR}/production"
mkdir -p \
    "${WRITE_TIMING_CACHE_DIR}" \
    "${WRITE_PRODUCTION_CACHE_DIR}" \
    "${READ_ONLY_TIMING_CACHE_DIR}" \
    "${READ_ONLY_PRODUCTION_CACHE_DIR}"

run_timing_case() {
    local case_name="$1"
    local cache_dir="$2"
    shift 2
    local case_dir="${OUTPUT_ROOT}/${case_name}"
    mkdir -p "${case_dir}"
    (
        cd "${REPO_ROOT}/Self-supervised"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ocl_train.py \
            --dataset_mode ctrate_subset \
            --data_root "${DATA_ROOT}" \
            --datalist_json "${DATALIST_JSON}" \
            --cache_dir "${cache_dir}" \
            --cache \
            --batch_size "${BATCH_SIZE}" \
            --workers 16 \
            --prefetch_factor 2 \
            --persistent_workers \
            --pin_memory \
            --num_steps "${TOTAL_STEPS}" \
            --eval_num "${TOTAL_STEPS}" \
            --log_every "${TOTAL_STEPS}" \
            --diagnose_gpu_gaps \
            --gap_profile_steps "${TOTAL_STEPS}" \
            --gap_timing_csv "${case_dir}/step_timing.csv" \
            --disable_training_logging \
            --disable_checkpoint \
            --disable_validation \
            --disable_encoder_export \
            --disable_final_artifacts \
            --logdir "${case_dir}/checkpoints" \
            "$@"
    ) 2>&1 | tee "${case_dir}/train.log"

    "${PYTHON_BIN}" "${REPO_ROOT}/perf_regression/analyze_step_timing.py" \
        "${case_dir}/step_timing.csv" \
        --warmup 0 --measure "${EPOCH_STEPS}" \
        --output "${case_dir}/first_epoch_summary.json"
    "${PYTHON_BIN}" "${REPO_ROOT}/perf_regression/analyze_step_timing.py" \
        "${case_dir}/step_timing.csv" \
        --warmup "${EPOCH_STEPS}" --measure "${STABLE_STEPS}" \
        --output "${case_dir}/stable_summary.json"
}

run_production_case() {
    local case_name="$1"
    local cache_dir="$2"
    local warmup_steps="$3"
    local measure_steps="$4"
    local total_steps="$5"
    shift 5
    local case_dir="${OUTPUT_ROOT}/${case_name}"
    local dmon_pid=""
    mkdir -p "${case_dir}"

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
            --cache_dir "${cache_dir}" \
            --cache \
            --batch_size "${BATCH_SIZE}" \
            --workers 16 \
            --prefetch_factor 2 \
            --persistent_workers \
            --pin_memory \
            --num_steps "${total_steps}" \
            --eval_num "${total_steps}" \
            --log_every "${total_steps}" \
            --throughput_warmup_steps "${warmup_steps}" \
            --throughput_measure_steps "${measure_steps}" \
            --throughput_output "${case_dir}/throughput.json" \
            --disable_training_logging \
            --disable_checkpoint \
            --disable_validation \
            --disable_encoder_export \
            --disable_final_artifacts \
            --logdir "${case_dir}/production_checkpoints" \
            "$@"
    ) 2>&1 | tee "${case_dir}/production.log"
    local training_status=${PIPESTATUS[0]}
    set -e

    if [[ -n "${dmon_pid}" ]]; then
        kill "${dmon_pid}" 2>/dev/null || true
        wait "${dmon_pid}" 2>/dev/null || true
    fi
    if (( training_status != 0 )); then
        return "${training_status}"
    fi
}

# CW0 writes each miss; CW1 performs the same deterministic transforms but
# never writes. Their first-epoch delta is order-sensitive because the first
# case warms the OS page cache. Run production capture before intrusive timing,
# then repeat the entire script in reverse order with fresh dedicated roots.
for cold_case in "${COLD_CASES[@]}"; do
    case "${cold_case}" in
        CW0)
            run_production_case \
                CW0_cold_cache_write \
                "${WRITE_PRODUCTION_CACHE_DIR}" \
                "${EPOCH_STEPS}" "${STABLE_STEPS}" "${TOTAL_STEPS}"
            ;;
        CW1)
            run_production_case \
                CW1_cold_cache_read_only \
                "${READ_ONLY_PRODUCTION_CACHE_DIR}" \
                "${EPOCH_STEPS}" "${STABLE_STEPS}" "${TOTAL_STEPS}" \
                --disable_cache_write
            ;;
    esac
done

for cold_case in "${COLD_CASES[@]}"; do
    case "${cold_case}" in
        CW0)
            run_timing_case \
                CW0_cold_cache_write \
                "${WRITE_TIMING_CACHE_DIR}"
            ;;
        CW1)
            run_timing_case \
                CW1_cold_cache_read_only \
                "${READ_ONLY_TIMING_CACHE_DIR}" \
                --disable_cache_write
            ;;
    esac
done

# Both WRITE_* caches are now warm. Capture the non-intrusive CW2 reference
# before its timing pass as well.
run_production_case \
    CW2_prewarmed_hits \
    "${WRITE_PRODUCTION_CACHE_DIR}" \
    0 "${STABLE_STEPS}" "${STABLE_STEPS}" \
    --disable_cache_write

case_dir="${OUTPUT_ROOT}/CW2_prewarmed_hits"
mkdir -p "${case_dir}"
(
    cd "${REPO_ROOT}/Self-supervised"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ocl_train.py \
        --dataset_mode ctrate_subset \
        --data_root "${DATA_ROOT}" \
        --datalist_json "${DATALIST_JSON}" \
        --cache_dir "${WRITE_TIMING_CACHE_DIR}" \
        --cache --disable_cache_write \
        --batch_size "${BATCH_SIZE}" \
        --workers 16 --prefetch_factor 2 --persistent_workers --pin_memory \
        --num_steps "${STABLE_STEPS}" \
        --eval_num "${STABLE_STEPS}" --log_every "${STABLE_STEPS}" \
        --diagnose_gpu_gaps --gap_profile_steps "${STABLE_STEPS}" \
        --gap_timing_csv "${case_dir}/step_timing.csv" \
        --disable_training_logging --disable_checkpoint --disable_validation \
        --disable_encoder_export --disable_final_artifacts \
        --logdir "${case_dir}/checkpoints"
) 2>&1 | tee "${case_dir}/train.log"
"${PYTHON_BIN}" "${REPO_ROOT}/perf_regression/analyze_step_timing.py" \
    "${case_dir}/step_timing.csv" \
    --warmup 0 --measure "${STABLE_STEPS}" \
    --output "${case_dir}/stable_summary.json" \
    --require-no-cache-miss

echo "CW0-vs-CW1 is an order-sensitive cache-write candidate; it is not by itself an isolated causal estimate."
echo "Repeat with CASE_ORDER reversed and fresh cache roots; report the order sensitivity."
echo "CW2 is the stable prewarmed hit path."
echo "Do not report CW0 first-epoch throughput as steady-state training performance."
