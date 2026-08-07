#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "usage: $0 TRAIN_PID" >&2
    exit 2
fi

TRAIN_PID="$1"
OUTPUT_DIR="${OUTPUT_DIR:-perf_regression/system_monitor_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
children=()

start_if_available() {
    local command_name="$1"
    local output_name="$2"
    shift 2
    if command -v "${command_name}" >/dev/null 2>&1; then
        "$@" >"${OUTPUT_DIR}/${output_name}" 2>&1 &
        children+=("$!")
    else
        echo "${command_name} is unavailable" >"${OUTPUT_DIR}/${output_name}"
    fi
}

cleanup() {
    for child in "${children[@]}"; do
        kill "${child}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_if_available nvidia-smi nvidia_dmon.txt \
    nvidia-smi dmon -s pucvmet -d 1 -o DT
start_if_available pidstat pidstat.txt \
    pidstat -dru -p "${TRAIN_PID}" 1
start_if_available iostat iostat.txt \
    iostat -xzt 1
start_if_available vmstat vmstat.txt \
    vmstat -t 1
start_if_available pidstat pidstat_threads.txt \
    pidstat -t -dru -p "${TRAIN_PID}" 1

echo "Monitoring PID ${TRAIN_PID}; outputs: ${OUTPUT_DIR}; press Ctrl-C to stop."
wait
