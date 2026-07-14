#!/usr/bin/env bash

set -u

HF_BIN="/home/bld/data/.venv/bin/hf"
REPO_ID="Luffy503/PreCT-160K"
LOCAL_DIR="/home/bld/data/dataset/PreCT-160K"
HF_TOKEN="${HF_TOKEN:-}"

export HF_DEBUG=1
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
unset HF_HUB_DISABLE_PROGRESS_BARS

mkdir -p "${LOCAL_DIR}"

while true; do
    "${HF_BIN}" download         --repo-type dataset         "${REPO_ID}"         --local-dir "${LOCAL_DIR}" --token "${HF_TOKEN}"

    rc=$?

    if [[ "${rc}" -eq 0 ]]; then
        echo "PreCT-160K 下载完成。"
        break
    fi

    echo "下载中断，退出码=${rc}；300 秒后重新执行并检查已有文件。"
    sleep 300
done