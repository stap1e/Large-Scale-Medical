"""Pre-build the MONAI PersistentDataset cache for a CT-RATE subset.

Run from the Self-supervised directory with the same ``--data_root``,
``--datalist_json`` and ``--cache_dir`` you will pass to training, so the
cache is fully populated before the first training step.

Example:
    python cache_warmup.py \
        --dataset_mode ctrate_subset \
        --data_root /ai/workspace/datasets/CT_RATE/train \
        --datalist_json /ai/workspace/datasets/CT_RATE/ct_rate_subset_256.json \
        --cache_dir /ai/workspace/cache/ctrate_subset_256 \
        --workers 8
"""

from __future__ import annotations

import argparse
from time import time

from utils.data_utils_ctrate_subset import _read_manifest, get_dataset
from utils.pretrain_common import add_data_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm up the CT-RATE PersistentDataset cache")
    add_data_arguments(parser)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--roi_x", default=64, type=int)
    parser.add_argument("--roi_y", default=64, type=int)
    parser.add_argument("--roi_z", default=64, type=int)
    parser.add_argument("--sw_batch_size", default=2, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.cache = True

    manifest, datalist = _read_manifest(args)
    print(f"datalist: {args.datalist_json}")
    print(f"entries: {len(datalist)} (numTraining={manifest['numTraining']})")
    print(f"cache_dir: {args.cache_dir}")

    dataset = get_dataset(args, datalist)
    start = time()
    for index in range(len(dataset)):
        dataset[index]
        if (index + 1) % 50 == 0 or index + 1 == len(dataset):
            elapsed = time() - start
            print(f"  {index + 1}/{len(dataset)}  ({elapsed:.1f}s, {(index + 1) / elapsed:.2f} it/s)")
    print(f"Cache ready at {args.cache_dir} in {time() - start:.1f}s")


if __name__ == "__main__":
    main()
