"""Pre-build the MONAI PersistentDataset cache for a CT-RATE subset.

Run from the Self-supervised directory with the same ``--data_root``,
``--datalist_json`` and ``--cache_dir`` you will pass to training.

This version deliberately does NOT replicate MONAI's internal cache hashing.
It simply accesses every item once and lets PersistentDataset load from cache
when a cache file already exists, or build and save it otherwise.  It is safe
to interrupt (Ctrl+C) and re-run at any time.

Example:
    python cache_warmup.py \
        --dataset_mode ctrate_subset \
        --data_root /ai/workspace/datasets/CT_RATE/train \
        --datalist_json /ai/workspace/datasets/CT_RATE/ct_rate_subset_2065.json \
        --cache_dir /ai/workspace/cache/ctrate_subset_2065
"""

from __future__ import annotations

import argparse
from time import time

import torch

from utils.data_utils_ctrate_subset import _read_manifest, get_dataset
from utils.pretrain_common import add_data_arguments


def _patch_torch_load_weights_only() -> None:
    # MONAI 1.3.0 PersistentDataset loads its cache via torch.load without
    # passing weights_only.  torch >= 2.6 defaults weights_only=True, which
    # cannot unpickle MONAI's cached MetaTensor/numpy objects.  The cache is
    # produced locally and trusted, so default to weights_only=False here.
    try:
        import inspect

        if "weights_only" not in inspect.signature(torch.load).parameters:
            return
        if getattr(torch.load, "_voc_weights_only_patched", False):
            return
        _original = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original(*args, **kwargs)

        _load._voc_weights_only_patched = True  # type: ignore[attr-defined]
        torch.load = _load
    except Exception:
        pass


_patch_torch_load_weights_only()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm up the CT-RATE PersistentDataset cache")
    add_data_arguments(parser)
    parser.add_argument("--workers", default=8, type=int, help="(unused; kept for CLI compatibility)")
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
    total = len(dataset)
    start = time()
    for index in range(total):
        dataset[index]
        if (index + 1) % 50 == 0 or index + 1 == total:
            elapsed = time() - start
            print(f"  {index + 1}/{total}  ({elapsed:.1f}s, {(index + 1) / elapsed:.2f} it/s)")
    print(f"Cache ready at {args.cache_dir} in {time() - start:.1f}s")


if __name__ == "__main__":
    main()
