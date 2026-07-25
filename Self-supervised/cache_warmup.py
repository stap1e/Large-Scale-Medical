"""Pre-build the MONAI PersistentDataset cache for a CT-RATE subset.

Run from the Self-supervised directory with the same ``--data_root``,
``--datalist_json`` and ``--cache_dir`` you will pass to training, so the
cache is fully populated before the first training step.

Already-cached items are skipped, so this is safe to interrupt (Ctrl+C) and
re-run: a re-run only builds the items that are still missing.  With
``--workers N > 1`` the missing items are spread across N processes.

Example:
    python cache_warmup.py \
        --dataset_mode ctrate_subset \
        --data_root /ai/workspace/datasets/CT_RATE/train \
        --datalist_json /ai/workspace/datasets/CT_RATE/ct_rate_subset_2065.json \
        --cache_dir /ai/workspace/cache/ctrate_subset_2065 \
        --workers 8
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path
from time import time

import torch
from monai.data.utils import pickle_hashing

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

_DATASET = None
_TOTAL = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm up the CT-RATE PersistentDataset cache")
    add_data_arguments(parser)
    parser.add_argument("--workers", default=8, type=int, help="parallel cache-building processes")
    parser.add_argument("--roi_x", default=64, type=int)
    parser.add_argument("--roi_y", default=64, type=int)
    parser.add_argument("--roi_z", default=64, type=int)
    parser.add_argument("--sw_batch_size", default=2, type=int)
    return parser


def _worker_init(args, datalist) -> None:
    global _DATASET, _TOTAL
    _patch_torch_load_weights_only()
    _DATASET = get_dataset(args, datalist)
    _TOTAL = len(_DATASET)


def _hash_file(dataset, index: int) -> Path:
    item = dataset.data[index]
    return dataset.cache_dir / f"{pickle_hashing(item)}.pt"


def _build_one(index: int) -> bool:
    if _hash_file(_DATASET, index).is_file():
        return False
    _DATASET[index]
    return True


def main() -> None:
    args = build_parser().parse_args()
    args.cache = True

    manifest, datalist = _read_manifest(args)
    print(f"datalist: {args.datalist_json}")
    print(f"entries: {len(datalist)} (numTraining={manifest['numTraining']})")
    print(f"cache_dir: {args.cache_dir}")
    print(f"workers: {max(1, args.workers)}")

    dataset = get_dataset(args, datalist)
    total = len(dataset)
    cache_dir = Path(dataset.cache_dir)
    existing = {p.name for p in cache_dir.glob("*.pt")} if cache_dir.is_dir() else set()

    # Verify the hashing scheme reproduces MONAI's cache file names before
    # trusting the cached/missing split below.
    if existing:
        if _hash_file(dataset, 0).name not in existing:
            raise RuntimeError(
                "computed cache hash does not match existing cache files; "
                "MONAI's PersistentDataset hashing scheme differs from pickle_hashing(item, transform)"
            )

    cached = sum(1 for i in range(total) if _hash_file(dataset, i).name in existing)
    print(f"already cached: {cached}/{total}; building {total - cached}")
    if cached == total:
        print(f"Cache already complete at {args.cache_dir}")
        return

    start = time()
    workers = max(1, args.workers)
    if workers == 1:
        _worker_init(args, datalist)
        done = 0
        for index in range(total):
            if _build_one(index):
                done += 1
                if done % 20 == 0:
                    elapsed = time() - start
                    print(f"  built {done}/{total - cached}  ({elapsed:.1f}s, {done / elapsed:.2f} it/s)")
    else:
        done = 0
        with Pool(
            workers,
            initializer=_worker_init,
            initargs=(args, datalist),
            maxtasksperchild=16,
        ) as pool:
            for built in pool.imap_unordered(_build_one, range(total), chunksize=4):
                if built:
                    done += 1
                    if done % 20 == 0 or done == total - cached:
                        elapsed = time() - start
                        print(f"  built {done}/{total - cached}  ({elapsed:.1f}s, {done / elapsed:.2f} it/s)")
    print(f"Cache ready at {args.cache_dir}; built {total - cached} items in {time() - start:.1f}s")


if __name__ == "__main__":
    main()
