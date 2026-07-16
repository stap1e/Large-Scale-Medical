"""Independent CT-RATE subset loader for VoCo/OCL pipeline validation.

This module intentionally imports neither ``utils.data_utils`` nor any of the
abdomen/head-neck/chest dataset inventory modules.  It reuses only the existing
chest transform and therefore preserves the current VoCo/OCL view contract.
"""

from __future__ import annotations

import json
import pickle
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from monai.data import DataLoader, Dataset, PersistentDataset, load_decathlon_datalist
from monai.transforms import Compose
from torch.utils.data.distributed import DistributedSampler

from utils.data_trans import get_chest_trans


PATIENT_DIR_RE = re.compile(r"^train_(\d+)$")


def _read_manifest(args) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_root = Path(args.data_root).expanduser()
    datalist_json = Path(args.datalist_json).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(f"CT-RATE data_root is not a directory: {data_root}")
    if not datalist_json.is_file():
        raise FileNotFoundError(f"CT-RATE datalist_json does not exist: {datalist_json}")

    with datalist_json.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    raw_training = manifest.get("training")
    if not isinstance(raw_training, list) or not raw_training:
        raise ValueError(f"{datalist_json} must contain a non-empty 'training' list")
    if manifest.get("numTraining") != len(raw_training):
        raise ValueError(
            f"numTraining={manifest.get('numTraining')!r} does not match "
            f"training entries={len(raw_training)} in {datalist_json}"
        )

    patient_ids: set[int] = set()
    resolved_paths: list[Path] = []
    for index, item in enumerate(raw_training):
        if not isinstance(item, dict) or not isinstance(item.get("image"), str):
            raise ValueError(f"training[{index}] must be an object with a string 'image' path")
        relative = PurePosixPath(item["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"training[{index}] must use a safe path relative to data_root: {item['image']!r}")
        if not relative.parts:
            raise ValueError(f"training[{index}] has an empty image path")
        patient = relative.parts[0]
        match = PATIENT_DIR_RE.fullmatch(patient)
        if match is None:
            raise ValueError(
                f"training[{index}] must start with a train_<numeric_id> patient directory: {item['image']!r}"
            )
        patient_id = int(match.group(1))
        if patient_id in patient_ids:
            raise ValueError(
                f"patient {patient!r} appears more than once in {datalist_json}; "
                "CT-RATE subset mode requires one volume per patient"
            )
        patient_ids.add(patient_id)
        resolved_paths.append(data_root.joinpath(*relative.parts))

    missing = [path for path in resolved_paths if not path.is_file()]
    if missing:
        preview = "\n  ".join(str(path) for path in missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} of {len(resolved_paths)} CT-RATE paths do not exist:\n  {preview}{more}"
        )

    datalist = load_decathlon_datalist(
        str(datalist_json),
        is_segmentation=False,
        data_list_key="training",
        base_dir=str(data_root),
    )
    if len(datalist) != len(raw_training):
        raise RuntimeError(
            f"MONAI loaded {len(datalist)} entries, but the manifest contains {len(raw_training)}"
        )
    return manifest, datalist


def get_dataset(args, datalist: list[dict[str, Any]] | None = None):
    if datalist is None:
        _, datalist = _read_manifest(args)
    transform = Compose(get_chest_trans(args))
    if args.cache:
        cache_dir = Path(args.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return PersistentDataset(
            data=datalist,
            transform=transform,
            cache_dir=str(cache_dir),
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )
    return Dataset(data=datalist, transform=transform)


def _make_loader(args, dataset) -> DataLoader:
    is_ocl = getattr(args, "pretraining_method", "voco").lower() == "ocl"
    # OCL performs an in-batch contrastive loss.  Full per-rank batches prevent
    # rank-dependent shapes, especially on the final batch of a DDP epoch.
    drop_last = is_ocl
    sampler = None
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=getattr(args, "seed", 2026),
            drop_last=drop_last,
        )

    workers = int(args.workers)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        num_workers=workers,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        # PyTorch rejects persistent_workers=True when num_workers == 0.
        persistent_workers=workers > 0,
    )


def get_loader(args) -> DataLoader:
    _, datalist = _read_manifest(args)
    dataset = get_dataset(args, datalist)
    loader = _make_loader(args, dataset)
    if not len(loader):
        raise RuntimeError(
            f"CT-RATE loader has zero batches (dataset={len(dataset)}, batch_size={args.batch_size}, "
            f"drop_last={loader.drop_last})"
        )
    return loader


def _finite_and_stats(value: Any, name: str) -> bool:
    if torch.is_tensor(value):
        tensor = value.detach()
        finite = bool(torch.isfinite(tensor).all()) if (tensor.is_floating_point() or tensor.is_complex()) else True
        minimum = tensor.min().item() if tensor.numel() else "empty"
        maximum = tensor.max().item() if tensor.numel() else "empty"
        print(
            f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"min={minimum}, max={maximum}, finite={finite}"
        )
        return finite
    if isinstance(value, np.ndarray):
        finite = bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
        minimum = value.min().item() if value.size else "empty"
        maximum = value.max().item() if value.size else "empty"
        print(
            f"  {name}: shape={value.shape}, dtype={value.dtype}, "
            f"min={minimum}, max={maximum}, finite={finite}"
        )
        return finite
    if isinstance(value, Mapping):
        return all(_finite_and_stats(item, f"{name}.{key}") for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_and_stats(item, f"{name}[{index}]") for index, item in enumerate(value))
    print(f"  {name}: {type(value).__name__}={value!r}")
    return True


def _concat_views(views) -> torch.Tensor:
    output = torch.cat([view["image"] for view in views], dim=1)
    batch_size, view_count, x, y, z = output.shape
    return output.reshape(batch_size * view_count, 1, x, y, z)


def run_data_check(args) -> None:
    """Run an end-to-end CPU-safe manifest, transform, and collate check."""

    manifest, datalist = _read_manifest(args)
    print("=== CT-RATE subset data check ===")
    print(f"JSON: {Path(args.datalist_json).expanduser()}")
    print(f"JSON entries: {len(datalist)} (numTraining={manifest['numTraining']})")
    print(f"All {len(datalist)} paths exist under: {Path(args.data_root).expanduser()}")
    print(f"Cache: {Path(args.cache_dir).expanduser() if args.cache else 'disabled'}")

    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("data_check_only requires nibabel to inspect NIfTI headers") from exc

    sample_count = min(int(getattr(args, "data_check_samples", 3)), len(datalist))
    rng = random.Random(getattr(args, "seed", 2026))
    print(f"Random NIfTI headers ({sample_count}):")
    for item in rng.sample(datalist, sample_count):
        path = Path(item["image"])
        image = nib.load(str(path), mmap=False)
        shape = tuple(int(size) for size in image.shape)
        spacing = tuple(float(value) for value in image.header.get_zooms()[: len(shape)])
        orientation = "".join(axis or "?" for axis in nib.aff2axcodes(image.affine))
        print(
            f"  path={path}; shape={shape}; dtype={np.dtype(image.get_data_dtype())}; "
            f"spacing={spacing}; orientation={orientation}"
        )

    dataset = get_dataset(args, datalist)
    print("Transformed sample tensors:")
    transformed = dataset[0]
    if not _finite_and_stats(transformed, "sample"):
        raise RuntimeError("NaN or Inf found in transformed sample")

    loader = _make_loader(args, dataset)
    if not len(loader):
        raise RuntimeError(
            f"CT-RATE loader has zero batches (dataset={len(dataset)}, batch_size={args.batch_size}, "
            f"drop_last={loader.drop_last})"
        )
    batch = next(iter(loader))
    print("Collated batch tensors:")
    if not _finite_and_stats(batch, "batch"):
        raise RuntimeError("NaN or Inf found in collated batch")

    random_views, labels, base_views = batch
    random_crops = _concat_views(random_views)
    base_crops = _concat_views(base_views)
    expected_random = labels.shape[0] * int(args.sw_batch_size)
    expected_bases = labels.shape[0] * 9
    if tuple(random_crops.shape[1:]) != (1, 64, 64, 64) or random_crops.shape[0] != expected_random:
        raise RuntimeError(f"unexpected random-crop training shape: {tuple(random_crops.shape)}")
    if tuple(base_crops.shape[1:]) != (1, 64, 64, 64) or base_crops.shape[0] != expected_bases:
        raise RuntimeError(f"unexpected base-crop training shape: {tuple(base_crops.shape)}")
    if tuple(labels.shape[1:]) != (int(args.sw_batch_size), 9):
        raise RuntimeError(f"unexpected VoCo label shape: {tuple(labels.shape)}")

    print("Training-loop batch contract:")
    print(f"  random crops: {tuple(random_crops.shape)}")
    print(f"  base crops:   {tuple(base_crops.shape)}")
    print(f"  labels:       {tuple(labels.shape)}")
    if getattr(args, "pretraining_method", "voco").lower() == "ocl":
        print("  OCL consumes random crops; its existing model code creates the two masked views.")
    print("Data check passed.")
