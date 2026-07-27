"""Shared runtime helpers for the VoCo and OCL pre-training entry points."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

import torch


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


def str_to_bool(value: Any) -> bool:
    """Parse both modern flags and legacy ``--flag True/False`` syntax."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: str,
) -> None:
    """Add ``--name [BOOL]`` and ``--no-name`` without breaking old scripts."""

    option = name.replace("_", "-")
    positive = {f"--{name}", f"--{option}"}
    negative = {f"--no-{option}", f"--no_{name}"}
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        *sorted(positive),
        dest=name,
        nargs="?",
        const=True,
        default=default,
        type=str_to_bool,
        help=help,
    )
    group.add_argument(*sorted(negative), dest=name, action="store_false", help=argparse.SUPPRESS)


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset_mode",
        "--dataset-mode",
        choices=("original", "ctrate_subset"),
        default="original",
        help="original multi-dataset loader or isolated CT-RATE subset loader",
    )
    parser.add_argument(
        "--data_root",
        "--data-root",
        default="/home/bld/data/dataset/CT_RATE/train",
        help="CT-RATE train directory used by ctrate_subset mode",
    )
    parser.add_argument(
        "--datalist_json",
        "--datalist-json",
        default="/home/bld/data/dataset/CT_RATE/ct_rate_subset_256.json",
        help="subset Decathlon datalist used by ctrate_subset mode",
    )
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        default="/home/bld/data/cache/ctrate_subset_256",
        help="PersistentDataset cache location used by ctrate_subset mode",
    )
    parser.add_argument(
        "--data_check_only",
        "--data-check-only",
        action="store_true",
        help="validate the CT-RATE datalist, transforms, and one collated batch, then exit",
    )
    parser.add_argument(
        "--data_check_samples",
        "--data-check-samples",
        type=int,
        default=3,
        help="number of raw NIfTI headers printed by data_check_only",
    )
    parser.add_argument("--seed", type=int, default=2026, help="data-check and sampler seed")


def get_pretraining_loader(args):
    if args.dataset_mode == "ctrate_subset":
        from utils.data_utils_ctrate_subset import get_loader
    else:
        # This intentionally stays local: original mode retains the old loader,
        # while CT-only mode never imports its eager multi-dataset inventory.
        from utils.data_utils import get_loader
    return get_loader(args)


def run_data_check_only(args) -> None:
    if args.dataset_mode != "ctrate_subset":
        raise ValueError("--data_check_only currently requires --dataset_mode ctrate_subset")
    # Data check is deliberately single-process and runs before CUDA/DDP/model
    # initialization in both training entry points.
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    from utils.data_utils_ctrate_subset import run_data_check

    run_data_check(args)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def resolve_resume_path(value: Any, logdir: str) -> Path | None:
    if value is None or value is False:
        return None
    normalized = str(value).strip()
    if normalized.lower() in {"", "0", "false", "no", "off", "none"}:
        return None
    if normalized.lower() in {"1", "true", "yes", "on", "auto"}:
        return Path(logdir) / "model_current_epoch.pt"
    return Path(normalized).expanduser()


def checkpoint_payload(model, optimizer, scheduler, scaler, global_step: int, args) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 2,
        "global_step": int(global_step),
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "pretraining_method": args.pretraining_method,
        "feature_size": args.feature_size,
        "num_steps": args.num_steps,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def save_checkpoint(path: str | Path, model, optimizer, scheduler, scaler, global_step: int, args) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(checkpoint_payload(model, optimizer, scheduler, scaler, global_step, args), temporary)
    os.replace(temporary, destination)


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def restore_checkpoint(path: Path, model, optimizer, scheduler, scaler, device) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"resume checkpoint is missing state_dict: {path}")

    state_dict = _strip_module_prefix(checkpoint["state_dict"])
    model.load_state_dict(state_dict, strict=True)
    global_step = int(checkpoint.get("global_step", 0))

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        warnings.warn("legacy checkpoint has no optimizer state; optimizer starts fresh", stacklevel=2)

    if scheduler is not None:
        if "scheduler" in checkpoint:
            runtime_schedule = {
                name: getattr(scheduler, name)
                for name in ("warmup_steps", "t_total", "cycles")
                if hasattr(scheduler, name)
            }
            scheduler.load_state_dict(checkpoint["scheduler"])
            loaded_schedule = {
                name: getattr(scheduler, name) for name in runtime_schedule
            }
            # A short smoke checkpoint may intentionally be resumed with a
            # larger target. Keep the current command's schedule horizon while
            # restoring its step counters and then recompute LR at global_step.
            for name, value in runtime_schedule.items():
                setattr(scheduler, name, value)
            if loaded_schedule != runtime_schedule:
                warnings.warn(
                    f"resume schedule target changed; using current settings {runtime_schedule} "
                    f"instead of checkpoint settings {loaded_schedule}",
                    stacklevel=2,
                )
        elif global_step:
            warnings.warn(
                "legacy checkpoint has no scheduler state; aligning scheduler from global_step",
                stacklevel=2,
            )
        # LambdaLR supports the closed-form epoch argument, avoiding an
        # O(global_step) replay and ensuring the optimizer LR is recalculated
        # for the current target even when num_steps changed on resume.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scheduler.step(global_step)

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    if scheduler is not None and scheduler.last_epoch != global_step:
        raise RuntimeError(
            f"scheduler/global_step mismatch after resume: {scheduler.last_epoch} != {global_step}"
        )
    print(f"Resumed {path} at global_step={global_step}")
    return global_step


def save_final_artifacts(logdir: str, model, optimizer, scheduler, scaler, global_step: int, args) -> None:
    root = Path(logdir)
    raw_model = unwrap_model(model)
    save_checkpoint(root / "model_final_epoch.pt", model, optimizer, scheduler, scaler, global_step, args)
    torch.save(raw_model.state_dict(), root / "final_model.pt")
    # Plain backbone keys (swinViT.*, encoder1.*, ...) match downstream
    # SwinUNETR loaders directly, without requiring a backbone. prefix strip.
    torch.save(
        {"global_step": int(global_step), "state_dict": raw_model.backbone.state_dict()},
        root / "encoder_final.pt",
    )
