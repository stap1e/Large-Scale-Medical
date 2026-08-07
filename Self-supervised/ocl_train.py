"""OCL (MiCL) pre-training with the existing VoCo 3D backbone and views.

Runtime/data-source selection and GPU-gap diagnostics live here.  OCL view
generation and the default loss semantics remain in ``models.ocl_head.OCLHead3D``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import statistics
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter, time as wall_time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from models.ocl_head import OCLHead3D
from optimizers.lr_scheduler import WarmupCosineSchedule
from utils.pretrain_common import (
    add_bool_argument,
    add_data_arguments,
    get_pretraining_loader,
    resolve_resume_path,
    restore_checkpoint,
    run_data_check_only,
    save_checkpoint,
    save_checkpoint_pair,
    save_final_artifacts,
    str_to_bool,
)
from utils.perf_diagnostics import (
    CudaPhaseTimer,
    StepTimingRecorder,
    create_gap_profiler,
    phase_range,
    prefetch_queue_depth,
    prefetch_tasks_outstanding,
    summarize_batch_metadata,
)


torch.multiprocessing.set_sharing_strategy("file_system")
try:  # Linux workers benefit from a larger file-descriptor limit.
    import resource

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, hard_limit), hard_limit))
except (ImportError, OSError, ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    roi = 64
    parser = argparse.ArgumentParser(description="OCL (MiCL) 3D Pre-Training")
    parser.add_argument("--logdir", default="logs_ocl", type=str, help="directory to save logs")
    parser.add_argument("--num_steps", "--num-steps", default=2_000_000, type=int)
    parser.add_argument("--eval_num", "--eval-num", default=20_000, type=int, help="checkpoint frequency")
    parser.add_argument("--log_every", "--log-every", default=1_000, type=int)
    add_bool_argument(
        parser,
        "sync_timing",
        default=False,
        help="synchronize CUDA for accurate smoke-test step timing",
    )
    add_bool_argument(
        parser,
        "diagnose_gpu_gaps",
        default=False,
        help="collect bounded per-stage CPU/CUDA timings for GPU starvation",
    )
    parser.add_argument(
        "--gap_profile_steps",
        "--gap-profile-steps",
        default=500,
        type=int,
        help="maximum number of diagnostic steps written to the timing CSV",
    )
    parser.add_argument(
        "--slow_step_threshold_ms",
        "--slow-step-threshold-ms",
        default=None,
        type=float,
        help="optional absolute slow-step threshold in addition to 1.5x rolling median",
    )
    add_bool_argument(
        parser,
        "save_gap_trace",
        default=False,
        help="capture a torch.profiler trace with a configurable long schedule",
    )
    add_bool_argument(
        parser,
        "emit_nvtx",
        default=False,
        help="emit phase NVTX ranges without enabling per-step CUDA timing/sync",
    )
    parser.add_argument("--profiler_wait_steps", type=int, default=20)
    parser.add_argument("--profiler_warmup_steps", type=int, default=10)
    parser.add_argument("--profiler_active_steps", type=int, default=200)
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--gap_timing_csv",
        "--gap-timing-csv",
        default=str(repository_root / "perf_regression" / "step_timing.csv"),
        help="diagnostic step CSV (nonzero DDP ranks receive a rank suffix)",
    )
    parser.add_argument(
        "--gap_trace_dir",
        "--gap-trace-dir",
        default=str(repository_root / "perf_regression" / "traces"),
        help="torch.profiler trace directory",
    )
    parser.add_argument(
        "--throughput_warmup_steps",
        "--throughput-warmup-steps",
        default=0,
        type=int,
        help="warmup steps excluded from the optional non-intrusive throughput window",
    )
    parser.add_argument(
        "--throughput_output",
        "--throughput-output",
        default=None,
        help="write a boundary-synchronized throughput JSON without per-step CUDA sync",
    )
    parser.add_argument(
        "--throughput_measure_steps",
        "--throughput-measure-steps",
        default=500,
        type=int,
        help="successful optimizer updates measured after throughput warmup",
    )
    parser.add_argument("--warmup_steps", "--warmup-steps", default=5_000, type=int)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--dropout_path_rate", default=0.0, type=float)
    add_bool_argument(parser, "use_checkpoint", default=True, help="use gradient checkpointing")
    parser.add_argument("--spatial_dims", default=3, type=int)
    parser.add_argument("--a_min", default=-175.0, type=float)
    parser.add_argument("--a_max", default=250.0, type=float)
    parser.add_argument("--b_min", default=0.0, type=float)
    parser.add_argument("--b_max", default=1.0, type=float)
    parser.add_argument("--space_x", default=1.5, type=float)
    parser.add_argument("--space_y", default=1.5, type=float)
    parser.add_argument("--space_z", default=1.5, type=float)
    parser.add_argument("--roi_x", default=roi, type=int)
    parser.add_argument("--roi_y", default=roi, type=int)
    parser.add_argument("--roi_z", default=roi, type=int)
    parser.add_argument("--batch_size", "--batch-size", default=4, type=int)
    parser.add_argument("--sw_batch_size", "--sw-batch-size", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--decay", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    add_bool_argument(parser, "lrdecay", default=True, help="enable learning-rate decay")
    parser.add_argument(
        "--workers", "--num_workers", "--num-workers", default=16, type=int
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--opt", choices=("adam", "adamw", "sgd"), default="adamw")
    parser.add_argument("--lr_schedule", choices=("warmup_cosine", "poly"), default="warmup_cosine")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume path; bare --resume or legacy --resume True uses <logdir>/model_current_epoch.pt",
    )
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0)
    add_bool_argument(parser, "grad_clip", default=False, help="clip gradient norm")
    add_bool_argument(
        parser,
        "strict_step_control",
        default=True,
        help="advance global step/scheduler only when GradScaler calls optimizer.step",
    )
    add_bool_argument(
        parser,
        "merge_ocl_views",
        default=False,
        help="encode both masked OCL views in one concatenated backbone call",
    )
    add_bool_argument(
        parser,
        "legacy_amp_scale_polling",
        default=False,
        help="A/B only: reproduce the regressed per-step GradScaler.get_scale host syncs",
    )
    add_bool_argument(
        parser,
        "cpu_concat_before_h2d",
        default=False,
        help="A/B only: reproduce CPU view concatenation before the device copy",
    )
    add_bool_argument(
        parser,
        "duplicate_checkpoint_serialization",
        default=False,
        help="A/B only: reproduce serializing identical current/step checkpoints twice",
    )
    add_bool_argument(
        parser,
        "full_periodic_checkpoint",
        default=True,
        help=(
            "include optimizer/scheduler/scaler in periodic checkpoints; disable "
            "only for an explicitly lightweight diagnostic snapshot"
        ),
    )
    add_bool_argument(
        parser,
        "legacy_loss_item_each_step",
        default=False,
        help="A/B only: reproduce the old per-step loss.item host synchronization",
    )
    parser.add_argument(
        "--disable_final_artifacts",
        "--disable-final-artifacts",
        action="store_true",
        help="A/B only: omit end-of-run saves while retaining periodic checkpoint behavior",
    )
    parser.add_argument("--noamp", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--dist-url", default="env://")
    add_bool_argument(parser, "cache", default=True, help="use MONAI PersistentDataset cache")
    add_data_arguments(parser)
    # OCL's short CT subset needs persistent prefetch workers by default. Keep
    # the shared VoCo parser's legacy default unchanged.
    parser.set_defaults(persistent_workers=True)

    # Existing OCL/MiCL algorithm hyperparameters.
    parser.add_argument("--mask_drop", default=0.3, type=float)
    parser.add_argument("--kappa", default=1.0 / 64.0, type=float)
    parser.add_argument("--temperature", default=0.07, type=float)
    return parser


def make_optimizer(args, model):
    if args.opt == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if args.opt == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)
    return optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay
    )


def make_scheduler(args, optimizer):
    if not args.lrdecay:
        return None
    if args.lr_schedule == "warmup_cosine":
        return WarmupCosineSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps
        )

    def polynomial(step: int) -> float:
        progress = min(float(step) / float(max(1, args.num_steps)), 1.0)
        return (1.0 - progress) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial)


def _unpack_ocl_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) not in (3, 4):
        raise RuntimeError(
            "OCL DataLoader must return (image_views, labels, base_views[, diagnostic_metadata])"
        )
    metadata = batch[3] if len(batch) == 4 else None
    return batch[0], batch[1], batch[2], metadata


def _view_tensors(image_views) -> list[torch.Tensor]:
    tensors = []
    for view in image_views:
        value = view["image"] if isinstance(view, Mapping) else view
        if hasattr(value, "as_tensor"):
            value = value.as_tensor()
        if not torch.is_tensor(value):
            raise TypeError(f"OCL image view must be a tensor, got {type(value).__name__}")
        tensors.append(value)
    if not tensors:
        raise RuntimeError("OCL batch contains no image views")
    return tensors


def _h2d_views(
    sources: list[torch.Tensor], device: torch.device
) -> tuple[list[torch.Tensor], bool]:
    pinned = all(source.device.type == "cpu" and source.is_pinned() for source in sources)
    moved = [source.to(device, non_blocking=pinned) for source in sources]
    return moved, pinned


def _concat_device_views(moved: list[torch.Tensor]) -> torch.Tensor:
    output = torch.concatenate(moved, dim=1)
    batch_size, view_count, x_size, y_size, z_size = output.shape
    return output.reshape(batch_size * view_count, 1, x_size, y_size, z_size)


def _concat_cpu_views(sources: list[torch.Tensor]) -> torch.Tensor:
    output = torch.concatenate(sources, dim=1)
    batch_size, view_count, x_size, y_size, z_size = output.shape
    return output.reshape(batch_size * view_count, 1, x_size, y_size, z_size)


def _throughput_boundary_sync(args) -> None:
    torch.cuda.synchronize(args.device)
    if args.distributed:
        dist.barrier()


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile + 0.999999) - 1))
    return float(ordered[index])


def _write_effective_config(args, train_loader) -> None:
    """Persist actual argv and effective loader settings for regression A/B runs."""

    worker_init_fn = getattr(train_loader, "worker_init_fn", None)
    if worker_init_fn is None:
        worker_init_fn_name = None
    else:
        worker_init_fn_name = ".".join(
            part
            for part in (
                getattr(worker_init_fn, "__module__", None),
                getattr(worker_init_fn, "__qualname__", None)
                or getattr(worker_init_fn, "__name__", None),
            )
            if part
        ) or repr(worker_init_fn)
    loader_context = getattr(train_loader, "multiprocessing_context", None)
    if loader_context is None:
        loader_context_name = None
    elif hasattr(loader_context, "get_start_method"):
        loader_context_name = loader_context.get_start_method()
    else:
        loader_context_name = str(loader_context)

    repository_root = Path(__file__).resolve().parents[1]
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
        git_dirty = None
    try:
        import monai

        monai_version = monai.__version__
    except (ImportError, AttributeError):
        monai_version = "unavailable"
    config = {
        "argv": sys.argv,
        "git_commit": git_commit,
        "git_worktree_dirty": git_dirty,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "monai_version": monai_version,
        "gpu_name": torch.cuda.get_device_name(args.device),
        "dataset_mode": args.dataset_mode,
        "batch_size": args.batch_size,
        "num_workers": int(getattr(train_loader, "num_workers", args.workers)),
        "prefetch_factor": getattr(train_loader, "prefetch_factor", None),
        "persistent_workers": bool(
            getattr(train_loader, "persistent_workers", False)
        ),
        "pin_memory": bool(getattr(train_loader, "pin_memory", False)),
        "drop_last": bool(
            getattr(train_loader, "drop_last", False)
            or getattr(train_loader, "voco_continuous", False)
        ),
        "multiprocessing_context": loader_context_name,
        "worker_init_fn": worker_init_fn_name,
        "cache_enabled": bool(args.cache),
        "cache_rate": "N/A (PersistentDataset)",
        "cache_num": "N/A (PersistentDataset)",
        "continuous_dataloader": bool(getattr(train_loader, "voco_continuous", False)),
        "ocl_drop_last": args.ocl_drop_last,
        "logical_batches_per_epoch": int(
            getattr(train_loader, "voco_batches_per_epoch", len(train_loader))
        ),
        "log_interval": args.log_every,
        "checkpoint_interval": args.eval_num,
        "validation_interval": None,
        "strict_step_control": args.strict_step_control,
        "sync_timing": args.sync_timing,
        "diagnose_gpu_gaps": args.diagnose_gpu_gaps,
        "gap_profile_steps": args.gap_profile_steps,
        "save_gap_trace": args.save_gap_trace,
        "emit_nvtx": args.emit_nvtx,
        "throughput_warmup_steps": args.throughput_warmup_steps,
        "throughput_measure_steps": args.throughput_measure_steps,
        "profiler_schedule": {
            "wait": args.profiler_wait_steps,
            "warmup": args.profiler_warmup_steps,
            "active": args.profiler_active_steps,
        },
        "disable_training_logging": args.disable_training_logging,
        "disable_checkpoint": args.disable_checkpoint,
        "disable_validation": args.disable_validation,
        "disable_cache_write": args.disable_cache_write,
        "disable_encoder_export": args.disable_encoder_export,
        "disable_data_integrity_check": args.disable_data_integrity_check,
        "skip_unused_ocl_crops": args.skip_unused_ocl_crops,
        "merge_ocl_views": args.merge_ocl_views,
        "legacy_amp_scale_polling": args.legacy_amp_scale_polling,
        "cpu_concat_before_h2d": args.cpu_concat_before_h2d,
        "duplicate_checkpoint_serialization": args.duplicate_checkpoint_serialization,
        "full_periodic_checkpoint": args.full_periodic_checkpoint,
        "legacy_loss_item_each_step": args.legacy_loss_item_each_step,
        "disable_final_artifacts": args.disable_final_artifacts,
    }
    outputs = {Path(args.logdir) / "effective_config.json"}
    if args.diagnose_gpu_gaps:
        outputs.add(Path(args.gap_timing_csv).parent / "effective_config.json")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, output)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.pretraining_method = "ocl"
    args.amp = not args.noamp
    if args.num_steps <= 0 or args.eval_num <= 0 or args.log_every <= 0:
        parser.error("num_steps, eval_num, and log_every must be positive")
    if args.workers < 0 or args.batch_size <= 0 or args.data_check_samples <= 0:
        parser.error("workers must be non-negative; batch_size and data_check_samples must be positive")
    if args.gap_profile_steps <= 0:
        parser.error("gap_profile_steps must be positive")
    if args.prefetch_factor <= 0:
        parser.error("prefetch_factor must be positive")
    if (
        args.profiler_wait_steps < 0
        or args.profiler_warmup_steps <= 0
        or args.profiler_active_steps <= 0
    ):
        parser.error("profiler wait must be non-negative; warmup/active must be positive")
    if args.slow_step_threshold_ms is not None and args.slow_step_threshold_ms <= 0:
        parser.error("slow_step_threshold_ms must be positive")
    if args.throughput_warmup_steps < 0 or args.throughput_measure_steps <= 0:
        parser.error(
            "throughput_warmup_steps must be non-negative and "
            "throughput_measure_steps must be positive"
        )

    if args.data_check_only:
        run_data_check_only(args)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; use --data_check_only for the CPU data check")

    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    ddp_cleanup = None
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.device = torch.device("cuda", args.local_rank)

        def cleanup_distributed() -> None:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()

        ddp_cleanup = cleanup_distributed
        atexit.register(ddp_cleanup)
    else:
        args.world_size = 1
        args.rank = 0
        args.device = torch.device("cuda", 0)
        torch.cuda.set_device(args.device)

    random.seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    torch.cuda.manual_seed_all(args.seed + args.rank)
    torch.backends.cudnn.benchmark = True

    logdir = Path(args.logdir)
    if args.rank == 0:
        logdir.mkdir(parents=True, exist_ok=True)
        mode = f"DDP ({args.world_size} ranks)" if args.distributed else "single GPU"
        print(f"Training OCL in {mode}; dataset_mode={args.dataset_mode}")
        if args.sync_timing:
            print(
                "WARNING: --sync_timing synchronizes CUDA around every step and must not be "
                "used for utilization/throughput measurements."
            )
        if args.throughput_output and (args.diagnose_gpu_gaps or args.sync_timing):
            print(
                "WARNING: throughput output is being collected with per-step synchronization; "
                "run a separate pass without --diagnose_gpu_gaps/--sync_timing for production metrics."
            )
        if (
            args.legacy_amp_scale_polling
            or args.cpu_concat_before_h2d
            or args.duplicate_checkpoint_serialization
            or args.legacy_loss_item_each_step
        ):
            print(
                "WARNING: one or more legacy regression-reproduction switches are active; "
                "this run is for A/B diagnosis, not production training."
            )

    model = OCLHead3D(args).to(args.device)
    if args.rank == 0:
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"Total trainable parameters: {parameters}")

    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)
    scaler = GradScaler(enabled=args.amp)
    optimizer_step_state = {"called": False}
    optimizer_step_hook = None
    if args.amp and args.strict_step_control:
        if not hasattr(optimizer, "register_step_post_hook"):
            raise RuntimeError(
                "strict step control requires Optimizer.register_step_post_hook; "
                "do not reintroduce GradScaler.get_scale(), which synchronizes CUDA"
            )

        def mark_optimizer_step(_optimizer, _args, _kwargs):
            optimizer_step_state["called"] = True

        optimizer_step_hook = optimizer.register_step_post_hook(mark_optimizer_step)

    global_step = 0
    resume_path = resolve_resume_path(args.resume, args.logdir)
    if resume_path is not None:
        global_step = restore_checkpoint(
            resume_path, model, optimizer, scheduler, scaler if args.amp else None, args.device
        )
    if global_step > args.num_steps:
        raise ValueError(
            f"checkpoint global_step={global_step} exceeds target num_steps={args.num_steps}"
        )
    remaining_steps = args.num_steps - global_step
    throughput_required_steps = (
        args.throughput_warmup_steps + args.throughput_measure_steps
    )
    if args.throughput_output and remaining_steps < throughput_required_steps:
        raise ValueError(
            "throughput window does not fit after resume: "
            f"remaining={remaining_steps}, required={throughput_required_steps} "
            f"(warmup={args.throughput_warmup_steps}, "
            f"measure={args.throughput_measure_steps})"
        )
    profiler_required_steps = (
        args.profiler_wait_steps
        + args.profiler_warmup_steps
        + args.profiler_active_steps
    )
    if args.save_gap_trace and remaining_steps < profiler_required_steps:
        raise ValueError(
            "profiler schedule does not fit after resume: "
            f"remaining={remaining_steps}, required={profiler_required_steps}"
        )

    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )

    train_loader = get_pretraining_loader(args)
    if args.rank == 0:
        _write_effective_config(args, train_loader)
    recorder = (
        StepTimingRecorder(
            args.gap_timing_csv,
            rank=args.rank,
            max_steps=args.gap_profile_steps,
            slow_step_threshold_ms=args.slow_step_threshold_ms,
        )
        if args.diagnose_gpu_gaps
        else None
    )
    profiler = create_gap_profiler(args, args.rank)
    if profiler is not None:
        profiler.start()
    profiler_range_steps = (
        args.profiler_wait_steps
        + args.profiler_warmup_steps
        + args.profiler_active_steps
    )

    running_loss_sum = (
        torch.zeros((), device=args.device)
        if args.rank == 0 and not args.disable_training_logging
        else None
    )
    running_loss_cpu_sum = 0.0
    running_loss_count = 0
    total_step_time = 0.0
    total_data_time = 0.0
    measured_steps = 0
    loader_epoch = 0
    iterator_batch_index = 0
    consumed_batches = 0
    data_iterator = None
    continuous_loader = bool(getattr(train_loader, "voco_continuous", False))
    logical_batches_per_epoch = int(
        getattr(train_loader, "voco_batches_per_epoch", len(train_loader))
    )
    throughput_started_at = None
    throughput_finished_at = None
    throughput_started_wall = None
    throughput_finished_wall = None
    throughput_samples = 0
    throughput_query_crops = 0
    throughput_views = 0
    throughput_start_event = None
    throughput_step_end_events = []
    previous_step_tail_event = None
    previous_triggered_checkpoint = False
    previous_triggered_validation = False
    slow_gap_details = []
    disabled_cuda_timer = CudaPhaseTimer(False)
    torch.cuda.reset_peak_memory_stats(args.device)
    model.train()
    if args.throughput_output and args.throughput_warmup_steps == 0:
        _throughput_boundary_sync(args)
        throughput_start_event = torch.cuda.Event(enable_timing=True)
        throughput_start_event.record()
        throughput_started_wall = wall_time()
        throughput_started_at = perf_counter()

    try:
        while global_step < args.num_steps:
            timing_active = bool(recorder is not None and recorder.active)
            step_ranges_enabled = bool(
                args.emit_nvtx
                or timing_active
                or (
                    profiler is not None
                    and consumed_batches < profiler_range_steps
                )
            )
            step_start_unix = wall_time() if timing_active else 0.0
            step_wall_started = perf_counter()
            with phase_range("dataloader_wait", step_ranges_enabled):
                while True:
                    if data_iterator is None:
                        sampler = getattr(train_loader, "sampler", None)
                        if hasattr(sampler, "set_epoch"):
                            sampler.set_epoch(loader_epoch)
                        data_iterator = iter(train_loader)
                        iterator_batch_index = 0
                    try:
                        batch = next(data_iterator)
                        break
                    except StopIteration:
                        if iterator_batch_index == 0:
                            raise RuntimeError("training loader produced no batches")
                        loader_epoch += 1
                        data_iterator = None
            iterator_wait_ms = (perf_counter() - step_wall_started) * 1000.0

            if continuous_loader:
                logical_epoch = consumed_batches // logical_batches_per_epoch
                batch_index = consumed_batches % logical_batches_per_epoch
            else:
                logical_epoch = loader_epoch
                batch_index = iterator_batch_index
            epoch_boundary = batch_index == 0
            queue_depth = (
                prefetch_queue_depth(data_iterator) if timing_active else -1
            )
            tasks_outstanding = (
                prefetch_tasks_outstanding(data_iterator) if timing_active else -1
            )
            cuda_timer = CudaPhaseTimer(True) if timing_active else disabled_cuda_timer

            cpu_prepare_started = perf_counter() if timing_active else None
            with phase_range("batch_prepare", step_ranges_enabled):
                image_views, labels, _unused_base_views, worker_metadata = _unpack_ocl_batch(
                    batch
                )
                image_sources = _view_tensors(image_views)
                source_tensors_pinned = (
                    all(
                        source.device.type == "cpu" and source.is_pinned()
                        for source in image_sources
                    )
                    if timing_active
                    else False
                )
                cpu_concatenated_image = (
                    _concat_cpu_views(image_sources)
                    if args.cpu_concat_before_h2d
                    else None
                )
                metadata_summary = (
                    summarize_batch_metadata(worker_metadata)
                    if timing_active
                    else None
                )
                batch_samples = int(labels.shape[0])
            cpu_prepare_ms = (
                (perf_counter() - cpu_prepare_started) * 1000.0
                if cpu_prepare_started is not None
                else 0.0
            )

            if args.sync_timing and not timing_active:
                torch.cuda.synchronize(args.device)

            h2d_started = perf_counter() if timing_active else None
            cuda_timer.start("h2d")
            with phase_range("h2d", step_ranges_enabled):
                if cpu_concatenated_image is not None:
                    non_blocking_effective = cpu_concatenated_image.is_pinned()
                    image = cpu_concatenated_image.to(
                        # Reproduce the tracked legacy call exactly for A/B.
                        # The CSV separately reports whether its pageable
                        # source can make that request genuinely asynchronous.
                        args.device, non_blocking=True
                    )
                    device_views = None
                else:
                    device_views, non_blocking_effective = _h2d_views(
                        image_sources, args.device
                    )
                    source_tensors_pinned = non_blocking_effective
            cuda_timer.end("h2d")
            h2d_submit_ms = (
                (perf_counter() - h2d_started) * 1000.0
                if h2d_started is not None
                else 0.0
            )

            with phase_range("optimizer_zero_grad", step_ranges_enabled):
                optimizer.zero_grad(set_to_none=True)
            cuda_timer.start("forward")
            if device_views is not None:
                with phase_range("forward_input_concat", step_ranges_enabled):
                    image = _concat_device_views(device_views)
            with autocast(enabled=args.amp):
                loss = model(
                    image,
                    profile_events=cuda_timer.model_events,
                    ranges_enabled=step_ranges_enabled,
                )

            strict_step_sync_ms = 0.0
            legacy_scale_before = None
            if args.amp and args.legacy_amp_scale_polling:
                strict_sync_started = perf_counter()
                with phase_range("amp_scale_check", step_ranges_enabled):
                    legacy_scale_before = scaler.get_scale()
                strict_step_sync_ms += (
                    perf_counter() - strict_sync_started
                ) * 1000.0

            cuda_timer.start("backward")
            with phase_range("backward", step_ranges_enabled):
                if args.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            cuda_timer.end("backward")

            cuda_timer.start("optimizer")
            with phase_range("optimizer", step_ranges_enabled):
                if args.amp:
                    if args.grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    optimizer_step_state["called"] = False
                    scaler.step(optimizer)
                    scaler.update()
                    if args.legacy_amp_scale_polling:
                        strict_sync_started = perf_counter()
                        with phase_range("amp_scale_check", step_ranges_enabled):
                            legacy_scale_after = scaler.get_scale()
                        strict_step_sync_ms += (
                            perf_counter() - strict_sync_started
                        ) * 1000.0
                        update_succeeded = (
                            legacy_scale_after >= legacy_scale_before
                            if args.strict_step_control
                            else True
                        )
                    else:
                        update_succeeded = (
                            optimizer_step_state["called"]
                            if args.strict_step_control
                            else True
                        )
                else:
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    update_succeeded = True
                cuda_timer.end("optimizer")
            if update_succeeded and scheduler is not None:
                with phase_range("scheduler", step_ranges_enabled):
                    scheduler.step()

            if update_succeeded:
                global_step += 1
                measured_steps += 1
            step_for_triggers = global_step

            triggered_logging = bool(
                args.rank == 0
                and not args.disable_training_logging
                and (
                    not update_succeeded
                    or global_step == 1
                    or global_step % args.log_every == 0
                    or global_step == args.num_steps
                )
            )
            logging_started = perf_counter() if timing_active else None
            with phase_range("logging", step_ranges_enabled):
                legacy_loss_scalar = None
                if (
                    update_succeeded
                    and args.legacy_loss_item_each_step
                    and not args.disable_training_logging
                ):
                    legacy_loss_scalar = loss.item()
                if (
                    update_succeeded
                    and args.rank == 0
                    and not args.disable_training_logging
                ):
                    if legacy_loss_scalar is None:
                        running_loss_sum.add_(loss.detach(), alpha=batch_samples)
                    else:
                        running_loss_cpu_sum += legacy_loss_scalar * batch_samples
                    running_loss_count += batch_samples
                    if triggered_logging:
                        average_loss = (
                            running_loss_sum.item()
                            if legacy_loss_scalar is None
                            else running_loss_cpu_sum
                        ) / max(1, running_loss_count)
                        lr = optimizer.param_groups[0]["lr"]
                        memory = torch.cuda.max_memory_allocated(args.device) / (1024**2)
                        print(
                            f"Step:{global_step}/{args.num_steps}, "
                            f"Rank0LocalLoss:{average_loss:.4f}, "
                            f"lr:{lr:.8f}, IteratorWait:{iterator_wait_ms / 1000.0:.4f}s, "
                            f"MaxMem:{memory:.1f}MiB"
                        )
                elif (
                    not update_succeeded
                    and args.rank == 0
                    and not args.disable_training_logging
                ):
                    print(
                        "AMP overflow: optimizer update skipped; global_step and scheduler unchanged"
                    )
            logging_ms = (
                (perf_counter() - logging_started) * 1000.0
                if logging_started is not None
                else 0.0
            )
            step_tail_event = None
            if timing_active:
                # Start the inter-step candidate after deferred loss
                # accumulation and logging-side CUDA work.
                step_tail_event = torch.cuda.Event(enable_timing=True)
                step_tail_event.record()

            triggered_checkpoint = bool(
                update_succeeded
                and args.rank == 0
                and not args.disable_checkpoint
                and global_step % args.eval_num == 0
            )
            checkpoint_started = perf_counter() if timing_active else None
            with phase_range("checkpoint", step_ranges_enabled):
                if triggered_checkpoint:
                    if args.duplicate_checkpoint_serialization:
                        save_checkpoint(
                            logdir / "model_current_epoch.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler if args.amp else None,
                            global_step,
                            args,
                        )
                        save_checkpoint(
                            logdir / f"model_step{global_step}.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler if args.amp else None,
                            global_step,
                            args,
                        )
                    else:
                        save_checkpoint_pair(
                            logdir / "model_current_epoch.pt",
                            logdir / f"model_step{global_step}.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler if args.amp else None,
                            global_step,
                            args,
                            include_training_state=args.full_periodic_checkpoint,
                        )
            checkpoint_ms = (
                (perf_counter() - checkpoint_started) * 1000.0
                if checkpoint_started is not None
                else 0.0
            )

            # OCL has no validation hook.  Keep an explicit range/column so an
            # eventual hook cannot be mistaken for an unexplained GPU gap.
            triggered_validation = False
            validation_started = perf_counter() if timing_active else None
            if step_ranges_enabled:
                with phase_range("validation", True):
                    pass
            validation_ms = (
                (perf_counter() - validation_started) * 1000.0
                if validation_started is not None
                else 0.0
            )

            inter_step_cuda_gap_ms = 0.0
            if timing_active:
                cuda_times = cuda_timer.synchronize_and_read(args.device)
                if previous_step_tail_event is not None:
                    inter_step_cuda_gap_ms = float(
                        previous_step_tail_event.elapsed_time(
                            cuda_timer.start_event("h2d")
                        )
                    )
                previous_step_tail_event = step_tail_event
            else:
                cuda_times = {
                    "h2d_cuda_time": 0.0,
                    "forward_cuda_time": 0.0,
                    "loss_cuda_time": 0.0,
                    "backward_cuda_time": 0.0,
                    "optimizer_cuda_time": 0.0,
                }
                previous_step_tail_event = None
                if args.sync_timing:
                    torch.cuda.synchronize(args.device)

            # profiler.step() may synchronously export a completed trace. Keep
            # that host-side pause inside this step's total wall time so the
            # diagnostic cannot create an invisible inter-step GPU gap.
            if profiler is not None:
                profiler.step()
            total_wall_ms = (perf_counter() - step_wall_started) * 1000.0
            step_end_unix = wall_time() if timing_active else 0.0
            if update_succeeded:
                if args.throughput_output:
                    if (
                        throughput_started_at is None
                        and measured_steps == args.throughput_warmup_steps
                    ):
                        _throughput_boundary_sync(args)
                        throughput_start_event = torch.cuda.Event(enable_timing=True)
                        throughput_start_event.record()
                        throughput_started_wall = wall_time()
                        throughput_started_at = perf_counter()
                    if (
                        throughput_started_at is not None
                        and measured_steps > args.throughput_warmup_steps
                        and len(throughput_step_end_events)
                        < args.throughput_measure_steps
                    ):
                        throughput_samples += batch_samples
                        throughput_query_crops += int(image.shape[0])
                        throughput_views += int(image.shape[0]) * 2
                        step_end_event = torch.cuda.Event(enable_timing=True)
                        step_end_event.record()
                        throughput_step_end_events.append(step_end_event)
                        if (
                            len(throughput_step_end_events)
                            == args.throughput_measure_steps
                        ):
                            _throughput_boundary_sync(args)
                            throughput_finished_at = perf_counter()
                            throughput_finished_wall = wall_time()
            total_step_time += total_wall_ms / 1000.0
            total_data_time += iterator_wait_ms / 1000.0

            if timing_active:
                row = {
                    "attempt_index": consumed_batches + 1,
                    "global_step": step_for_triggers,
                    "step_start_unix": step_start_unix,
                    "step_end_unix": step_end_unix,
                    "loader_epoch": logical_epoch,
                    "batch_index": batch_index,
                    "epoch_boundary": epoch_boundary,
                    "iterator_wait_time": iterator_wait_ms,
                    "cpu_batch_prepare_time": cpu_prepare_ms,
                    "h2d_submit_time": h2d_submit_ms,
                    **cuda_times,
                    "logging_time": logging_ms,
                    "checkpoint_time": checkpoint_ms,
                    "validation_time": validation_ms,
                    "total_wall_time": total_wall_ms,
                    "inter_step_cuda_gap_time": inter_step_cuda_gap_ms,
                    "prefetch_queue_depth": queue_depth,
                    "prefetch_tasks_outstanding": tasks_outstanding,
                    "batch_size": batch_samples,
                    # Each query crop is masked twice and both views are
                    # encoded by OCLHead3D.
                    "effective_training_views": int(image.shape[0]) * 2,
                    "batch_tensor_shape": json.dumps(list(image.shape)),
                    "source_tensor_pinned": source_tensors_pinned,
                    "h2d_non_blocking_effective": non_blocking_effective,
                    "triggered_logging": triggered_logging,
                    "triggered_checkpoint": triggered_checkpoint,
                    "triggered_validation": triggered_validation,
                    "update_succeeded": update_succeeded,
                    "amp_overflow": not update_succeeded,
                    "gap_preceded_by_checkpoint": previous_triggered_checkpoint,
                    "gap_preceded_by_validation": previous_triggered_validation,
                    "strict_step_control": args.strict_step_control,
                    # Strict success detection now uses an optimizer post-hook;
                    # no GradScaler.get_scale() host read remains.
                    "strict_step_sync_time_ms": strict_step_sync_ms,
                    "allocated_gpu_memory_bytes": torch.cuda.memory_allocated(
                        args.device
                    ),
                    "reserved_gpu_memory_bytes": torch.cuda.memory_reserved(
                        args.device
                    ),
                    **metadata_summary,
                }
                slow, classification = recorder.record(row)
                if not recorder.active:
                    diagnostic_event = getattr(
                        train_loader, "voco_diagnostic_event", None
                    )
                    if diagnostic_event is not None:
                        diagnostic_event.clear()
                if slow and args.rank == 0:
                    detail = {
                        key: row[key]
                        for key in (
                            "attempt_index",
                            "global_step",
                            "step_start_unix",
                            "step_end_unix",
                            "loader_epoch",
                            "batch_index",
                            "epoch_boundary",
                            "iterator_wait_time",
                            "cpu_batch_prepare_time",
                            "h2d_submit_time",
                            "h2d_cuda_time",
                            "forward_cuda_time",
                            "loss_cuda_time",
                            "backward_cuda_time",
                            "optimizer_cuda_time",
                            "logging_time",
                            "checkpoint_time",
                            "validation_time",
                            "total_wall_time",
                            "inter_step_cuda_gap_time",
                            "prefetch_queue_depth",
                            "prefetch_tasks_outstanding",
                            "batch_tensor_shape",
                            "source_tensor_pinned",
                            "h2d_non_blocking_effective",
                            "triggered_logging",
                            "triggered_checkpoint",
                            "triggered_validation",
                            "triggered_cache_write",
                            "update_succeeded",
                            "amp_overflow",
                            "allocated_gpu_memory_bytes",
                            "reserved_gpu_memory_bytes",
                        )
                    }
                    detail["classification"] = classification
                    detail["sample_ids"] = metadata_summary["sample_id"]
                    detail["paths"] = metadata_summary["data_file_path"]
                    detail["cache_status"] = metadata_summary["cache_status"]
                    detail["workers"] = metadata_summary["dataloader_worker_id"]
                    detail["worker_getitem_time_ms"] = metadata_summary[
                        "worker_getitem_time_ms"
                    ]
                    slow_gap_details.append(detail)

            previous_triggered_checkpoint = triggered_checkpoint
            previous_triggered_validation = triggered_validation
            iterator_batch_index += 1
            consumed_batches += 1
    finally:
        try:
            if profiler is not None:
                profiler.stop()
        finally:
            try:
                if recorder is not None:
                    recorder.close()
            finally:
                if optimizer_step_hook is not None:
                    optimizer_step_hook.remove()

    if args.rank == 0:
        # Persist the recoverable final state before optional summaries so a
        # reporting/JSON failure cannot discard an otherwise completed run.
        if not args.disable_checkpoint and not args.disable_final_artifacts:
            save_checkpoint(
                logdir / "model_current_epoch.pt",
                model,
                optimizer,
                scheduler,
                scaler if args.amp else None,
                global_step,
                args,
            )
            save_final_artifacts(
                args.logdir,
                model,
                optimizer,
                scheduler,
                scaler if args.amp else None,
                global_step,
                args,
            )
        for detail in slow_gap_details:
            print(f"[GPU-GAP] {json.dumps(detail, ensure_ascii=False)}")
        if args.throughput_output:
            if (
                throughput_started_at is None
                or throughput_finished_at is None
                or throughput_start_event is None
                or len(throughput_step_end_events)
                != args.throughput_measure_steps
            ):
                raise RuntimeError("throughput measurement window did not complete")
            throughput_seconds = throughput_finished_at - throughput_started_at
            previous_event = throughput_start_event
            step_cycle_times_ms = []
            for step_end_event in throughput_step_end_events:
                step_cycle_times_ms.append(
                    float(previous_event.elapsed_time(step_end_event))
                )
                previous_event = step_end_event
            throughput_summary = {
                "warmup_steps": args.throughput_warmup_steps,
                "measured_steps": len(throughput_step_end_events),
                "elapsed_seconds": throughput_seconds,
                "measurement_start_unix": throughput_started_wall,
                "measurement_end_unix": throughput_finished_wall,
                "world_size": args.world_size,
                "raw_samples_per_rank": throughput_samples,
                "raw_samples_global": throughput_samples * args.world_size,
                "input_query_crops_per_rank": throughput_query_crops,
                "effective_training_views_per_rank": throughput_views,
                "samples_per_second_per_rank": throughput_samples / throughput_seconds,
                "samples_per_second_global": (
                    throughput_samples * args.world_size / throughput_seconds
                ),
                "views_per_second_per_rank": throughput_views / throughput_seconds,
                "median_step_cycle_ms": statistics.median(step_cycle_times_ms),
                "p95_step_cycle_ms": _percentile(step_cycle_times_ms, 0.95),
                "step_cycle_definition": (
                    "deferred CUDA event interval between consecutive post-step "
                    "markers; includes device idle caused by host-side gaps"
                ),
                "per_step_cuda_synchronize": bool(
                    args.diagnose_gpu_gaps or args.sync_timing
                ),
            }
            throughput_path = Path(args.throughput_output)
            throughput_path.parent.mkdir(parents=True, exist_ok=True)
            throughput_temporary = throughput_path.with_suffix(
                throughput_path.suffix + ".tmp"
            )
            throughput_temporary.write_text(
                json.dumps(throughput_summary, indent=2), encoding="utf-8"
            )
            os.replace(throughput_temporary, throughput_path)
        average_time = total_step_time / measured_steps if measured_steps else 0.0
        average_data_time = total_data_time / measured_steps if measured_steps else 0.0
        peak_memory = torch.cuda.max_memory_allocated(args.device) / (1024**2)
        checkpoint_summary = (
            str(logdir / "model_current_epoch.pt")
            if not args.disable_checkpoint and not args.disable_final_artifacts
            else "final artifacts disabled (periodic checkpoints retained)"
            if args.disable_final_artifacts and not args.disable_checkpoint
            else "disabled"
        )
        print(
            f"Training complete at global_step={global_step}; average data={average_data_time:.4f}s; "
            f"average host step wall={average_time:.4f}s; "
            f"max memory={peak_memory:.1f}MiB; checkpoint={checkpoint_summary}"
        )

    if ddp_cleanup is not None:
        ddp_cleanup()
        atexit.unregister(ddp_cleanup)


if __name__ == "__main__":
    main()
