# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

"""VoCo pre-training entry point."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from models.voco_head import VoCoHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from utils.ops import concat_image
from utils.pretrain_common import (
    add_bool_argument,
    add_data_arguments,
    get_pretraining_loader,
    resolve_resume_path,
    restore_checkpoint,
    run_data_check_only,
    save_checkpoint,
    save_final_artifacts,
    str_to_bool,
)
from utils.utils import AverageMeter


torch.multiprocessing.set_sharing_strategy("file_system")
try:  # Linux workers benefit from a larger file-descriptor limit.
    import resource

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, hard_limit), hard_limit))
except (ImportError, OSError, ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    roi = 64
    parser = argparse.ArgumentParser(description="PyTorch VoCo Pre-Training")
    parser.add_argument("--logdir", default="logs", type=str, help="directory to save logs")
    parser.add_argument("--num_steps", "--num-steps", default=2_000_000, type=int)
    parser.add_argument("--eval_num", "--eval-num", default=20_000, type=int, help="checkpoint frequency")
    parser.add_argument("--log_every", "--log-every", default=1_000, type=int)
    add_bool_argument(
        parser,
        "sync_timing",
        default=False,
        help="synchronize CUDA for accurate smoke-test step timing",
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
    parser.add_argument("--workers", default=16, type=int)
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
    parser.add_argument("--noamp", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--dist-url", default="env://")
    add_bool_argument(parser, "cache", default=True, help="use MONAI PersistentDataset cache")
    add_data_arguments(parser)
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.pretraining_method = "voco"
    args.amp = not args.noamp
    if args.num_steps <= 0 or args.eval_num <= 0 or args.log_every <= 0:
        parser.error("num_steps, eval_num, and log_every must be positive")
    if args.workers < 0 or args.batch_size <= 0 or args.data_check_samples <= 0:
        parser.error("workers must be non-negative; batch_size and data_check_samples must be positive")

    if args.data_check_only:
        run_data_check_only(args)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; use --data_check_only for the CPU data check")

    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.device = torch.device("cuda", args.local_rank)
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
        print(f"Training VoCo in {mode}; dataset_mode={args.dataset_mode}")

    model = VoCoHead(args).to(args.device)
    if args.rank == 0:
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"Total trainable parameters: {parameters}")

    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)
    scaler = GradScaler(enabled=args.amp)
    global_step = 0
    resume_path = resolve_resume_path(args.resume, args.logdir)
    if resume_path is not None:
        global_step = restore_checkpoint(
            resume_path, model, optimizer, scheduler, scaler if args.amp else None, args.device
        )

    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )

    train_loader = get_pretraining_loader(args)
    run_loss = AverageMeter()
    total_step_time = 0.0
    total_data_time = 0.0
    measured_steps = 0
    loader_epoch = 0
    torch.cuda.reset_peak_memory_stats(args.device)

    while global_step < args.num_steps:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(loader_epoch)
        step_at_epoch_start = global_step
        model.train()
        data_wait_started = time()

        for batch in train_loader:
            data_elapsed = time() - data_wait_started
            if global_step >= args.num_steps:
                break
            if args.sync_timing:
                torch.cuda.synchronize(args.device)
            started = time()
            img, labels, crops = batch
            img, crops = concat_image(img), concat_image(crops)
            img = img.to(args.device, non_blocking=True)
            crops = crops.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                loss = model(img, crops, labels)

            if args.amp:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                update_succeeded = scaler.get_scale() >= scale_before
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                update_succeeded = True

            if not update_succeeded:
                if args.rank == 0:
                    print("AMP overflow: skipped optimizer update; global_step and scheduler unchanged")
                data_wait_started = time()
                continue
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            if args.sync_timing:
                torch.cuda.synchronize(args.device)
            elapsed = time() - started
            total_step_time += elapsed
            total_data_time += data_elapsed
            measured_steps += 1
            run_loss.update(loss.item(), n=labels.shape[0])

            if args.rank == 0 and (
                global_step == 1
                or global_step % args.log_every == 0
                or global_step == args.num_steps
            ):
                lr = optimizer.param_groups[0]["lr"]
                memory = torch.cuda.max_memory_allocated(args.device) / (1024**2)
                print(
                    f"Step:{global_step}/{args.num_steps}, Loss:{run_loss.avg:.4f}, "
                    f"lr:{lr:.8f}, Data:{data_elapsed:.4f}s, Compute:{elapsed:.4f}s, "
                    f"MaxMem:{memory:.1f}MiB"
                )

            if args.rank == 0 and global_step % args.eval_num == 0:
                current = logdir / "model_current_epoch.pt"
                save_checkpoint(current, model, optimizer, scheduler, scaler if args.amp else None, global_step, args)
                save_checkpoint(
                    logdir / f"model_step{global_step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler if args.amp else None,
                    global_step,
                    args,
                )
            if global_step >= args.num_steps:
                break
            data_wait_started = time()

        if global_step == step_at_epoch_start:
            raise RuntimeError("training loader produced no batches")
        loader_epoch += 1

    if args.rank == 0:
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
            args.logdir, model, optimizer, scheduler, scaler if args.amp else None, global_step, args
        )
        average_time = total_step_time / measured_steps if measured_steps else 0.0
        average_data_time = total_data_time / measured_steps if measured_steps else 0.0
        peak_memory = torch.cuda.max_memory_allocated(args.device) / (1024**2)
        print(
            f"Training complete at global_step={global_step}; average data={average_data_time:.4f}s; "
            f"average compute={average_time:.4f}s; "
            f"max memory={peak_memory:.1f}MiB; checkpoint={logdir / 'model_current_epoch.pt'}"
        )

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
