"""GPU-gap diagnostics shared by the OCL training loop and data loaders.

The diagnostics are deliberately opt-in.  CUDA events are synchronized only
while ``--diagnose_gpu_gaps`` is collecting its bounded profiling window; the
normal training path never synchronizes for these timers.
"""

from __future__ import annotations

import bisect
import csv
import json
import os
import statistics
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, get_worker_info


CUDA_PHASES = ("h2d", "forward", "loss", "backward", "optimizer")
TIME_FIELDS = (
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
)

CSV_FIELDS = (
    "attempt_index",
    "global_step",
    "step_start_unix",
    "step_end_unix",
    "loader_epoch",
    "batch_index",
    "epoch_boundary",
    "timing_unit",
    *TIME_FIELDS,
    "inter_step_cuda_gap_time",
    "unattributed_wall_time",
    "slow_step",
    "slow_threshold_ms",
    "gap_classification",
    "sample_id",
    "volume_id",
    "data_file_path",
    "cache_hit",
    "cache_miss",
    "cache_status",
    "dataloader_worker_id",
    "worker_getitem_time_ms",
    "max_worker_getitem_time_ms",
    "prefetch_queue_depth",
    "prefetch_tasks_outstanding",
    "batch_size",
    "effective_training_views",
    "batch_tensor_shape",
    "source_tensor_pinned",
    "h2d_non_blocking_effective",
    "triggered_logging",
    "triggered_checkpoint",
    "triggered_validation",
    "triggered_cache_write",
    "update_succeeded",
    "amp_overflow",
    "gap_preceded_by_checkpoint",
    "gap_preceded_by_validation",
    "strict_step_control",
    "strict_step_sync_time_ms",
    "allocated_gpu_memory_bytes",
    "reserved_gpu_memory_bytes",
)


_NOOP_RANGE = nullcontext()


def phase_range(name: str, enabled: bool = True):
    """Return a zero-work context in production or an NVTX/profiler range."""

    if not enabled:
        return _NOOP_RANGE
    return _active_phase_range(name)


@contextmanager
def _active_phase_range(name: str):
    pushed = False
    with torch.autograd.profiler.record_function(name):
        if torch.cuda.is_available():
            try:
                torch.cuda.nvtx.range_push(name)
                pushed = True
            except (AttributeError, RuntimeError):
                pushed = False
        try:
            yield
        finally:
            if pushed:
                torch.cuda.nvtx.range_pop()


class CudaPhaseTimer:
    """CUDA-event timer with one synchronization after all step work is queued."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        if self.enabled:
            self._events = {
                phase: (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for phase in CUDA_PHASES
            }

    def start(self, phase: str) -> None:
        if self.enabled:
            self._events[phase][0].record()

    def end(self, phase: str) -> None:
        if self.enabled:
            self._events[phase][1].record()

    def start_event(self, phase: str) -> torch.cuda.Event | None:
        return self._events[phase][0] if self.enabled else None

    def end_event(self, phase: str) -> torch.cuda.Event | None:
        return self._events[phase][1] if self.enabled else None

    @property
    def model_events(self) -> dict[str, torch.cuda.Event] | None:
        """Events recorded inside OCLHead3D to separate encoder and loss time."""

        if not self.enabled:
            return None
        return {
            "forward_end": self._events["forward"][1],
            "loss_start": self._events["loss"][0],
            "loss_end": self._events["loss"][1],
        }

    def synchronize_and_read(self, device: torch.device) -> dict[str, float]:
        if not self.enabled:
            return {f"{phase}_cuda_time": 0.0 for phase in CUDA_PHASES}
        torch.cuda.synchronize(device)
        return {
            f"{phase}_cuda_time": float(start.elapsed_time(end))
            for phase, (start, end) in self._events.items()
        }


def _resolve_leaf_dataset(dataset: Dataset, index: int) -> tuple[Dataset, int]:
    """Resolve nested ConcatDataset/Subset objects without calling __getitem__."""

    if isinstance(dataset, ConcatDataset):
        if index < 0:
            index += len(dataset)
        dataset_index = bisect.bisect_right(dataset.cumulative_sizes, index)
        previous = 0 if dataset_index == 0 else dataset.cumulative_sizes[dataset_index - 1]
        return _resolve_leaf_dataset(dataset.datasets[dataset_index], index - previous)
    if isinstance(dataset, Subset):
        return _resolve_leaf_dataset(dataset.dataset, int(dataset.indices[index]))
    return dataset, index


def _raw_dataset_item(dataset: Dataset, index: int) -> Any:
    data = getattr(dataset, "data", None)
    if data is None:
        return None
    try:
        return data[index]
    except (IndexError, KeyError, TypeError):
        return None


def persistent_cache_path(dataset: Dataset, index: int) -> Path | None:
    """Best-effort MONAI PersistentDataset cache path resolution."""

    leaf, leaf_index = _resolve_leaf_dataset(dataset, index)
    cache_dir = getattr(leaf, "cache_dir", None)
    hash_func = getattr(leaf, "hash_func", None)
    if cache_dir is None or not callable(hash_func):
        return None
    raw_item = _raw_dataset_item(leaf, leaf_index)
    if raw_item is None:
        return None
    try:
        digest = hash_func(raw_item)
        if isinstance(digest, bytes):
            digest = digest.decode("utf-8")
        digest = str(digest) + str(getattr(leaf, "transform_hash", ""))
        return Path(cache_dir) / f"{digest}.pt"
    except (OSError, TypeError, ValueError):
        return None


def _find_first_path(value: Any) -> str:
    if isinstance(value, Mapping):
        for preferred in ("image", "volume", "path", "file"):
            candidate = value.get(preferred)
            if isinstance(candidate, (str, Path)):
                return str(candidate)
        for candidate in value.values():
            found = _find_first_path(candidate)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for candidate in value:
            found = _find_first_path(candidate)
            if found:
                return found
    return ""


def _sample_id(raw_item: Any, path: str, index: int) -> str:
    if isinstance(raw_item, Mapping):
        for key in ("sample_id", "volume_id", "patient_id", "id"):
            if key in raw_item:
                return str(raw_item[key])
    if path:
        source = Path(path)
        for parent in (source.parent, *source.parents):
            if parent.name.startswith("train_"):
                return parent.name
        name = source.name
        return name[:-7] if name.endswith(".nii.gz") else source.stem
    return str(index)


class InstrumentedDataset(Dataset):
    """Attach per-worker __getitem__, source-path, and cache metadata to a sample."""

    def __init__(self, dataset: Dataset, enabled_event: Any | None = None):
        self.dataset = dataset
        self.enabled_event = enabled_event

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.enabled_event is not None and not self.enabled_event.is_set():
            sample = self.dataset[index]
            metadata = {
                "sample_id": "",
                "volume_id": "",
                "data_file_path": "",
                "cache_hit": -1,
                "cache_miss": -1,
                "cache_status": "diagnostic_window_complete",
                "cache_write": 0,
                "worker_id": -1,
                "getitem_time_ms": 0.0,
            }
            if isinstance(sample, tuple):
                return (*sample, metadata)
            if isinstance(sample, list):
                return (*sample, metadata)
            return sample, metadata

        started = time.perf_counter()
        leaf, leaf_index = _resolve_leaf_dataset(self.dataset, index)
        raw_item = _raw_dataset_item(leaf, leaf_index)
        path = _find_first_path(raw_item)
        cache_path = persistent_cache_path(self.dataset, index)
        cache_existed_before = bool(cache_path is not None and cache_path.is_file())
        sample = self.dataset[index]
        cache_exists_after = bool(cache_path is not None and cache_path.is_file())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        worker = get_worker_info()
        metadata = {
            "sample_id": _sample_id(raw_item, path, index),
            "volume_id": _sample_id(raw_item, path, index),
            "data_file_path": path,
            "cache_hit": int(cache_existed_before),
            "cache_miss": int(cache_path is not None and not cache_existed_before),
            "cache_status": (
                "hit"
                if cache_existed_before
                else "miss_written"
                if cache_path is not None and cache_exists_after
                else "miss_read_only"
                if cache_path is not None
                else "disabled_or_unknown"
            ),
            "cache_write": int(
                cache_path is not None and not cache_existed_before and cache_exists_after
            ),
            "worker_id": worker.id if worker is not None else -1,
            "getitem_time_ms": elapsed_ms,
        }
        if isinstance(sample, tuple):
            return (*sample, metadata)
        if isinstance(sample, list):
            return (*sample, metadata)
        return sample, metadata


def _to_python_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def summarize_batch_metadata(metadata: Any) -> dict[str, Any]:
    """Turn collated worker metadata into compact CSV-safe values."""

    if not isinstance(metadata, Mapping):
        return {
            "sample_id": "",
            "volume_id": "",
            "data_file_path": "",
            "cache_hit": "",
            "cache_miss": "",
            "cache_status": "",
            "dataloader_worker_id": "",
            "worker_getitem_time_ms": "",
            "max_worker_getitem_time_ms": 0.0,
            "triggered_cache_write": False,
        }

    def encoded(key: str) -> str:
        return json.dumps(_to_python_list(metadata.get(key)), ensure_ascii=False)

    getitem_times = [float(value) for value in _to_python_list(metadata.get("getitem_time_ms"))]
    cache_writes = [int(value) for value in _to_python_list(metadata.get("cache_write"))]
    return {
        "sample_id": encoded("sample_id"),
        "volume_id": encoded("volume_id"),
        "data_file_path": encoded("data_file_path"),
        "cache_hit": encoded("cache_hit"),
        "cache_miss": encoded("cache_miss"),
        "cache_status": encoded("cache_status"),
        "dataloader_worker_id": encoded("worker_id"),
        "worker_getitem_time_ms": json.dumps(getitem_times),
        "max_worker_getitem_time_ms": max(getitem_times, default=0.0),
        "triggered_cache_write": any(cache_writes),
    }


def prefetch_queue_depth(data_iterator: Any) -> int:
    """Return the private DataLoader result-queue depth for diagnostics only."""

    queue = getattr(data_iterator, "_data_queue", None)
    if queue is None:
        return -1
    try:
        return int(queue.qsize())
    except (AttributeError, NotImplementedError, OSError):
        return -1


def prefetch_tasks_outstanding(data_iterator: Any) -> int:
    """Best-effort count of DataLoader tasks submitted but not yet consumed."""

    try:
        return int(getattr(data_iterator, "_tasks_outstanding"))
    except (AttributeError, TypeError, ValueError):
        return -1


class StepTimingRecorder:
    """Write step-level timings and classify slow GPU-gap candidates."""

    def __init__(
        self,
        path: str | Path,
        *,
        rank: int,
        max_steps: int,
        slow_step_threshold_ms: float | None,
    ):
        requested = Path(path)
        if rank:
            requested = requested.with_name(f"{requested.stem}_rank{rank}{requested.suffix}")
        requested.parent.mkdir(parents=True, exist_ok=True)
        self.path = requested
        self.max_steps = int(max_steps)
        self.explicit_threshold_ms = slow_step_threshold_ms
        self.rows_written = 0
        self.successful_rows_written = 0
        self._history = {field: deque(maxlen=100) for field in TIME_FIELDS}
        self._rows: list[dict[str, Any]] = []
        self._closed = False
        self._write_thread: threading.Thread | None = None
        self._write_error: BaseException | None = None

    @property
    def active(self) -> bool:
        return self.successful_rows_written < self.max_steps

    def _median(self, field: str) -> float:
        values = self._history[field]
        return float(statistics.median(values)) if values else 0.0

    def _is_elevated(self, row: Mapping[str, Any], field: str) -> bool:
        value = float(row.get(field, 0.0))
        baseline = self._median(field)
        if value <= 0.0:
            return False
        if baseline <= 0.0:
            return value >= 5.0
        return value > max(5.0, baseline * 1.5)

    def _classify(self, row: Mapping[str, Any], slow: bool) -> str:
        if not slow:
            return "NORMAL"
        if bool(row.get("triggered_checkpoint")) and self._is_elevated(
            row, "checkpoint_time"
        ):
            return "E_CHECKPOINT"
        if bool(row.get("triggered_validation")) and self._is_elevated(
            row, "validation_time"
        ):
            return "F_VALIDATION"
        if bool(row.get("triggered_logging")) and self._is_elevated(row, "logging_time"):
            return "D_LOGGING"
        if self._is_elevated(row, "iterator_wait_time"):
            return "A_ITERATOR_WAIT"
        if self._is_elevated(row, "cpu_batch_prepare_time"):
            return "B_CPU_BATCH_PREPARE"
        if self._is_elevated(row, "h2d_submit_time") or self._is_elevated(
            row, "h2d_cuda_time"
        ):
            return "C_H2D"
        if float(row.get("strict_step_sync_time_ms", 0.0)) >= 1.0:
            return "G_UNATTRIBUTED_CPU_OR_SYNC"

        compute_fields = (
            "forward_cuda_time",
            "loss_cuda_time",
            "backward_cuda_time",
            "optimizer_cuda_time",
        )
        if any(self._is_elevated(row, field) for field in compute_fields):
            # CUDA Event elapsed time covers the whole launch interval. A
            # long value can therefore mean either slower kernels or a Python
            # launch gap inside the instrumented phase; only a timeline can
            # distinguish them.
            return "G_CUDA_PHASE_OR_PYTHON_LAUNCH"
        return "G_UNATTRIBUTED_CPU_OR_SYNC"

    def record(self, row: dict[str, Any]) -> tuple[bool, str]:
        total_ms = float(row["total_wall_time"])
        dynamic_threshold = self._median("total_wall_time") * 1.5
        thresholds = [value for value in (dynamic_threshold, self.explicit_threshold_ms) if value]
        threshold_ms = min(thresholds) if thresholds else 0.0
        slow = bool(threshold_ms and total_ms > threshold_ms)

        accounted = sum(
            float(row.get(field, 0.0))
            for field in (
                "iterator_wait_time",
                "cpu_batch_prepare_time",
                "h2d_cuda_time",
                "forward_cuda_time",
                "loss_cuda_time",
                "backward_cuda_time",
                "optimizer_cuda_time",
                "logging_time",
                "checkpoint_time",
                "validation_time",
            )
        )
        row["unattributed_wall_time"] = max(0.0, total_ms - accounted)
        row["slow_step"] = slow
        row["slow_threshold_ms"] = threshold_ms
        classification = self._classify(row, slow)
        row["gap_classification"] = classification
        row["timing_unit"] = "ms"
        self._rows.append({field: row.get(field, "") for field in CSV_FIELDS})
        self.rows_written += 1
        if bool(row.get("update_succeeded", True)):
            self.successful_rows_written += 1
        for field in TIME_FIELDS:
            self._history[field].append(float(row.get(field, 0.0)))
        # Start persistence as soon as the bounded window is complete. The
        # writer is asynchronous so its small CSV write cannot create a
        # main-thread GPU starvation gap immediately after the window.
        if self.successful_rows_written >= self.max_steps:
            self._start_async_write()
        return slow, classification

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=CSV_FIELDS, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, self.path)
        except BaseException as exc:
            self._write_error = exc

    def _start_async_write(self) -> None:
        if self._write_thread is not None:
            return
        rows = list(self._rows)
        self._write_thread = threading.Thread(
            target=self._write_rows,
            args=(rows,),
            name="gpu-gap-csv-writer",
            daemon=False,
        )
        self._write_thread.start()

    def close(self) -> None:
        if self._closed:
            return
        self._start_async_write()
        assert self._write_thread is not None
        self._write_thread.join()
        if self._write_error is not None:
            raise RuntimeError(
                f"failed to write GPU-gap timing CSV: {self.path}"
            ) from self._write_error
        self._closed = True


def create_gap_profiler(args: Any, rank: int):
    """Create a long-window torch.profiler capture, or return None."""

    if not getattr(args, "save_gap_trace", False):
        return None
    wait_steps = int(getattr(args, "profiler_wait_steps", 20))
    warmup_steps = int(getattr(args, "profiler_warmup_steps", 10))
    active_steps = int(getattr(args, "profiler_active_steps", 200))
    trace_root = Path(args.gap_trace_dir) / f"rank{rank}"
    trace_root.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_root)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
