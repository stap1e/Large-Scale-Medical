#!/usr/bin/env python3
"""Build a deterministic, patient-disjoint CT-RATE subset datalist.

The source NIfTI files are only inspected. They are never copied, renamed, or
moved.  Paths written to the datalist are POSIX paths relative to ``data_root``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import re
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("/home/bld/data/dataset/CT_RATE/train")
DEFAULT_OUTPUT_JSON = Path("/home/bld/data/dataset/CT_RATE/ct_rate_subset_256.json")
PATIENT_DIR_RE = re.compile(r"^train_(\d+)$")


class InvalidNiftiHeader(ValueError):
    """Raised when a file does not contain a readable NIfTI header."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _derive_sidecar(output_json: Path, suffix: str) -> Path:
    return output_json.with_name(f"{output_json.stem}_{suffix}")


def _lightweight_nifti_header_check(path: Path) -> None:
    """Validate core NIfTI-1 header fields without loading voxel data.

    nibabel is preferred when installed.  This small fallback keeps the subset
    inventory tool usable before the full MONAI environment is activated.
    """

    opener = gzip.open if path.name.lower().endswith(".gz") else open
    try:
        with opener(path, "rb") as stream:
            header = stream.read(352)
    except (OSError, EOFError) as exc:
        raise InvalidNiftiHeader(f"cannot decompress/read header: {exc}") from exc

    if len(header) < 348:
        raise InvalidNiftiHeader(f"header is truncated ({len(header)} bytes; expected at least 348)")

    little = struct.unpack("<i", header[0:4])[0]
    big = struct.unpack(">i", header[0:4])[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise InvalidNiftiHeader("sizeof_hdr is not 348")

    dims = struct.unpack(f"{endian}8h", header[40:56])
    ndim = dims[0]
    if ndim != 3 or any(size <= 0 for size in dims[1:4]):
        raise InvalidNiftiHeader(f"invalid dimensions: {dims}")

    datatype, bitpix = struct.unpack(f"{endian}2h", header[70:74])
    if datatype <= 0 or bitpix <= 0:
        raise InvalidNiftiHeader(f"invalid datatype/bitpix: {datatype}/{bitpix}")

    pixdim = struct.unpack(f"{endian}8f", header[76:108])
    if any(not math.isfinite(value) or value <= 0 for value in pixdim[1:4]):
        raise InvalidNiftiHeader(f"invalid spatial spacing: {pixdim[1:4]}")

    if header[344:348] not in (b"n+1\x00", b"ni1\x00"):
        raise InvalidNiftiHeader(f"invalid NIfTI magic: {header[344:348]!r}")


def validate_nifti_header(path: Path) -> str:
    """Validate a NIfTI header and return the validation backend name."""

    try:
        import nibabel as nib  # type: ignore
        import numpy as np
    except ImportError:
        _lightweight_nifti_header_check(path)
        return "builtin"

    try:
        image = nib.load(str(path), mmap=False)
        shape = tuple(int(size) for size in image.shape)
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise InvalidNiftiHeader(f"expected a non-empty 3D image, got shape {shape}")
        # Force access to the parsed header fields, but deliberately do not read
        # the multi-gigabyte voxel array.
        image.header.get_data_dtype()
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if any(not math.isfinite(value) or value <= 0 for value in spacing):
            raise InvalidNiftiHeader(f"invalid spatial spacing: {spacing}")
        affine = np.asarray(image.affine)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise InvalidNiftiHeader("affine is missing or contains NaN/Inf")
        if abs(float(np.linalg.det(affine[:3, :3]))) <= np.finfo(float).eps:
            raise InvalidNiftiHeader("affine is spatially degenerate")
        orientation = nib.aff2axcodes(affine)
        if len(orientation) != 3 or any(axis is None for axis in orientation):
            raise InvalidNiftiHeader(f"cannot determine orientation from affine: {orientation}")
    except Exception as exc:
        raise InvalidNiftiHeader(str(exc)) from exc
    return "nibabel"


def inventory_patients(data_root: Path, validate_header: bool) -> tuple[dict[str, list[Path]], list[dict[str, str]], Counter[str]]:
    valid: dict[str, list[Path]] = {}
    skipped: list[dict[str, str]] = []
    validators: Counter[str] = Counter()

    patient_dirs = sorted(
        (path for path in data_root.iterdir() if path.is_dir() and PATIENT_DIR_RE.fullmatch(path.name)),
        key=lambda path: (
            int(PATIENT_DIR_RE.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
            path.name,
        ),
    )
    patient_id_counts = Counter(
        int(PATIENT_DIR_RE.fullmatch(path.name).group(1))  # type: ignore[union-attr]
        for path in patient_dirs
    )

    for patient_dir in patient_dirs:
        patient_id = int(PATIENT_DIR_RE.fullmatch(patient_dir.name).group(1))  # type: ignore[union-attr]
        if patient_id_counts[patient_id] > 1:
            skipped.append(
                {
                    "patient": patient_dir.name,
                    "path": patient_dir.relative_to(data_root).as_posix(),
                    "reason": f"ambiguous duplicate numeric patient id: {patient_id}",
                }
            )
            continue
        candidates = sorted(path for path in patient_dir.rglob("*.nii.gz") if path.is_file())
        if not candidates:
            skipped.append(
                {
                    "patient": patient_dir.name,
                    "path": patient_dir.relative_to(data_root).as_posix(),
                    "reason": "empty patient directory (no .nii.gz files)",
                }
            )
            continue

        patient_volumes: list[Path] = []
        for volume in candidates:
            relative = volume.relative_to(data_root).as_posix()
            try:
                if volume.stat().st_size == 0:
                    raise ValueError("zero-byte file")
                if validate_header:
                    validators[validate_nifti_header(volume)] += 1
            except (OSError, ValueError, InvalidNiftiHeader) as exc:
                skipped.append({"patient": patient_dir.name, "path": relative, "reason": str(exc)})
                continue
            patient_volumes.append(volume)

        if patient_volumes:
            valid[patient_dir.name] = patient_volumes
        else:
            skipped.append(
                {
                    "patient": patient_dir.name,
                    "path": patient_dir.relative_to(data_root).as_posix(),
                    "reason": "patient has no valid NIfTI volumes",
                }
            )

    return valid, skipped, validators


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic one-volume-per-patient CT-RATE subset datalist."
    )
    parser.add_argument("--data_root", "--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_json", "--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--num_patients", "--num-patients", type=_positive_int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--selection_mode",
        "--selection-mode",
        choices=("one_volume_per_patient",),
        default="one_volume_per_patient",
    )
    header_group = parser.add_mutually_exclusive_group()
    header_group.add_argument(
        "--validate_header",
        "--validate-header",
        dest="validate_header",
        action="store_true",
        help="validate every candidate NIfTI header (default)",
    )
    header_group.add_argument(
        "--no_validate_header",
        "--no-validate-header",
        dest="validate_header",
        action="store_false",
        help="skip header validation (file existence and non-zero size are still checked)",
    )
    parser.set_defaults(validate_header=True)
    parser.add_argument(
        "--patients_output",
        "--patients-output",
        type=Path,
        default=None,
        help="selected patient list (default: <output_json stem>_patients.txt)",
    )
    parser.add_argument(
        "--skipped_output",
        "--skipped-output",
        type=Path,
        default=None,
        help="invalid/empty input report (default: <output_json stem>_skipped.json)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    patients_output = (
        args.patients_output.expanduser().resolve()
        if args.patients_output is not None
        else _derive_sidecar(output_json, "patients.txt")
    )
    skipped_output = (
        args.skipped_output.expanduser().resolve()
        if args.skipped_output is not None
        else _derive_sidecar(output_json, "skipped.json")
    )

    if not data_root.is_dir():
        raise FileNotFoundError(f"CT-RATE data root does not exist or is not a directory: {data_root}")

    outputs = (output_json, patients_output, skipped_output)
    if len(set(outputs)) != len(outputs):
        raise ValueError("output_json, patients_output, and skipped_output must be different paths")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        formatted = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing output file(s). Pass --overwrite to replace them:\n  " + formatted
        )

    valid, skipped, validators = inventory_patients(data_root, args.validate_header)
    report = {
        "data_root": data_root.as_posix(),
        "validate_header": args.validate_header,
        "header_validators": dict(validators),
        "valid_patients": len(valid),
        "requested_patients": args.num_patients,
        "selected_patients": 0,
        "skipped_records": len(skipped),
        "skipped": skipped,
    }
    if len(valid) < args.num_patients:
        reason_counts = Counter(item["reason"] for item in skipped)
        common = "; ".join(f"{reason}={count}" for reason, count in reason_counts.most_common(5))
        _write_json(skipped_output, report)
        raise RuntimeError(
            f"Requested {args.num_patients} valid patients, but only {len(valid)} were found under "
            f"{data_root}. Skipped records: {len(skipped)}"
            + (f" ({common})" if common else "")
            + f". Diagnostic report: {skipped_output}"
        )

    rng = random.Random(args.seed)
    selected_patients = rng.sample(sorted(valid), args.num_patients)
    training = []
    for patient in selected_patients:
        # One source volume per patient prevents _1/_2 reconstructions of the
        # same scan from entering an OCL batch as independent patient negatives.
        volume = rng.choice(valid[patient])
        training.append({"image": volume.relative_to(data_root).as_posix()})

    datalist = {
        "name": f"ct_rate_subset_{len(training)}",
        "description": "CT-RATE subset for pretraining pipeline validation",
        "modality": {"0": "CT"},
        "numTraining": len(training),
        "seed": args.seed,
        "selection": args.selection_mode,
        "training": training,
    }
    report["selected_patients"] = len(selected_patients)

    _write_json(output_json, datalist)
    _write_text(patients_output, "\n".join(selected_patients) + "\n")
    _write_json(skipped_output, report)

    print(f"Wrote datalist: {output_json}")
    print(f"Wrote patients: {patients_output}")
    print(f"Wrote skipped-input report: {skipped_output}")
    print(
        f"Selected {len(training)} of {len(valid)} valid patients with seed={args.seed}; "
        f"skipped records={len(skipped)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
