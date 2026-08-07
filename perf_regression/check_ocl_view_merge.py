#!/usr/bin/env python3
"""Verify separate-vs-merged OCL view encoding on the target GPU.

This deliberately supplies already-created masked views so the comparison is
not confounded by random masking.  It compares embeddings, the unchanged
current contrastive loss, and every parameter gradient.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SELF_SUPERVISED = REPOSITORY_ROOT / "Self-supervised"
sys.path.insert(0, str(SELF_SUPERVISED))

from models.ocl_head import OCLHead3D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument(
        "--output",
        default=str(REPOSITORY_ROOT / "perf_regression" / "view_merge_equivalence.json"),
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the OCL view-merge equivalence check")
    device = torch.device("cuda", 0)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    model_args = SimpleNamespace(
        in_channels=1,
        feature_size=cli.feature_size,
        spatial_dims=3,
        dropout_path_rate=0.0,
        use_checkpoint=cli.use_checkpoint,
        temperature=0.07,
        kappa=1.0 / 64.0,
        mask_drop=0.3,
        merge_ocl_views=False,
        diagnose_gpu_gaps=False,
        save_gap_trace=False,
    )
    model = OCLHead3D(model_args).to(device).train()
    x1 = torch.randn(
        cli.batch_size, 1, 64, 64, 64, device=device, requires_grad=True
    )
    x2 = torch.randn(
        cli.batch_size, 1, 64, 64, 64, device=device, requires_grad=True
    )
    atol = cli.atol if cli.atol is not None else (5e-3 if cli.amp else 1e-5)
    rtol = cli.rtol if cli.rtol is not None else (5e-3 if cli.amp else 1e-4)

    def run(merge: bool):
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.cuda.amp.autocast(enabled=cli.amp):
            e1, e2 = model.encode_views(x1, x2, merge_views=merge)
            loss = model.contrastive_loss(e1, e2)
        loss.backward()
        outputs = (e1.detach().float().cpu(), e2.detach().float().cpu())
        gradients = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        return outputs, float(loss.detach().cpu()), gradients, peak_memory_bytes

    (
        separate_outputs,
        separate_loss,
        separate_gradients,
        separate_peak_memory,
    ) = run(False)
    merged_outputs, merged_loss, merged_gradients, merged_peak_memory = run(True)

    output_max_abs = max(
        float((left - right).abs().max())
        for left, right in zip(separate_outputs, merged_outputs)
    )
    gradient_max_abs = max(
        (
            float(
                (
                    separate_gradients[name] - merged_gradients[name]
                ).abs().max()
            )
            for name in separate_gradients.keys() & merged_gradients.keys()
        ),
        default=0.0,
    )
    outputs_close = all(
        torch.allclose(left, right, atol=atol, rtol=rtol)
        for left, right in zip(separate_outputs, merged_outputs)
    )
    loss_close = abs(separate_loss - merged_loss) <= atol + rtol * abs(separate_loss)
    gradients_close = (
        separate_gradients.keys() == merged_gradients.keys()
        and all(
            torch.allclose(
                separate_gradients[name],
                merged_gradients[name],
                atol=atol,
                rtol=rtol,
            )
            for name in separate_gradients
        )
    )
    result = {
        "passed": outputs_close and loss_close and gradients_close,
        "amp": cli.amp,
        "use_checkpoint": cli.use_checkpoint,
        "batch_size": cli.batch_size,
        "feature_size": cli.feature_size,
        "atol": atol,
        "rtol": rtol,
        "separate_loss": separate_loss,
        "merged_loss": merged_loss,
        "loss_abs_difference": abs(separate_loss - merged_loss),
        "output_max_abs_difference": output_max_abs,
        "gradient_max_abs_difference": gradient_max_abs,
        "separate_peak_memory_bytes": separate_peak_memory,
        "merged_peak_memory_bytes": merged_peak_memory,
        "outputs_close": outputs_close,
        "loss_close": loss_close,
        "gradients_close": gradients_close,
    }
    destination = Path(cli.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
