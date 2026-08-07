# OCL (MiCL) pre-training head adapted to VoCo's 3D SwinUNETR backbone.
#
# This file is a NEW addition to the VoCo project. It does NOT modify any
# existing VoCo file. The backbone (Swin) and the 3D data pipeline are reused
# unchanged; only the self-supervised *algorithm logic* is replaced by OCL's
# Masked Image Contrastive Learning (MiCL), read from external/OCL (read-only).
#
# OCL's algorithm (external/OCL/models_mae.py, MiCLAutoencoderViT):
#   1. randomly mask patches of an image to generate two different views
#   2. encode both views with the SAME encoder
#   3. contrast the two view representations within a mini-batch (InfoNCE style,
#      with a temperature-scaled positive similarity transform tSP)
# No EMA teacher, no geometric-context crops (these are VoCo-specific and removed).

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.voco_head import Swin  # identical 3D SwinUNETR backbone, unchanged
from utils.ops import patch_rand_drop  # 3D block dropout, reused from VoCo utils
from utils.perf_diagnostics import phase_range


class OCLHead3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = Swin(args)  # backbone identical to VoCo
        self.temperature = getattr(args, "temperature", 0.07)
        self.kappa = getattr(args, "kappa", 1.0 / 64.0)
        self.merge_views = getattr(args, "merge_ocl_views", False)
        self.diagnostic_ranges = bool(
            getattr(args, "diagnose_gpu_gaps", False)
            or getattr(args, "save_gap_trace", False)
            or getattr(args, "emit_nvtx", False)
        )

    def compute_tSP(self, x):
        # OCL's positive-similarity transform (models_mae.py: compute_tSP)
        x = 0.5 * (1.0 + x) / (1.0 + (1.0 - x) * self.kappa)
        return x / self.temperature

    def _make_view(self, x):
        # x: [B, C, H, W, Z] -> per-sample random 3D block dropout (OCL-style masking)
        b = x.size(0)
        out = x.detach().clone()
        max_drop = getattr(self.args, "mask_drop", 0.3)
        for i in range(b):
            out[i] = patch_rand_drop(self.args, out[i], max_drop=max_drop)
        return out

    def encode_views(self, x1, x2, merge_views=None, ranges_enabled=None):
        """Encode two masked views, optionally in one larger backbone call.

        Swin uses per-sample InstanceNorm/LayerNorm rather than batch-coupled
        BatchNorm, so concatenating on batch dimension preserves the algorithm.
        The switch remains explicit so equivalence and memory can be A/B checked
        on the target GPU before it becomes a production setting.
        """

        merge = self.merge_views if merge_views is None else bool(merge_views)
        emit_ranges = (
            self.diagnostic_ranges
            if ranges_enabled is None
            else bool(ranges_enabled)
        )
        if merge:
            if emit_ranges:
                with phase_range("backbone_merged_views", True):
                    combined = self.backbone(torch.cat((x1, x2), dim=0))
            else:
                combined = self.backbone(torch.cat((x1, x2), dim=0))
            return combined.chunk(2, dim=0)
        if not emit_ranges:
            e1 = self.backbone(x1)
            e2 = self.backbone(x2)
            return e1, e2
        with phase_range("backbone_view1", True):
            e1 = self.backbone(x1)
        with phase_range("backbone_view2", True):
            e2 = self.backbone(x2)
        return e1, e2

    def contrastive_loss(self, e1, e2):
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = e2 / e2.norm(dim=-1, keepdim=True)

        # stack the two views as [2B, D] -> [B, 2, D] (mirrors OCL forward_loss)
        feats = torch.cat([e1, e2], dim=0)
        feats = feats.reshape(-1, 2, feats.shape[-1])
        m, n = feats[:, 0], feats[:, 1]

        sim_mn = self.compute_tSP(m @ n.T)
        sim_nm = self.compute_tSP(n @ m.T)

        labels = torch.arange(sim_mn.shape[0], dtype=torch.long, device=e1.device)
        loss = (F.cross_entropy(sim_mn, labels) +
                F.cross_entropy(sim_nm, labels)) / 2.0
        return loss

    def forward(self, x, profile_events=None, ranges_enabled=None):
        # x: [B, 1, H, W, Z]  (3D volume from VoCo's data pipeline)
        emit_ranges = (
            self.diagnostic_ranges
            if ranges_enabled is None
            else bool(ranges_enabled)
        )
        if emit_ranges:
            with phase_range("forward", True):
                with phase_range("view_generation", True):
                    x1 = self._make_view(x)
                    x2 = self._make_view(x)
                e1, e2 = self.encode_views(
                    x1, x2, ranges_enabled=True
                )
        else:
            x1 = self._make_view(x)
            x2 = self._make_view(x)
            e1, e2 = self.encode_views(x1, x2, ranges_enabled=False)
        if profile_events is not None:
            profile_events["forward_end"].record()
            profile_events["loss_start"].record()
        if emit_ranges:
            with phase_range("loss", True):
                loss = self.contrastive_loss(e1, e2)
        else:
            loss = self.contrastive_loss(e1, e2)
        if profile_events is not None:
            profile_events["loss_end"].record()
        return loss
