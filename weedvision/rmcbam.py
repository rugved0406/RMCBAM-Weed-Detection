"""Residual Multi-Scale CBAM (RMCBAM) and its integration into YOLOv8 C2f blocks.

RMCBAM(x):
    1. ECA-style channel attention   Mc = sigmoid(Conv1d_k(GAP(x)) + Conv1d_k(GMP(x)))      (no FC bottleneck)
    2. Multi-scale spatial attention  on [mean_c, max_c](x * Mc) with three parallel branches:
         3x3 conv, 7x7 conv, 5x5 conv with dilation 2 (receptive field 9x9),
       fused by a 1x1 conv -> Ms = sigmoid(...)
    3. Residual attention gating      out = x + gamma ⊙ (x * Mc * Ms)
       gamma is a learnable per-channel gate initialised to 0, so a freshly inserted
       RMCBAM is an exact identity and pretrained YOLOv8s weights are preserved at the
       start of fine-tuning (ReZero-style).

Integration: selected ``C2f`` modules of a YOLOv8 ``DetectionModel`` are converted
in place to ``C2f_RMCBAM`` (a C2f subclass that applies RMCBAM to its output). Because
the original C2f sub-module names are kept (``cv1``, ``cv2``, ``m``) and the attention is
added under ``att``, baseline YOLOv8s weights load directly into the modified model.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f

# Default insertion points for YOLOv8 (n/s/m/l/x share the same layer indices):
#   backbone stages P3 (4), P4 (6), P5 (8) and the three neck outputs feeding Detect (15, 18, 21).
DEFAULT_RMCBAM_LAYERS: tuple[int, ...] = (4, 6, 8, 15, 18, 21)


def eca_kernel_size(channels: int, gamma: int = 2, b: int = 1) -> int:
    """Adaptive 1-D kernel size from the ECA-Net paper (always odd, >= 3)."""
    t = int(abs((math.log2(channels) + b) / gamma))
    k = t if t % 2 else t + 1
    return max(3, k)


class ECAChannelAttention(nn.Module):
    """Efficient channel attention using both average- and max-pooled descriptors."""

    def __init__(self, channels: int, k: int | None = None):
        super().__init__()
        k = k or eca_kernel_size(channels)
        self.k = k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(2, 3)).unsqueeze(1)  # B,1,C
        mx = x.amax(dim=(2, 3)).unsqueeze(1)  # B,1,C
        a = torch.sigmoid(self.conv(avg) + self.conv(mx))  # B,1,C
        return a.transpose(1, 2).unsqueeze(-1)  # B,C,1,1


class MultiScaleSpatialAttention(nn.Module):
    """Spatial attention from 3x3, 7x7 and dilated 5x5 branches over [mean, max] channel maps."""

    def __init__(self):
        super().__init__()

        def branch(k: int, d: int = 1) -> nn.Sequential:
            pad = d * (k - 1) // 2
            return nn.Sequential(nn.Conv2d(2, 1, k, padding=pad, dilation=d, bias=False), nn.BatchNorm2d(1))

        self.b3 = branch(3)
        self.b7 = branch(7)
        self.b5d = branch(5, d=2)
        self.fuse = nn.Conv2d(3, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)  # B,2,H,W
        m = torch.cat([self.b3(s), self.b7(s), self.b5d(s)], dim=1)  # B,3,H,W
        return torch.sigmoid(self.fuse(m))  # B,1,H,W


class RMCBAM(nn.Module):
    """Residual Multi-Scale CBAM block (channel-preserving)."""

    def __init__(self, channels: int, eca_k: int | None = None):
        super().__init__()
        self.channels = channels
        self.ca = ECAChannelAttention(channels, eca_k)
        self.sa = MultiScaleSpatialAttention()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # Attention capture for explainability (off by default; no effect on training).
        self.store_attention = False
        self.last_channel_att: torch.Tensor | None = None
        self.last_spatial_att: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc = self.ca(x)
        xc = x * mc
        ms = self.sa(xc)
        if self.store_attention:
            self.last_channel_att = mc.detach()
            self.last_spatial_att = ms.detach()
        return x + self.gamma * (xc * ms)


class C2f_RMCBAM(C2f):  # noqa: N801  (name mirrors Ultralytics module naming)
    """C2f block followed by RMCBAM on its output. Keeps C2f parameter names for weight compatibility."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.att = RMCBAM(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.att(super().forward(x))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        return self.att(super().forward_split(x))


def c2f_out_channels(m: C2f) -> int:
    return m.cv2.conv.out_channels


def inject_rmcbam(model: nn.Module, layers: Iterable[int] = DEFAULT_RMCBAM_LAYERS) -> list[int]:
    """Convert the C2f modules at ``layers`` of a DetectionModel into C2f_RMCBAM, in place.

    Returns the list of layer indices actually converted. Raises ValueError if an index is
    not a plain C2f module.
    """
    seq = model.model
    done: list[int] = []
    for i in layers:
        m = seq[i]
        if isinstance(m, C2f_RMCBAM):
            done.append(i)
            continue
        if type(m) is not C2f:
            raise ValueError(f"Layer {i} is {type(m).__name__}, not C2f; cannot insert RMCBAM there.")
        p = next(m.parameters())
        m.__class__ = C2f_RMCBAM
        m.add_module("att", RMCBAM(c2f_out_channels(m)).to(device=p.device, dtype=p.dtype))
        m.type = f"{C2f_RMCBAM.__module__}.{C2f_RMCBAM.__name__}"
        m.np = sum(x.numel() for x in m.parameters())
        done.append(i)
    if isinstance(getattr(model, "yaml", None), dict):
        model.yaml["rmcbam_layers"] = sorted(done)
    return sorted(done)


def find_rmcbam(model: nn.Module) -> list[tuple[str, RMCBAM]]:
    """All RMCBAM modules in a model as (qualified_name, module)."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, RMCBAM)]


def set_attention_capture(model: nn.Module, enabled: bool) -> None:
    for _, m in find_rmcbam(model):
        m.store_attention = enabled
        if not enabled:
            m.last_channel_att = m.last_spatial_att = None


def build_rmcbam_model(
    cfg: str | dict = "yolov8s.yaml",
    nc: int = 5,
    weights: str | nn.Module | None = None,
    layers: Iterable[int] = DEFAULT_RMCBAM_LAYERS,
    verbose: bool = False,
):
    """Build a YOLOv8 DetectionModel with RMCBAM inserted, optionally loading (baseline or RMCBAM) weights."""
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint

    model = DetectionModel(cfg, nc=nc, verbose=verbose)
    if weights is None:
        inject_rmcbam(model, layers)
        return model
    if isinstance(weights, (str,)):
        weights, _ = load_checkpoint(weights)
    if find_rmcbam(weights):  # RMCBAM -> RMCBAM: same structure first, then load
        inject_rmcbam(model, layers)
        model.load(weights, verbose=verbose)
    else:  # baseline -> RMCBAM: load C2f weights, then add zero-gated attention
        model.load(weights, verbose=verbose)
        inject_rmcbam(model, layers)
    return model


def make_rmcbam_trainer(layers: Iterable[int] = DEFAULT_RMCBAM_LAYERS):
    """Return a DetectionTrainer subclass whose models carry RMCBAM at ``layers``.

    Usage: ``YOLO("yolov8s.pt").train(data=..., trainer=make_rmcbam_trainer())``
    """
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import RANK

    layer_list = tuple(layers)

    class RMCBAMTrainer(DetectionTrainer):
        rmcbam_layers = layer_list

        def get_model(self, cfg=None, weights=None, verbose=True):
            if isinstance(cfg, dict) and cfg.get("rmcbam_layers"):
                use_layers = cfg["rmcbam_layers"]
            else:
                use_layers = self.rmcbam_layers
            model = build_rmcbam_model(
                cfg, nc=self.data["nc"], weights=weights, layers=use_layers, verbose=verbose and RANK == -1
            )
            if hasattr(self, "set_model_names_for_load"):
                model = self.set_model_names_for_load(model)
            return model

    return RMCBAMTrainer
