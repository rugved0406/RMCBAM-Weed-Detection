"""Checkpoint discovery, inspection, compatibility checks and model statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import weedvision.rmcbam as rm  # noqa: F401  (must be importable before unpickling RMCBAM checkpoints)
from weedvision.config import CLASS_NAMES, WEED_CLASS


@dataclass
class CheckpointInfo:
    path: str
    ok: bool
    error: str = ""
    size_mb: float = 0.0
    names: dict[int, str] = field(default_factory=dict)
    nc: int = 0
    is_rmcbam: bool = False
    rmcbam_layers: list[int] = field(default_factory=list)
    rmcbam_gate_mean_abs: dict[str, float] = field(default_factory=dict)
    n_params: int = 0
    gflops: float | None = None
    epoch: int | None = None
    best_fitness: float | None = None
    date: str | None = None
    ultralytics_version: str | None = None
    train_args: dict = field(default_factory=dict)
    arch: str = ""

    @property
    def label(self) -> str:
        p = Path(self.path)
        if "uploaded" in str(p).lower():
            return f"uploaded/{p.name}"
        return f"{p.parent.parent.name}/{p.name}" if "weights" in str(p) else p.name

    @property
    def class_list(self) -> list[str]:
        return [self.names[k] for k in sorted(self.names)]

    @property
    def has_weed_class(self) -> bool:
        return any(n.lower() == WEED_CLASS.lower() for n in self.names.values())

    @property
    def missing_project_classes(self) -> list[str]:
        have = {n.lower() for n in self.names.values()}
        return [c for c in CLASS_NAMES if c.lower() not in have]

    @property
    def is_smoke(self) -> bool:
        name = str(self.train_args.get("name", "")).lower()
        epochs = self.train_args.get("epochs", None)
        return (
            "smoke" in self.path.lower()
            or "smoke" in name
            or (isinstance(epochs, (int, float)) and epochs <= 8)
        )

    @property
    def is_project_model(self) -> bool:
        return self.ok and self.has_weed_class

    def compatibility_notes(self) -> list[tuple[str, str]]:
        """(level, message) pairs; level in {ok, warn, error}."""
        if not self.ok:
            return [("error", f"Could not load checkpoint: {self.error}")]
        notes: list[tuple[str, str]] = []
        if self.is_smoke:
            notes.append(("warn", "PIPELINE SMOKE-TEST checkpoint: trained for a few epochs on 56 partially-labelled images "
                          "(Weed only). Its detections, heatmaps and metrics are meaningless and exist only to demonstrate the dashboard."))
        if self.has_weed_class:
            notes.append(("ok", "Checkpoint contains a 'Weed' class."))
        else:
            notes.append(
                (
                    "warn",
                    "No 'Weed' class in this checkpoint — outputs are NOT weed detections "
                    f"(classes: {', '.join(self.class_list[:8])}{'…' if self.nc > 8 else ''}).",
                )
            )
        miss = self.missing_project_classes
        if self.has_weed_class and miss:
            notes.append(("warn", f"Project classes absent from checkpoint: {', '.join(miss)}."))
        if self.is_rmcbam:
            notes.append(("ok", f"RMCBAM modules found at layers {self.rmcbam_layers}."))
        else:
            notes.append(("ok", "Plain YOLO architecture (no RMCBAM modules)."))
        return notes


def discover_checkpoints(*dirs: str | Path) -> list[Path]:
    """All *.pt files directly in model dirs, subdirectories (e.g. uploaded/), and Ultralytics run weights."""
    found: list[Path] = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        found += sorted(d.rglob("*.pt"))
        found += sorted(d.glob("**/weights/*.pt"))
    uniq: dict[str, Path] = {}
    for p in found:
        uniq[str(p.resolve())] = p

    def _rank(p: Path):
        name = p.name.lower()
        full = str(p).lower()
        is_smoke = "smoke" in full or name in ("yolov8s_baseline_best.pt", "yolov8s_rmcbam_best.pt")
        is_tuned = "tuned" in name or "weed_detection_best" in name
        is_uploaded = "uploaded" in full
        if is_uploaded and not is_smoke:
            tier = 0
        elif is_tuned and not is_smoke:
            tier = 1
        elif not is_smoke:
            tier = 2
        else:
            tier = 3
        return (tier, p.name != "best.pt", str(p))

    return sorted(uniq.values(), key=_rank)


def _load_raw(path: str | Path) -> tuple[dict, torch.nn.Module]:
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, _ = torch_safe_load(str(path))
    model = ckpt.get("ema") or ckpt.get("model")
    if model is None:
        raise ValueError("Checkpoint has no 'model' or 'ema' entry (not an Ultralytics checkpoint).")
    return ckpt, model


def model_gflops(model: torch.nn.Module, imgsz: int = 640) -> float | None:
    try:
        from ultralytics.utils.torch_utils import get_flops

        g = get_flops(model, imgsz)
        return float(g) if g else None
    except Exception:
        return None


def inspect_checkpoint(path: str | Path, compute_flops: bool = True) -> CheckpointInfo:
    path = Path(path)
    info = CheckpointInfo(path=str(path), ok=False, size_mb=round(path.stat().st_size / 1e6, 2) if path.exists() else 0)
    try:
        ckpt, model = _load_raw(path)
        model = model.float()
        names = getattr(model, "names", None) or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        info.names = {int(k): str(v) for k, v in names.items()}
        info.nc = len(info.names)
        mods = rm.find_rmcbam(model)
        info.is_rmcbam = bool(mods)
        info.rmcbam_layers = sorted({int(n.split(".")[1]) for n, _ in mods if n.startswith("model.")})
        info.rmcbam_gate_mean_abs = {n: float(m.gamma.detach().abs().mean()) for n, m in mods}
        info.n_params = int(sum(p.numel() for p in model.parameters()))
        info.gflops = model_gflops(model) if compute_flops else None
        info.epoch = ckpt.get("epoch")
        bf = ckpt.get("best_fitness")
        info.best_fitness = float(bf) if bf is not None else None
        info.date = ckpt.get("date")
        info.ultralytics_version = ckpt.get("version")
        ta = ckpt.get("train_args") or {}
        keep = ("data", "epochs", "imgsz", "batch", "optimizer", "lr0", "model", "name", "pretrained", "device", "seed")
        info.train_args = {k: ta.get(k) for k in keep if k in ta}
        y = getattr(model, "yaml", {}) or {}
        info.arch = str(y.get("yaml_file", "")) or type(model).__name__
        info.ok = True
    except Exception as e:  # report, never crash the UI
        info.error = f"{type(e).__name__}: {e}"
    return info


def load_yolo(path: str | Path):
    """Load an Ultralytics YOLO wrapper (RMCBAM classes are registered via this module's import)."""
    from ultralytics import YOLO

    return YOLO(str(path))


def benchmark_latency(yolo, imgsz: int, device: str, runs: int = 10, warmup: int = 2) -> dict:
    """Measured end-to-end predict latency on a random image (ms). Real timing, this machine only."""
    img = (np.random.default_rng(0).random((imgsz, imgsz, 3)) * 255).astype(np.uint8)
    for _ in range(warmup):
        yolo.predict(img, imgsz=imgsz, device=device, verbose=False)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        r = yolo.predict(img, imgsz=imgsz, device=device, verbose=False)
        times.append((time.perf_counter() - t) * 1000)
    sp = r[0].speed
    return {
        "device": device,
        "imgsz": imgsz,
        "runs": runs,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "fps": float(1000 / np.mean(times)),
        "last_preprocess_ms": sp.get("preprocess"),
        "last_inference_ms": sp.get("inference"),
        "last_postprocess_ms": sp.get("postprocess"),
    }


def architecture_table(model: torch.nn.Module) -> list[dict]:
    """Per-layer table of a DetectionModel (index, from, type, params)."""
    rows = []
    for m in getattr(model, "model", []):
        rows.append(
            {
                "layer": getattr(m, "i", None),
                "from": str(getattr(m, "f", "")),
                "module": str(getattr(m, "type", type(m).__name__)).replace("ultralytics.nn.modules.", ""),
                "params": int(sum(p.numel() for p in m.parameters())),
                "rmcbam": isinstance(m, rm.C2f_RMCBAM),
            }
        )
    return rows
