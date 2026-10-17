"""Project-wide constants and user-adjustable settings."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The five target classes, in the canonical YOLO index order used by this project.
CLASS_NAMES: list[str] = ["Weed", "Tomato", "Rose", "Marigold", "Cauliflower"]
WEED_CLASS = "Weed"
CROP_CLASSES: list[str] = ["Tomato", "Rose", "Marigold", "Cauliflower"]

# Colours (BGR for OpenCV, hex for Plotly) per class name; unknown classes fall back to grey.
# Validated categorical order (dataviz reference palette, dark steps) assigned in class-index order:
# passes CVD separation (>=8 ΔE) and normal-vision floor on the dark surface.
CLASS_COLORS_HEX: dict[str, str] = {
    "Weed": "#e11d48",
    "Tomato": "#d95926",
    "Rose": "#199e70",
    "Marigold": "#c98500",
    "Cauliflower": "#d55181",
    "Crop": "#16a34a",  # unspecified crop (e.g. recovered patches)
}
DEFAULT_COLOR_HEX = "#898781"

# Status palette (reserved for severity; always shown with an icon + label).
SEVERITY_COLORS: dict[str, str] = {"Low": "#0ca30c", "Moderate": "#fab219", "High": "#ec835a", "Severe": "#d03b3b"}
SEVERITY_ICONS: dict[str, str] = {"Low": "●", "Moderate": "▲", "High": "◆", "Severe": "✖"}

SETTINGS_FILE = PROJECT_ROOT / "settings.json"


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def class_color_hex(name: str) -> str:
    for k, v in CLASS_COLORS_HEX.items():
        if k.lower() == str(name).lower():
            return v
    return DEFAULT_COLOR_HEX


@dataclass
class Settings:
    """Settings editable from the dashboard's Settings page (persisted to settings.json)."""

    models_dir: str = str(PROJECT_ROOT / "models")
    runs_dir: str = str(PROJECT_ROOT / "runs")
    dataset_dir: str = str(PROJECT_ROOT / "data")
    device: str = "auto"  # auto | cpu | cuda:0 ...
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    weed_class: str = WEED_CLASS
    crop_classes: list[str] = field(default_factory=lambda: list(CROP_CLASSES))
    # Severity thresholds on estimated infestation % (union of weed boxes / image area).
    severity_low: float = 5.0
    severity_moderate: float = 15.0
    severity_high: float = 30.0
    # Crop association: maximum centre-to-edge distance, as a fraction of the image diagonal.
    proximity_frac: float = 0.15
    # Optional ground sampling distance (cm / pixel) to express density per m².
    gsd_cm_per_px: float = 0.0
    gradcam_layers: list[int] = field(default_factory=lambda: [15, 18, 21])
    # Accuracy enhancement toggles
    augment: bool = False  # Test-Time Augmentation (TTA multi-scale & horizontal flip)
    agnostic_nms: bool = False  # Class-agnostic box suppression (avoids weed/crop double-hits)
    ensemble: bool = False  # Multi-model fusion (fuses RMCBAM-tuned + Baseline-best)
    default_checkpoint: str = ""  # Explicit user-selected default checkpoint

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                s = cls(**known)
            except Exception:  # corrupt file -> defaults
                pass
        # Environment overrides (used by run_demo.*): WV_MODELS_DIR, WV_RUNS_DIR, WV_DATASET_DIR, WV_CONF
        env = {"models_dir": "WV_MODELS_DIR", "runs_dir": "WV_RUNS_DIR", "dataset_dir": "WV_DATASET_DIR"}
        for attr, var in env.items():
            if os.environ.get(var):
                setattr(s, attr, str((PROJECT_ROOT / os.environ[var]).resolve()) if not os.path.isabs(os.environ[var]) else os.environ[var])
        if os.environ.get("WV_CONF"):
            s.conf = float(os.environ["WV_CONF"])
        return s

    @property
    def demo_mode(self) -> bool:
        return os.environ.get("WV_DEMO") == "1"

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2))

    def severity(self, infestation_pct: float) -> str:
        if infestation_pct < self.severity_low:
            return "Low"
        if infestation_pct < self.severity_moderate:
            return "Moderate"
        if infestation_pct < self.severity_high:
            return "High"
        return "Severe"


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device
