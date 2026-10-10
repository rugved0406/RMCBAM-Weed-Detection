"""Train baseline YOLOv8s and/or YOLOv8s + RMCBAM on the same dataset with identical settings.

Usage:
  python tools/train.py --data data/yolo/data.yaml --variant both --epochs 100 --imgsz 640 --batch 16
  python tools/train.py --data data/yolo/data.yaml --variant rmcbam --layers 4 6 8 15 18 21

Runs are written to runs/detect/<name>; copy weights/best.pt into models/ (or point the dashboard
Settings at runs/) to use them. RMCBAM checkpoints require the `weedvision` package to be importable
when loading (the dashboard and these tools handle that).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO  # noqa: E402

from weedvision.config import PROJECT_ROOT  # noqa: E402
from weedvision.rmcbam import DEFAULT_RMCBAM_LAYERS, make_rmcbam_trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", choices=["baseline", "rmcbam", "both"], default="both")
    ap.add_argument("--weights", default="yolov8s.pt", help="Starting weights (COCO-pretrained YOLOv8s by default)")
    ap.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_RMCBAM_LAYERS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", default="auto", choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"], help="Optimizer")
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (e.g. 0.01 for SGD, 0.001 for AdamW)")
    ap.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction (lr0 * lrf)")
    ap.add_argument("--cos-lr", action="store_true", help="Use cosine learning rate schedule")
    ap.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs")
    ap.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic augmentation for the final N epochs")
    ap.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "detect"))
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--nbs", type=int, default=16, help="Nominal batch size for gradient accumulation (set = --batch on tiny datasets)")
    ap.add_argument("--suffix", default="", help="Suffix for run names, e.g. _SMOKE")
    a = ap.parse_args()

    device = a.device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    common = dict(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device, workers=a.workers,
        seed=a.seed, deterministic=True, project=str(Path(a.project).resolve()), exist_ok=True, patience=a.patience, plots=True, nbs=a.nbs,
        optimizer=a.optimizer, lr0=a.lr0, lrf=a.lrf, cos_lr=a.cos_lr, warmup_epochs=a.warmup_epochs, close_mosaic=a.close_mosaic,
    )
    if a.variant in ("baseline", "both"):
        print("=== Training baseline YOLOv8s ===")
        YOLO(a.weights).train(name="yolov8s_baseline" + a.suffix, **common)
    if a.variant in ("rmcbam", "both"):
        print(f"=== Training YOLOv8s + RMCBAM (layers {a.layers}) ===")
        YOLO(a.weights).train(name="yolov8s_rmcbam" + a.suffix, trainer=make_rmcbam_trainer(a.layers), **common)


if __name__ == "__main__":
    main()
