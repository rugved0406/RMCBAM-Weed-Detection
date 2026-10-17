"""Evaluation and Benchmarking tool for WeedVision experiments.

Evaluates any YOLO checkpoint on validation and independent test splits,
calculating mAP@50, mAP@50-95, Precision, Recall, F1-Score, and inference latency.

Usage:
  python tools/evaluate_experiments.py --model models/weed_detection_best.pt --data data/yolo_weed_v3/data.yaml --split val
  python tools/evaluate_experiments.py --model models/weed_detection_best.pt --data data/yolo_weed_v3/data.yaml --split test
"""

import argparse
import json
import time
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
import weedvision.rmcbam  # noqa: F401


def evaluate_checkpoint(model_path: str, data_yaml: str, split: str = "val", imgsz: int = 640, device: str = None) -> dict:
    if device is None:
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = YOLO(model_path)
    
    # Measure latency on dummy batch of same resolution
    dummy_img = (np.random.default_rng(42).random((imgsz, imgsz, 3)) * 255).astype(np.uint8)
    # Warmup
    for _ in range(3):
        model.predict(dummy_img, imgsz=imgsz, device=device, verbose=False)
    
    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        model.predict(dummy_img, imgsz=imgsz, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    
    mean_latency = float(np.mean(times))
    fps = float(1000.0 / mean_latency) if mean_latency > 0 else 0.0

    # Run formal Ultralytics validation
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, verbose=False)
    
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    
    # Compute F1
    if precision + recall > 0:
        f1 = float(2 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    results = {
        "model_path": str(model_path),
        "split": split,
        "imgsz": imgsz,
        "device": str(device),
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": mean_latency,
        "fps": fps,
    }
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--data", default="data/yolo_weed_v3/data.yaml", help="Dataset yaml path")
    ap.add_argument("--split", choices=["val", "test", "train"], default="val", help="Split to evaluate on")
    ap.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    ap.add_argument("--device", default=None, help="Device to use (mps, cuda, cpu)")
    ap.add_argument("--json", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    print(f"=== Evaluating {args.model} on '{args.split}' split (imgsz={args.imgsz}) ===")
    res = evaluate_checkpoint(args.model, args.data, split=args.split, imgsz=args.imgsz, device=args.device)

    print(f"Split:       {res['split']}")
    print(f"mAP@50:      {res['mAP50']*100:.2f}%")
    print(f"mAP@50-95:   {res['mAP50_95']*100:.2f}%")
    print(f"Precision:   {res['precision']*100:.2f}%")
    print(f"Recall:      {res['recall']*100:.2f}%")
    print(f"F1-Score:    {res['f1']:.4f}")
    print(f"Latency:     {res['latency_ms']:.2f} ms ({res['fps']:.1f} FPS)")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2))
        print(f"Saved results to {p}")


if __name__ == "__main__":
    main()
