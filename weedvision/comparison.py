"""Model comparison from real artefacts: Ultralytics run folders (results.csv + plots) or live validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METRIC_COLS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}

PLOT_FILES = {
    "Confusion matrix": ["confusion_matrix.png"],
    "Confusion matrix (normalised)": ["confusion_matrix_normalized.png"],
    "PR curve": ["BoxPR_curve.png", "PR_curve.png"],
    "F1 curve": ["BoxF1_curve.png", "F1_curve.png"],
    "Precision curve": ["BoxP_curve.png", "P_curve.png"],
    "Recall curve": ["BoxR_curve.png", "R_curve.png"],
    "Training curves": ["results.png"],
}


def find_runs(*roots: str | Path) -> list[Path]:
    """Ultralytics run directories (contain results.csv or weights/)."""
    runs = set()
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for p in r.rglob("results.csv"):
            runs.add(p.parent)
        for p in r.rglob("weights"):
            if p.is_dir():
                runs.add(p.parent)
    return sorted(runs)


def load_results(run_dir: str | Path) -> pd.DataFrame | None:
    p = Path(run_dir) / "results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def best_epoch_metrics(df: pd.DataFrame) -> dict:
    """Metrics at the epoch Ultralytics would pick as best (fitness = 0.1*mAP50 + 0.9*mAP50-95)."""
    if df is None or df.empty or METRIC_COLS["mAP50-95"] not in df:
        return {}
    fit = 0.1 * df[METRIC_COLS["mAP50"]] + 0.9 * df[METRIC_COLS["mAP50-95"]]
    i = int(fit.idxmax())
    row = df.loc[i]
    out = {k: float(row[v]) for k, v in METRIC_COLS.items() if v in df}
    p, r = out.get("precision", 0), out.get("recall", 0)
    out["F1"] = 2 * p * r / (p + r) if (p + r) else 0.0
    out["best_epoch"] = int(row["epoch"]) if "epoch" in df else i
    out["epochs_logged"] = int(len(df))
    return out


def run_plots(run_dir: str | Path) -> dict[str, Path]:
    out = {}
    for title, names in PLOT_FILES.items():
        for n in names:
            p = Path(run_dir) / n
            if p.exists():
                out[title] = p
                break
    return out


def validate(yolo, data_yaml: str, imgsz: int, device: str, split: str = "val", conf: float = 0.001, iou: float = 0.6, batch: int = 8, project: str | None = None) -> dict:
    """Run Ultralytics validation and return metrics, per-class AP, curves and confusion matrix (all real)."""
    m = yolo.val(data=data_yaml, imgsz=imgsz, device=device, split=split, conf=conf, iou=iou, batch=batch, plots=True, verbose=False, project=project, exist_ok=True)
    box = m.box
    names = m.names if hasattr(m, "names") else yolo.names
    res = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "speed_ms": dict(getattr(m, "speed", {}) or {}),
        "save_dir": str(getattr(m, "save_dir", "")),
        "per_class": [],
        "curves": {},
        "confusion_matrix": None,
        "names": names,
    }
    p, r = res["precision"], res["recall"]
    res["F1"] = 2 * p * r / (p + r) if (p + r) else 0.0
    try:
        for j, ci in enumerate(box.ap_class_index):
            res["per_class"].append(
                {
                    "class": names[int(ci)],
                    "precision": float(box.p[j]),
                    "recall": float(box.r[j]),
                    "AP50": float(box.ap50[j]),
                    "AP50-95": float(box.ap[j]),
                }
            )
    except Exception:
        pass
    try:
        for key, (x, y, xl, yl) in zip(m.curves, m.curves_results):
            res["curves"][key] = {"x": np.asarray(x).tolist(), "y": np.asarray(y).tolist(), "xlabel": xl, "ylabel": yl}
    except Exception:
        pass
    try:
        res["confusion_matrix"] = np.asarray(m.confusion_matrix.matrix).tolist()
    except Exception:
        pass
    return res
