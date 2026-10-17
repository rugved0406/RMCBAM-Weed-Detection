"""Dataset annotation audit tool for WeedVision.

Inspects all field images, recovered bounding boxes, label consistency,
coordinate bounds, near-duplicate leakage, and class distributions.
Generates an audit report and builds versioned dataset labels.

Usage:
  python tools/audit_dataset.py --out-report reports/dataset_audit_report.json
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weedvision.config import CLASS_NAMES, PROJECT_ROOT
from weedvision.dataset import convert_labelme_dataset, list_images, validate_yolo_dataset


def compute_ahash(img_path: Path) -> np.ndarray:
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return np.zeros((16, 16), dtype=bool)
    small = cv2.resize(im, (16, 16), interpolation=cv2.INTER_AREA)
    return small > small.mean()


def run_audit(root_dir: Path = PROJECT_ROOT) -> dict:
    orig_dir = root_dir / "data" / "Himani mam images" / "Origanal"
    labelme_dir = root_dir / "data" / "recovered" / "labelme"
    rec_csv_path = root_dir / "data" / "recovered" / "recovered_boxes.csv"
    yolo_dir = root_dir / "data" / "yolo_weed_partial"

    orig_imgs = list_images(orig_dir) if orig_dir.exists() else []
    json_files = sorted(labelme_dir.glob("*.json")) if labelme_dir.exists() else []
    rec_df = pd.read_csv(rec_csv_path) if rec_csv_path.exists() else pd.DataFrame()

    report = {
        "summary": {
            "total_raw_images": len(orig_imgs),
            "annotated_images": len(json_files),
            "unannotated_images": len(orig_imgs) - len(json_files),
            "total_recovered_boxes": len(rec_df),
            "accepted_boxes": int(rec_df["accepted"].sum()) if not rec_df.empty else 0,
            "mean_template_match_score": float(rec_df["score"].mean()) if not rec_df.empty else 0.0,
            "min_template_match_score": float(rec_df["score"].min()) if not rec_df.empty else 0.0,
        },
        "class_distribution": {str(k): int(v) for k, v in rec_df["label"].value_counts().items()} if not rec_df.empty else {},
        "issues_found": {
            "missing_image_files": [],
            "unreadable_images": [],
            "dimension_mismatches": [],
            "out_of_bounds_boxes": [],
            "degenerate_boxes": [],
            "overlapping_boxes": [],
            "single_label_images": [],
        },
        "leakage_audit": {
            "train_val_near_duplicate_pairs": [],
            "train_val_leakage_detected": False,
        },
        "image_details": [],
    }

    # 1. Audit bounding boxes in LabelMe JSON files
    for jp in json_files:
        js = json.loads(jp.read_text(encoding="utf-8"))
        stem = jp.stem
        img_p = orig_dir / f"{stem}.jpg"
        if not img_p.exists():
            report["issues_found"]["missing_image_files"].append(str(img_p))
            continue
        im = cv2.imread(str(img_p))
        if im is None:
            report["issues_found"]["unreadable_images"].append(str(img_p))
            continue
        H, W = im.shape[:2]
        jw, jh = js.get("imageWidth", W), js.get("imageHeight", H)
        if (jw, jh) != (W, H):
            report["issues_found"]["dimension_mismatches"].append(
                f"{stem}: JSON ({jw}x{jh}) != Image ({W}x{H})"
            )

        shapes = js.get("shapes", [])
        if len(shapes) == 1:
            report["issues_found"]["single_label_images"].append(
                f"{stem}: Only 1 annotation ({shapes[0].get('label')})"
            )

        boxes = []
        for s in shapes:
            pts = s.get("points", [])
            label = s.get("label", "unknown")
            if len(pts) < 2:
                report["issues_found"]["degenerate_boxes"].append(f"{stem}: <2 points for {label}")
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            w, h = x2 - x1, y2 - y1
            if w < 5 or h < 5:
                report["issues_found"]["degenerate_boxes"].append(f"{stem}: Degenerate box {w}x{h} for {label}")
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                report["issues_found"]["out_of_bounds_boxes"].append(
                    f"{stem}: Box [{x1},{y1},{x2},{y2}] exceeds image boundaries ({W}x{H})"
                )
            boxes.append((x1, y1, x2, y2, label))

        # Check IoU overlap between boxes in the same image
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                b1, b2 = boxes[i], boxes[j]
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                if ix2 > ix1 and iy2 > iy1:
                    i_area = (ix2 - ix1) * (iy2 - iy1)
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = i_area / max(1.0, (a1 + a2 - i_area))
                    if iou > 0.05:
                        report["issues_found"]["overlapping_boxes"].append(
                            f"{stem}: {b1[4]} and {b2[4]} overlap with IoU={iou:.3f}"
                        )

        report["image_details"].append({
            "image": f"{stem}.jpg",
            "width": W,
            "height": H,
            "num_shapes": len(shapes),
            "labels": [s.get("label") for s in shapes],
        })

    # 2. Audit train vs val split leakage in yolo_weed_partial
    if yolo_dir.exists():
        t_imgs = list((yolo_dir / "images" / "train").glob("*.jpg"))
        v_imgs = list((yolo_dir / "images" / "val").glob("*.jpg"))
        t_hashes = {p.name: compute_ahash(p) for p in t_imgs}
        v_hashes = {p.name: compute_ahash(p) for p in v_imgs}
        for vn, vh in v_hashes.items():
            for tn, th in t_hashes.items():
                diff = int(np.count_nonzero(vh != th))
                if diff <= 15:  # Hamming distance <= 15 bits out of 256
                    report["leakage_audit"]["train_val_near_duplicate_pairs"].append(
                        f"VAL {vn} <-> TRAIN {tn} (diff = {diff}/256 bits)"
                    )
        report["leakage_audit"]["train_val_leakage_detected"] = len(
            report["leakage_audit"]["train_val_near_duplicate_pairs"]
        ) > 0

    return report


def build_versioned_dataset(
    version_name: str = "yolo_weed_v2",
    val_frac: float = 0.2,
    seed: int = 42,
    root_dir: Path = PROJECT_ROOT,
):
    """Generate a clean, verified versioned dataset with validated manifests."""
    orig_dir = root_dir / "data" / "Himani mam images" / "Origanal"
    labelme_dir = root_dir / "data" / "recovered" / "labelme"
    out_dir = root_dir / "data" / version_name

    rep = convert_labelme_dataset(
        images_dir=orig_dir,
        out_dir=out_dir,
        json_dir=labelme_dir,
        class_names=CLASS_NAMES,
        val_frac=val_frac,
        seed=seed,
        include_unlabeled=False,
    )
    val_info = validate_yolo_dataset(out_dir / "data.yaml")
    return rep, val_info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-report", default="reports/dataset_audit_report.json")
    ap.add_argument("--build-v2", action="store_true", help="Build data/yolo_weed_v2 clean versioned dataset")
    args = ap.parse_args()

    print("=== Running Complete WeedVision Dataset Audit ===")
    report = run_audit(PROJECT_ROOT)

    out_p = Path(args.out_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ok] Audit report saved to {out_p}")

    s = report["summary"]
    print(f"Total Raw Images: {s['total_raw_images']}")
    print(f"Annotated Images: {s['annotated_images']}")
    print(f"Unannotated Field Images: {s['unannotated_images']}")
    print(f"Recovered Boxes: {s['total_recovered_boxes']} (Accepted: {s['accepted_boxes']}, Mean Match: {s['mean_template_match_score']:.4f})")
    print(f"Train/Val Data Leakage: {'DETECTED' if report['leakage_audit']['train_val_leakage_detected'] else 'CLEAN (No Leakage)'}")
    print(f"Single-label images: {len(report['issues_found']['single_label_images'])}")

    if args.build_v2:
        print("\n=== Generating Versioned Clean Dataset (data/yolo_weed_v2) ===")
        rep, val_info = build_versioned_dataset("yolo_weed_v2", val_frac=0.2, seed=42)
        print(f"[ok] Created data/yolo_weed_v2: {rep.train_images} train / {rep.val_images} val images.")
        print(f"[ok] YOLO Validation: {len(val_info['issues'])} issues, splits: {val_info['splits']}")


if __name__ == "__main__":
    main()
