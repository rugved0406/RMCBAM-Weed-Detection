"""Weed infestation metrics and crop-wise weed association."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from weedvision.config import Settings
from weedvision.inference import Detection


def split_weeds_crops(dets: list[Detection], s: Settings) -> tuple[list[Detection], list[Detection]]:
    weed = s.weed_class.lower()
    crops = {c.lower() for c in s.crop_classes}
    return [d for d in dets if d.cls_name.lower() == weed], [d for d in dets if d.cls_name.lower() in crops]


def union_area(dets: list[Detection], w: int, h: int) -> int:
    """Pixel area covered by the union of boxes (overlaps counted once)."""
    if not dets:
        return 0
    mask = np.zeros((h, w), dtype=bool)
    for d in dets:
        x1, y1 = max(0, int(math.floor(d.x1))), max(0, int(math.floor(d.y1)))
        x2, y2 = min(w, int(math.ceil(d.x2))), min(h, int(math.ceil(d.y2)))
        mask[y1:y2, x1:x2] = True
    return int(mask.sum())


def weed_metrics(weeds: list[Detection], w: int, h: int, s: Settings) -> dict:
    img_area = float(w * h)
    u = union_area(weeds, w, h)
    infest = 100.0 * u / img_area
    confs = np.array([d.conf for d in weeds]) if weeds else np.array([])
    areas = np.array([d.area for d in weeds]) if weeds else np.array([])
    m = {
        "weed_count": len(weeds),
        "image_px": int(img_area),
        "density_per_mpx": len(weeds) / (img_area / 1e6),
        "sum_box_area_px": float(areas.sum()) if len(areas) else 0.0,
        "union_box_area_px": u,
        "infestation_pct": infest,
        "mean_box_area_px": float(areas.mean()) if len(areas) else 0.0,
        "median_box_area_px": float(np.median(areas)) if len(areas) else 0.0,
        "mean_conf": float(confs.mean()) if len(confs) else None,
        "min_conf": float(confs.min()) if len(confs) else None,
        "max_conf": float(confs.max()) if len(confs) else None,
        "severity": s.severity(infest),
        "density_per_m2": None,
        "ground_area_m2": None,
    }
    if s.gsd_cm_per_px and s.gsd_cm_per_px > 0:
        ground_m2 = img_area * (s.gsd_cm_per_px / 100.0) ** 2
        m["ground_area_m2"] = ground_m2
        m["density_per_m2"] = len(weeds) / ground_m2 if ground_m2 else None
    return m


def _dist_point_to_box(px: float, py: float, d: Detection) -> float:
    dx = max(d.x1 - px, 0.0, px - d.x2)
    dy = max(d.y1 - py, 0.0, py - d.y2)
    return math.hypot(dx, dy)


def crop_association(weeds: list[Detection], crops: list[Detection], w: int, h: int, s: Settings):
    """Associate each weed with the nearest crop detection (weed centre → crop box edge distance).

    Returns (per_weed_df, per_crop_class_df). Both are empty if there are no crop detections.
    """
    max_d = s.proximity_frac * math.hypot(w, h)
    rows = []
    for wd in weeds:
        best, bd = None, float("inf")
        for c in crops:
            dist = _dist_point_to_box(wd.cx, wd.cy, c)
            if dist < bd:
                best, bd = c, dist
        assoc = best is not None and bd <= max_d
        rows.append(
            {
                "weed_id": wd.uid,
                "nearest_crop_id": best.uid if best else None,
                "nearest_crop_class": best.cls_name if best else None,
                "distance_px": round(bd, 1) if best else None,
                "inside_crop_box": bool(best and bd == 0),
                "associated": bool(assoc),
                "associated_class": best.cls_name if assoc else "Unassociated",
            }
        )
    per_weed = pd.DataFrame(rows)
    if not crops:
        return per_weed, pd.DataFrame()
    summ = []
    for cls in s.crop_classes:
        inst = [c for c in crops if c.cls_name.lower() == cls.lower()]
        n_w = int((per_weed["associated_class"].str.lower() == cls.lower()).sum()) if len(per_weed) else 0
        dists = per_weed.loc[per_weed["associated_class"].str.lower() == cls.lower(), "distance_px"] if len(per_weed) else []
        summ.append(
            {
                "crop_class": cls,
                "crop_instances": len(inst),
                "associated_weeds": n_w,
                "weeds_per_crop_instance": (n_w / len(inst)) if inst else None,
                "mean_distance_px": float(np.mean(dists)) if len(dists) else None,
            }
        )
    return per_weed, pd.DataFrame(summ)
