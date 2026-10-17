"""Detection inference, ID assignment, drawing and export helpers."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from weedvision.config import class_color_hex, hex_to_bgr


@dataclass
class Detection:
    uid: str
    cls_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(w=self.w, h=self.h, cx=self.cx, cy=self.cy, area_px=self.area)
        return d


def load_image_rgb(data: bytes | str) -> np.ndarray:
    """Read an uploaded file (bytes) or path into an RGB uint8 array, honouring EXIF orientation."""
    from PIL import ImageOps

    im = Image.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im)


def _prefix(name: str) -> str:
    return (name[:1] or "X").upper()


def assign_ids(dets: list[Detection]) -> list[Detection]:
    """Stable per-class IDs (W-001, M-002…) ordered top-to-bottom, then left-to-right."""
    out: list[Detection] = []
    by_cls: dict[str, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls_name, []).append(d)
    for name, group in by_cls.items():
        group.sort(key=lambda d: (round(d.cy / 40), d.cx))  # row bands of 40 px
        for k, d in enumerate(group, 1):
            d.uid = f"{_prefix(name)}-{k:03d}"
            out.append(d)
    out.sort(key=lambda d: (d.cls_name, d.uid))
    return out


def run_detection(
    yolo,
    image_rgb: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    max_det: int = 300,
    augment: bool = False,
    agnostic_nms: bool = False,
):
    """Run YOLO prediction with optional Test-Time Augmentation (TTA) and class-agnostic NMS.
    Returns (detections, speed_ms_dict).
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    res = yolo.predict(
        bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        augment=augment,
        agnostic_nms=agnostic_nms,
        verbose=False,
    )[0]
    names = res.names
    dets: list[Detection] = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, cf):
            dets.append(Detection("", int(c), str(names[int(c)]), float(s), *map(float, b)))
    return assign_ids(dets), dict(res.speed)


def run_ensemble_detection(
    models: list,
    image_rgb: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    max_det: int = 300,
    augment: bool = False,
    agnostic_nms: bool = False,
):
    """Run multi-model prediction and fuse bounding boxes using batched NMS for enhanced precision & recall."""
    if not models:
        return [], {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    if len(models) == 1:
        return run_detection(
            models[0],
            image_rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            augment=augment,
            agnostic_nms=agnostic_nms,
        )

    import torch
    import torchvision

    all_dets: list[Detection] = []
    combined_speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

    for m in models:
        dets, spd = run_detection(
            m,
            image_rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            augment=augment,
            agnostic_nms=agnostic_nms,
        )
        all_dets.extend(dets)
        for k, v in spd.items():
            combined_speed[k] = combined_speed.get(k, 0.0) + float(v)

    if not all_dets:
        return [], combined_speed

    boxes = torch.tensor([[d.x1, d.y1, d.x2, d.y2] for d in all_dets], dtype=torch.float32)
    scores = torch.tensor([d.conf for d in all_dets], dtype=torch.float32)
    classes = torch.tensor([d.cls_id for d in all_dets], dtype=torch.int64)

    if agnostic_nms:
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=iou)
    else:
        keep = torchvision.ops.batched_nms(boxes, scores, classes, iou_threshold=iou)

    keep_indices = keep[:max_det].cpu().numpy().tolist()
    final_dets = [all_dets[i] for i in keep_indices]
    return assign_ids(final_dets), combined_speed


def draw_detections(image_rgb: np.ndarray, dets: list[Detection], show_conf: bool = True, highlight: str | None = None) -> np.ndarray:
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = img.shape[:2]
    # Restrained scale: font scale between 0.38 and 0.65, thickness 1 or 2
    avg_dim = (H + W) / 2
    t = max(1, min(2, int(round(0.0012 * avg_dim))))
    fs = max(0.38, min(0.60, 0.00045 * avg_dim))
    
    # Pre-render subtle overlay for labels
    overlay = img.copy()
    for d in dets:
        col = hex_to_bgr(class_color_hex(d.cls_name))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, col, t + (2 if highlight and d.uid == highlight else 0), cv2.LINE_AA)
        label = f"{d.uid} {d.cls_name}" + (f" {d.conf:.2f}" if show_conf else "")
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y0 = p1[1] - th - bl - 4 if p1[1] - th - bl - 4 > 0 else p1[1] + 2
        cv2.rectangle(overlay, (p1[0], y0), (p1[0] + tw + 6, y0 + th + bl + 4), col, -1)
    
    # Blend overlay with 85% opacity for label badges
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    # Draw crisp white antialiased text on top
    for d in dets:
        p1 = (int(d.x1), int(d.y1))
        label = f"{d.uid} {d.cls_name}" + (f" {d.conf:.2f}" if show_conf else "")
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y0 = p1[1] - th - bl - 4 if p1[1] - th - bl - 4 > 0 else p1[1] + 2
        cv2.putText(img, label, (p1[0] + 3, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detections_df(dets: list[Detection], img_w: int, img_h: int) -> pd.DataFrame:
    cols = ["uid", "cls_name", "conf", "x1", "y1", "x2", "y2", "cx", "cy", "w", "h", "area_px", "area_pct"]
    if not dets:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame([d.to_dict() for d in dets])
    df["area_pct"] = 100 * df["area_px"] / float(img_w * img_h)
    return df[cols].round({"conf": 4, "x1": 1, "y1": 1, "x2": 1, "y2": 1, "cx": 1, "cy": 1, "w": 1, "h": 1, "area_px": 0, "area_pct": 4})


def to_png_bytes(image_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return buf.getvalue()


def detections_json(dets: list[Detection], image_name: str, img_w: int, img_h: int, model_name: str, params: dict) -> str:
    return json.dumps(
        {
            "image": image_name,
            "width": img_w,
            "height": img_h,
            "model": model_name,
            "params": params,
            "detections": [d.to_dict() for d in dets],
        },
        indent=2,
    )
