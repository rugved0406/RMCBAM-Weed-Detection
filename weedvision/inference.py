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


def suppress_duplicate_boxes(dets: list[Detection], iou_thresh: float = 0.50, containment_thresh: float = 0.80) -> list[Detection]:
    """Suppress redundant, heavily overlapping, or contained detections."""
    if len(dets) <= 1:
        return dets
    sorted_dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep: list[Detection] = []
    for d in sorted_dets:
        discard = False
        for k in keep:
            ix1 = max(d.x1, k.x1)
            iy1 = max(d.y1, k.y1)
            ix2 = min(d.x2, k.x2)
            iy2 = min(d.y2, k.y2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                d_area = max(1.0, d.area)
                k_area = max(1.0, k.area)
                union_area = d_area + k_area - inter_area
                iou = inter_area / max(1.0, union_area)
                containment = inter_area / d_area
                if iou >= iou_thresh or (containment >= containment_thresh and (d.cls_id == k.cls_id or k.conf > d.conf + 0.2)):
                    discard = True
                    break
        if not discard:
            keep.append(d)
    return keep


def run_detection(yolo, image_rgb: np.ndarray, conf: float, iou: float, imgsz: int, device: str, max_det: int = 300):
    """Run YOLO prediction with non-maximum suppression and duplicate filtering. Returns (detections, speed_ms_dict)."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    res = yolo.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, device=device, max_det=max_det, verbose=False)[0]
    names = res.names
    dets: list[Detection] = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, cf):
            dets.append(Detection("", int(c), str(names[int(c)]), float(s), *map(float, b)))
    
    # Suppress redundant duplicate sub-boxes
    filtered_dets = suppress_duplicate_boxes(dets, iou_thresh=iou)
    return assign_ids(filtered_dets), dict(res.speed)


def _rects_overlap(r1: tuple[int, int, int, int], r2: tuple[int, int, int, int], margin: int = 1) -> bool:
    """Check if two rectangles [x1, y1, x2, y2] intersect."""
    return not (r1[2] + margin < r2[0] or r1[0] - margin > r2[2] or r1[3] + margin < r2[1] or r1[1] - margin > r2[3])


def draw_detections(
    image_rgb: np.ndarray,
    dets: list[Detection],
    show_conf: bool = True,
    highlight: str | None = None,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_ids: bool = False,
    max_detections: int | None = None,
    conf_threshold: float = 0.0,
    compact: bool = True,
) -> np.ndarray:
    """Draw ultra-clean, thin precision bounding boxes with sleek translucent micro-badges."""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = img.shape[:2]

    # Filter detections
    filtered = [d for d in dets if d.conf >= conf_threshold]
    filtered.sort(key=lambda d: d.conf, reverse=True)
    if max_detections is not None and max_detections > 0:
        filtered = filtered[:max_detections]

    if not filtered:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    diag = (H + W) / 2
    stroke = max(1, min(2, int(round(0.0011 * diag))))
    fs = float(np.clip(0.00036 * diag, 0.30, 0.42))
    txt_thickness = 1
    pad_x, pad_y = 3, 2

    # 1. Draw Clean Thin Bounding Boxes
    if show_boxes:
        for d in filtered:
            col = hex_to_bgr(class_color_hex(d.cls_name))
            p1 = (int(np.clip(d.x1, 0, W - 1)), int(np.clip(d.y1, 0, H - 1)))
            p2 = (int(np.clip(d.x2, 0, W - 1)), int(np.clip(d.y2, 0, H - 1)))
            is_hl = highlight and d.uid == highlight
            s_width = stroke + (1 if is_hl else 0)
            cv2.rectangle(img, p1, p2, col, s_width, cv2.LINE_AA)

            # Subtle modern corner tick
            c_len = max(3, min(8, int(min(d.w, d.h) * 0.12)))
            if c_len > 3:
                cv2.line(img, p1, (p1[0] + c_len, p1[1]), col, s_width + 1, cv2.LINE_AA)
                cv2.line(img, p1, (p1[0], p1[1] + c_len), col, s_width + 1, cv2.LINE_AA)
                cv2.line(img, p2, (p2[0] - c_len, p2[1]), col, s_width + 1, cv2.LINE_AA)
                cv2.line(img, p2, (p2[0], p2[1] - c_len), col, s_width + 1, cv2.LINE_AA)

    # 2. Draw Translucent Non-Overlapping Micro-Badges
    if show_labels:
        overlay = img.copy()
        occupied_label_rects: list[tuple[int, int, int, int]] = []
        is_dense = len(filtered) > 12

        for d in filtered:
            is_hl = highlight and d.uid == highlight
            col = (0, 215, 255) if is_hl else hex_to_bgr(class_color_hex(d.cls_name))

            
            # Format text compactly e.g. "Weed 87%" or "W-001 Weed 87%"
            parts = []
            if show_ids and d.uid and not is_dense:
                parts.append(d.uid)
            name_str = d.cls_name if not (is_dense and compact) else (d.cls_name[:1] if d.cls_name == "Weed" else d.cls_name)
            if show_conf:
                parts.append(f"{name_str} {int(round(d.conf * 100))}%")
            else:
                parts.append(name_str)
            label_text = " ".join(parts)
            # Ensure label_text contains only standard ASCII to prevent OpenCV '??' rendering artifacts
            label_text = label_text.replace("·", " ").replace("•", " ").replace("—", "-")
            label_text = "".join(c for c in label_text if 32 <= ord(c) <= 126)

            (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fs, txt_thickness)
            bw = tw + 2 * pad_x + 4
            bh = th + bl + 2 * pad_y

            bx1, by1, bx2, by2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)

            # Potential placement slots
            candidates = [
                # Top-left inside box (cleanest, doesn't spill outside)
                (bx1 + 1, by1 + 1, bx1 + bw + 1, by1 + bh + 1, False),
                # Top-left above box
                (bx1, by1 - bh - 1, bx1 + bw, by1 - 1, False),
                # Bottom-left inside box
                (bx1 + 1, by2 - bh - 1, bx1 + bw + 1, by2 - 1, False),
                # Top-right inside box
                (max(0, bx2 - bw - 1), by1 + 1, max(bw + 1, bx2 - 1), by1 + bh + 1, False),
                # Bottom-left below box
                (bx1, by2 + 1, bx1 + bw, by2 + bh + 1, False),
                # Displaced above
                (bx1, max(0, by1 - bh - 8), bx1 + bw, max(bh, by1 - 8), True),
            ]

            chosen = None
            needs_leader = False
            for cand_x1, cand_y1, cand_x2, cand_y2, leader in candidates:
                if cand_x1 < 0 or cand_y1 < 0 or cand_x2 > W or cand_y2 > H:
                    continue
                cand_rect = (cand_x1, cand_y1, cand_x2, cand_y2)
                if not any(_rects_overlap(cand_rect, occ, margin=1) for occ in occupied_label_rects):
                    chosen = cand_rect
                    needs_leader = leader
                    break

            if chosen is None:
                cx1 = np.clip(bx1, 0, max(0, W - bw))
                cy1 = np.clip(by1 + 1, 0, max(0, H - bh))
                chosen = (int(cx1), int(cy1), int(cx1 + bw), int(cy1 + bh))

            occupied_label_rects.append(chosen)

            # Leader line if displaced
            if needs_leader and show_boxes:
                badge_anchor = ((chosen[0] + chosen[2]) // 2, (chosen[1] + chosen[3]) // 2)
                box_anchor = (int(np.clip((d.x1 + d.x2) / 2, 0, W)), int(np.clip(d.y1, 0, H)))
                cv2.line(img, box_anchor, badge_anchor, col, 1, cv2.LINE_AA)

            lx1, ly1, lx2, ly2 = chosen
            # Semi-transparent dark background
            cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (18, 22, 19), -1)
            cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), col, 1 if not is_hl else 2, cv2.LINE_AA)

            # Indicator dot
            cv2.rectangle(overlay, (lx1, ly1), (lx1 + 3, ly2), col, -1)

            # Text
            tx = lx1 + 5
            ty = ly1 + th + pad_y - 1
            cv2.putText(overlay, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (250, 252, 250), txt_thickness, cv2.LINE_AA)


        # Blend overlay (alpha = 0.88) to preserve underlying crop image visibility
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

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

