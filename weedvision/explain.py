"""Gradient-based explanations for YOLOv8 detections + RMCBAM attention-map extraction.

Grad-CAM for a detector: the score being explained is the (sigmoid) class score of the
target class, summed over the anchor predictions whose decoded boxes overlap the
detection being explained (IoU >= ``iou_match``). Gradients of that score w.r.t. the
activations of chosen layers give channel weights (Grad-CAM) or element-wise
weights (HiResCAM, the default: it is faithful to the
model and localises far better than Grad-CAM on dense detectors, where averaging gradients over
the whole feature map smears the evidence). Everything is computed from the model's real forward/backward pass.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from weedvision.inference import Detection
from weedvision.rmcbam import find_rmcbam, set_attention_capture


@dataclass
class Letterbox:
    tensor: torch.Tensor  # 1,3,S,S float 0..1
    ratio: float
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    size: int


def letterbox(image_rgb: np.ndarray, size: int, device: str) -> Letterbox:
    """Same geometry as Ultralytics LetterBox(auto=False, center=True), grey (114) padding."""
    h, w = image_rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    img = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else image_rgb
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    bottom, right = size - nh - top, size - nw - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div(255).unsqueeze(0).to(device)
    return Letterbox(t, r, left, top, nw, nh, size)


def _box_to_lb(d: Detection, lb: Letterbox) -> torch.Tensor:
    return torch.tensor(
        [d.x1 * lb.ratio + lb.pad_x, d.y1 * lb.ratio + lb.pad_y, d.x2 * lb.ratio + lb.pad_x, d.y2 * lb.ratio + lb.pad_y]
    )


def _xywh2xyxy(b: torch.Tensor) -> torch.Tensor:
    x, y, w, h = b.unbind(-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1)


def _iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


class DetectorCAM:
    """Grad-CAM / HiResCAM for Ultralytics detection models (works for baseline and RMCBAM)."""

    def __init__(self, nn_model: torch.nn.Module, layers: list[int], device: str = "cpu"):
        self.device = device
        self.model = copy.deepcopy(nn_model).float().to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        n = len(self.model.model)
        self.layers = [i for i in layers if 0 <= i < n - 1]
        if not self.layers:
            raise ValueError(f"No valid target layers in {layers} (model has {n} layers).")
        self.nc = len(self.model.names)
        self._acts: dict[int, torch.Tensor] = {}
        self._hooks = [self.model.model[i].register_forward_hook(self._hook(i)) for i in self.layers]

    def _hook(self, i):
        def fn(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                out.retain_grad()
                self._acts[i] = out

        return fn

    def close(self):
        for h in self._hooks:
            h.remove()

    def _forward(self, lb: Letterbox):
        self._acts.clear()
        x = lb.tensor.clone().requires_grad_(True)
        out = self.model(x)
        y = out[0] if isinstance(out, (tuple, list)) else out  # 1, 4+nc, N  (xywh in letterbox px, sigmoid scores)
        if y.shape[1] != 4 + self.nc:
            raise RuntimeError(f"Unexpected head output shape {tuple(y.shape)} for nc={self.nc}.")
        return y[0]

    def _cam_from_grads(self, lb: Letterbox, method: str) -> np.ndarray:
        maps = []
        for i in self.layers:
            a = self._acts.get(i)
            if a is None or a.grad is None:
                continue
            g = a.grad[0]
            act = a[0].detach()
            if method == "hirescam":
                cam = (g * act).sum(0)
            else:
                cam = (g.mean(dim=(1, 2), keepdim=True) * act).sum(0)
            cam = F.relu(cam)
            cam = F.interpolate(cam[None, None], size=(lb.size, lb.size), mode="bilinear", align_corners=False)[0, 0]
            if cam.max() > 0:
                cam = cam / cam.max()
            maps.append(cam)
        if not maps:
            return np.zeros((lb.size, lb.size), np.float32)
        cam = torch.stack(maps).mean(0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()

    @staticmethod
    def _unletterbox(cam: np.ndarray, lb: Letterbox, out_w: int, out_h: int) -> np.ndarray:
        crop = cam[lb.pad_y : lb.pad_y + lb.new_h, lb.pad_x : lb.pad_x + lb.new_w]
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    def explain(
        self,
        image_rgb: np.ndarray,
        targets: list[Detection],
        imgsz: int = 640,
        method: str = "hirescam",
        iou_match: float = 0.5,
        per_target: bool = True,
    ) -> dict:
        """Return {'combined': HxW map, 'per_target': {uid: HxW map}, 'anchors': {uid: n_matched}}.

        ``combined`` explains the sum of all targets' scores; ``per_target`` explains each detection alone.
        """
        H, W = image_rgb.shape[:2]
        lb = letterbox(image_rgb, imgsz, self.device)
        result = {"combined": None, "per_target": {}, "anchors": {}, "method": method, "layers": self.layers}
        if not targets:
            return result

        def score_for(y: torch.Tensor, d: Detection) -> tuple[torch.Tensor, int]:
            boxes = _xywh2xyxy(y[:4].T)
            ious = _iou(_box_to_lb(d, lb).to(boxes.device), boxes)
            sel = (ious >= iou_match).nonzero().flatten()
            if sel.numel() == 0:
                sel = ious.argmax().view(1)
            return y[4 + d.cls_id, sel].sum(), int(sel.numel())

        # combined map in one backward pass
        y = self._forward(lb)
        total = 0
        for d in targets:
            s, n = score_for(y, d)
            total = total + s
            result["anchors"][d.uid] = n
        total.backward()
        result["combined"] = self._unletterbox(self._cam_from_grads(lb, method), lb, W, H)

        if per_target:
            for d in targets:
                y = self._forward(lb)
                s, _ = score_for(y, d)
                s.backward()
                result["per_target"][d.uid] = self._unletterbox(self._cam_from_grads(lb, method), lb, W, H)
        return result

    def explain_class(self, image_rgb: np.ndarray, cls_id: int, imgsz: int = 640, topk: int = 100, method: str = "hirescam"):
        """Class-evidence map when nothing was detected: explains the sum of the top-k class scores."""
        H, W = image_rgb.shape[:2]
        lb = letterbox(image_rgb, imgsz, self.device)
        y = self._forward(lb)
        s = y[4 + cls_id].topk(min(topk, y.shape[1])).values.sum()
        s.backward()
        return self._unletterbox(self._cam_from_grads(lb, method), lb, W, H)


def rmcbam_attention_maps(nn_model: torch.nn.Module, image_rgb: np.ndarray, imgsz: int = 640, device: str = "cpu") -> list[dict]:
    """Run one forward pass and collect the real spatial/channel attention of every RMCBAM block.

    Returns [] if the model has no RMCBAM modules.
    """
    model = copy.deepcopy(nn_model).float().to(device).eval()
    mods = find_rmcbam(model)
    if not mods:
        return []
    H, W = image_rgb.shape[:2]
    lb = letterbox(image_rgb, imgsz, device)
    set_attention_capture(model, True)
    with torch.no_grad():
        model(lb.tensor)
    out = []
    for name, m in mods:
        sa = m.last_spatial_att[0, 0]
        sa_up = F.interpolate(sa[None, None], size=(lb.size, lb.size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        out.append(
            {
                "name": name,
                "layer": int(name.split(".")[1]),
                "feat_hw": tuple(sa.shape),
                "spatial": DetectorCAM._unletterbox(sa_up, lb, W, H),
                "spatial_min": float(sa.min()),
                "spatial_max": float(sa.max()),
                "channel": m.last_channel_att[0, :, 0, 0].cpu().numpy(),
                "gamma_mean_abs": float(m.gamma.detach().abs().mean()),
                "gamma": m.gamma.detach()[0, :, 0, 0].cpu().numpy(),
            }
        )
    set_attention_capture(model, False)
    return out


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45, normalize: bool = True) -> np.ndarray:
    c = cam.astype(np.float32)
    if normalize:
        lo, hi = float(c.min()), float(c.max())
        c = (c - lo) / (hi - lo) if hi > lo else np.zeros_like(c)
    heat = cv2.applyColorMap((np.clip(c, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (image_rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)


def energy_inside_box(cam: np.ndarray, d: Detection) -> float:
    """Fraction of total CAM energy that falls inside the detection box (a simple localisation sanity score)."""
    tot = float(cam.sum())
    if tot <= 0:
        return 0.0
    x1, y1, x2, y2 = (int(max(0, v)) for v in (d.x1, d.y1, d.x2, d.y2))
    return float(cam[y1:y2, x1:x2].sum()) / tot
