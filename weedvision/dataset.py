"""Dataset inspection, LabelMe → YOLO conversion, label validation, and patch-box recovery.

Original files are never modified: conversion copies images and writes labels into a
separate output directory.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from weedvision.config import CLASS_NAMES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Accepted spellings for each canonical class (case-insensitive).
CLASS_ALIASES: dict[str, list[str]] = {
    "Weed": ["weed", "weeds", "weed_plant", "grass_weed", "broadleaf_weed"],
    "Tomato": ["tomato", "tomatoes", "tomato_plant", "tamatar"],
    "Rose": ["rose", "roses", "rose_plant", "gulab"],
    "Marigold": ["marigold", "marigolds", "genda", "gende", "marigold_plant", "tagetes"],
    "Cauliflower": ["cauliflower", "cauliflowers", "cauli", "gobhi", "phool_gobhi", "cauliflower_plant"],
}


def normalise_label(label: str, class_names: list[str] = CLASS_NAMES) -> str | None:
    key = re.sub(r"[\s\-]+", "_", label.strip().lower())
    for canon in class_names:
        if key == canon.lower() or key in CLASS_ALIASES.get(canon, []):
            return canon
    return None


def _walk(d: Path):
    import os

    for dp, _dn, fn in os.walk(d, followlinks=True):
        for f in fn:
            yield Path(dp) / f


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in _walk(Path(d)) if p.suffix.lower() in IMG_EXTS)


# ----------------------------------------------------------------------------------------------
# LabelMe conversion
# ----------------------------------------------------------------------------------------------
@dataclass
class ConversionReport:
    images_total: int = 0
    images_with_json: int = 0
    images_without_json: list[str] = field(default_factory=list)
    json_without_image: list[str] = field(default_factory=list)
    instances_per_class: dict[str, int] = field(default_factory=dict)
    unknown_labels: dict[str, int] = field(default_factory=dict)
    skipped_shapes: dict[str, int] = field(default_factory=dict)
    degenerate_boxes: int = 0
    clipped_boxes: int = 0
    size_mismatch: list[str] = field(default_factory=list)
    missing_classes: list[str] = field(default_factory=list)
    train_images: int = 0
    val_images: int = 0
    out_dir: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def shape_to_xyxy(shape: dict) -> tuple[float, float, float, float] | None:
    pts = np.asarray(shape.get("points") or [], dtype=float)
    st = (shape.get("shape_type") or "polygon").lower()
    if pts.size == 0:
        return None
    if st == "circle" and len(pts) >= 2:
        (cx, cy), (ex, ey) = pts[0], pts[1]
        r = float(np.hypot(ex - cx, ey - cy))
        return cx - r, cy - r, cx + r, cy + r
    if st in {"rectangle", "polygon", "linestrip", "mask"} or len(pts) >= 2:
        if st in {"point", "line"}:
            return None
        return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())
    return None


def labelme_to_yolo_lines(js: dict, img_w: int, img_h: int, class_names: list[str], rep: ConversionReport) -> list[str]:
    lines = []
    for sh in js.get("shapes", []):
        st = (sh.get("shape_type") or "polygon").lower()
        canon = normalise_label(str(sh.get("label", "")), class_names)
        if canon is None:
            lab = str(sh.get("label", ""))
            rep.unknown_labels[lab] = rep.unknown_labels.get(lab, 0) + 1
            continue
        if st in {"point", "line"}:
            rep.skipped_shapes[st] = rep.skipped_shapes.get(st, 0) + 1
            continue
        b = shape_to_xyxy(sh)
        if b is None:
            rep.skipped_shapes[st] = rep.skipped_shapes.get(st, 0) + 1
            continue
        x1, y1, x2, y2 = b
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            rep.clipped_boxes += 1
        if cx2 - cx1 < 1 or cy2 - cy1 < 1:
            rep.degenerate_boxes += 1
            continue
        cid = class_names.index(canon)
        xc, yc = (cx1 + cx2) / 2 / img_w, (cy1 + cy2) / 2 / img_h
        bw, bh = (cx2 - cx1) / img_w, (cy2 - cy1) / img_h
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        rep.instances_per_class[canon] = rep.instances_per_class.get(canon, 0) + 1
    return lines


def find_json_for_image(img: Path, json_dirs: list[Path]) -> Path | None:
    for d in [img.parent, *json_dirs]:
        p = d / (img.stem + ".json")
        if p.exists():
            return p
    return None


def convert_labelme_dataset(
    images_dir: str | Path,
    out_dir: str | Path,
    json_dir: str | Path | None = None,
    class_names: list[str] = CLASS_NAMES,
    val_frac: float = 0.2,
    seed: int = 42,
    include_unlabeled: bool = False,
) -> ConversionReport:
    """Convert a LabelMe-annotated folder into an Ultralytics YOLO dataset with train/val split + data.yaml."""
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    if out_dir.resolve() == images_dir.resolve():
        raise ValueError("Output directory must differ from the source directory (originals are preserved).")
    json_dirs = [Path(json_dir)] if json_dir else []
    rep = ConversionReport(out_dir=str(out_dir))
    imgs = list_images(images_dir)
    imgs = [p for p in imgs if out_dir.resolve() not in p.resolve().parents]
    rep.images_total = len(imgs)
    items: list[tuple[Path, list[str]]] = []
    seen_json = set()
    for img in imgs:
        jp = find_json_for_image(img, json_dirs)
        if jp is None:
            rep.images_without_json.append(str(img.relative_to(images_dir)))
            if include_unlabeled:
                items.append((img, []))
            continue
        seen_json.add(jp.resolve())
        js = json.loads(jp.read_text(encoding="utf-8"))
        im = cv2.imread(str(img))
        if im is None:
            rep.size_mismatch.append(f"{img.name}: unreadable image")
            continue
        h, w = im.shape[:2]
        jw, jh = js.get("imageWidth"), js.get("imageHeight")
        if jw and jh and (int(jw), int(jh)) != (w, h):
            rep.size_mismatch.append(f"{img.name}: json {jw}x{jh} vs image {w}x{h}")
            # LabelMe coords refer to the annotated size; rescale to actual image.
            sx, sy = w / float(jw), h / float(jh)
            for sh in js.get("shapes", []):
                sh["points"] = [[x * sx, y * sy] for x, y in sh.get("points", [])]
        rep.images_with_json += 1
        items.append((img, labelme_to_yolo_lines(js, w, h, class_names, rep)))
    for d in {images_dir, *json_dirs}:
        for jp in d.rglob("*.json"):
            if jp.resolve() in seen_json:
                continue
            try:
                js = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(js, dict) and "shapes" in js:
                rep.json_without_image.append(str(jp))
    rep.missing_classes = [c for c in class_names if rep.instances_per_class.get(c, 0) == 0]

    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = int(round(len(items) * val_frac)) if len(items) > 1 else 0
    splits = {"val": items[:n_val], "train": items[n_val:]}
    for split, its in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lines in its:
            dst = out_dir / "images" / split / img.name
            shutil.copy2(img, dst)
            (out_dir / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    rep.train_images, rep.val_images = len(splits["train"]), len(splits["val"])
    write_data_yaml(out_dir, class_names)
    (out_dir / "conversion_report.json").write_text(json.dumps(rep.to_dict(), indent=2))
    return rep


def write_data_yaml(root: Path, class_names: list[str]) -> Path:
    p = Path(root) / "data.yaml"
    # No "path" key: Ultralytics then resolves train/val relative to this yaml's folder (portable).
    data = {
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {i: n for i, n in enumerate(class_names)},
    }
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------
# YOLO label validation
# ----------------------------------------------------------------------------------------------
def validate_yolo_dataset(data_yaml: str | Path) -> dict:
    """Validate a YOLO dataset described by data.yaml. Returns a dict with per-split stats and issues."""
    data_yaml = Path(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    names = cfg.get("names")
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names or [])
    nc = int(cfg.get("nc", len(names)))
    out = {"root": str(root), "names": names, "nc": nc, "splits": {}, "issues": [], "instances": Counter(), "boxes": []}
    if nc != len(names):
        out["issues"].append(f"nc={nc} but {len(names)} names listed")
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if not rel:
            continue
        rels = rel if isinstance(rel, list) else [rel]
        imgs: list[Path] = []
        for r in rels:
            d = (root / r) if not Path(r).is_absolute() else Path(r)
            if d.is_dir():
                imgs += list_images(d)
            elif d.suffix == ".txt" and d.exists():
                imgs += [Path(line.strip()) for line in d.read_text().splitlines() if line.strip()]
            else:
                out["issues"].append(f"{split}: path not found: {d}")
        st = {"images": len(imgs), "labelled": 0, "empty_label": 0, "missing_label": 0, "instances": 0}
        for img in imgs:
            lp = img.with_suffix(".txt")
            parts = list(img.parts)
            if "images" in parts:
                i = len(parts) - 1 - parts[::-1].index("images")
                parts[i] = "labels"
                lp = Path(*parts).with_suffix(".txt")
            if not lp.exists():
                st["missing_label"] += 1
                continue
            lines = [line for line in lp.read_text().splitlines() if line.strip()]
            if not lines:
                st["empty_label"] += 1
                continue
            st["labelled"] += 1
            seen = set()
            for ln, line in enumerate(lines, 1):
                v = line.split()
                if len(v) != 5:
                    out["issues"].append(f"{lp.name}:{ln} expected 5 values, got {len(v)}")
                    continue
                try:
                    c = int(float(v[0]))
                    x, y, w, h = map(float, v[1:])
                except ValueError:
                    out["issues"].append(f"{lp.name}:{ln} non-numeric values")
                    continue
                if not 0 <= c < nc:
                    out["issues"].append(f"{lp.name}:{ln} class id {c} out of range 0..{nc - 1}")
                    continue
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    out["issues"].append(f"{lp.name}:{ln} coordinates outside [0,1] or zero size")
                if tuple(v) in seen:
                    out["issues"].append(f"{lp.name}:{ln} duplicate box")
                seen.add(tuple(v))
                out["instances"][names[c] if c < len(names) else str(c)] += 1
                out["boxes"].append({"split": split, "class": names[c] if c < len(names) else str(c), "w": w, "h": h, "xc": x, "yc": y})
                st["instances"] += 1
        out["splits"][split] = st
    out["missing_classes"] = [n for n in names if out["instances"].get(n, 0) == 0]
    out["instances"] = dict(out["instances"])
    return out


# ----------------------------------------------------------------------------------------------
# Folder scan + image quality statistics
# ----------------------------------------------------------------------------------------------
def scan_folder(root: str | Path) -> dict:
    """Inventory of a dataset folder: images, LabelMe JSONs, YOLO txt labels, data.yaml files."""
    root = Path(root)
    if not root.exists():
        return {"exists": False, "root": str(root)}
    imgs = list_images(root)
    files = list(_walk(root))
    jsons = sorted(p for p in files if p.suffix.lower() == ".json")
    labelme = []
    for j in jsons:
        try:
            js = json.loads(j.read_text(encoding="utf-8"))
            if isinstance(js, dict) and "shapes" in js:
                labelme.append(j)
        except Exception:
            pass
    txts = [t for t in files if t.suffix == ".txt" and "labels" in t.parts]
    yamls = sorted(p for p in files if p.suffix.lower() in (".yaml", ".yml"))
    by_folder = Counter(str(p.parent.relative_to(root)) for p in imgs)
    return {
        "exists": True,
        "root": str(root),
        "images": [str(p) for p in imgs],
        "n_images": len(imgs),
        "labelme_json": [str(p) for p in labelme],
        "n_labelme_json": len(labelme),
        "n_yolo_txt": len(txts),
        "yaml_files": sorted({str(p) for p in yamls}),
        "images_by_folder": dict(by_folder),
    }


def ahash(gray_small: np.ndarray) -> int:
    g = cv2.resize(gray_small, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (g > g.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def image_quality(path: str | Path) -> dict:
    from PIL import Image

    p = Path(path)
    try:
        with Image.open(p) as im:
            w, h = im.size  # header only, no full decode
    except Exception:
        return {"path": str(p), "file": p.name, "folder": p.parent.name, "readable": False}
    flag = cv2.IMREAD_REDUCED_COLOR_4 if max(w, h) >= 2048 else (cv2.IMREAD_REDUCED_COLOR_2 if max(w, h) >= 1024 else cv2.IMREAD_COLOR)
    img = cv2.imread(str(p), flag)
    if img is None:
        return {"path": str(p), "file": p.name, "folder": p.parent.name, "readable": False}
    hh, ww = img.shape[:2]
    scale = min(1.0, 512 / max(hh, ww))
    small = cv2.resize(img, (max(1, int(ww * scale)), max(1, int(hh * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    black = float((small.max(axis=2) < 8).mean())  # padding from rotation/affine augmentation
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = float(((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50)).mean())
    return {
        "path": str(p),
        "file": p.name,
        "folder": p.parent.name,
        "readable": True,
        "width": w,
        "height": h,
        "megapixels": round(w * h / 1e6, 3),
        "size_kb": round(p.stat().st_size / 1024, 1),
        "brightness": float(gray.mean()),
        "sharpness_lapvar": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "black_border_frac": black,
        "green_frac": green,
        "ahash": ahash(gray),
    }


def quality_table(paths: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([image_quality(p) for p in paths])
    if df.empty:
        return df
    df["likely_augmented"] = df["black_border_frac"] > 0.02
    df["dup_group"] = df.groupby("ahash").ngroup()
    df["dup_count"] = df.groupby("ahash")["ahash"].transform("count")
    return df


# ----------------------------------------------------------------------------------------------
# Patch-box recovery (template matching crop patches back into their source images)
# ----------------------------------------------------------------------------------------------
PATCH_RE = re.compile(r"(img\d+)", re.IGNORECASE)


def recover_patch_boxes(
    originals_dir: str | Path,
    patch_dirs: dict[str, str | Path],
    min_score: float = 0.9,
) -> pd.DataFrame:
    """Locate each patch inside its source image (matched by the 'imgNNN' token in the file name).

    ``patch_dirs`` maps an annotation label (e.g. 'Weed', 'Crop') to a folder of patches.
    Returns one row per patch with the recovered pixel box and the normalised cross-correlation score.
    """
    originals_dir = Path(originals_dir)
    src_index = {p.stem.lower(): p for p in list_images(originals_dir)}
    rows = []
    cache: dict[str, np.ndarray] = {}
    for label, d in patch_dirs.items():
        for patch in list_images(Path(d)):
            m = PATCH_RE.search(patch.stem)
            row = {"patch": patch.name, "patch_path": str(patch), "label": label, "source": None, "score": None, "accepted": False}
            if not m or m.group(1).lower() not in src_index:
                row["note"] = "source image not found"
                rows.append(row)
                continue
            src = src_index[m.group(1).lower()]
            if str(src) not in cache:
                cache = {str(src): cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)}
            I = cache[str(src)]
            T = cv2.imread(str(patch), cv2.IMREAD_GRAYSCALE)
            row["source"] = src.name
            row["source_path"] = str(src)
            if T is None or I is None or T.shape[0] > I.shape[0] or T.shape[1] > I.shape[1]:
                row["note"] = "patch larger than source or unreadable"
                rows.append(row)
                continue
            r = cv2.matchTemplate(I, T, cv2.TM_CCOEFF_NORMED)
            _, mx, _, (x, y) = cv2.minMaxLoc(r)
            th, tw = T.shape[:2]
            row.update(
                score=round(float(mx), 4),
                x1=x, y1=y, x2=x + tw, y2=y + th,
                img_w=I.shape[1], img_h=I.shape[0],
                accepted=bool(mx >= min_score),
                note="" if mx >= min_score else "low match score",
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_labelme_from_boxes(df: pd.DataFrame, out_dir: str | Path, originals_dir: str | Path) -> list[Path]:
    """Write one LabelMe JSON per source image (accepted rows only). imagePath points at the original."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    ok = df[df["accepted"]]
    for src, g in ok.groupby("source"):
        src_path = Path(g.iloc[0]["source_path"])
        rel = Path(_relpath(src_path, out_dir))
        js = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [
                {
                    "label": r["label"],
                    "points": [[float(r["x1"]), float(r["y1"])], [float(r["x2"]), float(r["y2"])]],
                    "group_id": None,
                    "description": f"recovered from patch {r['patch']} (NCC={r['score']})",
                    "shape_type": "rectangle",
                    "flags": {},
                }
                for _, r in g.iterrows()
            ],
            "imagePath": str(rel),
            "imageData": None,
            "imageHeight": int(g.iloc[0]["img_h"]),
            "imageWidth": int(g.iloc[0]["img_w"]),
        }
        p = out_dir / (Path(src).stem + ".json")
        p.write_text(json.dumps(js, indent=2))
        written.append(p)
    return written


def _relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(os.path.abspath(target), os.path.abspath(start))
