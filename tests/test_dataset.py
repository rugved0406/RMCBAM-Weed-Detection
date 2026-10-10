"""Dataset tool tests. JSON/image fixtures here are synthetic test inputs created in a temp dir."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weedvision.dataset import (  # noqa: E402
    convert_labelme_dataset,
    normalise_label,
    quality_table,
    recover_patch_boxes,
    shape_to_xyxy,
    validate_yolo_dataset,
    write_labelme_from_boxes,
)


def _img(p: Path, w=200, h=100, seed=0):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(p), (rng.random((h, w, 3)) * 255).astype(np.uint8))


def _lm(p: Path, shapes, w=200, h=100, image="a.jpg"):
    p.write_text(json.dumps({"shapes": shapes, "imagePath": image, "imageWidth": w, "imageHeight": h}))


def test_normalise_label():
    assert normalise_label("weeds") == "Weed"
    assert normalise_label(" Marigold ") == "Marigold"
    assert normalise_label("genda") == "Marigold"
    assert normalise_label("phool gobhi") == "Cauliflower"
    assert normalise_label("Crop") is None


def test_shape_to_xyxy():
    assert shape_to_xyxy({"shape_type": "rectangle", "points": [[10, 20], [5, 40]]}) == (5, 20, 10, 40)
    assert shape_to_xyxy({"shape_type": "polygon", "points": [[1, 1], [9, 2], [4, 8]]}) == (1, 1, 9, 8)
    assert shape_to_xyxy({"shape_type": "circle", "points": [[50, 50], [60, 50]]}) == (40, 40, 60, 60)
    assert shape_to_xyxy({"shape_type": "point", "points": [[1, 1]]}) is None


def test_convert_and_validate(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _img(src / "a.jpg")
    _img(src / "b.jpg", seed=1)
    _img(src / "c.jpg", seed=2)  # no json
    _lm(src / "a.json", [
        {"label": "weed", "shape_type": "rectangle", "points": [[20, 10], [60, 50]]},
        {"label": "Tomato", "shape_type": "polygon", "points": [[100, 10], [150, 20], [120, 90]]},
        {"label": "Weed", "shape_type": "rectangle", "points": [[-10, -5], [30, 20]]},  # clipped
        {"label": "stone", "shape_type": "rectangle", "points": [[0, 0], [5, 5]]},  # unknown
        {"label": "Rose", "shape_type": "point", "points": [[5, 5]]},  # skipped
    ])
    _lm(src / "b.json", [{"label": "Weed", "shape_type": "rectangle", "points": [[0, 0], [100, 50]]}], w=100, h=50, image="b.jpg")  # half size → rescaled
    before = {p.name: p.stat().st_mtime for p in src.iterdir()}
    out = tmp_path / "yolo"
    rep = convert_labelme_dataset(src, out, val_frac=0.5, seed=0)
    assert {p.name: p.stat().st_mtime for p in src.iterdir()} == before, "originals must be untouched"
    assert rep.images_with_json == 2 and rep.images_without_json == ["c.jpg"]
    assert rep.instances_per_class == {"Weed": 3, "Tomato": 1}
    assert rep.unknown_labels == {"stone": 1} and rep.skipped_shapes == {"point": 1}
    assert rep.clipped_boxes == 1 and len(rep.size_mismatch) == 1
    assert set(rep.missing_classes) == {"Rose", "Marigold", "Cauliflower"}
    lbl = {p.stem: p.read_text().split("\n") for p in out.rglob("labels/*/*.txt")}
    a0 = [float(v) for v in lbl["a"][0].split()]
    assert a0 == [0, 0.2, 0.3, 0.2, 0.4]  # (20..60)/200, (10..50)/100
    b0 = [float(v) for v in lbl["b"][0].split()]
    assert b0 == [0, 0.5, 0.5, 1.0, 1.0]  # rescaled to full image
    d = yaml.safe_load((out / "data.yaml").read_text())
    assert d["nc"] == 5 and d["names"][0] == "Weed"
    v = validate_yolo_dataset(out / "data.yaml")
    assert v["issues"] == []
    assert v["instances"] == {"Weed": 3, "Tomato": 1}


def test_validator_flags_bad_lines(tmp_path):
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    _img(tmp_path / "images" / "val" / "x.jpg")
    _img(tmp_path / "images" / "val" / "y.jpg")
    (tmp_path / "labels" / "val" / "x.txt").write_text("7 0.5 0.5 0.1 0.1\n0 1.5 0.5 0.1 0.1\n0 0.5 0.5\n0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"path": str(tmp_path), "train": "images/val", "val": "images/val", "names": {0: "Weed", 1: "Tomato"}}))
    v = validate_yolo_dataset(tmp_path / "data.yaml")
    text = " | ".join(v["issues"])
    assert "out of range" in text and "outside [0,1]" in text and "expected 5 values" in text and "duplicate" in text
    assert v["splits"]["val"]["missing_label"] == 1


def test_recover_patch_boxes(tmp_path):
    (tmp_path / "orig").mkdir()
    (tmp_path / "weeds").mkdir()
    rng = np.random.default_rng(3)
    big = (rng.random((300, 400, 3)) * 255).astype(np.uint8)
    big = cv2.GaussianBlur(big, (5, 5), 0)
    cv2.imwrite(str(tmp_path / "orig" / "img7.png"), big)
    cv2.imwrite(str(tmp_path / "weeds" / "weed_001_img7.png"), big[120:200, 50:170])
    cv2.imwrite(str(tmp_path / "weeds" / "weed_002_img99.png"), big[0:20, 0:20])  # no source
    df = recover_patch_boxes(tmp_path / "orig", {"Weed": tmp_path / "weeds"})
    ok = df[df["accepted"]].iloc[0]
    assert (ok.x1, ok.y1, ok.x2, ok.y2) == (50, 120, 170, 200) and ok.score > 0.99
    assert (~df["accepted"]).sum() == 1
    files = write_labelme_from_boxes(df, tmp_path / "lm", tmp_path / "orig")
    js = json.loads(files[0].read_text())
    assert js["shapes"][0]["points"] == [[50.0, 120.0], [170.0, 200.0]]
    assert (tmp_path / "lm" / js["imagePath"]).resolve() == (tmp_path / "orig" / "img7.png").resolve()


def test_quality_table(tmp_path):
    a = np.zeros((100, 100, 3), np.uint8)
    a[10:90, 10:90] = 120
    cv2.imwrite(str(tmp_path / "a.png"), a)
    cv2.imwrite(str(tmp_path / "b.png"), a)
    q = quality_table([str(tmp_path / "a.png"), str(tmp_path / "b.png")])
    assert q["likely_augmented"].all() and (q["dup_count"] == 2).all()
