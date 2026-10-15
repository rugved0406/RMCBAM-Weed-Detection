"""Analysis maths, plus end-to-end inference and Grad-CAM pipeline tests.

The pipeline tests use the stock COCO-pretrained yolov8s.pt and Ultralytics' bundled bus.jpg
only to prove the code paths work on a real detector. They say nothing about weed accuracy.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weedvision.analysis import crop_association, split_weeds_crops, union_area, weed_metrics  # noqa: E402
from weedvision.config import Settings  # noqa: E402
from weedvision.inference import Detection, assign_ids, detections_df, draw_detections  # noqa: E402


def D(cls, x1, y1, x2, y2, conf=0.9):
    return Detection("", 0 if cls == "Weed" else 1, cls, conf, x1, y1, x2, y2)


def test_union_area_counts_overlap_once():
    a, b = D("Weed", 0, 0, 10, 10), D("Weed", 5, 5, 15, 15)
    assert union_area([a, b], 100, 100) == 175


def test_weed_metrics_and_severity():
    s = Settings()
    w = assign_ids([D("Weed", 0, 0, 10, 10, 0.5), D("Weed", 50, 50, 60, 60, 0.7)])
    m = weed_metrics(w, 100, 100, s)
    assert m["weed_count"] == 2 and math.isclose(m["infestation_pct"], 2.0)
    assert m["severity"] == "Low" and math.isclose(m["mean_conf"], 0.6)
    assert math.isclose(m["density_per_mpx"], 2 / 0.01)
    s.gsd_cm_per_px = 1.0  # 100x100 px = 1 m²
    assert math.isclose(weed_metrics(w, 100, 100, s)["density_per_m2"], 2.0)
    assert s.severity(4.9) == "Low" and s.severity(5) == "Moderate" and s.severity(15) == "High" and s.severity(30) == "Severe"


def test_ids_unique_and_ordered():
    ds = assign_ids([D("Weed", 300, 0, 310, 10), D("Weed", 0, 0, 10, 10), D("Rose", 0, 200, 10, 210), D("Weed", 0, 300, 10, 310)])
    weeds = [d for d in ds if d.cls_name == "Weed"]
    assert [d.uid for d in sorted(weeds, key=lambda d: (d.y1, d.x1))] == ["W-001", "W-002", "W-003"]
    assert {d.uid for d in ds if d.cls_name == "Rose"} == {"R-001"}


def test_crop_association():
    s = Settings()
    dets = assign_ids([D("Weed", 10, 10, 20, 20), D("Weed", 900, 900, 910, 910), D("Rose", 25, 10, 60, 40), D("Marigold", 0, 600, 50, 650)])
    weeds, crops = split_weeds_crops(dets, s)
    pw, pc = crop_association(weeds, crops, 1000, 1000, s)
    near = pw.set_index("weed_id").loc["W-001"]
    assert near["nearest_crop_class"] == "Rose" and near["associated"] and math.isclose(near["distance_px"], 10.0)
    far = pw.set_index("weed_id").loc["W-002"]
    assert not far["associated"] and far["associated_class"] == "Unassociated"
    assert int(pc.set_index("crop_class").loc["Rose", "associated_weeds"]) == 1
    _, empty = crop_association(weeds, [], 1000, 1000, s)
    assert empty.empty


def test_draw_and_df():
    img = np.zeros((100, 200, 3), np.uint8)
    ds = assign_ids([D("Weed", 10, 10, 50, 50)])
    out = draw_detections(img, ds)
    assert out.shape == img.shape and out.sum() > 0
    df = detections_df(ds, 200, 100)
    assert df.loc[0, "uid"] == "W-001" and math.isclose(df.loc[0, "area_pct"], 8.0)


def test_draw_detections_anti_collision():
    img = np.zeros((400, 400, 3), np.uint8)
    # 4 highly clustered weed detections at almost the same location
    clustered = assign_ids([
        D("Weed", 100, 100, 140, 140, conf=0.95),
        D("Weed", 102, 102, 142, 142, conf=0.88),
        D("Weed", 105, 105, 145, 145, conf=0.75),
        D("Weed", 108, 108, 148, 148, conf=0.60),
    ])
    out = draw_detections(img, clustered, show_conf=True, show_labels=True, show_boxes=True)
    assert out.shape == img.shape and out.sum() > 0


def test_draw_detections_filtering_and_toggles():
    img = np.zeros((300, 300, 3), np.uint8)
    ds = assign_ids([
        D("Weed", 10, 10, 50, 50, conf=0.90),
        D("Weed", 60, 60, 100, 100, conf=0.40),
        D("Rose", 120, 120, 160, 160, conf=0.20),
    ])
    
    # Confidence filter: only conf >= 0.50 should render
    filtered_out = draw_detections(img, ds, conf_threshold=0.50)
    assert filtered_out.shape == img.shape and filtered_out.sum() > 0

    # Max detections limit: top 1
    top1_out = draw_detections(img, ds, max_detections=1)
    assert top1_out.shape == img.shape and top1_out.sum() > 0

    # Toggles: boxes only, labels only, no annotations
    boxes_only = draw_detections(img, ds, show_boxes=True, show_labels=False)
    labels_only = draw_detections(img, ds, show_boxes=False, show_labels=True)
    none_rendered = draw_detections(img, ds, show_boxes=False, show_labels=False)
    
    assert boxes_only.sum() > 0
    assert labels_only.sum() > 0
    assert none_rendered.sum() == 0  # Black background untouched

    # Empty list handling
    empty_out = draw_detections(img, [])
    assert empty_out.shape == img.shape and empty_out.sum() == 0



# ------------------------------------------------------------------ pipeline tests on a real detector
WEIGHTS = Path(__file__).resolve().parent.parent / "yolov8s.pt"


@pytest.fixture(scope="module")
def coco():
    if not WEIGHTS.exists():
        pytest.skip("yolov8s.pt not present (place it in the project root to run pipeline tests)")
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    from weedvision.inference import load_image_rgb

    return YOLO(str(WEIGHTS)), load_image_rgb(str(ASSETS / "bus.jpg"))


def test_inference_pipeline(coco):
    from weedvision.inference import run_detection

    yolo, img = coco
    dets, speed = run_detection(yolo, img, 0.25, 0.45, 640, "cpu")
    assert any(d.cls_name == "person" for d in dets)
    assert len({d.uid for d in dets}) == len(dets)
    assert "inference" in speed


def test_gradcam_localises_on_detection(coco):
    """Real gradients: the Grad-CAM mass for a detected person should concentrate inside its box."""
    from weedvision.explain import DetectorCAM, energy_inside_box
    from weedvision.inference import run_detection

    yolo, img = coco
    dets, _ = run_detection(yolo, img, 0.25, 0.45, 640, "cpu")
    person = max((d for d in dets if d.cls_name == "person"), key=lambda d: d.conf)
    cam = DetectorCAM(yolo.model, [15, 18, 21], "cpu")
    out = cam.explain(img, [person], 640, method="gradcam")
    m = out["per_target"][person.uid]
    assert m.shape == img.shape[:2] and np.isfinite(m).all() and m.max() > 0
    box_frac = person.area / (img.shape[0] * img.shape[1])
    assert energy_inside_box(m, person) > box_frac, "CAM should be denser inside the box than a uniform map"
    assert out["anchors"][person.uid] >= 1
    hi = cam.explain(img, [person], 640, method="hirescam", per_target=False)["combined"]
    assert hi.shape == img.shape[:2]
    # HiResCAM (default) must be strongly localised: >3x the density of a uniform map
    assert energy_inside_box(hi, person) > 3 * box_frac
    cam.close()


def test_rmcbam_attention_maps():
    from weedvision.explain import rmcbam_attention_maps
    from weedvision.rmcbam import DEFAULT_RMCBAM_LAYERS, build_rmcbam_model

    m = build_rmcbam_model("yolov8s.yaml", nc=5).eval()
    img = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
    maps = rmcbam_attention_maps(m, img, 320, "cpu")
    assert [x["layer"] for x in maps] == sorted(DEFAULT_RMCBAM_LAYERS)
    assert all(x["spatial"].shape == (240, 320) for x in maps)
    from ultralytics.nn.tasks import DetectionModel

    assert rmcbam_attention_maps(DetectionModel("yolov8s.yaml", nc=5, verbose=False), img, 320) == []
