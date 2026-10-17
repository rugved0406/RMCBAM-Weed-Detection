"""Integration tests for the WeedVision FastAPI backend adapter."""

import io
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app, PROJECT_ROOT
from weedvision.config import CLASS_NAMES

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["project"] == "WeedVision — RMCBAM-YOLOv8"
    assert data["classes"] == CLASS_NAMES
    assert data["checkpoints_count"] >= 4


def test_checkpoints_discovery():
    response = client.get("/api/checkpoints")
    assert response.status_code == 200
    data = response.json()
    checkpoints = data.get("checkpoints", [])
    assert len(checkpoints) >= 4
    
    filenames = [c["filename"] for c in checkpoints]
    assert "weed_detection_best.pt" in filenames
    assert "yolov8s_rmcbam_best.pt" in filenames

    rmcbam_ckpts = [c for c in checkpoints if c["is_rmcbam"]]
    assert len(rmcbam_ckpts) >= 2
    for rc in rmcbam_ckpts:
        assert len(rc["rmcbam_layers"]) > 0


def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    samples = data.get("samples", [])
    assert len(samples) > 0
    assert "img112.jpg" in samples


def test_predict_real_image():
    img_path = PROJECT_ROOT / "data" / "yolo_weed_partial" / "images" / "train" / "img112.jpg"
    assert img_path.exists(), f"Missing test image at {img_path}"
    
    with open(img_path, "rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("img112.jpg", f, "image/jpeg")},
            data={"conf": 0.25, "iou": 0.45},
        )
    
    assert response.status_code == 200
    res = response.json()
    assert res["filename"] == "img112.jpg"
    assert res["width"] > 0
    assert res["height"] > 0
    assert "speed" in res
    assert "detections" in res
    assert "infestation" in res
    assert "original_image" in res and res["original_image"].startswith(("data:image/png;base64,", "data:image/jpeg;base64,"))
    assert "annotated_image" in res and res["annotated_image"].startswith(("data:image/png;base64,", "data:image/jpeg;base64,"))

    for d in res["detections"]:
        assert d["cls_name"] in CLASS_NAMES
        assert 0.0 <= d["conf"] <= 1.0
        assert 0.0 <= d["x1"] <= d["x2"] <= res["width"]
        assert 0.0 <= d["y1"] <= d["y2"] <= res["height"]


def test_predict_sample_endpoint():
    response = client.post(
        "/api/predict-sample",
        data={"filename": "img112.jpg", "conf": 0.25},
    )
    assert response.status_code == 200
    res = response.json()
    assert res["filename"] == "img112.jpg"
    assert len(res["detections"]) >= 1


def test_explain_rmcbam_endpoint():
    img_path = PROJECT_ROOT / "data" / "yolo_weed_partial" / "images" / "train" / "img112.jpg"
    rmcbam_path = PROJECT_ROOT / "models" / "yolov8s_rmcbam_best.pt"
    
    with open(img_path, "rb") as f:
        response = client.post(
            "/api/explain",
            files={"file": ("img112.jpg", f, "image/jpeg")},
            data={"checkpoint": str(rmcbam_path), "method": "hirescam"},
        )
    
    assert response.status_code == 200
    res = response.json()
    assert res["method"] == "hirescam"
    assert "combined_cam" in res
    assert "energy_scores" in res
    assert "rmcbam_maps" in res
    assert len(res["rmcbam_maps"]) == 6  # 6 RMCBAM C2f modules


def test_dataset_and_model_info():
    res_ds = client.get("/api/dataset")
    assert res_ds.status_code == 200
    ds_data = res_ds.json()
    assert "inventory" in ds_data

    res_mi = client.get("/api/model-info")
    assert res_mi.status_code == 200
    mi_data = res_mi.json()
    assert "checkpoint_info" in mi_data
    assert "architecture_table" in mi_data


def test_error_handling_invalid_image():
    response = client.post(
        "/api/predict",
        files={"file": ("fake.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Invalid image" in response.json()["detail"]


def test_error_handling_missing_checkpoint():
    img_path = PROJECT_ROOT / "data" / "yolo_weed_partial" / "images" / "train" / "img112.jpg"
    with open(img_path, "rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("img112.jpg", f, "image/jpeg")},
            data={"checkpoint": "models/nonexistent_model.pt"},
        )
    assert response.status_code == 404


def test_accuracy_recommendations_endpoint():
    response = client.get("/api/accuracy/recommendations")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "recommendations" in data
    assert "model_benchmarks" in data
    assert len(data["model_benchmarks"]) >= 3


def test_predict_with_accuracy_options():
    response = client.post(
        "/api/predict-sample",
        data={
            "filename": "img112.jpg",
            "augment": "true",
            "agnostic_nms": "true",
            "ensemble": "true",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["params"]["augment"] is True
    assert data["params"]["agnostic_nms"] is True
    assert data["params"]["ensemble"] is True
    assert "detections" in data
