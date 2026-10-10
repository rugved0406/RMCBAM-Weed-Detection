"""Headless UI test: renders all 9 dashboard pages with Streamlit's AppTest and presses their action buttons.

Uses the demo smoke-test checkpoints (demo/smoke_runs) if present, otherwise skips.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEMO = ROOT / "demo" / "smoke_runs"
PAGES = ["Dashboard Overview", "Image Detection", "Weed Explainability", "Weed Analysis", "Weed Localization",
         "Model Comparison", "Dataset Analytics", "Model Information", "Settings"]
SKIP_BUTTONS = {"Convert", "Clear cached models & data", "Clear session results", "Run validation on both",
                "Apply & save as default", "Run detection", "Compute image quality statistics"}


@pytest.fixture(scope="module")
def session():
    ckpts = sorted(DEMO.glob("*rmcbam*/weights/best.pt"))
    if not ckpts:
        pytest.skip("demo smoke-test checkpoints not present")
    os.environ.update(WV_MODELS_DIR=str(ROOT / "demo" / "models"), WV_RUNS_DIR=str(DEMO), WV_DATASET_DIR=str(ROOT / "data"))
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    from weedvision.config import Settings
    from weedvision.inference import load_image_rgb, run_detection

    img_path = next((ROOT / "data").rglob("img331.jpg"), ASSETS / "bus.jpg")
    img = load_image_rgb(str(img_path))
    model = str(ckpts[0])
    dets, sp = run_detection(YOLO(model), img, 0.25, 0.45, 640, "cpu", 50)
    res = {"k": dict(name=img_path.name, image=img, dets=dets, speed=sp, model=model, params=dict(conf=0.25, iou=0.45, imgsz=640))}
    return Settings.load(), res, model


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(session, page):
    from streamlit.testing.v1 import AppTest

    settings, res, model = session
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.session_state["settings"] = settings
    at.session_state["results"] = dict(res)
    at.session_state["nav"] = page
    at.session_state["active_model"] = model
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    for b in list(at.button):
        if b.label in SKIP_BUTTONS or b.disabled:
            continue
        b.click()
        at.run()
        assert not at.exception, f"{page} / {b.label}: {[e.value for e in at.exception]}"
