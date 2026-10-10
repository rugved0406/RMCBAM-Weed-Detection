from __future__ import annotations

import platform

import streamlit as st
import torch

from weedvision import ui
from weedvision.config import CLASS_NAMES, Settings


def render():
    ui.hero("Settings", "Paths, inference defaults, analysis thresholds and compute device.")
    s = ui.settings()
    with st.form("settings_form"):
        st.markdown("**Paths**")
        c = st.columns(3)
        models_dir = c[0].text_input("Models directory", s.models_dir)
        runs_dir = c[1].text_input("Runs directory (Ultralytics)", s.runs_dir)
        dataset_dir = c[2].text_input("Dataset directory", s.dataset_dir)

        st.markdown("**Inference**")
        c = st.columns(5)
        devs = ["auto", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        device = c[0].selectbox("Device", devs, index=devs.index(s.device) if s.device in devs else 0)
        imgsz = c[1].selectbox("Image size", [320, 416, 512, 640, 768, 960, 1280], index=[320, 416, 512, 640, 768, 960, 1280].index(s.imgsz) if s.imgsz in [320, 416, 512, 640, 768, 960, 1280] else 3)
        conf = c[2].slider("Confidence", 0.01, 0.95, float(s.conf), 0.01)
        iou = c[3].slider("NMS IoU", 0.1, 0.95, float(s.iou), 0.01)
        max_det = c[4].number_input("Max detections", 1, 3000, int(s.max_det))

        st.markdown("**Classes**")
        c = st.columns(2)
        weed_class = c[0].text_input("Weed class name", s.weed_class)
        crop_classes = c[1].multiselect("Crop classes", sorted(set(CLASS_NAMES[1:] + s.crop_classes)), default=s.crop_classes)

        st.markdown("**Severity thresholds** (estimated infestation %)")
        c = st.columns(3)
        lo = c[0].number_input("Low → Moderate at", 0.0, 100.0, float(s.severity_low), 0.5)
        mo = c[1].number_input("Moderate → High at", 0.0, 100.0, float(s.severity_moderate), 0.5)
        hi = c[2].number_input("High → Severe at", 0.0, 100.0, float(s.severity_high), 0.5)

        st.markdown("**Analysis**")
        c = st.columns(3)
        prox = c[0].slider("Crop association radius (fraction of image diagonal)", 0.01, 0.5, float(s.proximity_frac), 0.01)
        gsd = c[1].number_input("Ground sampling distance (cm/px, 0 = unknown)", 0.0, 100.0, float(s.gsd_cm_per_px), 0.01)
        layers = c[2].text_input("Grad-CAM default layers", ",".join(map(str, s.gradcam_layers)))

        c = st.columns([1, 1, 4])
        apply = c[0].form_submit_button("Apply", type="primary")
        save = c[1].form_submit_button("Apply & save as default")
    if apply or save:
        if not (lo <= mo <= hi):
            st.error("Severity thresholds must be increasing.")
            return
        try:
            gl = [int(x) for x in layers.replace(" ", "").split(",") if x]
        except ValueError:
            st.error("Grad-CAM layers must be comma-separated integers.")
            return
        new = Settings(models_dir, runs_dir, dataset_dir, device, int(imgsz), float(conf), float(iou), int(max_det), weed_class, crop_classes, lo, mo, hi, prox, gsd, gl)
        st.session_state["settings"] = new
        if save:
            new.save()
            st.success("Settings applied and saved to settings.json.")
        else:
            st.success("Settings applied for this session.")

    st.markdown("**Environment**")
    import ultralytics

    st.code(
        f"python       {platform.python_version()}\n"
        f"torch        {torch.__version__}\n"
        f"ultralytics  {ultralytics.__version__}\n"
        f"CUDA         {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}\n"
        f"device used  {ui.device()}",
        language="text",
    )
    c = st.columns(3)
    if c[0].button("Clear cached models & data"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Caches cleared.")
    if c[1].button("Clear session results"):
        st.session_state["results"] = {}
        st.session_state.pop("last_batch", None)
        st.success("Session results cleared.")
