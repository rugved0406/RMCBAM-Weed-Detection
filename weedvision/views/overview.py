from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from weedvision import ui
from weedvision.analysis import split_weeds_crops, weed_metrics
from weedvision.config import CLASS_NAMES


def render():
    s = ui.settings()
    ui.hero(
        "RMCBAM Weed &amp; Crop Detection",
        "YOLOv8s with Residual Multi-Scale CBAM attention · Weed, Tomato, Rose, Marigold, Cauliflower",
    )

    paths = ui.checkpoint_paths()
    infos = [ui.get_info(p) for p in paths]
    project = [i for i in infos if i.is_project_model]
    rm = [i for i in project if i.is_rmcbam]
    base = [i for i in project if not i.is_rmcbam]

    ds = Path(s.dataset_dir)
    from weedvision.dataset import scan_folder

    scan = _scan(str(ds))
    res = ui.results()
    n_weeds = 0
    for r in res.values():
        w, _ = split_weeds_crops(r["dets"], s)
        n_weeds += len(w)

    c = st.columns(4)
    ui.card(c[0], "Checkpoints", f"{len(paths)}", f"{len(project)} with Weed class · {len(rm)} RMCBAM")
    ui.card(c[1], "Dataset images", f"{scan.get('n_images', 0)}" if scan.get("exists") else "—", f"{scan.get('n_labelme_json', 0)} LabelMe JSON · {len(scan.get('yaml_files', []))} yaml" if scan.get("exists") else f"folder not found: {ds}")
    ui.card(c[2], "Compute", ui.device().upper(), f"torch {torch.__version__} · CUDA {'yes' if torch.cuda.is_available() else 'no'}")
    ui.card(c[3], "This session", f"{len(res)} images", f"{n_weeds} weeds detected")

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Readiness")
        lines = [
            ui.check_line(bool(base), "Trained <b>baseline YOLOv8s</b> checkpoint with a Weed class"),
            ui.check_line(bool(rm), "Trained <b>YOLOv8s + RMCBAM</b> checkpoint with a Weed class"),
            ui.check_line(scan.get("n_labelme_json", 0) > 0, f"LabelMe annotations in dataset folder ({scan.get('n_labelme_json', 0)} found)"),
            ui.check_line(bool(scan.get("yaml_files")), "<code>data.yaml</code> present (needed for live validation)"),
        ]
        if project:
            miss = sorted({m for i in project for m in i.missing_project_classes})
            lines.append(ui.check_line(not miss, "All 5 classes declared in model" + (f" — missing: {', '.join(miss)}" if miss else "")))
        st.markdown("".join(lines), unsafe_allow_html=True)
        if not project:
            st.info(
                "No checkpoint with a **Weed** class was found, so no weed detections or accuracy metrics can be shown yet. "
                "Train with `python tools/train.py --data data/yolo/data.yaml` or copy existing `best.pt` files into `models/`."
            )
    with right:
        st.subheader("Pipeline")
        st.markdown(
            """
1. **Dataset Analytics** — scan images, convert LabelMe → YOLO, validate labels.
2. **Training** — `tools/train.py` trains baseline and RMCBAM with identical settings.
3. **Image Detection** — boxes, confidence, unique IDs (W-001…), downloads.
4. **Explainability** — Grad-CAM / HiResCAM from real gradients; RMCBAM attention maps.
5. **Weed Analysis & Localization** — count, density, infestation %, severity, crop association, coordinates.
6. **Model Comparison** — results.csv, plots, or live validation on your data.yaml.
"""
        )
        st.markdown(ui.class_legend(CLASS_NAMES), unsafe_allow_html=True)

    if res:
        st.subheader("Session summary")
        rows = []
        for r in res.values():
            H, W = r["image"].shape[:2]
            w, cr = split_weeds_crops(r["dets"], s)
            m = weed_metrics(w, W, H, s)
            rows.append({"image": r["name"], "model": ui.short_name(r["model"]), "weeds": m["weed_count"], "crops": len(cr), "infestation_%": round(m["infestation_pct"], 2), "severity": m["severity"], "mean_conf": None if m["mean_conf"] is None else round(m["mean_conf"], 3)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


@st.cache_data(ttl=60, show_spinner=False)
def _scan(path: str) -> dict:
    from weedvision.dataset import scan_folder

    d = scan_folder(path)
    d.pop("images", None)
    return d
