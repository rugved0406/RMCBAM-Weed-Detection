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
        "RMCBAM Weed & Crop Detection",
        "YOLOv8s with Residual Multi-Scale CBAM attention · Weed, Tomato, Rose, Marigold, Cauliflower",
    )

    paths = ui.checkpoint_paths()
    infos = [ui.get_info(p) for p in paths]
    project = [i for i in infos if i.is_project_model]
    rm = [i for i in project if i.is_rmcbam]
    base = [i for i in project if not i.is_rmcbam]

    ds = Path(s.dataset_dir)
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
    left, right = st.columns(2, gap="medium")
    with left:
        lines = [
            ui.check_line(bool(base), "Trained <b>baseline YOLOv8s</b> checkpoint with Weed class"),
            ui.check_line(bool(rm), "Trained <b>YOLOv8s + RMCBAM</b> checkpoint with Weed class"),
            ui.check_line(scan.get("n_labelme_json", 0) > 0, f"LabelMe annotations ({scan.get('n_labelme_json', 0)} JSON files)"),
            ui.check_line(bool(scan.get("yaml_files")), "<code>data.yaml</code> configuration file present"),
        ]
        if project:
            miss = sorted({m for i in project for m in i.missing_project_classes})
            lines.append(ui.check_line(not miss, "All 5 classes declared in model" + (f" (missing: {', '.join(miss)})" if miss else "")))
        if not project:
            lines.append(
                "<div style='margin-top:10px; padding:8px 12px; border-radius:6px; background:#fef3c7; border:1px solid #fde68a; color:#92400e; font-size:0.82rem;'>"
                "⚠️ No model with <b>Weed</b> class found. Train using <code>tools/train.py</code> or copy <code>best.pt</code> to <code>models/</code>."
                "</div>"
            )
        st.markdown(ui.section_card("System & Model Readiness", "".join(lines)), unsafe_allow_html=True)
    with right:
        pipeline_html = """
        <ol style="margin: 0; padding-left: 18px; font-size: 0.85rem; line-height: 1.65; color: #59665E;">
            <li><b style="color:#17231D">Dataset Analytics</b> — scan images, convert LabelMe → YOLO, validate.</li>
            <li><b style="color:#17231D">Training</b> — <code>tools/train.py</code> trains baseline and RMCBAM identically.</li>
            <li><b style="color:#17231D">Image Detection</b> — collision-free boxes, confidence, unique IDs (W-001…).</li>
            <li><b style="color:#17231D">Explainability</b> — Grad-CAM / HiResCAM and RMCBAM attention maps.</li>
            <li><b style="color:#17231D">Weed Localization</b> — count, density, infestation %, severity, coordinates.</li>
            <li><b style="color:#17231D">Model Comparison</b> — precision, recall, mAP@50, live test validation.</li>
        </ol>
        <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #E5ECE6;">
        """ + ui.class_legend(CLASS_NAMES) + "</div>"
        st.markdown(ui.section_card("Inspection Pipeline", pipeline_html), unsafe_allow_html=True)

    if res:
        st.write("")
        st.subheader("Session Summary")
        rows = []
        for r in res.values():
            H, W = r["image"].shape[:2]
            w, cr = split_weeds_crops(r["dets"], s)
            m = weed_metrics(w, W, H, s)
            rows.append({
                "Image": r["name"],
                "Model": ui.short_name(r["model"]),
                "Weeds": m["weed_count"],
                "Crops": len(cr),
                "Infestation %": m["infestation_pct"] / 100.0,
                "Severity": m["severity"],
                "Mean Conf": m["mean_conf"] if m["mean_conf"] is not None else 0.0,
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Image": st.column_config.TextColumn("Image"),
                "Model": st.column_config.TextColumn("Model"),
                "Weeds": st.column_config.NumberColumn("Weeds", format="%d"),
                "Crops": st.column_config.NumberColumn("Crops", format="%d"),
                "Infestation %": st.column_config.ProgressColumn("Infestation", min_value=0.0, max_value=1.0, format="%.1f%%"),
                "Severity": st.column_config.TextColumn("Severity"),
                "Mean Conf": st.column_config.NumberColumn("Mean Confidence", format="%.2f"),
            },
        )


@st.cache_data(ttl=60, show_spinner=False)
def _scan(path: str) -> dict:
    from weedvision.dataset import scan_folder

    d = scan_folder(path)
    d.pop("images", None)
    return d
