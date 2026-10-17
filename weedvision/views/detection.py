from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from weedvision import ui
from weedvision.analysis import split_weeds_crops, weed_metrics
from weedvision.inference import (
    Detection,
    detections_df,
    detections_json,
    draw_detections,
    load_image_rgb,
    run_detection,
    to_png_bytes,
)


def render():
    ui.hero(
        "Precision Weed & Crop Detection",
        "AI-assisted field inspection with sub-object localization, anti-collision labeling, and density estimation.",
    )
    s = ui.settings()
    paths = ui.checkpoint_paths()
    if not paths and not ui.active_model():
        st.error("No checkpoints available. Add a `.pt` file to `models/` or upload one in the sidebar.")
        return

    # Top Control Bar (Model & Detection Parameters)
    with st.container():
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        opts = paths if ui.active_model() in paths else paths + [ui.active_model()]
        model = c1.selectbox("Model Checkpoint", opts, index=opts.index(ui.active_model()) if ui.active_model() in opts else 0, format_func=ui.short_name)
        conf = c2.slider("Model Confidence", 0.05, 0.95, float(s.conf), 0.05, help="Inference confidence threshold.")
        iou = c3.slider("NMS IoU", 0.10, 0.90, float(s.iou), 0.05, help="Non-maximum suppression threshold.")
        imgsz = c4.select_slider("Resolution", [320, 416, 512, 640, 768, 800, 960], value=s.imgsz if s.imgsz in [320, 416, 512, 640, 768, 800, 960] else 640)
        ui.model_banner(model)

    # Image Upload Bar
    files = st.file_uploader("Upload Field Photos", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"], accept_multiple_files=True)
    
    col_btn, col_info = st.columns([1.2, 3.8])
    with col_btn:
        run = st.button("Run Inspection", type="primary", disabled=not files, use_container_width=True)
    with col_info:
        if files:
            st.caption(f"📁 {len(files)} image(s) uploaded and ready for analysis.")

    res = ui.results()
    if run and files:
        yolo = ui.get_yolo(model)
        prog = st.progress(0.0, text="Running model inference…")
        st.session_state["last_batch"] = []
        for i, f in enumerate(files):
            data = f.getvalue()
            try:
                img = load_image_rgb(data)
            except Exception as e:
                st.error(f"{f.name}: unreadable image file ({e})")
                continue
            dets, speed = run_detection(yolo, img, conf, iou, imgsz, ui.device(), s.max_det)
            k = ui.file_key(f.name, data) + f"|{ui.short_name(model)}|{conf}|{iou}|{imgsz}"
            res[k] = dict(name=f.name, image=img, dets=dets, speed=speed, model=model, params=dict(conf=conf, iou=iou, imgsz=imgsz))
            st.session_state["last_batch"].append(k)
            prog.progress((i + 1) / len(files), text=f"Inspecting {i + 1} of {len(files)}…")
        prog.empty()

    batch = [k for k in st.session_state.get("last_batch", []) if k in res]
    if not batch:
        st.info("Upload field photos above and click **Run Inspection** to evaluate weed and crop detections.")
        return

    # Global Batch Actions
    if len(batch) > 1:
        st.markdown("---")
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for k in batch:
                r = res[k]
                H, W = r["image"].shape[:2]
                stem = r["name"].rsplit(".", 1)[0]
                ann = draw_detections(r["image"], r["dets"], show_conf=True, show_boxes=True, show_labels=True, compact=True)
                z.writestr(f"{stem}_annotated.png", to_png_bytes(ann))
                z.writestr(f"{stem}_detections.csv", detections_df(r["dets"], W, H).to_csv(index=False))
                z.writestr(f"{stem}_detections.json", detections_json(r["dets"], r["name"], W, H, ui.short_name(r["model"]), r["params"]))
        st.download_button(f"Download Batch Results ({len(batch)} photos, ZIP)", zbuf.getvalue(), "batch_weedvision_results.zip", "application/zip")

    # Render Each Image in a Balanced 2-Column Responsive Layout
    for k in batch:
        r = res[k]
        img, dets = r["image"], r["dets"]
        H, W = img.shape[:2]

        st.markdown("---")
        st.markdown(f"#### 📷 {r['name']} ({W}×{H} px)")

        # Left (Image Viewer) vs Right (Analytics & Display Settings)
        col_view, col_panel = st.columns([1.35, 1.0], gap="medium")

        with col_panel:
            # Display & Filter Settings
            st.markdown("**Visual Display Controls**")
            
            c_style, c_mode = st.columns(2)
            with c_style:
                label_style = st.selectbox(
                    "Label Format",
                    ["Compact (Weed 87%)", "Minimal (Boxes Only)", "Detailed (ID + Conf)", "Top-5 Weeds Only"],
                    index=0,
                    key=f"lbl_{k}"
                )
            with c_mode:
                view_mode = st.selectbox(
                    "View Mode",
                    ["Annotated Detection", "Original Field Image", "Side-by-Side Split", "Interactive Zoom"],
                    index=0,
                    key=f"vmode_{k}"
                )

            c_filter1, c_filter2 = st.columns(2)
            with c_filter1:
                filter_conf = st.slider("Display Min Conf", 0.0, 1.0, float(conf), 0.05, key=f"fconf_{k}")
            with c_filter2:
                class_filter = st.selectbox("Class Filter", ["All Classes", "Weeds Only", "Crops Only"], index=0, key=f"cfilter_{k}")

            # Highlight specific detection
            valid_uids = ["None"] + [d.uid for d in dets if d.uid]
            selected_uid = st.selectbox("Highlight Target", valid_uids, index=0, key=f"hl_{k}")
            highlight_id = None if selected_uid == "None" else selected_uid

            # Filter active detections for statistics and rendering
            active_dets = [d for d in dets if d.conf >= filter_conf]
            if class_filter == "Weeds Only":
                active_dets = [d for d in active_dets if d.cls_name == "Weed"]
            elif class_filter == "Crops Only":
                active_dets = [d for d in active_dets if d.cls_name != "Weed"]

            if label_style == "Top-5 Weeds Only":
                active_dets = active_dets[:5]

            weeds, crops = split_weeds_crops(active_dets, s)
            m = weed_metrics(weeds, W, H, s)

            # Compact Statistics Grid (2x2)
            st.markdown("**Field Analytics**")
            s1, s2 = st.columns(2)
            with s1:
                ui.card(st, "Weed Detections", f"{len(weeds)}", f"{len(crops)} crops identified")
            with s2:
                avg_conf = (sum(d.conf for d in weeds) / len(weeds) * 100) if weeds else 0.0
                ui.card(st, "Avg Confidence", f"{avg_conf:.1f}%", f"{len(active_dets)} active targets")

            s3, s4 = st.columns(2)
            with s3:
                ui.card(st, "Infestation", f"{m['infestation_pct']:.2f}%", ui.severity_pill(m["severity"]))
            with s4:
                lat = r["speed"].get("inference", 0.0)
                fps = 1000.0 / lat if lat > 0 else 0.0
                ui.card(st, "Inference Latency", f"{lat:.1f} ms", f"~{fps:.0f} FPS · {ui.device()}")

            # Quick Action Downloads
            st.write("")
            stem = r["name"].rsplit(".", 1)[0]
            dl1, dl2, dl3 = st.columns(3)
            ann_raw = draw_detections(
                img, active_dets,
                show_conf=(label_style != "Minimal (Boxes Only)"),
                show_boxes=True,
                show_labels=(label_style != "Minimal (Boxes Only)"),
                show_ids=(label_style == "Detailed (ID + Conf)"),
                highlight=highlight_id,
                compact=(label_style == "Compact (Weed 87%)")
            )
            df = detections_df(active_dets, W, H)
            dl1.download_button("PNG", to_png_bytes(ann_raw), f"{stem}_annotated.png", "image/png", key=f"png_{k}", use_container_width=True)
            dl2.download_button("CSV", df.to_csv(index=False), f"{stem}_detections.csv", "text/csv", key=f"csv_{k}", use_container_width=True)
            dl3.download_button("JSON", detections_json(active_dets, r["name"], W, H, ui.short_name(r["model"]), r["params"]), f"{stem}_detections.json", "application/json", key=f"json_{k}", use_container_width=True)

        with col_view:
            # Responsive Framed Image Viewport (Max Height: 460px)
            ann_img = draw_detections(
                img, active_dets,
                show_conf=(label_style != "Minimal (Boxes Only)"),
                show_boxes=True,
                show_labels=(label_style != "Minimal (Boxes Only)"),
                show_ids=(label_style == "Detailed (ID + Conf)"),
                highlight=highlight_id,
                compact=(label_style == "Compact (Weed 87%)")
            )

            if view_mode == "Annotated Detection":
                st.image(ann_img, caption="Precision Bounding Boxes & Micro-Labels", use_container_width=True)

            elif view_mode == "Original Field Image":
                st.image(img, caption="Raw Field Photo (Untouched)", use_container_width=True)
            elif view_mode == "Side-by-Side Split":
                c_left, c_right = st.columns(2)
                c_left.image(img, caption="Original Photo", use_container_width=True)
                c_right.image(ann_img, caption="Annotated Overlay", use_container_width=True)
            elif view_mode == "Interactive Zoom":
                fig = ui.image_figure(ann_img, height=480)
                st.plotly_chart(fig, use_container_width=True)

        # Full-Width Sortable Detections Table
        st.markdown("**Detections Manifest & Target Inspection**")
        if df.empty:
            st.info("No targets meet the current confidence and class criteria.")
        else:
            q_col1, q_col2 = st.columns([2, 1])
            with q_col1:
                search_query = st.text_input("Filter Manifest (Search ID or Class)", "", key=f"sq_{k}")
            with q_col2:
                sort_col = st.selectbox("Sort By", ["Confidence (Desc)", "Confidence (Asc)", "Area % (Desc)", "ID"], index=0, key=f"sortc_{k}")

            display_table = df.copy()
            if search_query:
                display_table = display_table[
                    display_table["uid"].str.contains(search_query, case=False) |
                    display_table["cls_name"].str.contains(search_query, case=False)
                ]

            if sort_col == "Confidence (Desc)":
                display_table = display_table.sort_values(by="conf", ascending=False)
            elif sort_col == "Confidence (Asc)":
                display_table = display_table.sort_values(by="conf", ascending=True)
            elif sort_col == "Area % (Desc)":
                display_table = display_table.sort_values(by="area_pct", ascending=False)
            elif sort_col == "ID":
                display_table = display_table.sort_values(by="uid", ascending=True)

            st.markdown(ui.class_legend(sorted(display_table["cls_name"].unique())), unsafe_allow_html=True)
            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "uid": st.column_config.TextColumn("Target ID", width="small"),
                    "cls_name": st.column_config.TextColumn("Classification", width="medium"),
                    "conf": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                    "area_pct": st.column_config.NumberColumn("Canopy Area", format="%.3f%%"),
                    "x1": st.column_config.NumberColumn("x1", format="%.0f"),
                    "y1": st.column_config.NumberColumn("y1", format="%.0f"),
                    "x2": st.column_config.NumberColumn("x2", format="%.0f"),
                    "y2": st.column_config.NumberColumn("y2", format="%.0f"),
                },
                height=min(260, 40 + 32 * len(display_table))
            )


