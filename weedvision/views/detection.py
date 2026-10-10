from __future__ import annotations

import io
import zipfile

import streamlit as st

from weedvision import ui
from weedvision.analysis import split_weeds_crops, weed_metrics
from weedvision.inference import (
    detections_df,
    detections_json,
    draw_detections,
    load_image_rgb,
    run_detection,
    to_png_bytes,
)


def render():
    ui.hero("Image Detection", "Upload field images, run the selected model, and download annotated results.")
    s = ui.settings()
    paths = ui.checkpoint_paths()
    if not paths and not ui.active_model():
        st.error("No checkpoints available. Add a `.pt` file to `models/` or upload one in the sidebar.")
        return

    c = st.columns([2, 1, 1, 1])
    opts = paths if ui.active_model() in paths else paths + [ui.active_model()]
    model = c[0].selectbox("Model", opts, index=opts.index(ui.active_model()) if ui.active_model() in opts else 0, format_func=ui.short_name)
    conf = c[1].slider("Confidence", 0.01, 0.95, float(s.conf), 0.01)
    iou = c[2].slider("NMS IoU", 0.10, 0.95, float(s.iou), 0.01)
    imgsz = c[3].select_slider("Image size", [320, 416, 512, 640, 768, 960, 1280], value=s.imgsz if s.imgsz in [320, 416, 512, 640, 768, 960, 1280] else 640)
    ui.model_banner(model)

    files = st.file_uploader("Images", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"], accept_multiple_files=True)
    show_conf = st.toggle("Show confidence on boxes", value=True)
    run = st.button("Run detection", type="primary", disabled=not files)

    res = ui.results()
    if run and files:
        yolo = ui.get_yolo(model)
        prog = st.progress(0.0, text="Detecting…")
        st.session_state["last_batch"] = []
        for i, f in enumerate(files):
            data = f.getvalue()
            try:
                img = load_image_rgb(data)
            except Exception as e:
                st.error(f"{f.name}: cannot read image ({e})")
                continue
            dets, speed = run_detection(yolo, img, conf, iou, imgsz, ui.device(), s.max_det)
            k = ui.file_key(f.name, data) + f"|{ui.short_name(model)}|{conf}|{iou}|{imgsz}"
            res[k] = dict(name=f.name, image=img, dets=dets, speed=speed, model=model, params=dict(conf=conf, iou=iou, imgsz=imgsz))
            st.session_state["last_batch"].append(k)
            prog.progress((i + 1) / len(files), text=f"Detecting… {i + 1}/{len(files)}")
        prog.empty()

    batch = [k for k in st.session_state.get("last_batch", []) if k in res]
    if not batch:
        return

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for k in batch:
            r = res[k]
            H, W = r["image"].shape[:2]
            stem = r["name"].rsplit(".", 1)[0]
            z.writestr(f"{stem}_annotated.png", to_png_bytes(draw_detections(r["image"], r["dets"], show_conf)))
            z.writestr(f"{stem}_detections.csv", detections_df(r["dets"], W, H).to_csv(index=False))
            z.writestr(f"{stem}_detections.json", detections_json(r["dets"], r["name"], W, H, ui.short_name(r["model"]), r["params"]))
    st.download_button(f"Download all results ({len(batch)} images, ZIP)", zbuf.getvalue(), "detections.zip", "application/zip")

    for k in batch:
        r = res[k]
        img, dets = r["image"], r["dets"]
        H, W = img.shape[:2]
        weeds, crops = split_weeds_crops(dets, s)
        m = weed_metrics(weeds, W, H, s)
        st.markdown(f"#### {r['name']}")
        cols = st.columns([3, 1])
        ann = draw_detections(img, dets, show_conf)
        cols[0].image(ann, use_container_width=True)
        with cols[1]:
            ui.card(st, "Weeds", str(len(weeds)), f"{len(crops)} crop detections")
            st.write("")
            ui.card(st, "Infestation", f"{m['infestation_pct']:.2f}%", "union of weed boxes / image")
            st.write("")
            st.markdown(ui.severity_pill(m["severity"]), unsafe_allow_html=True)
            st.caption(f"Inference {r['speed'].get('inference', 0):.0f} ms · {W}×{H}px · {ui.device()}")
        df = detections_df(dets, W, H)
        if df.empty:
            st.info("No detections above the confidence threshold.")
        else:
            st.markdown(ui.class_legend(sorted(df["cls_name"].unique())), unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True, height=min(400, 38 + 35 * len(df)))
        stem = r["name"].rsplit(".", 1)[0]
        d = st.columns(3)
        d[0].download_button("Annotated PNG", to_png_bytes(ann), f"{stem}_annotated.png", "image/png", key=f"png_{k}")
        d[1].download_button("Detections CSV", df.to_csv(index=False), f"{stem}_detections.csv", "text/csv", key=f"csv_{k}")
        d[2].download_button("Detections JSON", detections_json(dets, r["name"], W, H, ui.short_name(r["model"]), r["params"]), f"{stem}_detections.json", "application/json", key=f"json_{k}")
        st.divider()
