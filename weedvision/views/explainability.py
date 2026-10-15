from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from weedvision import ui
from weedvision.analysis import split_weeds_crops
from weedvision.explain import energy_inside_box, overlay_heatmap, rmcbam_attention_maps
from weedvision.inference import draw_detections, to_png_bytes


def render():
    ui.hero(
        "Weed Explainability",
        "Grad-CAM / HiResCAM from the model's real gradients for each detected weed, plus RMCBAM attention maps when present.",
    )
    s = ui.settings()
    r = ui.pick_result("explain")
    if r is None:
        return
    model_path = r["model"]
    ui.model_banner(model_path)
    info = ui.get_info(model_path)
    img, dets = r["image"], r["dets"]
    weeds, _ = split_weeds_crops(dets, s)

    n_layers = len(ui.get_yolo(model_path).model.model)
    c = st.columns([1, 2, 1, 1])
    method = c[0].radio("Method", ["hirescam", "gradcam"], format_func={"gradcam": "Grad-CAM", "hirescam": "HiResCAM"}.get, horizontal=False)
    layer_opts = list(range(1, n_layers - 1))
    default_layers = [l for l in s.gradcam_layers if l in layer_opts] or layer_opts[-3:]
    layers = c[1].multiselect("Target layers (neck outputs 15/18/21 feed the detector)", layer_opts, default=default_layers)
    alpha = c[2].slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05)
    iou_match = c[3].slider("Anchor IoU match", 0.3, 0.9, 0.5, 0.05, help="Anchors whose decoded box overlaps the detection by at least this IoU contribute to the explained score.")
    if not layers:
        st.warning("Choose at least one target layer.")
        return

    target_cls = s.weed_class
    if not weeds:
        names = list(info.names.values())
        st.info(f"No **{s.weed_class}** detections in this image at the current threshold.")
        if not info.has_weed_class:
            target_cls = st.selectbox("This model has no Weed class. Explain another detected class instead:", sorted({d.cls_name for d in dets}) or names)
            weeds = [d for d in dets if d.cls_name == target_cls]
        if not weeds:
            if info.has_weed_class and st.button("Compute weed-class evidence map (top-100 Weed scores, no detection)"):
                cls_id = [k for k, v in info.names.items() if v.lower() == s.weed_class.lower()][0]
                cam = ui.get_cam(model_path, layers).explain_class(img, cls_id, r["params"]["imgsz"], method=method)
                st.image(overlay_heatmap(img, cam, alpha), caption="Weed-class evidence (below detection threshold)", use_container_width=True)
            _attention_section(model_path, img, r["params"]["imgsz"], info)
            return

    ids = [d.uid for d in sorted(weeds, key=lambda d: -d.conf)]
    pick = st.multiselect(f"{target_cls} detections to explain (top by confidence pre-selected)", ids, default=ids[: min(6, len(ids))])
    targets = [d for d in weeds if d.uid in pick]
    if not targets:
        return
    if st.button("Generate explanations", type="primary"):
        with st.spinner(f"Backpropagating through {len(layers)} layer(s) for {len(targets)} detection(s)…"):
            cam = ui.get_cam(model_path, layers)
            out = cam.explain(img, targets, r["params"]["imgsz"], method=method, iou_match=iou_match, per_target=True)
        st.session_state["cam_out"] = dict(key=id(r), out=out, uids=[d.uid for d in targets], alpha=alpha)

    co = st.session_state.get("cam_out")
    if not co or co["key"] != id(r) or co["uids"] != [d.uid for d in targets]:
        st.caption("Press **Generate explanations** to compute heatmaps.")
        _attention_section(model_path, img, r["params"]["imgsz"], info)
        return
    out = co["out"]
    combined = out["combined"]
    a, b = st.columns(2)
    a.image(draw_detections(img, targets), caption="Explained detections", use_container_width=True)
    ov = overlay_heatmap(draw_detections(img, targets, show_conf=False), combined, alpha)
    b.image(ov, caption=f"{'Grad-CAM' if out['method'] == 'gradcam' else 'HiResCAM'} — combined, layers {out['layers']}", use_container_width=True)
    b.download_button("Download combined heatmap (PNG)", to_png_bytes(ov), "weed_gradcam_combined.png", "image/png")

    st.subheader("Per-weed heatmaps")
    st.caption("Energy-in-box = share of the heatmap's total mass that falls inside that weed's box (higher → the evidence is localised on the weed).")
    cols = st.columns(3)
    rows = []
    for i, d in enumerate(targets):
        m = out["per_target"].get(d.uid)
        if m is None:
            continue
        e = energy_inside_box(m, d)
        rows.append({"weed_id": d.uid, "conf": round(d.conf, 3), "energy_in_box": round(e, 3), "anchors_matched": out["anchors"].get(d.uid)})
        pad = int(0.6 * max(d.w, d.h))
        H, W = img.shape[:2]
        x1, y1 = max(0, int(d.x1) - pad), max(0, int(d.y1) - pad)
        x2, y2 = min(W, int(d.x2) + pad), min(H, int(d.y2) + pad)
        crop = overlay_heatmap(draw_detections(img, [d], show_conf=False), m, alpha)[y1:y2, x1:x2]
        cols[i % 3].image(crop, caption=f"{d.uid} · conf {d.conf:.2f} · energy-in-box {e:.2f}", use_container_width=True)
    if rows:
        import pandas as pd

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    _attention_section(model_path, img, r["params"]["imgsz"], info)


def _attention_section(model_path, img, imgsz, info):
    st.subheader("RMCBAM attention maps")
    if not info.is_rmcbam:
        st.info("This checkpoint has no RMCBAM modules, so there are no attention maps to show (baseline YOLOv8).")
        return
    if not st.button("Extract RMCBAM attention for this image"):
        return
    with st.spinner("Running forward pass with attention capture…"):
        maps = rmcbam_attention_maps(ui.get_yolo(model_path).model, img, imgsz, ui.device())
    st.caption(
        "Spatial maps are the real sigmoid outputs of each block's multi-scale spatial attention (min–max normalised for display). "
        "γ is the learned residual gate: the block's output is x + γ·(attended x), so |γ|≈0 means that block barely changes the features."
    )
    cols = st.columns(3)
    for i, m in enumerate(maps):
        cols[i % 3].image(
            overlay_heatmap(img, m["spatial"], 0.5),
            caption=f"layer {m['layer']} · {m['feat_hw'][0]}×{m['feat_hw'][1]} · range {m['spatial_min']:.2f}–{m['spatial_max']:.2f} · mean|γ| {m['gamma_mean_abs']:.2e}",
            use_container_width=True,
        )
    sel = st.selectbox("Channel attention of layer", [m["layer"] for m in maps])
    m = next(x for x in maps if x["layer"] == sel)
    order = np.argsort(-m["channel"])
    fig = go.Figure(go.Bar(x=np.arange(len(order)), y=m["channel"][order], marker_color="#216B45", hovertemplate="channel %{customdata}<br>weight %{y:.3f}<extra></extra>", customdata=order))
    fig.update_layout(title=f"Layer {sel}: ECA channel weights (sorted)", xaxis_title="channel rank", yaxis_title="attention weight", height=300, bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

