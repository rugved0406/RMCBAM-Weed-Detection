from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from weedvision import ui
from weedvision.analysis import crop_association, split_weeds_crops, weed_metrics
from weedvision.config import SEVERITY_COLORS, class_color_hex


def render():
    ui.hero("Weed Analysis", "Count, density, area, infestation, confidence distribution, severity and crop-wise association.")
    s = ui.settings()
    r = ui.pick_result("analysis")
    if r is None:
        return
    ui.model_banner(r["model"])
    img, dets = r["image"], r["dets"]
    H, W = img.shape[:2]
    weeds, crops = split_weeds_crops(dets, s)
    m = weed_metrics(weeds, W, H, s)

    c = st.columns(5)
    ui.card(c[0], "Weed count", str(m["weed_count"]), f"{len(crops)} crops detected")
    if m["density_per_m2"] is not None:
        ui.card(c[1], "Density", f"{m['density_per_m2']:.2f}/m²", f"GSD {s.gsd_cm_per_px} cm/px · {m['ground_area_m2']:.1f} m²")
    else:
        ui.card(c[1], "Density", f"{m['density_per_mpx']:.2f}/MP", "per megapixel (set GSD for /m²)")
    ui.card(c[2], "Infestation", f"{m['infestation_pct']:.2f}%", "union of weed boxes / image area")
    ui.card(c[3], "Weed box area", f"{m['union_box_area_px'] / 1e3:.1f}k px", f"sum {m['sum_box_area_px'] / 1e3:.1f}k · mean {m['mean_box_area_px'] / 1e3:.1f}k")
    ui.card(c[4], "Mean confidence", "—" if m["mean_conf"] is None else f"{m['mean_conf']:.3f}", "" if m["mean_conf"] is None else f"min {m['min_conf']:.2f} · max {m['max_conf']:.2f}")
    st.write("")
    st.markdown(
        f"Severity: {ui.severity_pill(m['severity'])} &nbsp; <span style='color:#59665E'>thresholds: Low &lt; {s.severity_low}% ≤ Moderate &lt; {s.severity_moderate}% ≤ High &lt; {s.severity_high}% ≤ Severe (change in Settings)</span>",
        unsafe_allow_html=True,
    )
    st.caption("Infestation % is a box-based estimate: weed bounding boxes include some soil/crop pixels, so it is an upper bound on true weed cover.")

    if weeds:
        a, b = st.columns(2)
        conf = np.array([d.conf for d in weeds])
        fig = go.Figure(go.Histogram(x=conf, xbins=dict(start=0, end=1, size=0.05), marker_color=class_color_hex(s.weed_class), marker_line=dict(color="#DCE4DD", width=1), hovertemplate="conf %{x}<br>%{y} weeds<extra></extra>"))
        fig.update_layout(title="Weed confidence distribution", xaxis_title="confidence", yaxis_title="weeds", height=320, xaxis_range=[0, 1])
        a.plotly_chart(fig, use_container_width=True)
        area_pct = np.array([100 * d.area / (W * H) for d in weeds])
        fig = go.Figure(go.Histogram(x=area_pct, nbinsx=20, marker_color=class_color_hex(s.weed_class), marker_line=dict(color="#DCE4DD", width=1), hovertemplate="%{x}% of image<br>%{y} weeds<extra></extra>"))
        fig.update_layout(title="Weed box size (% of image)", xaxis_title="box area (% of image)", yaxis_title="weeds", height=320)
        b.plotly_chart(fig, use_container_width=True)

    st.subheader("Crop-wise weed association")
    if not crops:
        st.info(
            "No crop detections (Tomato / Rose / Marigold / Cauliflower) in this image, so weeds cannot be associated with crops. "
            "This needs a model trained with crop classes."
        )
    elif not weeds:
        st.info("No weeds detected — nothing to associate.")
    else:
        per_weed, per_crop = crop_association(weeds, crops, W, H, s)
        st.caption(f"Each weed is linked to the nearest crop box (weed centre → crop box edge) within {s.proximity_frac:.0%} of the image diagonal ({s.proximity_frac * np.hypot(W, H):.0f} px).")
        a, b = st.columns([1, 1])
        pc = per_crop[per_crop["crop_instances"] > 0]
        fig = go.Figure(go.Bar(x=pc["crop_class"], y=pc["associated_weeds"], marker_color=[class_color_hex(c) for c in pc["crop_class"]], text=pc["associated_weeds"], textposition="outside", textfont_color="#17231D", hovertemplate="%{x}: %{y} weeds<extra></extra>"))
        unassoc = int((per_weed["associated_class"] == "Unassociated").sum())
        fig.update_layout(title=f"Weeds associated per crop ({unassoc} unassociated)", yaxis_title="weeds", height=320, bargap=0.4)
        a.plotly_chart(fig, use_container_width=True)
        b.dataframe(per_crop, hide_index=True, use_container_width=True)
        st.dataframe(per_weed, hide_index=True, use_container_width=True)
        st.download_button("Download association CSV", per_weed.to_csv(index=False), "weed_crop_association.csv", "text/csv")

    res = ui.results()
    if len(res) > 1:
        st.subheader("All analysed images")
        rows = []
        for rr in res.values():
            hh, ww = rr["image"].shape[:2]
            w2, c2 = split_weeds_crops(rr["dets"], s)
            mm = weed_metrics(w2, ww, hh, s)
            rows.append({"image": rr["name"], "model": ui.short_name(rr["model"]), "weeds": mm["weed_count"], "crops": len(c2), "density_per_MP": round(mm["density_per_mpx"], 2), "infestation_%": round(mm["infestation_pct"], 2), "severity": mm["severity"]})
        df = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(x=df["image"], y=df["infestation_%"], marker_color=[SEVERITY_COLORS[x] for x in df["severity"]], text=[f"{v:.1f}% {sv}" for v, sv in zip(df["infestation_%"], df["severity"])], textposition="outside", textfont_color="#17231D"))
        fig.update_layout(title="Infestation by image (colour = severity)", yaxis_title="infestation %", height=340, bargap=0.4)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("Download batch summary CSV", df.to_csv(index=False), "weed_analysis_summary.csv", "text/csv")
