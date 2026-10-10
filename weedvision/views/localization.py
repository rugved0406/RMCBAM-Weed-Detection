from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from weedvision import ui
from weedvision.analysis import split_weeds_crops
from weedvision.config import class_color_hex
from weedvision.inference import detections_df


def render():
    ui.hero("Weed Localization", "Weed IDs, pixel coordinates, confidence and box area on an interactive image-coordinate plot.")
    s = ui.settings()
    r = ui.pick_result("loc")
    if r is None:
        return
    ui.model_banner(r["model"])
    img, dets = r["image"], r["dets"]
    H, W = img.shape[:2]
    weeds, crops = split_weeds_crops(dets, s)
    c = st.columns([1, 1, 2])
    show_crops = c[0].toggle("Show crop detections", value=bool(crops))
    min_conf = c[1].slider("Min confidence", 0.0, 1.0, 0.0, 0.01)
    shown = [d for d in weeds + (crops if show_crops else []) if d.conf >= min_conf]
    ids = [d.uid for d in weeds if d.conf >= min_conf]
    hl = c[2].selectbox("Highlight weed", ["—"] + ids)
    if not weeds:
        st.info(f"No **{s.weed_class}** detections in this image.")

    fig = ui.image_figure(img)
    for d in shown:
        col = class_color_hex(d.cls_name)
        width = 4 if d.uid == hl else 2
        fig.add_shape(type="rect", x0=d.x1, y0=d.y1, x1=d.x2, y1=d.y2, line=dict(color=col, width=width), fillcolor=col, opacity=0.18 if d.uid == hl else 0.0, layer="above")
        fig.add_shape(type="rect", x0=d.x1, y0=d.y1, x1=d.x2, y1=d.y2, line=dict(color=col, width=width), layer="above")
    by_cls: dict[str, list] = {}
    for d in shown:
        by_cls.setdefault(d.cls_name, []).append(d)
    for name, ds in by_cls.items():
        fig.add_trace(
            go.Scatter(
                x=[d.cx for d in ds], y=[d.cy for d in ds], mode="markers+text" if len(ds) <= 25 else "markers",
                text=[d.uid for d in ds], textposition="top center", textfont=dict(color="#ffffff", size=11),
                marker=dict(size=10, color=class_color_hex(name), line=dict(color="#1a1a19", width=2)), name=name,
                customdata=[[d.uid, d.conf, d.x1, d.y1, d.x2, d.y2, d.area] for d in ds],
                hovertemplate="<b>%{customdata[0]}</b><br>centre (%{x:.0f}, %{y:.0f})<br>box (%{customdata[2]:.0f}, %{customdata[3]:.0f}) – (%{customdata[4]:.0f}, %{customdata[5]:.0f})<br>conf %{customdata[1]:.3f}<br>area %{customdata[6]:,.0f} px²<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(legend=dict(orientation="h", y=1.04, x=0), title=f"{len(ids)} weeds · image {W}×{H} px (origin top-left)")
    st.plotly_chart(fig, use_container_width=True)

    df = detections_df([d for d in shown], W, H)
    if not df.empty:
        st.dataframe(df, hide_index=True, use_container_width=True, height=min(420, 38 + 35 * len(df)))
        stem = r["name"].rsplit(".", 1)[0]
        st.download_button("Download coordinates CSV", df.to_csv(index=False), f"{stem}_weed_coordinates.csv", "text/csv")
        if len(weeds) >= 2:
            wd = df[df["cls_name"].str.lower() == s.weed_class.lower()]
            fig2 = go.Figure(go.Histogram2d(x=wd["cx"], y=wd["cy"], xbins=dict(start=0, end=W, size=W / 8), ybins=dict(start=0, end=H, size=H / 6), colorscale=ui.SEQ_SCALE, hovertemplate="x %{x}<br>y %{y}<br>%{z} weeds<extra></extra>"))
            fig2.update_yaxes(autorange="reversed", title="y (px)")
            fig2.update_layout(title="Weed spatial distribution (count per grid cell)", xaxis_title="x (px)", height=360)
            st.plotly_chart(fig2, use_container_width=True)
