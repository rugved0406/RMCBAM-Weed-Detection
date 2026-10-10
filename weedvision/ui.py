"""Shared Streamlit UI helpers: styling, cached loaders, session state, Plotly theme."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from weedvision.config import PROJECT_ROOT, SEVERITY_COLORS, SEVERITY_ICONS, Settings, class_color_hex, resolve_device

# ---------------------------------------------------------------------------------------------- theme
INK, INK2, MUTED, GRID, AXIS, SURFACE, PAGE = "#f2f1ec", "#c3c2b7", "#898781", "#2c2c2a", "#383835", "#1a1a19", "#0d0d0d"

pio.templates["weedvision"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK2, family="Inter, system-ui, sans-serif", size=13),
        title=dict(font=dict(color=INK, size=15), x=0, xanchor="left"),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS, title_font=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS, title_font=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="#262624", bordercolor=AXIS, font=dict(color=INK)),
        margin=dict(l=48, r=16, t=48, b=44),
        colorway=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    )
)
pio.templates.default = "weedvision"

# Single-hue sequential scale (dark surface -> blue -> pale blue) for heatmaps and matrices.
SEQ_SCALE = [[0.0, "#1a1a19"], [0.5, "#2a6fc4"], [1.0, "#cfe2fb"]]

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.wv-hero {{ padding: 18px 22px; border: 1px solid {AXIS}; border-radius: 10px;
  background: linear-gradient(135deg, #14201a 0%, {SURFACE} 60%); margin-bottom: 14px; }}
.wv-hero h1 {{ margin: 0; font-size: 1.7rem; color: {INK}; }}
.wv-hero p {{ margin: 4px 0 0; color: {INK2}; }}
.wv-card {{ border: 1px solid {AXIS}; border-radius: 10px; padding: 14px 16px; background: {SURFACE}; height: 100%; }}
.wv-card .k {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
.wv-card .v {{ color: {INK}; font-size: 1.6rem; font-weight: 650; line-height: 1.2; margin-top: 2px; }}
.wv-card .s {{ color: {INK2}; font-size: .82rem; margin-top: 2px; }}
.wv-pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600;
  border: 1px solid {AXIS}; color: {INK}; background: #222220; }}
.wv-check {{ font-size: .92rem; margin: 3px 0; color: {INK2}; }}
.wv-check b {{ color: {INK}; }}
.wv-legend span {{ display:inline-flex; align-items:center; margin-right:14px; color:{INK2}; font-size:.85rem; }}
.wv-legend i {{ width:11px; height:11px; border-radius:3px; display:inline-block; margin-right:6px; }}
section[data-testid="stSidebar"] .wv-brand {{ font-weight: 700; font-size: 1.05rem; color: {INK}; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="wv-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(col, key: str, value: str, sub: str = ""):
    col.markdown(f'<div class="wv-card"><div class="k">{key}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def severity_pill(sev: str) -> str:
    c = SEVERITY_COLORS.get(sev, MUTED)
    return f'<span class="wv-pill" style="border-color:{c}"><span style="color:{c}">{SEVERITY_ICONS.get(sev, "")}</span> {sev}</span>'


def class_legend(names: list[str]) -> str:
    items = "".join(f'<span><i style="background:{class_color_hex(n)}"></i>{n}</span>' for n in names)
    return f'<div class="wv-legend">{items}</div>'


def check_line(ok: bool | None, text: str) -> str:
    icon = "✅" if ok else ("⚠️" if ok is None else "❌")
    return f'<div class="wv-check">{icon} {text}</div>'


# ---------------------------------------------------------------------------------------------- state
def settings() -> Settings:
    if "settings" not in st.session_state:
        st.session_state["settings"] = Settings.load()
    return st.session_state["settings"]


def device() -> str:
    return resolve_device(settings().device)


def results() -> dict:
    """Detection results stored by image key: {key: {name, image, dets, speed, model, params}}."""
    return st.session_state.setdefault("results", {})


def file_key(name: str, data: bytes) -> str:
    return f"{name}:{hashlib.md5(data).hexdigest()[:10]}"


# ---------------------------------------------------------------------------------------------- cached loaders
def _mtime(p: str) -> float:
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource(show_spinner="Loading model…")
def _load_yolo(path: str, mtime: float):
    from weedvision.models import load_yolo

    return load_yolo(path)


def get_yolo(path: str):
    return _load_yolo(path, _mtime(path))


@st.cache_data(show_spinner="Inspecting checkpoint…")
def _inspect(path: str, mtime: float):
    from weedvision.models import inspect_checkpoint

    return inspect_checkpoint(path)


def get_info(path: str):
    return _inspect(path, _mtime(path))


@st.cache_resource(show_spinner="Preparing Grad-CAM model…")
def _cam(path: str, mtime: float, layers: tuple[int, ...], dev: str):
    from weedvision.explain import DetectorCAM

    return DetectorCAM(_load_yolo(path, mtime).model, list(layers), dev)


def get_cam(path: str, layers: list[int]):
    return _cam(path, _mtime(path), tuple(sorted(layers)), device())


def checkpoint_paths() -> list[str]:
    from weedvision.models import discover_checkpoints

    s = settings()
    return [str(p) for p in discover_checkpoints(s.models_dir, s.runs_dir)]


def short_name(path: str) -> str:
    p = Path(path)
    if p.parent.name == "weights":
        return f"{p.parent.parent.name}/{p.name}"
    return p.name


# ---------------------------------------------------------------------------------------------- sidebar model picker
def sidebar_model_picker() -> str | None:
    s = settings()
    paths = checkpoint_paths()
    st.sidebar.markdown("**Active model**")
    up = st.sidebar.file_uploader("Add checkpoint (.pt)", type=["pt"], key="ckpt_upload", label_visibility="collapsed")
    if up is not None:
        dst = Path(s.models_dir) / "uploaded" / up.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != up.size:
            dst.write_bytes(up.getvalue())
        if str(dst) not in paths:
            paths.append(str(dst))
        st.session_state["active_model"] = str(dst)
    if not paths:
        st.sidebar.info("No checkpoints found. Put `.pt` files in `models/` or upload one above.")
        st.session_state["active_model"] = None
        return None
    cur = st.session_state.get("active_model")
    default = next((i for i, p in enumerate(paths) if "weed_detection_best.pt" in p), next((i for i, p in enumerate(paths) if "rmcbam" in short_name(p).lower() and p.endswith("best.pt")), 0))
    idx = paths.index(cur) if cur in paths else default
    choice = st.sidebar.selectbox("Checkpoint", paths, index=idx, format_func=short_name, label_visibility="collapsed")
    st.session_state["active_model"] = choice
    info = get_info(choice)
    if info.ok and info.is_smoke:
        st.sidebar.error("SMOKE-TEST checkpoint: outputs are meaningless (demo only).")
    if not info.ok:
        st.sidebar.error("Cannot load this checkpoint.")
    elif not info.has_weed_class:
        st.sidebar.warning("No 'Weed' class: outputs are not weed detections.")
    else:
        st.sidebar.caption(("RMCBAM model" if info.is_rmcbam else "Baseline model") + f" · {info.nc} classes")
    return choice


def active_model() -> str | None:
    return st.session_state.get("active_model")


def model_banner(path: str):
    info = get_info(path)
    for level, msg in info.compatibility_notes():
        if level == "error":
            st.error(msg)
        elif level == "warn":
            st.warning(msg)


# ---------------------------------------------------------------------------------------------- image source
def pick_result(page_key: str):
    """Select an analysed image from the session, or upload one and run detection with the active model.

    Returns the result dict or None.
    """
    from weedvision.inference import load_image_rgb, run_detection

    res = results()
    model = active_model()
    c1, c2 = st.columns([2, 1])
    with c2:
        up = st.file_uploader("Analyse a new image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"], key=f"up_{page_key}")
    if up is not None and model:
        data = up.getvalue()
        k = file_key(up.name, data)
        if k not in res or res[k]["model"] != model:
            s = settings()
            img = load_image_rgb(data)
            with st.spinner("Running detection…"):
                dets, speed = run_detection(get_yolo(model), img, s.conf, s.iou, s.imgsz, device(), s.max_det)
            res[k] = dict(name=up.name, image=img, dets=dets, speed=speed, model=model, params=dict(conf=s.conf, iou=s.iou, imgsz=s.imgsz))
        st.session_state[f"sel_{page_key}"] = k
    elif up is not None and not model:
        st.warning("Select or upload a checkpoint in the sidebar first.")
    if not res:
        with c1:
            st.info("No analysed images yet. Upload one here, or use **Image Detection**.")
        return None
    keys = list(res)
    cur = st.session_state.get(f"sel_{page_key}")
    with c1:
        k = st.selectbox(
            "Analysed image", keys, index=keys.index(cur) if cur in keys else len(keys) - 1,
            format_func=lambda x: f"{res[x]['name']} — {short_name(res[x]['model'])}", key=f"selbox_{page_key}",
        )
    st.session_state[f"sel_{page_key}"] = k
    return res[k]


def image_figure(img: np.ndarray, height: int | None = None) -> go.Figure:
    """Plotly figure with the image as background in pixel coordinates (y down)."""
    import base64
    import io

    from PIL import Image

    H, W = img.shape[:2]
    scale = min(1.0, 1600 / max(H, W))
    im = Image.fromarray(img)
    if scale < 1:
        im = im.resize((int(W * scale), int(H * scale)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    fig = go.Figure()
    fig.add_layout_image(dict(source=src, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H, sizing="stretch", layer="below"))
    fig.update_xaxes(range=[0, W], showgrid=False, title="x (px)", constrain="domain")
    fig.update_yaxes(range=[H, 0], showgrid=False, title="y (px)", scaleanchor="x", scaleratio=1)
    fig.update_layout(height=height or int(min(760, 900 * H / W)), margin=dict(l=50, r=10, t=30, b=40))
    return fig
