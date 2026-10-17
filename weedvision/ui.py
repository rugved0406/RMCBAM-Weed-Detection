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
INK = "#17231D"          # Primary dark text
INK2 = "#59665E"         # Secondary text
MUTED = "#76867D"        # Muted captions
BORDER = "#DCE4DD"       # Subtle border
BORDER_SUBTLE = "#E5ECE6"
SURFACE = "#FFFFFF"      # Clean white cards
SURFACE_MUTED = "#F8FAF8"
PAGE_BG = "#F5F7F5"      # Light neutral background
ACCENT_GREEN = "#216B45" # Forest Green
ACCENT_LIGHT = "#E8F3EC" # Pale green active/hover background

pio.templates["weedvision"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK2, family="Inter, system-ui, -apple-system, sans-serif", size=12),
        title=dict(font=dict(color=INK, size=14), x=0, xanchor="left"),
        xaxis=dict(gridcolor="#E5ECE6", linecolor="#DCE4DD", zerolinecolor="#DCE4DD", tickcolor="#DCE4DD", title_font=dict(color=MUTED)),
        yaxis=dict(gridcolor="#E5ECE6", linecolor="#DCE4DD", zerolinecolor="#DCE4DD", tickcolor="#DCE4DD", title_font=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#DCE4DD", font=dict(color=INK)),
        margin=dict(l=40, r=16, t=40, b=36),
        colorway=["#dc2626", "#ea580c", "#059669", "#d97706", "#7c3aed", "#0d9488"],
    )
)
pio.templates.default = "weedvision"

SEQ_SCALE = [[0.0, "#F5F7F5"], [0.5, "#4ade80"], [1.0, "#216B45"]]

CSS = f"""
<style>
/* Page container & background */
.stApp {{
  background-color: {PAGE_BG} !important;
  color: {INK} !important;
}}

.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
  max-width: 1360px;
}}

/* Typography */
body, [class*="css"] {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: {INK} !important;
}}

h1, h2, h3, h4, h5, h6, p, span, label {{
  color: {INK} !important;
}}

p, span, .stCaption {{
  color: {INK2};
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
  width: 240px !important;
  min-width: 240px !important;
  background-color: {SURFACE} !important;
  border-right: 1px solid {BORDER} !important;
}}

section[data-testid="stSidebar"] * {{
  color: {INK} !important;
}}

section[data-testid="stSidebar"] .wv-brand {{
  font-weight: 700;
  font-size: 1.05rem;
  color: {ACCENT_GREEN} !important;
  letter-spacing: -0.02em;
  padding: 4px 0;
}}

/* Sidebar Radio Navigation (Pale green active button) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
  padding: 6px 10px !important;
  border-radius: 6px !important;
  margin-bottom: 2px !important;
  transition: all 0.15s ease !important;
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
  background-color: {ACCENT_LIGHT} !important;
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[aria-checked="true"] {{
  background-color: {ACCENT_LIGHT} !important;
  border-left: 3px solid {ACCENT_GREEN} !important;
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[aria-checked="true"] p {{
  color: {ACCENT_GREEN} !important;
  font-weight: 600 !important;
}}

/* Premium Header Banner */
.wv-hero {{
  padding: 16px 20px;
  border: 1px solid {BORDER};
  border-left: 4px solid {ACCENT_GREEN};
  border-radius: 8px;
  background: {SURFACE};
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}}
.wv-hero h1 {{ margin: 0; font-size: 1.45rem; color: {INK} !important; font-weight: 700; }}
.wv-hero p {{ margin: 4px 0 0; color: {INK2} !important; font-size: 0.88rem; }}

/* Precision Agricultural Metric Cards */
.wv-metric-card {{
  border: 1px solid {BORDER};
  border-radius: 8px;
  padding: 14px 16px;
  background: {SURFACE};
  box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  height: 100%;
}}
.wv-metric-card .k {{
  color: {MUTED} !important;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
}}
.wv-metric-card .v {{
  color: {ACCENT_GREEN} !important;
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1.2;
  margin-top: 3px;
}}
.wv-metric-card .s {{
  color: {INK2} !important;
  font-size: 0.78rem;
  margin-top: 3px;
}}

/* Content Section Card */
.wv-section-card {{
  border: 1px solid {BORDER};
  border-radius: 8px;
  padding: 16px 18px;
  background: {SURFACE};
  box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  height: 100%;
}}
.wv-section-card h3 {{
  margin-top: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: {INK} !important;
  border-bottom: 1px solid {BORDER_SUBTLE};
  padding-bottom: 8px;
  margin-bottom: 12px;
}}

/* Responsive Image Viewport with Fixed Max-Height (460px) */
.wv-image-frame {{
  max-height: 460px;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #181c1a;
  border-radius: 8px;
  border: 1px solid {BORDER};
  overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}}
.wv-image-frame img {{
  max-height: 460px !important;
  width: auto !important;
  max-width: 100% !important;
  object-fit: contain !important;
}}

/* Clean Severity Badge */
.wv-pill {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  border: 1px solid {BORDER};
  background: {SURFACE_MUTED};
}}

/* Check Line items */
.wv-check {{ font-size: .90rem; margin: 6px 0; color: {INK2} !important; display: flex; align-items: center; gap: 8px; }}
.wv-check b {{ color: {INK} !important; }}

/* Class Legend */
.wv-legend {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 4px 0; }}
.wv-legend span {{ display:inline-flex; align-items:center; color:{INK2} !important; font-size:.82rem; font-weight: 500; }}
.wv-legend i {{ width:10px; height:10px; border-radius:2px; display:inline-block; margin-right:5px; }}

/* Form controls, inputs, buttons */
button[kind="primary"] {{
  background-color: {ACCENT_GREEN} !important;
  border-color: {ACCENT_GREEN} !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
}}

button[kind="secondary"] {{
  background-color: {SURFACE} !important;
  border-color: {BORDER} !important;
  color: {INK} !important;
}}

div[data-testid="stFileUploader"] {{
  border: 1px dashed {BORDER};
  border-radius: 8px;
  background: {SURFACE};
  padding: 8px;
}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="wv-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(col, key: str, value: str, sub: str = ""):
    col.markdown(f'<div class="wv-metric-card"><div class="k">{key}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def section_card(title: str, content_html: str) -> str:
    return f'<div class="wv-section-card"><h3>{title}</h3><div>{content_html}</div></div>'


def severity_pill(sev: str) -> str:
    c = SEVERITY_COLORS.get(sev, MUTED)
    return f'<span class="wv-pill" style="border-color:{c}; color:{c}"><span style="color:{c}">{SEVERITY_ICONS.get(sev, "")}</span> {sev}</span>'





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
