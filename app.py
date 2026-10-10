"""RMCBAM Weed & Crop Detection Dashboard — run with:  streamlit run app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="RMCBAM Weed Detection", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

import weedvision  # noqa: E402,F401  (registers RMCBAM for checkpoint unpickling)
from weedvision import ui  # noqa: E402
from weedvision.views import (  # noqa: E402
    analysis_page,
    comparison_page,
    dataset_page,
    detection,
    explainability,
    localization,
    model_info,
    overview,
    settings_page,
)

PAGES = {
    "Dashboard Overview": ("🏠", overview.render),
    "Image Detection": ("🔎", detection.render),
    "Weed Explainability": ("🔥", explainability.render),
    "Weed Analysis": ("📊", analysis_page.render),
    "Weed Localization": ("📍", localization.render),
    "Model Comparison": ("⚖️", comparison_page.render),
    "Dataset Analytics": ("🗂️", dataset_page.render),
    "Model Information": ("🧠", model_info.render),
    "Settings": ("⚙️", settings_page.render),
}


def main():
    ui.inject_css()
    ui.settings()
    st.sidebar.markdown('<div class="wv-brand">🌿 WeedVision · RMCBAM</div>', unsafe_allow_html=True)
    st.sidebar.caption("YOLOv8s + Residual Multi-Scale CBAM")
    if ui.settings().demo_mode:
        st.sidebar.warning("DEMO MODE: smoke-test models on partial labels. Use to explore the UI, not for results.")
    page = st.sidebar.radio("Navigate", list(PAGES), format_func=lambda p: f"{PAGES[p][0]}  {p}", label_visibility="collapsed", key="nav")
    st.sidebar.divider()
    ui.sidebar_model_picker()
    st.sidebar.divider()
    st.sidebar.caption(f"Device: **{ui.device()}** · conf {ui.settings().conf} · IoU {ui.settings().iou}")
    try:
        PAGES[page][1]()
    except Exception as e:  # keep the app alive; show the real error
        st.error(f"{type(e).__name__}: {e}")
        st.exception(e)


main()
