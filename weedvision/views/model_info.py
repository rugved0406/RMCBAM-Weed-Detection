from __future__ import annotations

import pandas as pd
import streamlit as st

from weedvision import ui
from weedvision.rmcbam import DEFAULT_RMCBAM_LAYERS, eca_kernel_size

RMCBAM_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#DCE4DD", fontcolor="#17231D", fontname="Helvetica", fontsize=13];
  edge [color="#216B45", fontcolor="#59665E", fontname="Helvetica", fontsize=11];
  x [label="C2f output x\\nC×H×W"];
  gap [label="GAP + GMP"]; eca [label="shared Conv1d(k)\\nECA, no FC"]; mc [label="σ → Mc\\nC×1×1"];
  mul1 [label="x ⊙ Mc", shape=ellipse];
  pool [label="mean_c ‖ max_c\\n2×H×W"];
  b3 [label="Conv 3×3"]; b7 [label="Conv 7×7"]; b5 [label="Conv 5×5\\ndilation 2"];
  fuse [label="concat → Conv 1×1\\nσ → Ms (1×H×W)"];
  mul2 [label="(x⊙Mc) ⊙ Ms", shape=ellipse];
  gate [label="× γ (learned,\\ninit 0)"]; add [label="+", shape=circle];
  out [label="RMCBAM(x)"];
  x -> gap -> eca -> mc -> mul1; x -> mul1;
  mul1 -> pool; pool -> b3; pool -> b7; pool -> b5; b3 -> fuse; b7 -> fuse; b5 -> fuse;
  fuse -> mul2; mul1 -> mul2; mul2 -> gate -> add; x -> add [label="residual"]; add -> out;
}
"""


def render():
    ui.hero("Model Information", "RMCBAM design, insertion points, and the structure of the selected checkpoint.")
    st.subheader("RMCBAM — Residual Multi-Scale CBAM")
    st.graphviz_chart(RMCBAM_DOT, use_container_width=True)
    st.markdown(
        f"""
- **Channel attention (ECA-style):** global average and max pooling share one 1-D convolution across channels (kernel size adapts to channel count, e.g. k={eca_kernel_size(128)} for 128 ch, k={eca_kernel_size(512)} for 512 ch). No fully-connected bottleneck, so no dimensionality reduction.
- **Multi-scale spatial attention:** the channel-wise mean and max maps go through **3×3**, **7×7** and **dilated 5×5 (rate 2, 9×9 receptive field)** convolutions in parallel, then a 1×1 fusion and a sigmoid. Small kernels capture thin grass-type weeds; large and dilated kernels capture broad-leaf clumps and their context.
- **Residual attention gating:** `out = x + γ ⊙ (x ⊙ Mc ⊙ Ms)` with a per-channel γ initialised to **0**. A newly inserted block is an exact identity, so COCO-pretrained YOLOv8s weights load unchanged and attention is learned gradually (verified: γ moves off zero once optimiser steps begin).
- **Placement:** wraps the C2f blocks at layers **{list(DEFAULT_RMCBAM_LAYERS)}**: backbone P3/P4/P5 stages (4, 6, 8) and the three neck outputs feeding the Detect head (15, 18, 21). Configurable via `tools/train.py --layers`.
- **Checkpoint compatibility:** converted blocks keep C2f parameter names (`cv1`, `cv2`, `m`) and add `att.*`, so baseline weights transfer 1:1. Checkpoints pickle `weedvision.rmcbam.C2f_RMCBAM`, so `weedvision` must be importable when loading (the dashboard and tools handle this).
"""
    )

    st.subheader("Reference architectures (built live, nc=5, untrained)")
    st.caption("Parameter and GFLOP cost of adding RMCBAM to YOLOv8s. These describe architecture only, not accuracy.")
    st.dataframe(_reference_arch(), hide_index=True, use_container_width=True)

    path = ui.active_model()
    st.subheader("Selected checkpoint")
    if not path:
        st.info("No checkpoint selected.")
        return
    info = ui.get_info(path)
    for level, msg in info.compatibility_notes():
        {"ok": st.success, "warn": st.warning, "error": st.error}[level](msg)
    if not info.ok:
        return
    c = st.columns(4)
    ui.card(c[0], "Parameters", f"{info.n_params / 1e6:.2f} M", f"{info.size_mb} MB on disk")
    ui.card(c[1], "GFLOPs @640", "—" if info.gflops is None else f"{info.gflops:.1f}", "")
    ui.card(c[2], "Classes", str(info.nc), ", ".join(info.class_list[:5]) + ("…" if info.nc > 5 else ""))
    ui.card(c[3], "Saved", (info.date or "—")[:10], f"Ultralytics {info.ultralytics_version or '?'}")
    st.write("")
    if info.train_args:
        with st.expander("Training arguments stored in checkpoint"):
            st.json(info.train_args)
    if info.rmcbam_gate_mean_abs:
        st.markdown("**RMCBAM residual gates** (mean |γ| per block, learned values)")
        st.dataframe(pd.DataFrame([{"block": k, "mean |gamma|": f"{v:.3e}"} for k, v in info.rmcbam_gate_mean_abs.items()]), hide_index=True, use_container_width=True)
    with st.expander("Layer table"):
        from weedvision.models import architecture_table

        st.dataframe(pd.DataFrame(architecture_table(ui.get_yolo(path).model)), hide_index=True, use_container_width=True, height=420)


@st.cache_data(show_spinner="Building reference models…")
def _reference_arch() -> pd.DataFrame:
    from ultralytics.nn.tasks import DetectionModel

    from weedvision.models import model_gflops
    from weedvision.rmcbam import find_rmcbam, inject_rmcbam

    rows = []
    b = DetectionModel("yolov8s.yaml", nc=5, verbose=False)
    pb = sum(p.numel() for p in b.parameters())
    rows.append({"model": "YOLOv8s (baseline)", "params (M)": round(pb / 1e6, 3), "GFLOPs@640": round(model_gflops(b) or 0, 2), "RMCBAM blocks": 0, "added params": 0})
    r = DetectionModel("yolov8s.yaml", nc=5, verbose=False)
    inject_rmcbam(r)
    pr = sum(p.numel() for p in r.parameters())
    rows.append({"model": "YOLOv8s + RMCBAM", "params (M)": round(pr / 1e6, 3), "GFLOPs@640": round(model_gflops(r) or 0, 2), "RMCBAM blocks": len(find_rmcbam(r)), "added params": pr - pb})
    return pd.DataFrame(rows)
