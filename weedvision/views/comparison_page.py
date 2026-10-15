from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from weedvision import ui
from weedvision.comparison import METRIC_COLS, best_epoch_metrics, find_runs, load_results, run_plots, validate

VARIANT_COLORS = {"Baseline": "#76867D", "RMCBAM": "#216B45"}


def render():
    ui.hero("Model Comparison", "YOLOv8s baseline vs YOLOv8s + RMCBAM — from training logs, live validation, and measured efficiency.")
    s = ui.settings()
    paths = ui.checkpoint_paths()
    runs = find_runs(s.runs_dir, s.models_dir)

    tab_logs, tab_val, tab_eff = st.tabs(["Training runs (results.csv)", "Live validation", "Efficiency"])

    with tab_logs:
        if not runs:
            st.info(f"No Ultralytics run folders with `results.csv` under `{s.runs_dir}` or `{s.models_dir}`. "
                    "Copy your `runs/detect/<name>/` folders there (or point Settings → runs directory at them).")
        else:
            c = st.columns(2)
            rb = c[0].selectbox("Baseline run", runs, index=_guess(runs, rmcbam=False), format_func=lambda p: str(Path(p).name))
            rr = c[1].selectbox("RMCBAM run", runs, index=_guess(runs, rmcbam=True), format_func=lambda p: str(Path(p).name))
            dfs = {"Baseline": load_results(rb), "RMCBAM": load_results(rr)}
            if "smoke" in (str(rb) + str(rr)).lower():
                st.error("These are PIPELINE SMOKE-TEST runs (few epochs, 56 partially-labelled Weed images). "
                         "The numbers below are real outputs of those runs but say nothing about RMCBAM vs baseline.")
            rows = []
            for k, df in dfs.items():
                m = best_epoch_metrics(df) if df is not None else {}
                rows.append({"model": k, "run": Path(rb if k == "Baseline" else rr).name, **{x: (round(v, 4) if isinstance(v, float) else v) for x, v in m.items()}})
            tab = pd.DataFrame(rows)
            if len(tab.columns) <= 2:
                st.warning("results.csv files found but they contain no validation metric columns.")
            else:
                st.markdown("**Metrics at best epoch** (fitness = 0.1·mAP50 + 0.9·mAP50-95, as Ultralytics selects `best.pt`)")
                st.dataframe(tab, hide_index=True, use_container_width=True)
                _bar(tab)
                metric = st.selectbox("Metric over epochs", list(METRIC_COLS))
                fig = go.Figure()
                for k, df in dfs.items():
                    if df is not None and METRIC_COLS[metric] in df:
                        fig.add_trace(go.Scatter(x=df["epoch"], y=df[METRIC_COLS[metric]], name=k, mode="lines", line=dict(color=VARIANT_COLORS[k], width=2), hovertemplate=f"{k}<br>epoch %{{x}}<br>{metric} %{{y:.4f}}<extra></extra>"))
                fig.update_layout(title=f"{metric} per epoch", xaxis_title="epoch", yaxis_title=metric, height=360, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Plots saved by Ultralytics**")
            pb, pr = run_plots(rb), run_plots(rr)
            titles = [t for t in pb if t in pr] + [t for t in pb if t not in pr] + [t for t in pr if t not in pb]
            if not titles:
                st.caption("No plot images in these run folders.")
            for t in dict.fromkeys(titles):
                st.markdown(f"*{t}*")
                a, b = st.columns(2)
                (a.image(str(pb[t]), caption="Baseline", use_container_width=True) if t in pb else a.caption("Baseline: not available"))
                (b.image(str(pr[t]), caption="RMCBAM", use_container_width=True) if t in pr else b.caption("RMCBAM: not available"))

    with tab_val:
        st.caption("Runs Ultralytics validation on your labelled split. Results are computed now, on this machine, from your data.")
        if not paths:
            st.info("No checkpoints found.")
        else:
            c = st.columns([2, 2, 2, 1])
            mb = c[0].selectbox("Baseline checkpoint", paths, index=_guess(paths, rmcbam=False), format_func=ui.short_name)
            mr = c[1].selectbox("RMCBAM checkpoint", paths, index=_guess(paths, rmcbam=True), format_func=ui.short_name)
            yamls = _yamls(s)
            data_yaml = c[2].selectbox("data.yaml", yamls) if yamls else c[2].text_input("data.yaml path")
            split = c[3].selectbox("Split", ["val", "test", "train"])
            if "partial" in str(data_yaml).lower() or "smoke" in (mb + mr).lower():
                st.warning("Smoke-test checkpoint and/or partial dataset selected: validation runs for real, but the metrics are not meaningful.")
            if st.button("Run validation on both", type="primary", disabled=not data_yaml):
                out = {}
                for k, p in (("Baseline", mb), ("RMCBAM", mr)):
                    info = ui.get_info(p)
                    if not info.ok:
                        st.error(f"{k}: {info.error}")
                        continue
                    with st.spinner(f"Validating {k}…"):
                        try:
                            out[k] = validate(ui.get_yolo(p), data_yaml, s.imgsz, ui.device(), split, project=str(Path(s.runs_dir) / "val"))
                        except Exception as e:
                            st.error(f"{k}: validation failed — {type(e).__name__}: {e}")
                st.session_state["val_out"] = out
            out = st.session_state.get("val_out")
            if out:
                _show_validation(out)

    with tab_eff:
        st.caption("Parameters and GFLOPs are read from the checkpoints; latency is measured live on this machine's device.")
        if not paths:
            st.info("No checkpoints found.")
        else:
            sel = st.multiselect("Checkpoints", paths, default=paths[: min(2, len(paths))], format_func=ui.short_name)
            runs_n = st.slider("Timed runs", 3, 50, 10)
            rows = []
            for p in sel:
                i = ui.get_info(p)
                rows.append({"checkpoint": ui.short_name(p), "RMCBAM": i.is_rmcbam, "classes": i.nc, "params (M)": round(i.n_params / 1e6, 3), "GFLOPs@640": None if i.gflops is None else round(i.gflops, 2), "size (MB)": i.size_mb})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if st.button("Measure latency") and sel:
                from weedvision.models import benchmark_latency

                lat = []
                for p in sel:
                    with st.spinner(f"Timing {ui.short_name(p)}…"):
                        b = benchmark_latency(ui.get_yolo(p), s.imgsz, ui.device(), runs=runs_n)
                    lat.append({"checkpoint": ui.short_name(p), **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in b.items()}})
                st.dataframe(pd.DataFrame(lat), hide_index=True, use_container_width=True)


def _guess(items, rmcbam: bool) -> int:
    for i, p in enumerate(items):
        tail = "/".join(Path(p).parts[-3:]).lower()
        if ("rmcbam" in tail) == rmcbam:
            return i
    return 0


@st.cache_data(ttl=60, show_spinner=False)
def _yamls_cached(dataset_dir: str, runs_dir: str) -> list[str]:
    out = []
    for d in (dataset_dir, runs_dir):
        if Path(d).exists():
            out += [str(p) for p in Path(d).rglob("data.yaml")]
    return sorted(set(out))


def _yamls(s) -> list[str]:
    return _yamls_cached(s.dataset_dir, s.runs_dir)


def _bar(tab: pd.DataFrame, title: str = "Best-epoch metrics"):
    mets = [m for m in ["precision", "recall", "F1", "mAP50", "mAP50-95"] if m in tab]
    fig = go.Figure()
    for _, row in tab.iterrows():
        fig.add_trace(go.Bar(name=row["model"], x=mets, y=[row[m] for m in mets], marker_color=VARIANT_COLORS.get(row["model"]), text=[f"{row[m]:.3f}" if pd.notna(row[m]) else "" for m in mets], textposition="outside", textfont_color="#17231D", marker_line=dict(color="#DCE4DD", width=1)))
    fig.update_layout(barmode="group", height=340, yaxis_range=[0, 1.08], yaxis_title="score", title=title, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)


def _show_validation(out: dict):
    tab = pd.DataFrame([{"model": k, **{m: round(v[m], 4) for m in ["precision", "recall", "F1", "mAP50", "mAP50-95"]}, "inference ms/img": round(v["speed_ms"].get("inference", float("nan")), 2)} for k, v in out.items()])
    st.dataframe(tab, hide_index=True, use_container_width=True)
    _bar(tab, "Validation metrics (computed now)")
    pcs = []
    for k, v in out.items():
        for row in v["per_class"]:
            pcs.append({"model": k, **{a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}})
    if pcs:
        st.markdown("**Per-class results**")
        st.dataframe(pd.DataFrame(pcs), hide_index=True, use_container_width=True)
    for key, title in (("Precision-Recall(B)", "Precision–Recall"), ("F1-Confidence(B)", "F1–Confidence")):
        fig = go.Figure()
        found = False
        for k, v in out.items():
            cv = v["curves"].get(key)
            if not cv:
                continue
            y = np.asarray(cv["y"])
            ym = y.mean(0) if y.ndim == 2 else y
            fig.add_trace(go.Scatter(x=cv["x"], y=ym, name=f"{k} (all classes)", mode="lines", line=dict(color=VARIANT_COLORS[k], width=2)))
            fig.update_layout(xaxis_title=cv["xlabel"], yaxis_title=cv["ylabel"])
            found = True
        if found:
            fig.update_layout(title=title, height=360, yaxis_range=[0, 1.02], xaxis_range=[0, 1], hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
    cms = [(k, v) for k, v in out.items() if v.get("confusion_matrix") is not None]
    if cms:
        cols = st.columns(len(cms))
        for col, (k, v) in zip(cols, cms):
            cm = np.asarray(v["confusion_matrix"])
            names = [v["names"][i] for i in sorted(v["names"])] + ["background"]
            n = min(len(names), cm.shape[0])
            colsum = cm.sum(0, keepdims=True)
            norm = np.divide(cm, colsum, out=np.zeros_like(cm, dtype=float), where=colsum > 0)
            fig = go.Figure(go.Heatmap(z=norm[:n, :n], x=names[:n], y=names[:n], colorscale=ui.SEQ_SCALE, zmin=0, zmax=1, text=cm[:n, :n].astype(int), texttemplate="%{text}", hovertemplate="true %{x}<br>pred %{y}<br>%{z:.2f} of column (n=%{text})<extra></extra>"))
            fig.update_layout(title=f"{k}: confusion matrix (column-normalised, counts shown)", xaxis_title="True", yaxis_title="Predicted", height=420)
            fig.update_yaxes(autorange="reversed")
            col.plotly_chart(fig, use_container_width=True)
    for k, v in out.items():
        st.caption(f"{k}: plots saved to `{v['save_dir']}`")
