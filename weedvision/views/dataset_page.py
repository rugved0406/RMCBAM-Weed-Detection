from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from weedvision import ui
from weedvision.config import CLASS_NAMES, PROJECT_ROOT, class_color_hex
from weedvision.dataset import (
    convert_labelme_dataset,
    normalise_label,
    quality_table,
    scan_folder,
    validate_yolo_dataset,
)


@st.cache_data(show_spinner=False)
def _scan(root: str, _stamp: float):
    return scan_folder(root)


@st.cache_data(show_spinner=False)
def _quality(paths: tuple[str, ...]):
    return quality_table(list(paths))


@st.cache_data(show_spinner=False)
def _labelme_stats(paths: tuple[str, ...]):
    rows, raw = [], Counter()
    for p in paths:
        try:
            js = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        W, H = js.get("imageWidth") or 1, js.get("imageHeight") or 1
        for sh in js.get("shapes", []):
            lab = str(sh.get("label", ""))
            raw[lab] += 1
            pts = sh.get("points") or []
            if len(pts) >= 2:
                xs, ys = [q[0] for q in pts], [q[1] for q in pts]
                rows.append({"json": Path(p).name, "label": lab, "class": normalise_label(lab) or f"unknown: {lab}", "shape": sh.get("shape_type", "polygon"), "w_frac": (max(xs) - min(xs)) / W, "h_frac": (max(ys) - min(ys)) / H})
    return pd.DataFrame(rows), dict(raw)


def render():
    ui.hero("Dataset Analytics", "Inventory, annotation coverage, class distribution, label validation and image quality — computed from your files.")
    s = ui.settings()
    root = st.text_input("Dataset folder", s.dataset_dir, help="Any folder: raw images + LabelMe JSONs, or a YOLO dataset with data.yaml.")
    if not Path(root).exists():
        st.error(f"Folder not found: `{root}`. Put your images in `data/` or change the path.")
        return
    from weedvision.dataset import _walk

    stamp = max((p.stat().st_mtime for p in _walk(Path(root))), default=0)
    with st.spinner("Scanning…"):
        sc = _scan(root, stamp)

    c = st.columns(4)
    ui.card(c[0], "Images", str(sc["n_images"]), f"{len(sc['images_by_folder'])} folder(s)")
    ui.card(c[1], "LabelMe JSON", str(sc["n_labelme_json"]), "files with 'shapes'")
    ui.card(c[2], "YOLO label files", str(sc["n_yolo_txt"]), "under */labels/*")
    ui.card(c[3], "YAML files", str(len(sc["yaml_files"])), ", ".join(Path(y).name for y in sc["yaml_files"][:3]))
    st.write("")
    with st.expander("Images per folder"):
        st.dataframe(pd.DataFrame([{"folder": k, "images": v} for k, v in sc["images_by_folder"].items()]), hide_index=True, use_container_width=True)

    t1, t2, t3, t4, t5 = st.tabs(["Annotations", "YOLO validation", "Image quality", "Convert LabelMe → YOLO", "Recovered patch boxes"])

    with t1:
        if sc["n_labelme_json"] == 0:
            st.warning(
                f"**No LabelMe annotations found** in `{root}`. Class distribution and annotation statistics cannot be computed. "
                "All images are currently unlabelled for every class: " + ", ".join(CLASS_NAMES) + "."
            )
        else:
            df, raw = _labelme_stats(tuple(sc["labelme_json"]))
            stems = {Path(p).stem for p in sc["labelme_json"]}
            missing = [p for p in sc["images"] if Path(p).stem not in stems]
            counts = df["class"].value_counts() if len(df) else pd.Series(dtype=int)
            order = CLASS_NAMES + [k for k in counts.index if k not in CLASS_NAMES]
            vals = [int(counts.get(k, 0)) for k in order]
            fig = go.Figure(go.Bar(x=order, y=vals, marker_color=[class_color_hex(k) for k in order], text=vals, textposition="outside", textfont_color="#17231D", marker_line=dict(color="#DCE4DD", width=1)))
            fig.update_layout(title="Annotated instances per class", yaxis_title="instances", height=340, bargap=0.35)
            st.plotly_chart(fig, use_container_width=True)
            absent = [k for k in CLASS_NAMES if counts.get(k, 0) == 0]
            if absent:
                st.error("Classes with **no annotations**: " + ", ".join(absent) + ". A model cannot learn these from this dataset.")
            unk = {k: v for k, v in raw.items() if normalise_label(k) is None}
            if unk:
                st.warning(f"Unrecognised labels (skipped by the converter): {unk}")
            a, b = st.columns(2)
            a.metric("Images without a JSON", f"{len(missing)} / {sc['n_images']}")
            b.metric("Shapes", f"{len(df)}", f"{df['shape'].value_counts().to_dict() if len(df) else {}}")
            if len(df):
                fig = go.Figure()
                for k in [x for x in order if x in set(df["class"])]:
                    d = df[df["class"] == k]
                    fig.add_trace(go.Scatter(x=d["w_frac"], y=d["h_frac"], mode="markers", name=k, marker=dict(size=8, color=class_color_hex(k), line=dict(color="#FFFFFF", width=1.5)), hovertemplate=f"{k}<br>w %{{x:.3f}} · h %{{y:.3f}} of image<extra></extra>"))
                fig.update_layout(title="Box size (fraction of image)", xaxis_title="width / image width", yaxis_title="height / image height", height=380)
                st.plotly_chart(fig, use_container_width=True)
            if missing:
                with st.expander(f"Images without annotations ({len(missing)})"):
                    st.dataframe(pd.DataFrame({"image": [str(Path(p).relative_to(root)) for p in missing]}), hide_index=True, use_container_width=True)

    with t2:
        ys = [y for y in sc["yaml_files"] if Path(y).name == "data.yaml"] or sc["yaml_files"]
        if not ys:
            st.info("No `data.yaml` in this folder. Create one with the converter tab.")
        else:
            y = st.selectbox("data.yaml", ys)
            try:
                v = validate_yolo_dataset(y)
            except Exception as e:
                st.error(f"Could not read {y}: {e}")
                v = None
            if v:
                st.dataframe(pd.DataFrame(v["splits"]).T, use_container_width=True)
                if v["missing_classes"]:
                    st.error("No instances for: " + ", ".join(v["missing_classes"]))
                st.markdown(f"**Issues:** {len(v['issues'])}")
                if v["issues"]:
                    st.dataframe(pd.DataFrame({"issue": v["issues"][:500]}), hide_index=True, use_container_width=True)
                if v["instances"]:
                    names = v["names"]
                    fig = go.Figure(go.Bar(x=names, y=[v["instances"].get(n, 0) for n in names], marker_color=[class_color_hex(n) for n in names], text=[v["instances"].get(n, 0) for n in names], textposition="outside", textfont_color="#17231D", marker_line=dict(color="#DCE4DD", width=1)))
                    fig.update_layout(title="Instances per class (all splits)", height=320, bargap=0.35)
                    st.plotly_chart(fig, use_container_width=True)

    with t3:
        n = len(sc["images"])
        lim = st.slider("Images to analyse", 1, max(1, n), min(n, 300)) if n > 1 else n
        if n and st.button("Compute image quality statistics"):
            with st.spinner(f"Reading {lim} images…"):
                st.session_state["qdf"] = _quality(tuple(sc["images"][:lim]))
        q = st.session_state.get("qdf")
        if q is not None and len(q):
            q = q[q["readable"]]
            c = st.columns(4)
            ui.card(c[0], "Readable", f"{len(q)}", "images analysed")
            ui.card(c[1], "Likely augmented", f"{int(q['likely_augmented'].sum())}", "black padding from rotation/warp > 2%")
            ui.card(c[2], "Near-duplicate groups", f"{int((q.groupby('dup_group').size() > 1).sum())}", f"{int((q['dup_count'] > 1).sum())} images share an 8×8 hash")
            ui.card(c[3], "Resolutions", f"{q.groupby(['width', 'height']).ngroups}", f"median {int(q['width'].median())}×{int(q['height'].median())}")
            st.write("")
            a, b = st.columns(2)
            res = q.groupby(["width", "height"]).size().reset_index(name="n").sort_values("n", ascending=False)
            fig = go.Figure(go.Scatter(x=res["width"], y=res["height"], mode="markers", marker=dict(size=(res["n"] ** 0.5) * 4 + 6, color="#216B45", line=dict(color="#FFFFFF", width=1.5)), customdata=res["n"], hovertemplate="%{x}×%{y}<br>%{customdata} images<extra></extra>"))
            fig.update_layout(title="Image resolutions (marker area ∝ count)", xaxis_title="width (px)", yaxis_title="height (px)", height=340)
            a.plotly_chart(fig, use_container_width=True)
            fig = go.Figure(go.Histogram(x=q["black_border_frac"] * 100, nbinsx=30, marker_color="#216B45", marker_line=dict(color="#DCE4DD", width=1)))
            fig.update_layout(title="Black padding per image (% of pixels)", xaxis_title="% black padding", yaxis_title="images", height=340)
            b.plotly_chart(fig, use_container_width=True)
            a, b = st.columns(2)
            fig = go.Figure(go.Histogram(x=q["sharpness_lapvar"], nbinsx=30, marker_color="#216B45", marker_line=dict(color="#DCE4DD", width=1)))
            fig.update_layout(title="Sharpness (Laplacian variance, higher = sharper)", xaxis_title="variance", yaxis_title="images", height=320)
            a.plotly_chart(fig, use_container_width=True)
            fig = go.Figure(go.Histogram(x=q["green_frac"] * 100, nbinsx=30, marker_color="#216B45", marker_line=dict(color="#DCE4DD", width=1)))
            fig.update_layout(title="Vegetation cover proxy (% green pixels, HSV)", xaxis_title="% green", yaxis_title="images", height=320)
            b.plotly_chart(fig, use_container_width=True)
            if q["likely_augmented"].mean() > 0.2:
                st.warning(
                    f"{q['likely_augmented'].mean():.0%} of images show rotation/warp padding — they appear to be augmented copies. "
                    "If augmented copies of one scene land in both train and val, validation metrics will be inflated. "
                    "Split by original scene, and let Ultralytics do augmentation at training time."
                )
            st.dataframe(q.drop(columns=["path", "ahash"]), hide_index=True, use_container_width=True, height=320)
            st.download_button("Download quality table CSV", q.drop(columns=["ahash"]).to_csv(index=False), "image_quality.csv", "text/csv")

    with t4:
        st.caption("Copies images into a new YOLO dataset folder and writes labels + data.yaml. Source files are never modified.")
        c = st.columns(2)
        src = c[0].text_input("Images folder", root)
        jdir = c[1].text_input("JSON folder (blank = next to images)", "")
        c = st.columns(3)
        out = c[0].text_input("Output folder", str(PROJECT_ROOT / "data" / "yolo"))
        val = c[1].slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
        inc = c[2].toggle("Include unlabelled images as background", False)
        if st.button("Convert", type="primary"):
            try:
                rep = convert_labelme_dataset(src, out, jdir or None, CLASS_NAMES, val, 42, inc)
                d = rep.to_dict()
                st.success(f"Wrote {rep.train_images} train / {rep.val_images} val images to `{out}` (+ data.yaml).")
                if rep.missing_classes:
                    st.error("No annotations for: " + ", ".join(rep.missing_classes))
                if rep.unknown_labels:
                    st.warning(f"Unrecognised labels skipped: {rep.unknown_labels}")
                st.json({k: (v if not isinstance(v, list) or len(v) < 10 else f"{len(v)} items") for k, v in d.items()})
            except Exception as e:
                st.error(f"Conversion failed: {type(e).__name__}: {e}")

    with t5:
        rec = PROJECT_ROOT / "data" / "recovered" / "recovered_boxes.csv"
        st.caption(
            "Boxes recovered by locating each patch from `Crops/Weed_Crops` and `Crops/Crop_Flower_Crops` inside its source image "
            "(normalised cross-correlation). Generated by `tools/recover_patch_boxes.py`."
        )
        if not rec.exists():
            st.info("No recovered boxes yet. Run `python tools/recover_patch_boxes.py --root \"<folder with Origanal/ and Crops/>\"`.")
        else:
            df = pd.read_csv(rec)
            c = st.columns(4)
            ui.card(c[0], "Patches", str(len(df)), ", ".join(f"{k}: {v}" for k, v in df["label"].value_counts().items()))
            ui.card(c[1], "Matched", f"{int(df['accepted'].sum())}", f"NCC ≥ 0.9")
            ui.card(c[2], "Match score", f"{df['score'].min():.3f}", "minimum NCC (1.0 = exact)")
            ui.card(c[3], "Source images", str(df["source"].nunique()), "with ≥1 recovered box")
            st.write("")
            st.dataframe(df.drop(columns=[c for c in ("patch_path", "source_path") if c in df]), hide_index=True, use_container_width=True, height=260)

            st.markdown("#### Visual Annotation Reviewer")
            src_imgs = sorted(df["source"].dropna().unique())
            if src_imgs:
                pick_src = st.selectbox("Select image to review bounding boxes:", src_imgs)
                sub_df = df[df["source"] == pick_src]
                img_path = Path(sub_df.iloc[0]["source_path"])
                if img_path.exists():
                    from weedvision.inference import Detection, draw_detections, load_image_rgb
                    raw_rgb = load_image_rgb(str(img_path))
                    h_im, w_im = raw_rgb.shape[:2]
                    review_dets = []
                    for idx, r_box in sub_df.iterrows():
                        review_dets.append(
                            Detection(
                                cls_id=0 if r_box["label"] == "Weed" else 1,
                                cls_name=str(r_box["label"]),
                                conf=float(r_box["score"]),
                                x1=float(r_box["x1"]),
                                y1=float(r_box["y1"]),
                                x2=float(r_box["x2"]),
                                y2=float(r_box["y2"]),
                                uid=f"{r_box['label']}-{idx+1}",
                            )
                        )
                    ann_img = draw_detections(raw_rgb, review_dets, show_conf=True)
                    st.image(ann_img, caption=f"{pick_src} — {len(review_dets)} recovered boxes (Scores: {', '.join(f'{d.cls_name}:{d.conf:.3f}' for d in review_dets)})", use_container_width=True)
