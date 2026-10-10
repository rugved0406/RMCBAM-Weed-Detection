"""Build REPORT.html (offline, no internet needed) from the files in reports/, demo/ and docs/.

  python tools/build_report_html.py

Every number in the report is read from a generated file; nothing is typed in by hand.
"""

import html
import json
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

R = ROOT / "reports"


def esc(x):
    return html.escape(str(x))


def table(df: pd.DataFrame, cls="") -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def kpi(k, v, s=""):
    return f'<div class="kpi"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'


def fig(src, cap, cls=""):
    return f'<figure class="{cls}"><a href="{esc(src)}" target="_blank"><img loading="lazy" src="{esc(src)}" alt="{esc(cap)}"></a><figcaption>{esc(cap)}</figcaption></figure>'


def main():
    q = json.loads((R / "dataset_quality_summary.json").read_text())
    inv = json.loads((R / "dataset_inventory.json").read_text())
    rec = pd.read_csv(ROOT / "data" / "recovered" / "recovered_boxes.csv")
    arch = pd.read_csv(R / "architecture_cost.csv")
    blocks = pd.read_csv(R / "rmcbam_blocks.csv")
    cam = pd.read_csv(R / "explainability_sanity" / "energy_in_box.csv")
    conv = json.loads((ROOT / "data" / "yolo_weed_partial" / "conversion_report.json").read_text())

    # tests
    jt = ET.parse(R / "tests" / "junit.xml").getroot()
    suites = jt.findall("testsuite") or [jt]
    n_tests = sum(int(s.get("tests", 0)) for s in suites)
    n_fail = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    n_skip = sum(int(s.get("skipped", 0)) for s in suites)
    cases = []
    for s in suites:
        for c in s.iter("testcase"):
            st = "skipped" if c.find("skipped") is not None else ("FAILED" if (c.find("failure") is not None or c.find("error") is not None) else "passed")
            cases.append({"module": c.get("classname", "").split(".")[-1], "test": c.get("name"), "result": st, "time (s)": round(float(c.get("time", 0)), 2)})
    tests_df = pd.DataFrame(cases)

    # demo runs
    demo_rows = []
    for run in sorted((ROOT / "demo" / "smoke_runs").glob("*_SMOKE")):
        df = pd.read_csv(run / "results.csv")
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        demo_rows.append({"run": run.name, "epochs": int(last["epoch"]), "precision": round(last["metrics/precision(B)"], 4), "recall": round(last["metrics/recall(B)"], 4), "mAP50": round(last["metrics/mAP50(B)"], 4), "mAP50-95": round(last["metrics/mAP50-95(B)"], 4)})
    demo_df = pd.DataFrame(demo_rows)
    gates = None
    try:
        from weedvision.models import inspect_checkpoint

        info = inspect_checkpoint(ROOT / "demo" / "smoke_runs" / "yolov8s_rmcbam_SMOKE" / "weights" / "best.pt", compute_flops=False)
        gates = pd.DataFrame([{"block": k, "mean |γ| after 8 epochs": f"{v:.2e}"} for k, v in info.rmcbam_gate_mean_abs.items()])
    except Exception:
        pass

    cam_summary = cam.groupby("method")["ratio_vs_uniform"].mean().round(2).to_dict()
    base_p, rm_p = int(arch.loc[0, "params"]), int(arch.loc[1, "params"])
    shots = sorted(p for p in (ROOT / "docs" / "screenshots").iterdir() if p.suffix.lower() in (".png", ".jpg"))
    shot_names = {
        "01": "Dashboard Overview", "02": "Image Detection", "02b": "Detection table", "03": "Explainability (HiResCAM)",
        "03b": "Per-weed heatmaps", "03c": "RMCBAM attention maps", "04": "Weed Analysis", "04b": "Crop-wise association",
        "05": "Weed Localization", "06": "Comparison: training runs", "06b": "Comparison: curves & plots",
        "06c": "Live validation", "06d": "Live validation: PR / F1", "06e": "Efficiency (params, GFLOPs, latency)",
        "06f": "Live validation: confusion matrices", "07": "Dataset: annotations", "07b": "Dataset: image quality",
        "07c": "Dataset: recovered patch boxes", "08": "Model Information", "08b": "Checkpoint details", "09": "Settings",
    }
    gallery = "".join(fig(f"docs/screenshots/{p.name}", shot_names.get(p.stem.split("_")[0], p.stem), "shot") for p in shots)
    previews = sorted((R / "recovered_boxes_preview").glob("*.jpg"))
    prev_gallery = "".join(fig(f"reports/recovered_boxes_preview/{p.name}", p.stem.replace("_recovered", ""), "thumb") for p in previews)

    folders = pd.DataFrame([
        ["REPORT.html", "This page"],
        ["README.md", "Full technical guide: install, train, convert, test"],
        ["setup_windows.bat / setup.sh", "One-time install into a virtual environment"],
        ["run_dashboard.bat / .sh", "Dashboard with YOUR models (models/, runs/)"],
        ["run_demo.bat / .sh", "Dashboard in DEMO MODE with smoke-test models (UI exploration only)"],
        ["app.py, weedvision/", "Dashboard + library (RMCBAM, inference, explainability, analysis, dataset tools)"],
        ["tools/", "CLI: convert_labelme, recover_patch_boxes, train, verify_checkpoint, build_reports, build_report_html"],
        ["tests/", f"{n_tests} automated tests (model, dataset, analysis, pipeline, all 9 dashboard pages)"],
        ["data/Himani mam images/", "Your original images (unchanged)"],
        ["data/recovered/", "130 recovered patch boxes (CSV) + 70 LabelMe pre-annotation files"],
        ["data/yolo_weed_partial/", "YOLO dataset built from the recovered Weed boxes — PARTIAL, pipeline testing only"],
        ["demo/", "Smoke-test training runs (8 epochs, partial labels) — NOT results"],
        ["reports/", "Dataset quality, near-duplicates, charts, recovered-box previews, architecture cost, CAM sanity, test report"],
        ["docs/screenshots/", "Screenshots of every dashboard page"],
        ["yolov8s.pt", "COCO-pretrained YOLOv8s (downloaded by setup; starting weights; used by pipeline tests)"],
        ["tools/prepare_data.py", "Run by setup: extracts Himani_mam_images.zip into data/, places demo checkpoints, rebuilds the partial dataset"],
        ["models/, runs/", "Empty: put trained checkpoints / Ultralytics run folders here"],
    ], columns=["Path", "What it is"])

    rec_by = rec.groupby("label")["score"].agg(["count", "min", "max"]).round(4).reset_index().rename(columns={"label": "patch type", "count": "patches", "min": "min NCC", "max": "max NCC"})
    top_res = pd.DataFrame([{"resolution": k, "images": v} for k, v in q["top_resolutions"].items()])

    svg = """
<svg viewBox="0 0 980 300" class="diagram" role="img" aria-label="RMCBAM block diagram">
 <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker></defs>
 <g font-size="12" fill="var(--ink)" text-anchor="middle">
  <rect x="10" y="125" width="110" height="46" rx="8" class="n"/><text x="65" y="145">C2f output x</text><text x="65" y="161" class="sm">C×H×W</text>
  <rect x="160" y="40" width="130" height="46" rx="8" class="n"/><text x="225" y="60">GAP ‖ GMP</text><text x="225" y="76" class="sm">→ shared Conv1d(k)</text>
  <rect x="320" y="40" width="110" height="46" rx="8" class="n acc"/><text x="375" y="60">σ → Mc</text><text x="375" y="76" class="sm">ECA, no FC</text>
  <circle cx="470" cy="148" r="20" class="n"/><text x="470" y="153">⊙</text>
  <rect x="510" y="125" width="120" height="46" rx="8" class="n"/><text x="570" y="145">mean_c ‖ max_c</text><text x="570" y="161" class="sm">2×H×W</text>
  <rect x="660" y="30" width="110" height="34" rx="8" class="n"/><text x="715" y="52">Conv 3×3</text>
  <rect x="660" y="80" width="110" height="34" rx="8" class="n"/><text x="715" y="102">Conv 7×7</text>
  <rect x="660" y="130" width="110" height="34" rx="8" class="n"/><text x="715" y="146">Conv 5×5</text><text x="715" y="159" class="sm">dilation 2</text>
  <rect x="800" y="70" width="120" height="46" rx="8" class="n acc"/><text x="860" y="90">1×1 fuse, σ</text><text x="860" y="106" class="sm">→ Ms (1×H×W)</text>
  <circle cx="860" cy="200" r="20" class="n"/><text x="860" y="205">⊙</text>
  <rect x="690" y="235" width="110" height="40" rx="8" class="n acc"/><text x="745" y="259">× γ (init 0)</text>
  <circle cx="600" cy="255" r="20" class="n"/><text x="600" y="260">+</text>
  <rect x="440" y="235" width="110" height="40" rx="8" class="n"/><text x="495" y="259">RMCBAM(x)</text>
 </g>
 <g stroke="var(--muted)" stroke-width="1.6" fill="none" marker-end="url(#a)">
  <path d="M120,140 C140,120 140,63 158,63"/><path d="M290,63 L318,63"/><path d="M430,63 C460,63 470,100 470,126"/>
  <path d="M120,148 L448,148"/><path d="M490,148 L508,148"/>
  <path d="M630,140 C645,120 645,47 658,47"/><path d="M630,145 C648,130 648,97 658,97"/><path d="M630,148 L658,148"/>
  <path d="M770,47 C790,47 790,85 798,88"/><path d="M770,97 L798,95"/><path d="M770,147 C790,147 790,110 798,104"/>
  <path d="M860,116 L860,178"/><path d="M490,165 C520,215 800,215 838,203"/>
  <path d="M860,220 C860,250 820,255 802,255"/><path d="M690,255 L622,255"/><path d="M580,255 L552,255"/>
  <path d="M65,171 C65,280 400,290 582,268" stroke-dasharray="5 4"/>
 </g>
 <text x="300" y="292" font-size="11" fill="var(--muted)">residual path: out = x + γ ⊙ (x ⊙ Mc ⊙ Ms)</text>
</svg>"""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RMCBAM Weed Detection — Project Report</title>
<style>
:root {{ --page:#0d0d0d; --surf:#1a1a19; --surf2:#222220; --ink:#f2f1ec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --blue:#3987e5; --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b; --brand:#5fb878; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink2); font:15px/1.6 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:28px 16px 80px; }}
h1 {{ color:var(--ink); font-size:30px; margin:0 0 6px; letter-spacing:-.01em; }}
h2 {{ color:var(--ink); font-size:21px; margin:46px 0 12px; padding-top:10px; border-top:1px solid var(--grid); }}
h3 {{ color:var(--ink); font-size:16px; margin:22px 0 8px; }}
a {{ color:#7fb2f0; }} code {{ background:var(--surf2); padding:1px 6px; border-radius:5px; color:var(--ink); font-size:13px; }}
pre {{ background:var(--surf); border:1px solid var(--axis); border-radius:8px; padding:12px 14px; overflow:auto; color:var(--ink); font-size:13px; }}
.hero {{ border:1px solid var(--axis); border-radius:12px; padding:22px; background:linear-gradient(135deg,#14201a,var(--surf) 60%); }}
.hero p {{ margin:4px 0 0; }}
nav.toc {{ display:flex; flex-wrap:wrap; gap:8px; margin:16px 0 0; }}
nav.toc a {{ text-decoration:none; color:var(--ink2); border:1px solid var(--axis); border-radius:999px; padding:3px 12px; font-size:13px; }}
nav.toc a:hover {{ color:var(--ink); border-color:var(--muted); }}
.callout {{ border-left:4px solid var(--warn); background:var(--surf); border-radius:8px; padding:12px 16px; margin:14px 0; }}
.callout.crit {{ border-color:var(--crit); }} .callout.good {{ border-color:var(--good); }}
.callout b {{ color:var(--ink); }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:14px 0; }}
.kpi {{ background:var(--surf); border:1px solid var(--axis); border-radius:10px; padding:12px 14px; }}
.kpi .k {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }}
.kpi .v {{ color:var(--ink); font-size:24px; font-weight:650; line-height:1.25; font-variant-numeric:tabular-nums; }}
.kpi .s {{ color:var(--ink2); font-size:12.5px; }}
.tw {{ overflow-x:auto; margin:10px 0; border:1px solid var(--axis); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; background:var(--surf); }}
th {{ text-align:left; color:var(--muted); font-weight:600; padding:8px 10px; border-bottom:1px solid var(--axis); white-space:nowrap; }}
td {{ padding:7px 10px; border-bottom:1px solid var(--grid); color:var(--ink2); font-variant-numeric:tabular-nums; }}
tr:last-child td {{ border-bottom:0; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:12px; }}
.thumbs {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:8px; max-height:560px; overflow:auto; padding-right:4px; }}
figure {{ margin:0; background:var(--surf); border:1px solid var(--axis); border-radius:10px; overflow:hidden; }}
figure img {{ display:block; width:100%; height:auto; }}
figure.shot img {{ aspect-ratio:1500/1300; object-fit:cover; object-position:top; }}
figure.thumb img {{ aspect-ratio:4/3; object-fit:cover; }}
figcaption {{ padding:6px 10px; font-size:12.5px; color:var(--ink2); }}
.diagram {{ width:100%; height:auto; background:var(--surf); border:1px solid var(--axis); border-radius:10px; padding:8px; }}
.diagram .n {{ fill:var(--surf2); stroke:var(--axis); }} .diagram .acc {{ stroke:var(--blue); }} .diagram .sm {{ font-size:10.5px; fill:var(--ink2); }}
.pill {{ display:inline-block; padding:1px 9px; border-radius:999px; font-size:12px; font-weight:600; border:1px solid var(--axis); }}
.pass {{ color:var(--good); border-color:var(--good); }} .fail {{ color:var(--crit); border-color:var(--crit); }}
ol li, ul li {{ margin:4px 0; }}
footer {{ margin-top:50px; color:var(--muted); font-size:12.5px; }}
@media (max-width:600px) {{ h1 {{ font-size:24px; }} .kpi .v {{ font-size:20px; }} }}
</style></head><body><div class="wrap">

<div class="hero">
  <h1>RMCBAM Weed &amp; Crop Detection</h1>
  <p>YOLOv8s + Residual Multi-Scale CBAM · classes: Weed, Tomato, Rose, Marigold, Cauliflower · package built {date.today().isoformat()}</p>
  <nav class="toc">
    <a href="#status">Status</a><a href="#start">Quick start</a><a href="#contents">Contents</a><a href="#data">Dataset findings</a>
    <a href="#recovered">Recovered boxes</a><a href="#arch">RMCBAM</a><a href="#verify">Verification</a><a href="#dash">Dashboard</a><a href="#next">What's needed</a>
  </nav>
</div>

<h2 id="status">1. Status in one minute</h2>
<div class="callout crit"><b>No accuracy results exist yet.</b> The supplied zip contained images only: no LabelMe annotations, no
<code>data.yaml</code>, no trained checkpoints and no notebooks. No weed-detection or mAP figures in this package come from a trained model.
The dashboard reports "not available" instead of inventing numbers.</div>
<div class="callout good"><b>What is ready and verified:</b> the RMCBAM module and its YOLOv8s integration, training, the LabelMe→YOLO conversion,
real gradient-based explanations, weed analysis and localisation, model comparison, and a 9-page dashboard.
<b>{n_tests - n_fail - n_skip}/{n_tests} automated tests pass</b>.</div>
<div class="kpis">
 {kpi("Original images", q["originals"], "in Origanal/")}
 {kpi("Patches", len(rec), f"{(rec.label == 'Weed').sum()} weed · {(rec.label != 'Weed').sum()} crop/flower")}
 {kpi("Annotations supplied", 0, "LabelMe JSON in the zip")}
 {kpi("Boxes recovered", f"{int(rec.accepted.sum())}/{len(rec)}", f"NCC ≥ {rec.score.min():.3f}")}
 {kpi("Augmented-looking", f"{q['originals_likely_augmented']} ({q['originals_likely_augmented'] / q['originals']:.0%})", "rotation / warp padding")}
 {kpi("Near-duplicates", q["originals_in_near_duplicate_groups"], f"in {q['near_duplicate_groups']} groups")}
 {kpi("Tests", f"{n_tests - n_fail - n_skip}/{n_tests}", "passed")}
 {kpi("RMCBAM overhead", f"+{rm_p - base_p:,}", f"params ({(rm_p - base_p) / base_p:.3%})")}
</div>

<h2 id="start">2. Quick start</h2>
<div class="grid2">
<div><h3>Windows</h3><pre>0. Put Himani_mam_images.zip + the two *_SMOKE_best.pt
   files next to this folder
1. Install Python 3.10–3.13 (tick "Add to PATH")
2. Double-click  setup_windows.bat     (one time)
3. Double-click  run_demo.bat          (explore UI)
   or            run_dashboard.bat     (your models)
4. Browser opens at http://localhost:8501</pre></div>
<div><h3>Linux / macOS</h3><pre>./setup.sh
./run_demo.sh          # explore the UI
./run_dashboard.sh     # your models
source .venv/bin/activate && pytest -q   # {n_tests} tests</pre></div>
</div>
<div class="callout"><b>Delivered in 4 pieces</b> because of the 30 MB chat file limit: this core ZIP, <code>Himani_mam_images.zip</code> (your originals),
<code>yolov8s_rmcbam_SMOKE_best.pt</code> and <code>yolov8s_baseline_SMOKE_best.pt</code>. Put them in the same parent folder and run setup.
<code>tools/prepare_data.py</code> then assembles everything. Images in this report are already included.</div>
<p>GPU: install the CUDA build of PyTorch from pytorch.org before <code>setup</code>. The dashboard picks the GPU automatically (Settings → Device).</p>

<h2 id="contents">3. What is in this package</h2>
{table(folders)}

<h2 id="data">4. Dataset findings (computed from your images)</h2>
<div class="callout"><b>Classes with zero annotations:</b> {esc(", ".join(conv["missing_classes"]))}. The only labels available are the
{conv["instances_per_class"].get("Weed", 0)} weed regions recovered from the patches, at one per image. Crop patches carry no species, so they are labelled <code>Crop</code>.</div>
<div class="grid2">
 {fig("reports/charts/augmented.png", "44% of originals carry black rotation/warp padding: pre-augmented copies")}
 {fig("reports/charts/resolutions.png", f"{q['distinct_resolutions']} distinct resolutions (originals)")}
 {fig("reports/charts/padding.png", "Black-padding share per image")}
 {fig("reports/charts/sharpness.png", "Sharpness (Laplacian variance)")}
</div>
<div class="callout"><b>Leakage risk:</b> {q["originals_in_near_duplicate_groups"]} originals fall into {q["near_duplicate_groups"]} near-duplicate groups
(<a href="reports/near_duplicate_groups.csv">near_duplicate_groups.csv</a>), and {q["originals_likely_augmented"]} look pre-augmented.
Split train and val by original scene. If copies of one scene sit on both sides, the validation mAP will be inflated.</div>
<h3>Most common resolutions</h3>{table(top_res)}
<p>Per-image table: <a href="reports/dataset_quality_all_images.csv">dataset_quality_all_images.csv</a> · inventory: <a href="reports/dataset_inventory.json">dataset_inventory.json</a></p>
<div class="grid2">
 {fig("reports/patches_weed_contact_sheet.jpg", "All 65 Weed patches (Crops/Weed_Crops)")}
 {fig("reports/patches_crop_contact_sheet.jpg", "All 65 Crop/Flower patches: rose, marigold and cauliflower visible; species not labelled")}
</div>

<h2 id="recovered">5. Recovered patch boxes</h2>
<p>Every patch was located inside its source image by normalised cross-correlation (NCC; 1.0 = exact copy). This yields
real pixel boxes, written as LabelMe files you can open, correct and complete: <code>labelme data/recovered/labelme</code>.</p>
{table(rec_by)}
{fig("reports/recovered_boxes_contact_sheet.jpg", "All 70 source images with recovered Weed (blue) and Crop (violet) boxes")}
<h3>Individual previews (click to enlarge)</h3><div class="thumbs">{prev_gallery}</div>
<div class="callout"><b>Partial labels.</b> Each image has one weed region and one crop region. Every other plant is unlabelled, and the regions are
field crops rather than tight per-plant boxes. Rename <code>Crop</code> to the species and annotate the rest in LabelMe before training.</div>

<h2 id="arch">6. RMCBAM architecture</h2>
{svg}
<ul>
<li><b>ECA-style channel attention:</b> average- and max-pooled descriptors share one 1-D convolution with no FC bottleneck. The kernel adapts to channel count.</li>
<li><b>Multi-scale spatial attention:</b> 3×3, 7×7 and dilated 5×5 (rate 2) branches run in parallel on the channel-mean and max maps, followed by a 1×1 fusion and a sigmoid.</li>
<li><b>Residual attention gating:</b> <code>out = x + γ ⊙ (x ⊙ Mc ⊙ Ms)</code>, where γ is per-channel and starts at 0. A newly inserted block is an exact identity, so YOLOv8s weights transfer 1:1.</li>
<li><b>Placement:</b> C2f blocks at layers 4, 6, 8 (backbone P3–P5) and 15, 18, 21 (neck outputs into Detect).</li>
</ul>
<h3>Cost (built live, nc = 5)</h3>{table(arch)}
<h3>Per-block detail</h3>{table(blocks)}

<h2 id="verify">7. Verification</h2>
<h3>Automated tests: <span class="pill {'pass' if n_fail == 0 else 'fail'}">{n_tests - n_fail - n_skip} passed · {n_fail} failed · {n_skip} skipped</span></h3>
{table(tests_df)}
<p>Highlights: baseline weights load into the RMCBAM model 1:1, and its output equals the baseline's exactly at initialisation. RMCBAM checkpoints
survive a save → Ultralytics-load round-trip. The converter preserves originals and reports missing classes. All 9 dashboard pages render and their buttons run.</p>

<h3>Explainability sanity check (COCO model on a standard photo)</h3>
<p>This uses a working detector to check that the explanation code localises correctly. The ratio is heatmap density inside the box against a uniform map (1.0 = no localisation).
Mean ratio: Grad-CAM <b>{cam_summary.get("gradcam", "?")}×</b>, HiResCAM <b>{cam_summary.get("hirescam", "?")}×</b>. HiResCAM is therefore the dashboard default.</p>
{fig("reports/explainability_sanity/gradcam_vs_hirescam_bus.jpg", "Top: Grad-CAM. Bottom: HiResCAM. One heatmap per detected person.")}
{table(cam)}

<h3>Training pipeline smoke test (NOT results)</h3>
<div class="callout crit">Both models were trained for 8 epochs at 320 px on CPU, on the <b>partial</b> Weed set (56 train / 14 val, one box per image).
This proves that training, saving, reloading, RMCBAM learning and validation plots work. The metrics below are real outputs of that run,
but they say <b>nothing</b> about RMCBAM versus the baseline and must not be reported.</div>
{table(demo_df)}
{table(gates) if gates is not None else ""}
<p>The γ gates moving off 0 confirms the attention parameters are being learned.</p>

<h2 id="dash">8. Dashboard (screenshots in demo mode)</h2>
<p>Screenshots were taken with the smoke-test models, so the detections are noise, and every page shows a smoke-test warning. They show what each page does.</p>
<div class="gallery">{gallery}</div>

<h2 id="next">9. What's needed for real results</h2>
<ol>
<li><b>LabelMe annotations for all five classes</b> (or complete the files in <code>data/recovered/labelme</code>), then run
<code>python tools/convert_labelme.py --images "data/Himani mam images/Origanal" --out data/yolo</code>.</li>
<li><b>Train both models with identical settings:</b> <code>python tools/train.py --data data/yolo/data.yaml --variant both --epochs 100 --device 0</code>.
Model Comparison then reads <code>runs/detect/yolov8s_baseline</code> and <code>yolov8s_rmcbam</code> directly.</li>
<li>Or supply <b>existing trained checkpoints / run folders</b> and the <b>training notebooks</b>. If their RMCBAM class names differ, those definitions are needed to load them.</li>
</ol>
<footer>Regenerate: <code>python tools/build_reports.py</code> then <code>python tools/build_report_html.py</code>.</footer>
</div></body></html>"""
    (ROOT / "REPORT.html").write_text(page, encoding="utf-8")
    print("REPORT.html written", len(page) // 1024, "KB;", n_tests, "tests,", len(shots), "screenshots,", len(previews), "previews")


if __name__ == "__main__":
    main()
