# WeedVision — RMCBAM-YOLOv8 Weed & Crop Detection Dashboard

Dark-themed Streamlit dashboard and toolkit for detecting **Weed, Tomato, Rose, Marigold and Cauliflower**
with YOLOv8s and the proposed **RMCBAM** (Residual Multi-Scale CBAM) attention.

> **Status of the supplied data (inspected 8 Oct 2026).** `Himani_mam_images.zip` contains **images only**:
> 566 field images in `Origanal/` and 65 + 65 patches in `Crops/Weed_Crops` and `Crops/Crop_Flower_Crops`.
> It contains **no LabelMe JSON, no data.yaml, no trained checkpoints and no notebooks**.
> No accuracy numbers, confusion matrices or PR/F1 curves can exist until annotations or trained weights are
> supplied. The dashboard reports "not available" in those places; it never shows placeholder metrics.

**Start here:** open **`REPORT.html`** in any browser (works offline).

> **Delivered in 4 pieces** (the chat file limit is 30 MB). Put them all in the **same parent folder**:
> 1. `RMCBAM_Weed_Detection.zip` (this package: unzip it)
> 2. `Himani_mam_images.zip` (the original images you already have)
> 3. `yolov8s_rmcbam_SMOKE_best.pt` and 4. `yolov8s_baseline_SMOKE_best.pt` (demo checkpoints)
>
> Then run `setup_windows.bat` (or `./setup.sh`). It calls `tools/prepare_data.py`, which extracts the images to
> `data/Himani mam images/`, puts the demo checkpoints in `demo/smoke_runs/*/weights/`, rebuilds
> `data/yolo_weed_partial/` and downloads `yolov8s.pt`. Re-run `python tools/prepare_data.py` any time to check. It has the findings, charts, recovered boxes,
test results and screenshots of every dashboard page.

## 0. Package contents

| Path | What it is |
|---|---|
| `REPORT.html` | Offline project report (open first) |
| `setup_windows.bat` / `setup.sh` | One-time install into `.venv` |
| `run_dashboard.bat` / `.sh` | Dashboard using **your** models (`models/`, `runs/`) |
| `run_demo.bat` / `.sh` | Dashboard in **demo mode** with smoke-test models (UI exploration only) |
| `app.py`, `weedvision/` | Dashboard and library |
| `tools/` | `convert_labelme`, `recover_patch_boxes`, `train`, `verify_checkpoint`, `build_reports`, `build_report_html` |
| `tests/` | 33 automated tests, including all 9 dashboard pages |
| `data/Himani mam images/` | Original images (unchanged) |
| `data/recovered/` | 130 recovered patch boxes (CSV) and 70 LabelMe pre-annotation files |
| `data/yolo_weed_partial/` | YOLO dataset from the recovered Weed boxes. **Partial**: pipeline testing only |
| `demo/smoke_runs/` | 8-epoch smoke-test runs, baseline and RMCBAM. **Not results** |
| `reports/` | Quality CSVs, near-duplicates, charts, box previews, architecture cost, CAM sanity check, test report |
| `docs/screenshots/` | Every dashboard page |
| `yolov8s.pt` | COCO-pretrained YOLOv8s starting weights (downloaded by setup) |
| `tools/prepare_data.py` | Places images and demo checkpoints, rebuilds the partial dataset, downloads `yolov8s.pt` |

**Demo mode** (`run_demo.*`) loads the smoke-test checkpoints so every page has something to show. A red banner marks
these checkpoints: their detections, heatmaps and metrics are meaningless.

---

## 1. Install

Windows: double-click `setup_windows.bat`, then `run_demo.bat` or `run_dashboard.bat`.
Linux/macOS: `./setup.sh`, then `./run_demo.sh` or `./run_dashboard.sh`.
Manual:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# GPU users: install the CUDA build of torch/torchvision from pytorch.org first
pip install -r requirements.txt
```

## 2. Folder layout (already arranged in this package)

```
rmcbam_weed_dashboard/
├── app.py                      ← streamlit run app.py
├── data/
│   ├── Himani mam images/      ← your original images (included: Origanal/, Crops/)
│   └── recovered/              ← pre-annotations recovered from the patches (shipped)
├── models/                     ← put trained .pt files here (baseline + RMCBAM)
├── runs/                       ← Ultralytics run folders (results.csv, plots, weights/)
├── tools/                      ← CLI: convert, recover, train, verify
├── weedvision/                 ← library + dashboard pages
└── tests/
```

## 3. Run the dashboard

```bash
streamlit run app.py
```

| Page | What it does (all computed from your files and models) |
|---|---|
| Dashboard Overview | Checkpoints found, dataset inventory, readiness checklist, session summary |
| Image Detection | Multi-image upload, model choice, confidence/IoU/size, boxes with IDs (W-001…), PNG/CSV/JSON/ZIP downloads |
| Weed Explainability | **HiResCAM** (default) or **Grad-CAM** per weed and combined, from real gradients; energy-in-box score; RMCBAM spatial and channel attention + learned gates (RMCBAM checkpoints only) |
| Weed Analysis | Count, density (/MP or /m² with GSD), box area, infestation % (union of boxes), confidence and size histograms, configurable severity, crop-wise association |
| Weed Localization | Interactive image-coordinate plot with hover (ID, centre, box, conf, area), highlight, CSV export, spatial density grid |
| Model Comparison | Baseline vs RMCBAM from `results.csv` and plots, **live validation** on your `data.yaml` (P, R, F1, mAP50, mAP50-95, per-class, PR/F1 curves, confusion matrix), params/GFLOPs, measured latency |
| Dataset Analytics | Inventory, LabelMe class distribution, missing labels, YOLO label validation, image quality (resolution, augmentation padding, near-duplicates, sharpness), LabelMe→YOLO converter, recovered patch boxes |
| Model Information | RMCBAM diagram, live baseline-vs-RMCBAM architecture cost, checkpoint compatibility, layer table |
| Settings | Paths, device (auto/CPU/CUDA), thresholds, severity bands, association radius, GSD; save to `settings.json` |

A checkpoint without a `Weed` class (e.g. COCO `yolov8s.pt`) still runs, for testing the pipeline. Every page then
shows a warning that its outputs are not weed detections.

## 4. RMCBAM

```
x ─┬─ GAP & GMP → shared Conv1d(k) → σ = Mc ──┐
   │                                          × → x·Mc ─┬─ [mean_c ‖ max_c] → {3×3, 7×7, 5×5 dil 2} → 1×1 → σ = Ms
   │                                                    └──────────────────────────────────────────────× → x·Mc·Ms
   └──────────────────────── residual ───────────────────────────────── out = x + γ ⊙ (x·Mc·Ms),  γ ∈ R^C, init 0
```

* **ECA-style channel attention**: no FC bottleneck; the adaptive odd kernel `k = |(log2 C + 1)/2|odd`.
* **Multi-scale spatial attention**: parallel 3×3, 7×7 and dilated 5×5 (rate 2) branches, fused by 1×1 conv.
* **Residual attention gating**: per-channel γ starts at 0, so a newly inserted block is an exact identity.
  COCO-pretrained YOLOv8s weights then load unchanged, and the network learns how much attention to apply.
* **Integration**: `C2f` blocks at layers **4, 6, 8** (backbone P3–P5) and **15, 18, 21** (neck outputs into
  `Detect`) become `C2f_RMCBAM` in place. The C2f parameter names are kept, and attention is added under `att.*`.
* **Cost** (measured on this build, nc=5): baseline 11.138 M params / 28.65 GFLOPs, RMCBAM 11.140 M / 28.67 GFLOPs
  (+2,878 parameters).

**Checkpoint compatibility.** RMCBAM checkpoints pickle `weedvision.rmcbam.C2f_RMCBAM`. Load them from code that
has `import weedvision` first, as the dashboard and `tools/` do. A plain `yolo predict` from another folder will
not find the class.

## 5. Dataset preparation

### 5a. If you have LabelMe annotations
```bash
python tools/convert_labelme.py --images "data/Himani mam images/Origanal" --out data/yolo --val 0.2
# JSONs in a separate folder:  --json-dir path/to/jsons
```
* Rectangles, polygons and circles become boxes. Points and lines are skipped and reported.
* Labels are matched case-insensitively with aliases (`weeds`, `genda`→Marigold, `gobhi`→Cauliflower, …).
* Unknown labels, clipped or degenerate boxes, JSON/image size mismatches (rescaled automatically) and
  **classes with zero annotations** are all reported in `data/yolo/conversion_report.json`.
* Images are **copied**. Originals are never modified.

### 5b. Starter boxes recovered from your patches (already done)
Every patch in `Crops/` was located in its source image by normalised cross-correlation:
**130/130 matched**, with NCC between 0.993 and 0.9997. The results are in `data/recovered/recovered_boxes.csv`,
plus 70 LabelMe files in `data/recovered/labelme/` (one per source image).
```bash
python tools/recover_patch_boxes.py --root "data/Himani mam images" --out data/recovered   # to regenerate
```
These are **partial**: each image has one weed region and one crop region, and the crop species is not recorded
in the file names, so crop boxes are labelled `Crop`. Open them in LabelMe (`labelme data/recovered/labelme`),
rename `Crop` to the species, add the remaining plants, then convert as in 5a.

### 5c. Watch for augmentation leakage
44% of `Origanal/` (250/566) shows black rotation/warp padding, and 117 images fall into 52 near-duplicate
groups. These look like pre-augmented copies. If copies of one scene land in both train and val, the
validation mAP will be inflated. Split by original scene, and let Ultralytics augment during training.

## 6. Training & Fine-Tuning (Baseline and RMCBAM)

```bash
# Train both models with optimal SGD / AdamW settings on GPU/MPS:
python tools/train.py --data data/yolo_weed_partial/data.yaml --variant both --epochs 80 --imgsz 640 --batch 8 --nbs 8 --optimizer SGD --lr0 0.01 --cos-lr

# Validate model performance on validation split:
python -c "
from ultralytics import YOLO
import weedvision
from weedvision.comparison import validate
from pprint import pprint

res = validate(YOLO('models/weed_detection_best.pt'), 'data/yolo_weed_partial/data.yaml', imgsz=640)
pprint({k: v for k, v in res.items() if k not in ['curves', 'confusion_matrix']})
"

# Verify model structure & learned RMCBAM gates:
python tools/verify_checkpoint.py models/weed_detection_best.pt
python tools/verify_checkpoint.py models/yolov8s_rmcbam_tuned.pt
```

* Runs are written to `runs/detect/yolov8s_baseline` and `runs/detect/yolov8s_rmcbam`.
* The best-performing model is deployed to `models/weed_detection_best.pt` and loaded automatically by the dashboard.

## 7. Testing

```bash
pytest -q        # 33 tests (dashboard tests use demo/ checkpoints)
```
* **RMCBAM**: shapes, kernel config (3×3 / 7×7 / dilated 5×5), identity at init, gradient flow, injection into
  YOLOv8s, rejection of non-C2f layers, **exact baseline-weight transfer and output equality**,
  checkpoint save → Ultralytics load round-trip, attention capture.
* **Dataset**: label aliases, shape→box conversion, LabelMe→YOLO (clipping, rescaling, unknown labels, missing
  classes, originals untouched), validator error detection, patch recovery, image-quality flags.
* **Analysis**: union area, infestation, severity bands, density/m², ID assignment, crop association.
* **Dashboard**: all 9 pages render headlessly (Streamlit AppTest) and their action buttons run without exceptions.
* **Pipeline** (uses the bundled `yolov8s.pt`): inference on Ultralytics' bundled `bus.jpg`,
  plus a Grad-CAM/HiResCAM localisation check (HiResCAM must put more than 3× uniform density inside the
  detected box). These tests use the COCO model only to prove the code works; they say nothing about weed
  accuracy.

## 8. Explainability notes

* The explained score is the target-class sigmoid score summed over anchors whose decoded box overlaps the
  detection (IoU ≥ 0.5). Gradients flow back to the chosen layers (default 15/18/21).
* **HiResCAM is the default.** On a detector, classic Grad-CAM averages gradients over the whole map, which smears
  the evidence. In the sanity check (`reports/explainability_sanity/`), HiResCAM put on average 10.6× the uniform
  density inside each detected box, against 1.5× for Grad-CAM.
* RMCBAM attention maps are the block's real sigmoid outputs, captured in a forward pass. They are shown only when
  the checkpoint contains RMCBAM.

## 9. Regenerating the reports

```bash
python tools/recover_patch_boxes.py --root "data/Himani mam images" --out data/recovered
python tools/build_reports.py
pytest -v --junitxml=reports/tests/junit.xml
python tools/build_report_html.py
```

## 10. What I still need from you

1. **LabelMe JSON annotations** covering all five classes, plus `data.yaml` if one exists.
2. **Trained checkpoints**: baseline `yolov8s` and `yolov8s + RMCBAM` `best.pt`, or the complete `runs/detect/…`
   folders (`results.csv`, plots, `weights/`).
3. **Notebooks** used for training, so their RMCBAM variant can be checked against this one. If their module
   names differ, their checkpoints need those class definitions to load.
