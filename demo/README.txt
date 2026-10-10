DEMO / PIPELINE SMOKE-TEST ARTEFACTS — NOT RESULTS
=================================================
smoke_runs/yolov8s_baseline_SMOKE   and   smoke_runs/yolov8s_rmcbam_SMOKE
  - trained 8 epochs, imgsz 320, CPU, on data/yolo_weed_partial (56 train / 14 val images,
    ONE recovered weed box per image; every other plant unlabelled).
  - Purpose: prove that training, checkpoint saving/loading, RMCBAM, validation plots and every
    dashboard page work end-to-end, and let the team explore the UI (run_demo.bat / run_demo.sh).
  - Their detections, heatmaps and metrics are NOT meaningful and must not be compared or reported.
The dashboard shows a red "SMOKE-TEST" banner whenever one of these is selected.
