"""One-time data preparation after unzipping the core package (run automatically by setup scripts).

It looks for these files in the project folder, its parent folder, and ~/Downloads (or paths you pass):
  * Himani_mam_images.zip              -> extracted to data/Himani mam images/   (original images)
  * yolov8s_rmcbam_SMOKE_best.pt        -> demo/smoke_runs/yolov8s_rmcbam_SMOKE/weights/best.pt
  * yolov8s_baseline_SMOKE_best.pt      -> demo/smoke_runs/yolov8s_baseline_SMOKE/weights/best.pt
Then it rebuilds data/yolo_weed_partial/ from the recovered LabelMe boxes and downloads yolov8s.pt.

  python tools/prepare_data.py [--images-zip PATH] [--search DIR ...]
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO_WEIGHTS = {
    "yolov8s_rmcbam_SMOKE_best.pt": ROOT / "demo/smoke_runs/yolov8s_rmcbam_SMOKE/weights/best.pt",
    "yolov8s_baseline_SMOKE_best.pt": ROOT / "demo/smoke_runs/yolov8s_baseline_SMOKE/weights/best.pt",
}
IMAGES_DIR = ROOT / "data" / "Himani mam images"


def find(name: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        p = d / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-zip", default=None)
    ap.add_argument("--search", nargs="*", default=[])
    a = ap.parse_args()
    dirs = [Path(d) for d in a.search] + [ROOT, ROOT.parent, Path.home() / "Downloads"]
    ok = True

    # 1. original images
    if (IMAGES_DIR / "Origanal").exists():
        print(f"[ok]   images already present: {IMAGES_DIR}")
    else:
        z = Path(a.images_zip) if a.images_zip else find("Himani_mam_images.zip", dirs)
        if z and z.exists():
            print(f"[..]   extracting {z} ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(ROOT / "data")
            print(f"[ok]   images extracted to {IMAGES_DIR}")
        else:
            ok = False
            print("[MISS] Himani_mam_images.zip not found. Put it next to this folder (or pass --images-zip) and re-run.")

    # 2. demo checkpoints
    for name, dst in DEMO_WEIGHTS.items():
        if dst.exists():
            print(f"[ok]   demo checkpoint present: {dst.relative_to(ROOT)}")
            continue
        src = find(name, dirs)
        if src:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[ok]   {name} -> {dst.relative_to(ROOT)}")
        else:
            print(f"[MISS] {name} not found (only needed for demo mode / dashboard tests).")

    # 3. partial YOLO dataset (needs images)
    if (IMAGES_DIR / "Origanal").exists():
        from weedvision.config import CLASS_NAMES
        from weedvision.dataset import convert_labelme_dataset

        out = ROOT / "data" / "yolo_weed_partial"
        keep = (out / "README.txt").read_text() if (out / "README.txt").exists() else None
        rep = convert_labelme_dataset(IMAGES_DIR / "Origanal", out, ROOT / "data" / "recovered" / "labelme", CLASS_NAMES, 0.2, 42)
        if keep:
            (out / "README.txt").write_text(keep)
        print(f"[ok]   data/yolo_weed_partial rebuilt ({rep.train_images} train / {rep.val_images} val, PARTIAL labels)")

    # 4. COCO-pretrained starting weights
    w = ROOT / "yolov8s.pt"
    if w.exists():
        print("[ok]   yolov8s.pt present")
    else:
        try:
            import os

            from ultralytics import YOLO

            cwd = os.getcwd()
            os.chdir(ROOT)
            YOLO("yolov8s.pt")
            os.chdir(cwd)
            print("[ok]   yolov8s.pt downloaded")
        except Exception as e:
            print(f"[MISS] could not download yolov8s.pt ({e}); training will download it on first use.")

    print("\nReady." if ok else "\nFinished with missing items (see [MISS] above).")


if __name__ == "__main__":
    main()
