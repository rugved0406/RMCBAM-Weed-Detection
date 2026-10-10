"""Recover bounding boxes for Crops/Weed_Crops and Crops/Crop_Flower_Crops patches by locating
each patch inside its source image (Origanal/imgNNN.jpg) with normalised cross-correlation.

Outputs (originals untouched):
  <out>/recovered_boxes.csv        one row per patch: source image, box, NCC score
  <out>/labelme/<imgNNN>.json      LabelMe pre-annotations (labels 'Weed' and 'Crop') to open,
                                   correct, rename 'Crop' -> species, and complete in LabelMe.

NOTE: each source image has only ONE weed patch and ONE crop patch, so these are partial
annotations (other weeds/crops in the image are unlabelled). Do not train on them as-is.

Usage:
  python tools/recover_patch_boxes.py --root "path/to/Himani mam images" --out data/recovered
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weedvision.dataset import recover_patch_boxes, write_labelme_from_boxes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Folder containing Origanal/ and Crops/")
    ap.add_argument("--originals", default="Origanal")
    ap.add_argument("--weed-patches", default="Crops/Weed_Crops")
    ap.add_argument("--crop-patches", default="Crops/Crop_Flower_Crops")
    ap.add_argument("--crop-label", default="Crop", help="Label for crop patches (species is not recorded in the files)")
    ap.add_argument("--min-score", type=float, default=0.9)
    ap.add_argument("--out", default="data/recovered")
    a = ap.parse_args()

    root = Path(a.root)
    df = recover_patch_boxes(
        root / a.originals,
        {"Weed": root / a.weed_patches, a.crop_label: root / a.crop_patches},
        min_score=a.min_score,
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "recovered_boxes.csv", index=False)
    written = write_labelme_from_boxes(df, out / "labelme", root / a.originals)
    import os

    for col in ("patch_path", "source_path"):  # portable CSV: paths relative to the current directory
        if col in df:
            df[col] = [os.path.relpath(p) if isinstance(p, str) else p for p in df[col]]
    df.to_csv(out / "recovered_boxes.csv", index=False)
    acc = int(df["accepted"].sum()) if len(df) else 0
    print(f"patches: {len(df)}  accepted (NCC>={a.min_score}): {acc}  LabelMe files: {len(written)}")
    if len(df):
        print(df.groupby("label")["score"].describe().round(4).to_string())
        bad = df[~df["accepted"]]
        if len(bad):
            print("\nNot accepted:")
            print(bad[["patch", "label", "score", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
