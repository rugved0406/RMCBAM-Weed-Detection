"""Convert LabelMe annotations to an Ultralytics YOLO dataset (copies images; originals untouched).

Usage:
  python tools/convert_labelme.py --images path/to/images [--json-dir path/to/jsons] --out data/yolo --val 0.2

Rectangles, polygons and circles become axis-aligned boxes. Labels are matched case-insensitively
(with aliases, e.g. 'weeds', 'genda' -> Marigold). Unknown labels and missing classes are reported.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weedvision.config import CLASS_NAMES  # noqa: E402
from weedvision.dataset import convert_labelme_dataset, validate_yolo_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--json-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-unlabeled", action="store_true", help="Copy images without JSON as background (empty labels)")
    a = ap.parse_args()

    rep = convert_labelme_dataset(a.images, a.out, a.json_dir, CLASS_NAMES, a.val, a.seed, a.include_unlabeled)
    d = rep.to_dict()
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 15 else f"{len(v)} items") for k, v in d.items()}, indent=2))
    if rep.missing_classes:
        print(f"\nWARNING: no annotations found for classes: {', '.join(rep.missing_classes)}")
    if rep.unknown_labels:
        print(f"WARNING: unknown labels skipped: {rep.unknown_labels}")
    v = validate_yolo_dataset(Path(a.out) / "data.yaml")
    print(f"\nValidation: {len(v['issues'])} issues; splits: {v['splits']}")
    for i in v["issues"][:20]:
        print("  -", i)


if __name__ == "__main__":
    main()
