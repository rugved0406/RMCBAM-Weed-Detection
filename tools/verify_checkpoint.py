"""Inspect one or more checkpoints: classes, RMCBAM presence/gates, params, GFLOPs, a test forward pass.

Usage: python tools/verify_checkpoint.py models/*.pt
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from weedvision.models import _load_raw, inspect_checkpoint  # noqa: E402


def main(paths):
    if not paths:
        print(__doc__)
        return 1
    rc = 0
    for p in paths:
        info = inspect_checkpoint(p)
        print(f"\n=== {p} ===")
        for level, msg in info.compatibility_notes():
            print(f"[{level.upper():5}] {msg}")
        if not info.ok:
            rc = 1
            continue
        print(json.dumps({k: v for k, v in info.__dict__.items() if k not in ("rmcbam_gate_mean_abs",)}, indent=2, default=str))
        if info.rmcbam_gate_mean_abs:
            print("RMCBAM residual gate mean|gamma| per block (0 = attention not yet used):")
            for k, v in info.rmcbam_gate_mean_abs.items():
                print(f"  {k}: {v:.5f}")
        _, model = _load_raw(p)
        model = model.float().eval()
        with torch.no_grad():
            y = model(torch.zeros(1, 3, 640, 640))
        y0 = y[0] if isinstance(y, (tuple, list)) else y
        print(f"forward pass OK: output {tuple(y0.shape)} (expected (1, {4 + info.nc}, 8400))")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
