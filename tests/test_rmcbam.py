import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weedvision.rmcbam import (  # noqa: E402
    RMCBAM,
    C2f_RMCBAM,
    DEFAULT_RMCBAM_LAYERS,
    ECAChannelAttention,
    MultiScaleSpatialAttention,
    build_rmcbam_model,
    eca_kernel_size,
    find_rmcbam,
    inject_rmcbam,
    set_attention_capture,
)


def test_eca_kernel_is_odd():
    for c in (16, 64, 128, 256, 512, 1024):
        k = eca_kernel_size(c)
        assert k % 2 == 1 and k >= 3


def test_components_shapes():
    x = torch.randn(2, 64, 40, 40)
    assert ECAChannelAttention(64)(x).shape == (2, 64, 1, 1)
    assert MultiScaleSpatialAttention()(x).shape == (2, 1, 40, 40)
    assert RMCBAM(64)(x).shape == x.shape


def test_spatial_branches_kernel_config():
    sa = MultiScaleSpatialAttention()
    assert sa.b3[0].kernel_size == (3, 3) and sa.b3[0].dilation == (1, 1)
    assert sa.b7[0].kernel_size == (7, 7)
    assert sa.b5d[0].kernel_size == (5, 5) and sa.b5d[0].dilation == (2, 2)


def test_identity_at_init_and_gradients():
    r = RMCBAM(32)
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    y = r(x)
    assert torch.allclose(y, x), "gamma=0 must make a fresh RMCBAM an identity"
    y.pow(2).sum().backward()
    assert r.gamma.grad is not None and r.gamma.grad.abs().sum() > 0
    # once gamma is non-zero, attention parameters receive gradients too
    with torch.no_grad():
        r.gamma.fill_(0.5)
    r.zero_grad()
    r(x).pow(2).sum().backward()
    assert r.ca.conv.weight.grad.abs().sum() > 0
    assert r.sa.b5d[0].weight.grad.abs().sum() > 0


def test_attention_capture():
    r = RMCBAM(16)
    r.store_attention = True
    r(torch.randn(1, 16, 8, 8))
    assert r.last_channel_att.shape == (1, 16, 1, 1)
    assert r.last_spatial_att.shape == (1, 1, 8, 8)
    assert 0 <= r.last_spatial_att.min() and r.last_spatial_att.max() <= 1


def test_inject_into_yolov8s_and_forward():
    from ultralytics.nn.tasks import DetectionModel

    m = DetectionModel("yolov8s.yaml", nc=5, verbose=False)
    n0 = sum(p.numel() for p in m.parameters())
    done = inject_rmcbam(m)
    assert done == sorted(DEFAULT_RMCBAM_LAYERS)
    assert all(isinstance(m.model[i], C2f_RMCBAM) for i in done)
    assert len(find_rmcbam(m)) == len(done)
    assert sum(p.numel() for p in m.parameters()) > n0
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 320, 320))
    y0 = y[0] if isinstance(y, (tuple, list)) else y
    assert y0.shape == (1, 4 + 5, 2100)  # 40²+20²+10² anchors at 320


def test_inject_rejects_non_c2f():
    from ultralytics.nn.tasks import DetectionModel

    m = DetectionModel("yolov8s.yaml", nc=5, verbose=False)
    with pytest.raises(ValueError):
        inject_rmcbam(m, [0])  # layer 0 is Conv


def test_baseline_weights_transfer_and_identity():
    """Baseline weights load 1:1; with gamma=0 the RMCBAM model reproduces the baseline output exactly."""
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    base = DetectionModel("yolov8s.yaml", nc=5, verbose=False).eval()
    rm = build_rmcbam_model("yolov8s.yaml", nc=5, weights=base).eval()
    sd_b, sd_r = base.state_dict(), rm.state_dict()
    for k, v in sd_b.items():
        assert k in sd_r and torch.equal(v, sd_r[k]), k
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        assert torch.allclose(base(x)[0], rm(x)[0], atol=1e-5)


def test_checkpoint_round_trip(tmp_path):
    """Save the way Ultralytics does (pickled module) and load with Ultralytics' loader."""
    from ultralytics.nn.tasks import torch_safe_load

    from weedvision.models import inspect_checkpoint

    m = build_rmcbam_model("yolov8s.yaml", nc=5)
    m.names = {0: "Weed", 1: "Tomato", 2: "Rose", 3: "Marigold", 4: "Cauliflower"}
    with torch.no_grad():
        for _, a in find_rmcbam(m):
            a.gamma.fill_(0.123)
    p = tmp_path / "rm.pt"
    torch.save({"model": m.half(), "epoch": 3, "best_fitness": None, "train_args": {"imgsz": 640}}, p)
    ck, _ = torch_safe_load(str(p))
    mods = find_rmcbam(ck["model"])
    assert len(mods) == len(DEFAULT_RMCBAM_LAYERS)
    assert abs(float(mods[0][1].gamma.detach().float().mean()) - 0.123) < 1e-3
    info = inspect_checkpoint(p, compute_flops=False)
    assert info.ok and info.is_rmcbam and info.has_weed_class and info.rmcbam_layers == sorted(DEFAULT_RMCBAM_LAYERS)
    assert info.missing_project_classes == []


def test_set_attention_capture():
    m = build_rmcbam_model("yolov8s.yaml", nc=5).eval()
    set_attention_capture(m, True)
    with torch.no_grad():
        m(torch.rand(1, 3, 320, 320))
    for _, a in find_rmcbam(m):
        assert a.last_spatial_att is not None
    set_attention_capture(m, False)
    assert all(a.last_spatial_att is None for _, a in find_rmcbam(m))
