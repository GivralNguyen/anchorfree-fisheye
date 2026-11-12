"""Sanity tests: geometry, target encode/decode round trip, losses, evaluation.

    python -m pytest -q tests      (or: python tests/test_core.py)
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arpd.boxes import affine_boxes, box_vertices, canonicalize, rad2deg_boxes  # noqa: E402
from arpd.data.targets import encode_targets  # noqa: E402
from arpd.data.transforms import make_affine  # noqa: E402
from arpd.decode import decode  # noqa: E402
from arpd.eval.rotated_eval import ap_metrics, prf_counts, rotated_iou  # noqa: E402
from arpd.losses import angle_loss, periodic_diff  # noqa: E402

RNG = np.random.default_rng(0)


def random_boxes(n=20, size=500):
    b = np.stack([RNG.uniform(50, size - 50, n), RNG.uniform(50, size - 50, n),
                  RNG.uniform(10, 60, n), RNG.uniform(10, 120, n),
                  RNG.uniform(-math.pi, math.pi, n)], 1)
    return canonicalize(b)


def same_rect(a, b, tol=1e-6):
    """Rotated rectangles are equal iff their vertex sets are equal."""
    va, vb = box_vertices(a), box_vertices(b)
    for x, y in zip(va, vb):
        d = np.linalg.norm(x[:, None] - y[None], axis=2)
        if d.min(1).max() > tol:
            return False
    return True


def test_canonical_keeps_rectangle():
    b = np.stack([RNG.uniform(0, 100, 50), RNG.uniform(0, 100, 50), RNG.uniform(1, 50, 50),
                  RNG.uniform(1, 50, 50), RNG.uniform(-7, 7, 50)], 1)
    c = canonicalize(b)
    assert (c[:, 2] <= c[:, 3]).all()
    assert (c[:, 4] >= -math.pi / 2).all() and (c[:, 4] < math.pi / 2).all()
    assert same_rect(b, c)


def test_affine_matches_vertex_transform():
    """Transforming boxes == transforming their corners, incl. flips."""
    b = random_boxes()
    for hf in (False, True):
        for vf in (False, True):
            M = make_affine(640, 480, 512, scale=1.3, rot_deg=37.0, shift=(0.05, -0.1),
                            hflip=hf, vflip=vf)
            out = affine_boxes(b, M)
            v = box_vertices(b) @ M[:, :2].T + M[:, 2]
            vo = box_vertices(out)
            for x, y in zip(v, vo):
                assert np.linalg.norm(x[:, None] - y[None], axis=2).min(1).max() < 1e-6


def test_encode_decode_roundtrip():
    b = random_boxes(10, 512)
    t = encode_targets(b, 512, stride=4)
    H = 128
    hm = torch.full((1, 1, H, H), -10.0)
    reg = torch.zeros(1, 2, H, H)
    wh = torch.zeros(1, 2, H, H)
    ang = torch.zeros(1, 1, H, H)
    for k in range(int(t['mask'].sum())):
        y, x = divmod(int(t['ind'][k]), H)
        hm[0, 0, y, x] = 10.0
        reg[0, :, y, x] = torch.tensor(t['reg'][k])
        wh[0, :, y, x] = torch.tensor(t['wh'][k])
        ang[0, 0, y, x] = float(t['ang'][k, 0])
    d = decode({'hm': hm, 'reg': reg, 'wh': wh, 'ang': ang}, K=10)[0].numpy()
    d = d[d[:, 5] > 0.5]
    assert len(d) == len(b)
    for box in b:
        assert np.abs(d[:, :5] - box).sum(1).min() < 1e-3


def test_periodic_loss():
    t = torch.tensor([[[0.3]]])
    for k in (-2, -1, 1, 2):
        p = t + k * math.pi
        assert abs(float(periodic_diff(p, t))) < 1e-5
    m = torch.ones(1, 1)
    # nearly identical box expressed at the opposite end of the range -> ~0 loss
    a = angle_loss(torch.tensor([[[math.pi / 2 - 0.01]]]), torch.tensor([[[-math.pi / 2 + 0.01]]]),
                   m, 'periodic_smooth_l1')
    assert float(a) < 1e-3
    a = angle_loss(torch.tensor([[[math.pi / 2 - 0.01]]]), torch.tensor([[[-math.pi / 2 + 0.01]]]),
                   m, 'l1')
    assert float(a) > 3.0


def test_eval_perfect_and_iou():
    b = random_boxes(8, 500)
    deg = rad2deg_boxes(b).tolist()
    ious = rotated_iou(deg, deg, 500, 500)
    assert np.allclose(np.diag(ious), 1.0, atol=1e-6)
    # w/h swap + 90 deg is the same rectangle -> IoU 1
    sw = [[x, y, h, w, a + 90] for x, y, w, h, a in deg]
    assert np.allclose(np.diag(rotated_iou(sw, deg, 500, 500)), 1.0, atol=1e-6)
    gt = {'images': [{'id': 'a', 'width': 500, 'height': 500}],
          'annotations': [{'image_id': 'a', 'bbox': d, 'category_id': 1} for d in deg]}
    dts = [{'image_id': 'a', 'bbox': d, 'score': 0.9} for d in deg]
    m = ap_metrics(gt, dts)
    assert m['AP50'] > 0.99
    assert prf_counts(gt, dts) == (8, 0, 0)




def test_dcn_gridsample_matches_torchvision():
    from torchvision.ops import deform_conv2d
    from arpd.models.neck import deform_conv2d_gs
    torch.manual_seed(0)
    x = torch.randn(2, 5, 9, 11, dtype=torch.float64, requires_grad=True)
    off = (torch.randn(2, 18, 9, 11, dtype=torch.float64) * 2.5).requires_grad_()
    m = torch.rand(2, 9, 9, 11, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 5, 3, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, dtype=torch.float64, requires_grad=True)
    ref = deform_conv2d(x, off, w, b, padding=1, mask=m)
    got = deform_conv2d_gs(x, off, w, b, padding=1, mask=m)
    assert torch.allclose(ref, got, atol=1e-8), (ref - got).abs().max()
    g1 = torch.autograd.grad(ref.square().sum(), [x, off, m, w, b])
    g2 = torch.autograd.grad(got.square().sum(), [x, off, m, w, b])
    for a, c in zip(g1, g2):
        assert torch.allclose(a, c, atol=1e-6), (a - c).abs().max()


def test_angle_loss_matches_atan_form():
    """Remainder-based wrap + smooth L1 == the atan(sin/cos) form written literally."""
    def reference(pred, tgt):
        dp = torch.atan(torch.sin(pred - tgt) / torch.cos(pred - tgt))
        return torch.where(dp.abs() >= 1, dp.abs() - 0.5, 0.5 * dp ** 2)
    g = torch.Generator().manual_seed(0)
    p = (torch.rand(10000, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    t = (torch.rand(10000, generator=g, dtype=torch.float64) - 0.5) * math.pi
    p1, p2 = p.clone().requires_grad_(), p.clone().requires_grad_()
    l1 = reference(p1, t)
    l2 = torch.nn.functional.smooth_l1_loss(periodic_diff(p2, t), torch.zeros_like(t),
                                            reduction='none', beta=1.0)
    l1.sum().backward()
    l2.sum().backward()
    assert torch.allclose(l1, l2, atol=1e-12) and torch.allclose(p1.grad, p2.grad, atol=1e-12)


def test_augmentation_changes_across_epochs_with_persistent_workers():
    """Regression: persistent workers once froze the epoch, so every epoch replayed the
    same augmentation per image (COCO pretraining memorised one augmented copy)."""
    import tempfile
    import cv2
    from torch.utils.data import DataLoader, SequentialSampler
    from arpd.config import load_config
    from arpd.data.datasets import EpochTaggedSampler, Records, TrainDataset
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, 'configs/base.yaml'), ['input_size=128'])
    with tempfile.TemporaryDirectory() as d:
        recs = Records()
        for k in range(4):
            p = os.path.join(d, f'{k}.jpg')
            cv2.imwrite(p, np.random.default_rng(k).integers(0, 255, (200, 300, 3), dtype=np.uint8))
            recs.add(p, k, 'v', [[150, 100, 30, 60, 0.3]])
        ds = TrainDataset(recs.freeze(), cfg, seed=0)
        sampler = EpochTaggedSampler(SequentialSampler(ds))
        dl = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=2, persistent_workers=True)
        seen = []
        for epoch in (0, 1, 0):
            ds.set_epoch(epoch)       # what train.py does; workers never see this one
            sampler.set_epoch(epoch)  # this one reaches them
            seen.append(next(iter(dl))['input'])
        assert not torch.equal(seen[0], seen[1]), 'epoch 1 replayed epoch 0 augmentation'
        assert torch.equal(seen[0], seen[2]), 'augmentation must be deterministic per (epoch, index)'


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
