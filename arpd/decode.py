"""Decode network outputs into rotated boxes (no NMS: 3x3 max-pool peaks)."""
import numpy as np
import torch
import torch.nn.functional as F

from .boxes import affine_boxes, rad2deg_boxes
from .losses import gather


@torch.no_grad()
def decode(out, stride=4, K=100, angle_repr='tanh'):
    """Returns (B, K, 6) tensor [cx, cy, w, h, theta_rad, score] in input pixels."""
    hm = torch.sigmoid(out['hm'].float())
    B, C, H, W = hm.shape
    keep = (F.max_pool2d(hm, 3, stride=1, padding=1) == hm).float()
    scores, inds = (hm * keep).view(B, -1).topk(K)
    inds = inds % (H * W)  # single class
    ys = (inds // W).float()
    xs = (inds % W).float()
    reg = gather(out['reg'].float(), inds)
    wh = gather(out['wh'].float(), inds)
    ang = gather(out['ang'].float(), inds)
    if angle_repr == 'sincos':
        theta = 0.5 * torch.atan2(ang[..., 0], ang[..., 1])
    else:
        theta = ang[..., 0]
    cx = (xs + reg[..., 0]) * stride
    cy = (ys + reg[..., 1]) * stride
    w = wh[..., 0].clamp(min=0) * stride
    h = wh[..., 1].clamp(min=0) * stride
    return torch.stack([cx, cy, w, h, theta, scores], dim=2)


def to_image_coords(dets, Minv, score_thresh=0.0):
    """dets: (K, 6) numpy for one image -> list of dicts with CEPDOF degrees."""
    dets = dets[dets[:, 5] >= score_thresh]
    if len(dets) == 0:
        return []
    boxes = rad2deg_boxes(affine_boxes(dets[:, :5], Minv))
    return [{'bbox': [float(v) for v in b], 'score': float(s)}
            for b, s in zip(boxes, dets[:, 5])]
