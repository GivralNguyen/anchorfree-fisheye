"""CenterNet-style training targets for rotated boxes."""
import numpy as np


def gaussian_radius(h, w, min_overlap=0.7):
    """Gaussian radius from CornerNet / CenterNet (kept identical on purpose)."""
    a1, b1 = 1, h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + np.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def _gaussian2d(radius):
    d = 2 * radius + 1
    sigma = d / 6
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    return g


def draw_gaussian(heatmap, cx, cy, radius):
    g = _gaussian2d(radius)
    H, W = heatmap.shape
    left, right = min(cx, radius), min(W - cx, radius + 1)
    top, bottom = min(cy, radius), min(H - cy, radius + 1)
    hm = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    gg = g[radius - top:radius + bottom, radius - left:radius + right]
    if min(hm.shape) > 0 and min(gg.shape) > 0:
        np.maximum(hm, gg, out=hm)


def encode_targets(boxes, input_size, stride=4, num_classes=1, max_objs=128,
                   angle_repr='tanh', min_overlap=0.7):
    """boxes: (N, 5) canonical rotated boxes in network-input pixels, class 0.

    Returns a dict of numpy arrays:
        hm   (C, H, W)  gaussian heatmap
        ind  (K,)       flat index of each object centre
        mask (K,)       1 for valid entries
        reg  (K, 2)     sub-pixel centre offset
        wh   (K, 2)     (w, h) in output-stride units
        ang  (K, 1)     theta (radians)              if angle_repr == 'tanh'
             (K, 2)     (sin 2theta, cos 2theta)     if angle_repr == 'sincos'
    """
    out = input_size // stride
    hm = np.zeros((num_classes, out, out), dtype=np.float32)
    ind = np.zeros(max_objs, dtype=np.int64)
    mask = np.zeros(max_objs, dtype=np.float32)
    reg = np.zeros((max_objs, 2), dtype=np.float32)
    wh = np.zeros((max_objs, 2), dtype=np.float32)
    ang_dim = 2 if angle_repr == 'sincos' else 1
    ang = np.zeros((max_objs, ang_dim), dtype=np.float32)

    # Draw large boxes first so that small ones win the centre cell on collisions.
    order = np.argsort(-(boxes[:, 2] * boxes[:, 3])) if len(boxes) else []
    k = 0
    for i in order:
        if k >= max_objs:
            break
        cx, cy, w, h, th = boxes[i] / np.array([stride, stride, stride, stride, 1.0])
        ci, cj = int(cx), int(cy)
        if not (0 <= ci < out and 0 <= cj < out) or w <= 0 or h <= 0:
            continue
        radius = max(0, int(gaussian_radius(h, w, min_overlap)))
        draw_gaussian(hm[0], ci, cj, radius)
        ind[k] = cj * out + ci
        mask[k] = 1
        reg[k] = (cx - ci, cy - cj)
        wh[k] = (w, h)
        if angle_repr == 'sincos':
            ang[k] = (np.sin(2 * th), np.cos(2 * th))
        else:
            ang[k] = th
        k += 1
    return dict(hm=hm, ind=ind, mask=mask, reg=reg, wh=wh, ang=ang)
