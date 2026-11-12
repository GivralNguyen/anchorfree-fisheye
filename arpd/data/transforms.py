"""Geometric + photometric augmentation for rotated boxes.

Every geometric transform is a single 2x3 affine matrix (letterbox, uniform
scale, rotation, flips, shift). The image is warped once with cv2.warpAffine
and the boxes are transformed exactly with ``boxes.affine_boxes``. Because
the linear part is always isotropic, rotated rectangles stay rotated
rectangles: there is no approximation error in the targets.
"""
import math

import cv2
import numpy as np

from ..boxes import affine_boxes

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_affine(src_w, src_h, out_size, scale=1.0, rot_deg=0.0,
                shift=(0.0, 0.0), hflip=False, vflip=False):
    """Matrix mapping source pixels -> output pixels.

    1. move the source centre to the origin, 2. optional flips,
    3. letterbox scale (longest side -> out_size) times ``scale``,
    4. rotate by ``rot_deg`` (clockwise on screen),
    5. move to the output centre plus ``shift`` (fraction of out_size).
    """
    T1 = np.array([[1, 0, -src_w / 2], [0, 1, -src_h / 2], [0, 0, 1]], dtype=np.float64)
    F = np.diag([-1.0 if hflip else 1.0, -1.0 if vflip else 1.0, 1.0])
    s = out_size / max(src_w, src_h) * scale
    S = np.diag([s, s, 1.0])
    a = math.radians(rot_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0],
                  [math.sin(a), math.cos(a), 0],
                  [0, 0, 1]], dtype=np.float64)
    T2 = np.array([[1, 0, out_size / 2 + shift[0] * out_size],
                   [0, 1, out_size / 2 + shift[1] * out_size],
                   [0, 0, 1]], dtype=np.float64)
    return (T2 @ R @ S @ F @ T1)[:2]


def warp(img, M, out_size):
    return cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def color_jitter(img, rng, brightness=0.4, contrast=0.4, saturation=0.4):
    """CenterNet-style colour augmentation on a float32 RGB image in [0, 1]."""
    gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def _brightness(x):
        return x * (1 + rng.uniform(-brightness, brightness))

    def _contrast(x):
        return (x - gray.mean()) * (1 + rng.uniform(-contrast, contrast)) + gray.mean()

    def _saturation(x):
        g = (x @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        return (x - g) * (1 + rng.uniform(-saturation, saturation)) + g

    ops = [_brightness, _contrast, _saturation]
    for i in rng.permutation(len(ops)):
        img = ops[i](img)
    return np.clip(img, 0.0, 1.0)


def normalize(img_rgb_uint8):
    x = img_rgb_uint8.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class TrainTransform:
    """Random scale / shift / rotation / flips / colour, then encode to tensor."""

    def __init__(self, cfg):
        self.size = cfg['input_size']
        self.scale_range = cfg.get('scale_range', [0.6, 1.4])
        self.shift = cfg.get('shift', 0.1)
        self.rot_prob = cfg.get('rot_prob', 1.0)
        self.rot_range = cfg.get('rot_range', 180.0)  # uniform in [-r, r] degrees
        self.hflip = cfg.get('hflip', 0.5)
        self.vflip = cfg.get('vflip', 0.5)
        self.color = cfg.get('color_jitter', 0.4)
        self.min_box = cfg.get('min_box_size', 2.0)  # pixels at network input

    def __call__(self, img, boxes, rng):
        h, w = img.shape[:2]
        scale = rng.uniform(*self.scale_range)
        shift = rng.uniform(-self.shift, self.shift, size=2)
        rot = rng.uniform(-self.rot_range, self.rot_range) if rng.random() < self.rot_prob else 0.0
        M = make_affine(w, h, self.size, scale=scale, rot_deg=rot, shift=shift,
                        hflip=rng.random() < self.hflip, vflip=rng.random() < self.vflip)
        out = warp(img, M, self.size)
        b = affine_boxes(boxes, M)
        if len(b):
            keep = ((b[:, 0] >= 0) & (b[:, 0] < self.size) & (b[:, 1] >= 0) & (b[:, 1] < self.size)
                    & (b[:, 2] >= self.min_box))
            b = b[keep]
        x = out.astype(np.float32) / 255.0
        if self.color > 0:
            x = color_jitter(x, rng, self.color, self.color, self.color)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, b


class TestTransform:
    """Letterbox to a square input. Returns the matrix to map detections back."""

    def __init__(self, cfg):
        self.size = cfg['input_size']

    def __call__(self, img):
        h, w = img.shape[:2]
        M = make_affine(w, h, self.size)
        x = normalize(warp(img, M, self.size))
        Minv = cv2.invertAffineTransform(M)
        return x, Minv
