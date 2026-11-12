"""Rotated-box geometry.

Convention (same as CEPDOF / HABBOF / MW-R annotations):
    box = [cx, cy, w, h, theta]
    * (cx, cy) centre in pixels, image y-axis points down
    * w <= h  (h is the long side, i.e. a standing person has theta = 0)
    * theta: clockwise rotation (as seen on screen), radians internally,
      canonical range [-pi/2, pi/2). Annotation files store degrees.

The "up" unit vector of a box (centre -> middle of the top edge) is
(sin theta, -cos theta); the "right" vector is (cos theta, sin theta).
"""
import math

import numpy as np

HALF_PI = math.pi / 2


def wrap_angle(theta):
    """Wrap angle(s) in radians to the canonical range [-pi/2, pi/2)."""
    return (theta + HALF_PI) % math.pi - HALF_PI


def canonicalize(boxes):
    """Return a copy with w <= h and theta in [-pi/2, pi/2). boxes: (N, 5)."""
    b = np.array(boxes, dtype=np.float64).reshape(-1, 5)
    swap = b[:, 2] > b[:, 3]
    b[swap, 2], b[swap, 3] = b[swap, 3].copy(), b[swap, 2].copy()
    b[swap, 4] += HALF_PI  # same rectangle, rotated 90 deg with sides swapped
    b[:, 4] = wrap_angle(b[:, 4])
    return b


def deg2rad_boxes(boxes_deg):
    b = np.array(boxes_deg, dtype=np.float64).reshape(-1, 5)
    b[:, 4] = np.deg2rad(b[:, 4])
    return b


def rad2deg_boxes(boxes_rad):
    b = np.array(boxes_rad, dtype=np.float64).reshape(-1, 5)
    b[:, 4] = np.rad2deg(b[:, 4])
    return b


def affine_boxes(boxes, M):
    """Apply a 2x3 affine matrix with an isotropic linear part
    (rotation / reflection / uniform scale) to rotated boxes.

    Works for reflections too: the box orientation is recovered from the
    transformed "up" vector, so no sign bookkeeping is needed.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    if len(b) == 0:
        return b.copy()
    M = np.asarray(M, dtype=np.float64)
    A, t = M[:, :2], M[:, 2]
    scale = math.sqrt(abs(np.linalg.det(A)))
    centres = b[:, :2] @ A.T + t
    th = b[:, 4]
    up = np.stack([np.sin(th), -np.cos(th)], axis=1) @ A.T
    th_new = np.arctan2(up[:, 0], -up[:, 1])
    out = np.concatenate(
        [centres, b[:, 2:3] * scale, b[:, 3:4] * scale, th_new[:, None]], axis=1)
    return canonicalize(out)


def box_vertices(boxes):
    """(N, 5) -> (N, 4, 2) corners: top-left, top-right, bottom-right, bottom-left."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    c = b[:, None, :2]
    w, h, th = b[:, 2:3], b[:, 3:4], b[:, 4:5]
    up = np.concatenate([np.sin(th), -np.cos(th)], axis=1)[:, None] * (h[:, None] / 2)
    right = np.concatenate([np.cos(th), np.sin(th)], axis=1)[:, None] * (w[:, None] / 2)
    tl = c + up - right
    tr = c + up + right
    br = c - up + right
    bl = c - up - right
    return np.concatenate([tl, tr, br, bl], axis=1)
