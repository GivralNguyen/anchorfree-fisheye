"""Rotated-box evaluation (CEPDOF protocol).

* IoU between rotated boxes is computed by rasterising them to RLE masks
  with pycocotools, exactly like RAPiD's ``cepdof_api.py``.
* AP uses the COCO accumulation (101-point interpolated PR curve).
* Precision / recall / F1 at a fixed confidence threshold use greedy,
  score-ordered matching at IoU >= 0.5.
* Following RAPiD, metrics are computed per video and then averaged; pooled
  numbers are reported too. F1 is reported both as the per-video mean of F1
  and as the harmonic mean of the averaged P and R.
"""
import contextlib
import io
import json
from collections import defaultdict

import numpy as np
from pycocotools import cocoeval
from pycocotools import mask as mask_utils

from ..boxes import box_vertices, deg2rad_boxes


def rotated_iou(dts_deg, gts_deg, height, width):
    """dts, gts: lists of [cx, cy, w, h, degrees] -> IoU matrix (D, G)."""
    if len(dts_deg) == 0 or len(gts_deg) == 0:
        return np.zeros((len(dts_deg), len(gts_deg)))
    d = box_vertices(deg2rad_boxes(dts_deg)).reshape(len(dts_deg), 8).tolist()
    g = box_vertices(deg2rad_boxes(gts_deg)).reshape(len(gts_deg), 8).tolist()
    rd = mask_utils.frPyObjects(d, height, width)
    rg = mask_utils.frPyObjects(g, height, width)
    return mask_utils.iou(rd, rg, [0] * len(g))


class RotatedCOCOeval(cocoeval.COCOeval):
    """COCOeval with rotated-box IoU; mirrors RAPiD's CEPDOFeval."""

    def __init__(self, gt_json, dts):
        self.gt_json = gt_json
        self.dt_list = dts
        for i, g in enumerate(gt_json['annotations']):
            g.setdefault('id', i + 1)
            g['image_id'] = str(g['image_id'])
            g.setdefault('area', g['bbox'][2] * g['bbox'][3])
            g.setdefault('category_id', 1)
            g.setdefault('iscrowd', 0)
        for i, d in enumerate(dts):
            d['id'] = i + 1
            d['image_id'] = str(d['image_id'])
            d.setdefault('area', d['bbox'][2] * d['bbox'][3])
            d.setdefault('category_id', 1)
        self.img_info = {str(im['id']): im for im in gt_json['images']}
        self.params = cocoeval.Params(iouType='bbox')
        self.params.imgIds = sorted(self.img_info)
        self.params.catIds = [1]
        self.evalImgs = defaultdict(list)
        self.eval = {}
        self._paramsEval = {}
        self.stats = []
        self.ious = {}

    def _prepare(self):
        self._gts, self._dts = defaultdict(list), defaultdict(list)
        for g in self.gt_json['annotations']:
            g['ignore'] = g.get('ignore', False) or g.get('iscrowd', False)
            self._gts[g['image_id'], 1].append(g)
        for d in self.dt_list:
            self._dts[d['image_id'], 1].append(d)
        self.evalImgs = defaultdict(list)
        self.eval = {}

    def computeIoU(self, imgId, catId):
        gt, dt = self._gts[imgId, catId], self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return []
        order = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in order][:self.params.maxDets[-1]]
        info = self.img_info[imgId]
        return rotated_iou([d['bbox'] for d in dt], [g['bbox'] for g in gt],
                           info['height'], info['width'])


def ap_metrics(gt_json, dts):
    """Returns {'AP': AP@[.5:.95], 'AP50': ..., 'AP75': ...}."""
    if not dts:
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0}
    E = RotatedCOCOeval(gt_json, [dict(d) for d in dts])
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate()
        E.accumulate()
        E.summarize()
    return {'AP': float(E.stats[0]), 'AP50': float(E.stats[1]), 'AP75': float(E.stats[2])}


def prf_counts(gt_json, dts, conf=0.3, iou_thr=0.5):
    """Greedy matching at a fixed score threshold -> (TP, FP, FN)."""
    gts = defaultdict(list)
    for g in gt_json['annotations']:
        if not g.get('iscrowd', False):
            gts[str(g['image_id'])].append(g['bbox'])
    dt_by_img = defaultdict(list)
    for d in dts:
        if d['score'] >= conf:
            dt_by_img[str(d['image_id'])].append(d)
    tp = fp = fn = 0
    for im in gt_json['images']:
        iid = str(im['id'])
        g = gts[iid]
        d = sorted(dt_by_img[iid], key=lambda x: -x['score'])
        ious = rotated_iou([x['bbox'] for x in d], g, im['height'], im['width'])
        matched = np.zeros(len(g), dtype=bool)
        for i in range(len(d)):
            if len(g):
                cand = np.where(~matched, ious[i], -1)
                j = int(np.argmax(cand))
                if cand[j] >= iou_thr:
                    matched[j] = True
                    tp += 1
                    continue
            fp += 1
        fn += int((~matched).sum())
    return tp, fp, fn


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def filter_roi(dts, roi_path, img_info):
    """Discard detections whose centre lies outside the ROI map (WEPDTOF protocol)."""
    import cv2
    roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE) > 127
    keep = []
    for d in dts:
        im = img_info[str(d['image_id'])]
        sx, sy = roi.shape[1] / im['width'], roi.shape[0] / im['height']
        x = min(max(int(d['bbox'][0] * sx), 0), roi.shape[1] - 1)
        y = min(max(int(d['bbox'][1] * sy), 0), roi.shape[0] - 1)
        if roi[y, x]:
            keep.append(d)
    return keep


def evaluate_videos(gt_paths, dets_by_video, conf=0.3):
    """gt_paths: {video: gt_json_path}; dets_by_video: {video: [det dicts]}.

    Returns {'per_video': {...}, 'mean': {...}, 'pooled': {...}}.
    """
    per_video, pooled_gt, pooled_dt = {}, {'images': [], 'annotations': []}, []
    totals = np.zeros(3, dtype=np.int64)
    for video, path in gt_paths.items():
        with open(path) as f:
            gt = json.load(f)
        dts = dets_by_video.get(video, [])
        from ..data.datasets import roi_path_for
        roi = roi_path_for(path)
        if roi:
            dts = filter_roi(dts, roi, {str(im['id']): im for im in gt['images']})
        m = ap_metrics(gt, dts)
        tp, fp, fn = prf_counts(gt, dts, conf)
        totals += (tp, fp, fn)
        m['P'], m['R'], m['F1'] = _prf(tp, fp, fn)
        m.update(TP=tp, FP=fp, FN=fn, n_images=len(gt['images']), n_gt=len(gt['annotations']))
        per_video[video] = m
        # namespace ids so different videos cannot collide in the pooled eval
        for im in gt['images']:
            pooled_gt['images'].append(dict(im, id=f"{video}|{im['id']}"))
        for a in gt['annotations']:
            pooled_gt['annotations'].append(dict(a, image_id=f"{video}|{a['image_id']}"))
        pooled_dt += [dict(d, image_id=f"{video}|{d['image_id']}") for d in dts]

    keys = ['AP', 'AP50', 'AP75', 'P', 'R', 'F1']
    mean = {k: float(np.mean([v[k] for v in per_video.values()])) for k in keys}
    p, r = mean['P'], mean['R']
    mean['F1_of_meanPR'] = 2 * p * r / (p + r) if p + r else 0.0
    pooled = ap_metrics(pooled_gt, pooled_dt)
    pooled['P'], pooled['R'], pooled['F1'] = _prf(*totals)
    return {'per_video': per_video, 'mean': mean, 'pooled': pooled}
