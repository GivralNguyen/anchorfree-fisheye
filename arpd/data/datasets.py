"""Datasets: COCO person (perspective) and overhead fisheye videos.

All sources are flattened into compact numpy arrays (paths + concatenated boxes)
so that forked dataloader workers do not duplicate large Python object graphs.

Fisheye layout expected under ``root`` (CEPDOF's native layout; HABBOF and
MW-R are converted to it by the scripts in tools/):

    <root>/<DATASET>/annotations/<video>.json      CEPDOF-format json
    <root>/<DATASET>/<video>/<file_name>            frames
"""
import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..boxes import canonicalize, deg2rad_boxes
from .targets import encode_targets
from .transforms import TestTransform, TrainTransform

cv2.setNumThreads(0)  # dataloader workers provide the parallelism


class Records:
    """Flat container: image paths, ids, video names and per-image boxes (radians)."""

    def __init__(self):
        self.paths, self.ids, self.videos, self.boxes, self.rois = [], [], [], [], []

    def add(self, path, image_id, video, boxes, roi=''):
        self.paths.append(path)
        self.ids.append(image_id)
        self.videos.append(video)
        self.boxes.append(np.asarray(boxes, dtype=np.float32).reshape(-1, 5))
        self.rois.append(roi)

    def extend(self, other):
        for a in ('paths', 'ids', 'videos', 'boxes', 'rois'):
            getattr(self, a).extend(getattr(other, a))

    def freeze(self):
        lens = np.array([len(b) for b in self.boxes], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(lens)])
        self.all_boxes = (np.concatenate(self.boxes) if len(self.boxes)
                          else np.zeros((0, 5), np.float32))
        self.paths = np.array(self.paths)
        self.ids = np.array([str(i) for i in self.ids])
        self.videos = np.array(self.videos)
        self.rois = np.array(self.rois)
        del self.boxes
        return self

    def __len__(self):
        return len(self.paths)

    def get_boxes(self, i):
        return self.all_boxes[self.offsets[i]:self.offsets[i + 1]].astype(np.float64)


def load_coco_person(img_dir, ann_json, skip_crowd_images=True, min_size=4.0):
    """COCO 2017 person boxes as rotated boxes with theta = 0 (canonicalised)."""
    with open(ann_json) as f:
        data = json.load(f)
    person_ids = {c['id'] for c in data['categories'] if c['name'] == 'person'}
    per_img, crowd = {}, set()
    for a in data['annotations']:
        if a['category_id'] not in person_ids:
            continue
        if a.get('iscrowd', 0):
            crowd.add(a['image_id'])
            continue
        x, y, w, h = a['bbox']
        if w < min_size or h < min_size:
            continue
        per_img.setdefault(a['image_id'], []).append([x + w / 2, y + h / 2, w, h, 0.0])
    recs = Records()
    for img in data['images']:
        iid = img['id']
        if iid not in per_img or (skip_crowd_images and iid in crowd):
            continue
        recs.add(os.path.join(img_dir, img['file_name']), iid, 'coco',
                 canonicalize(per_img[iid]))
    return recs


def discover_videos(root, dataset):
    files = sorted(glob.glob(os.path.join(root, dataset, 'annotations', '*.json')))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def roi_path_for(ann_path):
    """<root>/<DS>/annotations/<v>.json -> <root>/<DS>/roi/<v>.png if it exists, else ''."""
    d, f = os.path.split(ann_path)
    p = os.path.join(os.path.dirname(d), 'roi', os.path.splitext(f)[0] + '.png')
    return p if os.path.exists(p) else ''


def load_roi(path, shape):
    """Binary ROI mask (H, W) resized to ``shape`` if needed."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m.shape[:2] != tuple(shape[:2]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def load_fisheye_video(root, dataset, video):
    """One CEPDOF-format video. Returns (Records, gt_json_path)."""
    ann_path = os.path.join(root, dataset, 'annotations', f'{video}.json')
    roi = roi_path_for(ann_path)
    with open(ann_path) as f:
        data = json.load(f)
    img_dir = os.path.join(root, dataset, video)
    per_img = {}
    for a in data['annotations']:
        per_img.setdefault(a['image_id'], []).append(a['bbox'])
    recs = Records()
    for img in data['images']:
        boxes = per_img.get(img['id'], [])
        boxes = canonicalize(deg2rad_boxes(boxes)) if boxes else np.zeros((0, 5))
        recs.add(os.path.join(img_dir, img['file_name']), img['id'], f'{dataset}/{video}', boxes, roi)
    return recs, ann_path


def build_fisheye(root, dataset, include=None, exclude=None):
    found = discover_videos(root, dataset)
    unknown = sorted(set(include or []) | set(exclude or []))
    unknown = [v for v in unknown if v not in found]
    if unknown:  # a typo must not silently change the training set
        raise ValueError(f'{dataset}: unknown videos {unknown}; available {found}')
    videos = [v for v in (include or found) if v not in set(exclude or [])]
    if not videos:
        raise FileNotFoundError(f'No videos found for {dataset} under {root}')
    recs, gts = Records(), {}
    for v in videos:
        r, gt = load_fisheye_video(root, dataset, v)
        recs.extend(r)
        gts[f'{dataset}/{v}'] = gt
    return recs, gts


def build_source(spec, data_cfg):
    """spec: {'type': 'coco'|'fisheye', ...}. Returns (Records, {video: gt_json})."""
    if spec['type'] == 'coco':
        root = data_cfg['coco_root']
        split = spec.get('split', 'train2017')
        recs = load_coco_person(os.path.join(root, split),
                                os.path.join(root, 'annotations', f'instances_{split}.json'),
                                skip_crowd_images=spec.get('skip_crowd_images', True))
        return recs, {}
    if spec['type'] == 'fisheye':
        return build_fisheye(data_cfg['fisheye_root'], spec['dataset'],
                             spec.get('include'), spec.get('exclude'))
    if spec['type'] == 'json':  # a single CEPDOF-format json + image folder
        name = spec.get('name', os.path.splitext(os.path.basename(spec['ann']))[0])
        with open(spec['ann']) as f:
            data = json.load(f)
        per_img = {}
        for a in data['annotations']:
            per_img.setdefault(a['image_id'], []).append(a['bbox'])
        recs = Records()
        for img in data['images']:
            b = per_img.get(img['id'], [])
            recs.add(os.path.join(spec['img_dir'], img['file_name']), img['id'], name,
                     canonicalize(deg2rad_boxes(b)) if b else np.zeros((0, 5)))
        return recs, {name: spec['ann']}
    raise ValueError(spec['type'])


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class EpochTaggedSampler(torch.utils.data.Sampler):
    """Wraps a sampler and yields (epoch, index) so augmentation seeds follow the epoch.

    The sampler lives in the main process, so set_epoch() here always takes effect,
    unlike TrainDataset.set_epoch() which persistent workers never see.
    """

    def __init__(self, base):
        self.base, self.epoch = base, 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.base, 'set_epoch'):
            self.base.set_epoch(epoch)

    def __iter__(self):
        e = self.epoch
        return ((e, int(j)) for j in self.base)

    def __len__(self):
        return len(self.base)


class TrainDataset(Dataset):
    def __init__(self, records, cfg, seed=0):
        self.recs = records
        self.tf = TrainTransform(cfg['augment'] | {'input_size': cfg['input_size']})
        self.cfg = cfg
        self.seed = seed
        self.epoch = 0
        self.repeat = int(cfg['train'].get('repeat', 1))  # >1 only for tiny debug sets

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.recs) * self.repeat

    def __getitem__(self, key):
        # key is (epoch, j) when fed by EpochTaggedSampler. The epoch must travel with the
        # index: with persistent DataLoader workers, set_epoch() only reaches the main
        # process's copy, and every epoch would replay the same augmentation per image.
        epoch, j = key if isinstance(key, tuple) else (self.epoch, key)
        rng = np.random.default_rng((self.seed, epoch, j))
        i = j % len(self.recs)
        img = read_rgb(self.recs.paths[i])
        roi = self.recs.rois[i]
        if roi and self.cfg['augment'].get('mask_outside_roi', True):
            # unannotated people outside the ROI must not become false negatives
            img[~load_roi(roi, img.shape)] = 0
        x, boxes = self.tf(img, self.recs.get_boxes(i), rng)
        t = encode_targets(boxes, self.cfg['input_size'], stride=self.cfg['stride'],
                           num_classes=1, max_objs=self.cfg['max_objs'],
                           angle_repr=self.cfg['model'].get('angle_repr', 'tanh'))
        out = {k: torch.from_numpy(v) for k, v in t.items()}
        out['input'] = torch.from_numpy(x.transpose(2, 0, 1).copy())
        return out


class TestDataset(Dataset):
    def __init__(self, records, cfg):
        self.recs = records
        self.tf = TestTransform({'input_size': cfg['input_size']})

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        img = read_rgb(self.recs.paths[i])
        x, Minv = self.tf(img)
        return {'input': torch.from_numpy(x.transpose(2, 0, 1).copy()),
                'Minv': torch.from_numpy(Minv), 'index': i}
