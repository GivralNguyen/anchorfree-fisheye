"""Draw rotated boxes.

  # augmented training samples (checks augmentation + target geometry)
  python tools/visualize.py aug configs/cv_test_WEPDTOF.yaml --n 4 --out vis/
  # detections from an evaluation run over ground truth
  python tools/visualize.py dets configs/cv_test_WEPDTOF.yaml --dets runs/cv_test_WEPDTOF/dets_final.json --n 8
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arpd.boxes import box_vertices, deg2rad_boxes  # noqa: E402
from arpd.config import load_config  # noqa: E402
from arpd.data.datasets import Records, build_source, read_rgb  # noqa: E402
from arpd.data.transforms import IMAGENET_MEAN, IMAGENET_STD, TrainTransform  # noqa: E402


def draw(img, boxes_rad, color, thick=2):
    for v in box_vertices(boxes_rad):
        cv2.polylines(img, [np.round(v).astype(np.int32)], True, color, thick, cv2.LINE_AA)
        top = (v[0] + v[1]) / 2  # mark the "head" edge
        cv2.circle(img, tuple(np.round(top).astype(int)), thick + 1, color, -1)
    return img


def records(cfg, key):
    recs, gts = Records(), {}
    for s in cfg['data'][key]:
        r, g = build_source(s, cfg['data'])
        recs.extend(r)
        gts.update(g)
    return recs.freeze(), gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['aug', 'dets'])
    ap.add_argument('config')
    ap.add_argument('--dets')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--out', default='vis')
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    if args.mode == 'aug':
        recs, _ = records(cfg, 'train')
        tf = TrainTransform(cfg['augment'] | {'input_size': cfg['input_size']})
        for k in range(args.n):
            i = int(rng.integers(len(recs)))
            x, b = tf(read_rgb(recs.paths[i]), recs.get_boxes(i), rng)
            img = np.clip((x * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255).astype(np.uint8)
            img = draw(np.ascontiguousarray(img[..., ::-1]), b, (0, 255, 0))
            cv2.imwrite(os.path.join(args.out, f'aug_{k}.jpg'), img)
    else:
        recs, _ = records(cfg, 'test')
        dets = json.load(open(args.dets))
        by_img = {}
        for v in dets.values():
            for d in v:
                if d['score'] >= args.conf:
                    by_img.setdefault(str(d['image_id']), []).append(d['bbox'])
        for k in rng.choice(len(recs), size=min(args.n, len(recs)), replace=False):
            img = cv2.imread(recs.paths[k])
            draw(img, recs.get_boxes(k), (0, 255, 0), 3)
            d = by_img.get(str(recs.ids[k]), [])
            if d:
                draw(img, deg2rad_boxes(d), (0, 0, 255), 3)
            cv2.imwrite(os.path.join(args.out, f'det_{os.path.basename(recs.paths[k])}'), img)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
