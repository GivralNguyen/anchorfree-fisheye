"""Rotated copy of a CEPDOF-format val set (e.g. COCO val2017 person) for rotation-robustness tests.

Each image is rotated about its centre by theta ~ U(-r, r) degrees (r = 180 matches
augment.rot_range in training), seeded by (seed, image index), onto a canvas just large
enough to hold the whole rotated image (black fill, as in training), so no person is
cropped. Boxes are transformed exactly with boxes.affine_boxes: a rotated image of an
axis-aligned box is the same rectangle at angle theta, so labels stay exact.

    python tools/make_rotated_val.py \
        --ann data/coco/annotations/person_val2017_rot.json --img_dir data/coco/val2017 \
        --out_img data/coco/val2017_rot180_s0 --out_ann data/coco/annotations/person_val2017_rot180_s0.json \
        --subset data/coco/annotations/person_val2017_monitor500.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arpd.boxes import affine_boxes, canonicalize, deg2rad_boxes, rad2deg_boxes  # noqa: E402


def rot_matrix(w, h, deg):
    """Rotate (clockwise on screen, as make_affine) about the centre onto an enlarged canvas."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    W = int(math.ceil(abs(w * c) + abs(h * s)))
    H = int(math.ceil(abs(w * s) + abs(h * c)))
    R = np.array([[c, -s], [s, c]])
    t = np.array([W / 2, H / 2]) - R @ np.array([w / 2, h / 2])
    return np.concatenate([R, t[:, None]], 1), W, H


def work(job):
    src, dst, w, h, deg, quality = job
    M, W, H = rot_matrix(w, h, deg)
    if not os.path.exists(dst):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        assert img is not None and img.shape[:2] == (h, w), src
        out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return M, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--out_img', required=True)
    ap.add_argument('--out_ann', required=True)
    ap.add_argument('--subset', nargs='*', default=[],
                    help='other jsons over the same images; each gets a rotated twin (*_rot180_s0.json)')
    ap.add_argument('--rot_range', type=float, default=180.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quality', type=int, default=95)
    ap.add_argument('--procs', type=int, default=4)
    args = ap.parse_args()

    data = json.load(open(args.ann))
    os.makedirs(args.out_img, exist_ok=True)
    angles, jobs = {}, []
    for k, im in enumerate(sorted(data['images'], key=lambda i: str(i['id']))):
        deg = float(np.random.default_rng((args.seed, k)).uniform(-args.rot_range, args.rot_range))
        angles[str(im['id'])] = deg
        jobs.append((os.path.join(args.img_dir, im['file_name']),
                     os.path.join(args.out_img, im['file_name']),
                     im['width'], im['height'], deg, args.quality))
    with mp.Pool(args.procs) as pool:
        geo = dict(zip(angles, pool.map(work, jobs, chunksize=32)))

    tag = os.path.splitext(os.path.basename(args.out_ann))[0]

    def rotate(d):
        out = {k: v for k, v in d.items() if k not in ('images', 'annotations')}
        out['images'] = []
        for im in d['images']:
            M, W, H = geo[str(im['id'])]
            out['images'].append(dict(im, width=W, height=H, rot_deg=angles[str(im['id'])]))
        out['annotations'] = []
        for a in d['annotations']:
            M = geo[str(a['image_id'])][0]
            b = affine_boxes(canonicalize(deg2rad_boxes([a['bbox']])), M)
            out['annotations'].append(dict(a, bbox=[round(float(x), 3) for x in rad2deg_boxes(b)[0]]))
        out['info'] = {'description': f'{tag}: rotated by U(-{args.rot_range}, {args.rot_range}) deg, '
                       f'seed {args.seed}, canvas enlarged (no crop)', 'source': args.ann}
        return out

    with open(args.out_ann, 'w') as f:
        json.dump(rotate(data), f)
    print('wrote', args.out_ann, len(data['images']), 'images', len(data['annotations']), 'boxes')
    for p in args.subset:
        sub = json.load(open(p))
        missing = {str(i['id']) for i in sub['images']} - set(geo)
        assert not missing, f'{p}: {len(missing)} images not in {args.ann}'
        q = p.replace('.json', '_rot180_s0.json') if args.rot_range == 180 and args.seed == 0 else \
            p.replace('.json', f'_rot{args.rot_range:g}_s{args.seed}.json')
        with open(q, 'w') as f:
            json.dump(rotate(sub), f)
        print('wrote', q, len(sub['images']), 'images', len(sub['annotations']), 'boxes')


if __name__ == '__main__':
    main()
