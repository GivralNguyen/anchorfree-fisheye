"""Bring the datasets into the layout the code expects.

Fisheye sets (CEPDOF's layout; frames are symlinked, not copied):

    data/fisheye/<DATASET>/annotations/<video>.json
    data/fisheye/<DATASET>/<video>/<frames>.jpg
    data/fisheye/<DATASET>/roi/<video>.png          (WEPDTOF only)

Sub-commands
------------
  organize     Given a folder of CEPDOF-format jsons (CEPDOF, WEPDTOF) and a folder
               (searched recursively) holding the frames, build the layout above.
  habbof       Convert HABBOF per-frame .txt labels ("person cx cy w h deg")
               into one CEPDOF-format json per video.
  coco_person  COCO 2017 val person boxes as a CEPDOF-format json (angle 0).
  monitor      Fixed monitoring subsets used during COCO pretraining (never for
               model selection): 500 random COCO val images, every 10th HABBOF frame.
  check        Print per-video statistics and verify every frame exists.

Examples
--------
  python tools/prepare_data.py organize --dataset CEPDOF --ann_dir <CEPDOF>/annotations \
      --frames <CEPDOF> --root data/fisheye
  python tools/prepare_data.py organize --dataset WEPDTOF --ann_dir <WEPDTOF>/annotations \
      --frames <WEPDTOF>/frames --roi_dir <WEPDTOF>/ROI_maps --root data/fisheye
  python tools/prepare_data.py habbof --src <HABBOF> --root data/fisheye
  python tools/prepare_data.py coco_person --coco data/coco
  python tools/prepare_data.py monitor --coco data/coco --root data/fisheye
  python tools/prepare_data.py check --root data/fisheye
"""
import argparse
import glob
import json
import os
import random

import cv2


def _write_json(data, path):
    """Write atomically so that an interrupted run never leaves a truncated file."""
    with open(path + '.tmp', 'w') as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + '.tmp', path)


def _index_frames(folder):
    idx = {}
    for p in glob.glob(os.path.join(folder, '**', '*'), recursive=True):
        if p.lower().endswith(('.jpg', '.jpeg', '.png')):
            idx.setdefault(os.path.basename(p), p)
    return idx


def organize(args):
    frames = _index_frames(args.frames)
    out_ann = os.path.join(args.root, args.dataset, 'annotations')
    os.makedirs(out_ann, exist_ok=True)
    jobs = []
    for js in sorted(glob.glob(os.path.join(args.ann_dir, '*.json'))):
        jobs.append((os.path.splitext(os.path.basename(js))[0], json.load(open(js))))
    for video, data in jobs:
        vdir = os.path.join(args.root, args.dataset, video)
        os.makedirs(vdir, exist_ok=True)
        missing = 0
        for im in data['images']:
            name = os.path.basename(im['file_name'])
            src = frames.get(name)
            if src is None:
                missing += 1
                continue
            dst = os.path.join(vdir, name)
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(src), dst)
            im['file_name'] = name
        _write_json(data, os.path.join(out_ann, f'{video}.json'))
        if args.roi_dir and os.path.exists(os.path.join(args.roi_dir, f'{video}.png')):
            os.makedirs(os.path.join(args.root, args.dataset, 'roi'), exist_ok=True)
            dst = os.path.join(args.root, args.dataset, 'roi', f'{video}.png')
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(os.path.join(args.roi_dir, f'{video}.png')), dst)
        print(f'{args.dataset}/{video}: {len(data["images"])} images, '
              f'{len(data["annotations"])} boxes, {missing} frames missing')


def habbof(args):
    out_ann = os.path.join(args.root, 'HABBOF', 'annotations')
    os.makedirs(out_ann, exist_ok=True)
    for vdir in sorted(d for d in glob.glob(os.path.join(args.src, '*')) if os.path.isdir(d)):
        video = os.path.basename(vdir)
        txts = sorted(glob.glob(os.path.join(vdir, '*.txt')))
        if not txts:
            continue
        images, anns = [], []
        h = w = None
        dst_dir = os.path.join(args.root, 'HABBOF', video)
        os.makedirs(dst_dir, exist_ok=True)
        for t in txts:
            stem = os.path.splitext(os.path.basename(t))[0]
            jpg = os.path.join(vdir, stem + '.jpg')
            if not os.path.exists(jpg):
                continue
            if h is None:
                h, w = cv2.imread(jpg).shape[:2]
            image_id = f'{video}_{stem}'
            name = f'{image_id}.jpg'
            dst = os.path.join(dst_dir, name)
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(jpg), dst)
            images.append({'file_name': name, 'id': image_id, 'width': w, 'height': h})
            for line in open(t):
                p = line.split()
                if len(p) < 6:
                    continue
                cx, cy, bw, bh, deg = map(float, p[1:6])
                anns.append({'image_id': image_id, 'bbox': [cx, cy, bw, bh, deg],
                             'area': bw * bh, 'category_id': 1, 'iscrowd': 0})
        data = {'images': images, 'annotations': anns,
                'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}]}
        _write_json(data, os.path.join(out_ann, f'{video}.json'))
        print(f'HABBOF/{video}: {len(images)} images, {len(anns)} boxes')


PERSON = [{'id': 1, 'name': 'person'}]


def coco_person(args):
    """instances_val2017.json -> person_val2017_rot.json: [cx, cy, w, h, 0] per person box.

    Crowd boxes are kept with their iscrowd flag (the evaluator ignores matches to them).
    """
    ann = os.path.join(args.coco, 'annotations')
    src = json.load(open(os.path.join(ann, f'instances_{args.split}.json')))
    pid = {c['id'] for c in src['categories'] if c['name'] == 'person'}
    images = [{'id': str(i['id']), 'file_name': i['file_name'], 'width': i['width'],
               'height': i['height']} for i in src['images']]
    anns = []
    for a in src['annotations']:
        if a['category_id'] in pid:
            x, y, w, h = a['bbox']
            anns.append({'image_id': str(a['image_id']), 'bbox': [x + w / 2, y + h / 2, w, h, 0.0],
                         'area': a['area'], 'iscrowd': a['iscrowd'], 'category_id': 1})
    out = os.path.join(ann, f'person_{args.split}_rot.json')
    _write_json({'images': images, 'annotations': anns, 'categories': PERSON}, out)
    print(f'{out}: {len(images)} images, {len(anns)} person boxes')


def monitor(args):
    ann = os.path.join(args.coco, 'annotations')
    full = json.load(open(os.path.join(ann, 'person_val2017_rot.json')))
    random.seed(0)
    imgs = random.sample(full['images'], 500)
    ids = {i['id'] for i in imgs}
    sub = {'images': imgs, 'annotations': [a for a in full['annotations'] if a['image_id'] in ids],
           'categories': full['categories']}
    _write_json(sub, os.path.join(ann, 'person_val2017_monitor500.json'))
    print(f'coco monitor: {len(sub["images"])} images, {len(sub["annotations"])} boxes')
    hab = os.path.join(args.root, 'HABBOF')
    out = {'images': [], 'annotations': [], 'categories': PERSON}
    for js in sorted(glob.glob(os.path.join(hab, 'annotations', '*.json'))):
        v = os.path.splitext(os.path.basename(js))[0]
        d = json.load(open(js))
        keep = d['images'][::10]
        kid = {i['id'] for i in keep}
        out['images'] += [dict(i, file_name=f"{v}/{i['file_name']}") for i in keep]
        out['annotations'] += [a for a in d['annotations'] if a['image_id'] in kid]
    _write_json(out, os.path.join(args.root, 'habbof_every10_monitor.json'))
    print(f'habbof monitor: {len(out["images"])} images, {len(out["annotations"])} boxes')


def check(args):
    for ds in sorted(os.listdir(args.root)):
        for js in sorted(glob.glob(os.path.join(args.root, ds, 'annotations', '*.json'))):
            video = os.path.splitext(os.path.basename(js))[0]
            data = json.load(open(js))
            miss = sum(not os.path.exists(os.path.join(args.root, ds, video, im['file_name']))
                       for im in data['images'])
            print(f'{ds:8s} {video:16s} images={len(data["images"]):6d} '
                  f'boxes={len(data["annotations"]):7d} missing_frames={miss}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('organize')
    a.add_argument('--dataset', required=True)
    a.add_argument('--ann_dir', required=True)
    a.add_argument('--frames', required=True)
    a.add_argument('--root', required=True)
    a.add_argument('--roi_dir', help='folder of <video>.png ROI maps (WEPDTOF)')
    a = sub.add_parser('habbof')
    a.add_argument('--src', required=True)
    a.add_argument('--root', required=True)
    a = sub.add_parser('coco_person')
    a.add_argument('--coco', required=True, help='COCO 2017 root (with annotations/)')
    a.add_argument('--split', default='val2017')
    a = sub.add_parser('monitor')
    a.add_argument('--coco', required=True)
    a.add_argument('--root', required=True, help='fisheye root')
    a = sub.add_parser('check')
    a.add_argument('--root', required=True)
    args = ap.parse_args()
    {'organize': organize, 'habbof': habbof, 'coco_person': coco_person, 'monitor': monitor,
     'check': check}[args.cmd](args)


if __name__ == '__main__':
    main()
