"""Evaluate an official RAPiD checkpoint with *our* evaluator (same-protocol baseline).

Inference follows RAPiD's own api.Detector._predict_pil exactly: resize the long
side to input_size, zero-pad to square, [0,1] tensor (no mean/std), keep
score >= conf, top-1000, rotated NMS at 0.45, map back to the original image.

    python tools/eval_rapid.py ../RAPiD/weights/pL1_HBCP608_Apr14_6000.ckpt \
        --datasets WEPDTOF --input_size 608 --tag HBCP608_in608
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from arpd.config import load_config  # noqa: E402
from arpd.eval.rotated_eval import evaluate_videos  # noqa: E402
from tools.evaluate import fmt_table  # noqa: E402
from tools.train import build_records  # noqa: E402

RAPID = None  # set in main; RAPiD's modules are imported from there


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torchvision.transforms.functional as tvf
        from utils.utils import rect_to_square
        img = Image.open(self.paths[i]).convert('RGB')
        sq, _, pad = rect_to_square(img, None, self.size, 0)
        return tvf.to_tensor(sq), pad, i


def _post(args):
    dts, pad, size = args
    from utils.utils import detection2original, nms
    dts = torch.from_numpy(dts)
    if len(dts) > 1000:
        dts = dts[torch.topk(dts[:, 5], k=1000)[1]]
    dts = nms(dts, is_degree=True, nms_thres=0.45, img_size=size)
    dts = detection2original(dts, torch.from_numpy(pad))
    return dts[:, :6].numpy()


def main():
    global RAPID
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--rapid', default=os.path.join(os.path.dirname(ROOT), 'RAPiD'))
    ap.add_argument('--datasets', nargs='*', default=[], help='fisheye dataset names')
    ap.add_argument('--json', nargs='*', default=[],
                    help='extra CEPDOF-format sets as name:img_dir:ann_json (e.g. COCO val person)')
    ap.add_argument('--input_size', type=int, default=608)
    ap.add_argument('--conf', type=float, default=None, help='default: eval.ap_conf of base.yaml')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--nms_procs', type=int, default=16)
    ap.add_argument('--out', default='runs/rapid_baseline')
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    RAPID = os.path.abspath(args.rapid)
    sys.path.insert(0, RAPID)

    cfg = load_config(os.path.join(ROOT, 'configs/base.yaml'), [])
    ec = cfg['eval']
    conf = args.conf if args.conf is not None else ec.get('ap_conf', 0.01)
    pool = mp.get_context('fork').Pool(args.nms_procs)  # before CUDA init

    from models.rapid import RAPiD
    model = RAPiD(backbone='dark53')
    try:
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    except Exception:  # older checkpoints pickle extra objects
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    # RAPiD's released checkpoints use either key (HBCP/MWCP: model_state_dict)
    sd = ck.get('model') or ck.get('model_state_dict')
    if sd is None:
        raise KeyError(f'no model weights in {args.ckpt}: keys {list(ck)}')
    model.load_state_dict(sd, strict=True)
    print(f'loaded {args.ckpt} (iter {ck.get("iter")})', flush=True)
    model.cuda().eval()

    os.makedirs(args.out, exist_ok=True)
    specs = [(d, {'type': 'fisheye', 'dataset': d}) for d in args.datasets]
    for j in args.json:
        name, img_dir, ann = j.split(':')
        specs.append((name, {'type': 'json', 'name': name, 'img_dir': img_dir, 'ann': ann}))
    for dsname, spec in specs:
        recs, gts = build_records([spec], cfg['data'])
        dl = torch.utils.data.DataLoader(Frames(recs.paths, args.input_size),
                                         batch_size=args.batch_size, num_workers=args.workers)
        jobs, index = [], []
        with torch.no_grad():
            for x, pads, idx in dl:
                out = model(x.cuda(non_blocking=True)).float().cpu()
                for b in range(len(idx)):
                    d = out[b]
                    d = d[d[:, 5] >= conf].numpy()
                    jobs.append((d, pads[b].numpy(), args.input_size))
                    index.append(int(idx[b]))
        results = pool.map(_post, jobs, chunksize=8)
        dets = {}
        for i, r in zip(index, results):
            v, iid = str(recs.videos[i]), str(recs.ids[i])
            dets.setdefault(v, []).extend(
                {'image_id': iid, 'bbox': [float(t) for t in row[:5]], 'score': float(row[5])}
                for row in r)
        res = evaluate_videos(gts, dets, conf=ec.get('conf', 0.3))
        tag = f'{args.tag}_{dsname}'
        with open(os.path.join(args.out, f'metrics_{tag}.json'), 'w') as f:
            json.dump(res, f, indent=1)
        with open(os.path.join(args.out, f'dets_{tag}.json'), 'w') as f:
            json.dump(dets, f)
        print(f'=== RAPiD {args.tag} on {dsname} (input {args.input_size}) ===')
        print(fmt_table(res), flush=True)
    pool.close()


if __name__ == '__main__':
    main()
