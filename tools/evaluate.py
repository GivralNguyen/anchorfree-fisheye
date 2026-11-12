"""Evaluate a checkpoint on fisheye test sets (CEPDOF protocol).

    python tools/evaluate.py runs/cv_test_WEPDTOF/final.pth          # test set stored in the checkpoint
    python tools/evaluate.py runs/cv_test_WEPDTOF/final.pth --tag in1024 input_size=1024   # any override
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arpd.config import _merge, load_config  # noqa: E402
from arpd.engine import predict  # noqa: E402
from arpd.eval.rotated_eval import evaluate_videos  # noqa: E402
from arpd.models.arpd import build_model  # noqa: E402
from tools.train import build_records  # noqa: E402


def fmt_table(res):
    rows = [f'{"video":28s} {"AP50":>6s} {"AP":>6s} {"AP75":>6s} {"P":>6s} {"R":>6s} {"F1":>6s}']
    for v, m in res['per_video'].items():
        rows.append(f'{v:28s} {m["AP50"]*100:6.1f} {m["AP"]*100:6.1f} {m["AP75"]*100:6.1f} '
                    f'{m["P"]:6.3f} {m["R"]:6.3f} {m["F1"]:6.3f}')
    for name in ('mean', 'pooled'):
        m = res[name]
        rows.append(f'{name.upper():28s} {m["AP50"]*100:6.1f} {m["AP"]*100:6.1f} {m["AP75"]*100:6.1f} '
                    f'{m["P"]:6.3f} {m["R"]:6.3f} {m["F1"]:6.3f}')
    rows.append(f'(F1 of mean P/R: {res["mean"]["F1_of_meanPR"]:.3f})')
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--config', help='defaults to the config stored in the checkpoint')
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--out', help='output directory (default: next to the checkpoint)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('overrides', nargs='*')
    args = ap.parse_intermixed_args()  # allow --flags and key=value overrides in any order

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(args.config, args.overrides) if args.config else _merge(
        ck['cfg'], load_config_from_overrides(args.overrides))
    model = build_model(cfg, pretrained=False)
    sd = ck['model'] if args.no_ema or not ck.get('ema') else ck['ema']
    model.load_state_dict(sd)
    model.cuda().eval()

    recs, gts = build_records(cfg['data']['test'], cfg['data'])
    ec = cfg['eval']
    dets = predict(model, recs, cfg, batch_size=ec.get('batch_size', 16),
                   workers=cfg['train']['workers'], conf=ec.get('ap_conf', 0.01))
    res = evaluate_videos(gts, dets, conf=ec.get('conf', 0.3))
    out = args.out or os.path.dirname(os.path.abspath(args.ckpt))
    tag = args.tag or f'eval_{cfg["input_size"]}'
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f'metrics_{tag}.json'), 'w'), indent=1)
    json.dump(dets, open(os.path.join(out, f'dets_{tag}.json'), 'w'))
    print(fmt_table(res))


def load_config_from_overrides(overrides):
    cfg = {}
    for o in overrides:
        key, val = o.split('=', 1)
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        import yaml
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


if __name__ == '__main__':
    main()
