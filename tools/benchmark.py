"""Measure inference speed at batch size 1: same device, same image, each model's own pipeline.

Reports three timings per configuration:
  net      : forward pass only
  net+dec  : forward + decoding (ARPD: peak extraction; RAPiD: score filter + rotated NMS)
  e2e      : JPEG read + resize/pad + host-to-device copy + forward + decoding + map to
             image coordinates (the "total" time reported in the README)

    python tools/benchmark.py --ckpt runs/cv_test_WEPDTOF/final.pth --sizes 512 608 1024 --dtype fp32 \
        --rapid ../RAPiD --rapid_ckpt ../RAPiD/weights/pL1_HBCP608_Apr14_6000.ckpt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arpd.config import load_config  # noqa: E402
from arpd.data.datasets import read_rgb  # noqa: E402
from arpd.data.transforms import TestTransform  # noqa: E402
from arpd.decode import decode, to_image_coords  # noqa: E402
from arpd.models.arpd import build_model  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def timeit(fn, n, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, 'configs/base.yaml'))
    ap.add_argument('--ckpt')
    ap.add_argument('--sizes', type=int, nargs='+', default=[512, 608, 1024])
    ap.add_argument('--dtype', nargs='+', default=['fp32', 'bf16'])
    ap.add_argument('--image', default='data/fisheye/WEPDTOF/exhibition/exhibition_000001.jpg')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--rapid', help='path to the RAPiD repo to also time its network')
    ap.add_argument('--rapid_ckpt', help='RAPiD .ckpt (needed to time its NMS realistically)')
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    cfg = load_config(args.config)
    model = build_model(cfg, pretrained=False)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ck.get('ema') or ck['model'])
    model = model.cuda().eval()
    img = read_rgb(args.image)
    print(f'torch {torch.__version__} | image {args.image} ({img.shape[1]}x{img.shape[0]})')
    print(f'{"model":8s} {"size":>5s} {"dtype":>5s} {"net ms":>8s} {"net+dec":>8s} {"e2e ms":>8s} {"e2e FPS":>8s}')

    for size in args.sizes:
        cfg_s = dict(cfg, input_size=size)
        tf = TestTransform({'input_size': size})
        x = torch.randn(1, 3, size, size, device='cuda')
        for dt in args.dtype:
            amp = dt == 'bf16'

            def net():
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    return model(x)

            def net_dec():
                return decode(net(), stride=cfg_s['stride'], K=100,
                              angle_repr=cfg['model']['angle_repr'])

            def e2e():
                im = read_rgb(args.image)
                xi, Minv = tf(im)
                t = torch.from_numpy(xi.transpose(2, 0, 1)[None].copy()).cuda(non_blocking=True)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    o = model(t)
                d = decode(o, stride=cfg_s['stride'], K=100,
                           angle_repr=cfg['model']['angle_repr'])[0].cpu().numpy()
                return to_image_coords(d.astype(np.float64), Minv, 0.3)

            a, b, c = timeit(net, args.n), timeit(net_dec, args.n), timeit(e2e, max(20, args.n // 4))
            print(f'{"ARPD":8s} {size:5d} {dt:>5s} {a*1e3:8.2f} {b*1e3:8.2f} {c*1e3:8.2f} {1/c:8.1f}'
                  f'  ({len(e2e())} dets)')

    if args.rapid:
        # RAPiD's own pipeline: network (box decoding inside forward) -> conf filter
        # -> rotated NMS (python loop + RLE IoU on CPU, as in RAPiD/api.py).
        # Without trained weights (--rapid_ckpt) the NMS timing is meaningless,
        # so only "net" is printed.
        rapid_root = os.path.abspath(args.rapid)
        sys.path.insert(0, rapid_root)
        cwd = os.getcwd()
        os.chdir(rapid_root)  # RAPiD loads ./weights/... relative to its root
        from models.rapid import RAPiD  # noqa: E402
        from utils.utils import detection2original, nms as rapid_nms  # noqa: E402
        rap = RAPiD(backbone='dark53')
        if args.rapid_ckpt:
            ck = torch.load(args.rapid_ckpt, map_location='cpu', weights_only=False)
            rap.load_state_dict(ck.get('model') or ck['model_state_dict'])  # both key styles released
        rap = rap.cuda().eval()
        os.chdir(cwd)
        from PIL import Image  # noqa: E402
        import torchvision.transforms.functional as tvf  # noqa: E402
        from utils.utils import rect_to_square  # noqa: E402
        pil = Image.open(args.image).convert('RGB')
        for size in args.sizes:
            # the same real image as ARPD, preprocessed as RAPiD's api.py (letterbox, [0,1]),
            # so the number of candidates entering NMS is realistic
            x = tvf.to_tensor(rect_to_square(pil, None, size, 0)[0])[None].cuda()
            for dt in args.dtype:
                amp = dt == 'bf16'

                def rnet():
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                        return rap(x)

                def rfull():
                    d = rnet().float().cpu().squeeze(0)
                    d = d[d[:, 5] >= 0.3]
                    if len(d) > 1000:
                        d = d[torch.topk(d[:, 5], k=1000)[1]]
                    return rapid_nms(d, is_degree=True, nms_thres=0.45, img_size=size)
                def re2e():  # JPEG read + letterbox + H2D + net + NMS + map to image coords
                    im = Image.open(args.image).convert('RGB')
                    sq, _, pad = rect_to_square(im, None, size, 0)
                    t = tvf.to_tensor(sq)[None].cuda(non_blocking=True)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                        d = rap(t)
                    d = d.float().cpu().squeeze(0)
                    d = d[d[:, 5] >= 0.3]
                    if len(d) > 1000:
                        d = d[torch.topk(d[:, 5], k=1000)[1]]
                    d = rapid_nms(d, is_degree=True, nms_thres=0.45, img_size=size)
                    return detection2original(d, pad)
                try:
                    a = timeit(rnet, args.n)
                    if args.rapid_ckpt:
                        b, c = timeit(rfull, max(20, args.n // 4)), timeit(re2e, max(20, args.n // 4))
                        print(f'{"RAPiD":8s} {size:5d} {dt:>5s} {a*1e3:8.2f} {b*1e3:8.2f} {c*1e3:8.2f} {1/c:8.1f}'
                              f'  (dec = conf filter + rotated NMS; {len(rfull())} dets)')
                    else:
                        print(f'{"RAPiD":8s} {size:5d} {dt:>5s} {a*1e3:8.2f}')
                except Exception as e:  # RAPiD code targets torch 1.x
                    print(f'RAPiD {size} {dt}: failed ({type(e).__name__}: {e})')


if __name__ == '__main__':
    main()
