"""Run a checkpoint on images and draw the detected people.

Box edges along the width are drawn in green, edges along the height in yellow.
--heatmap also saves the predicted person-centre heatmap over the image (<out>_heatmap.jpg).

    python tools/demo.py weights/cv_test_WEPDTOF.pth path/to/frame.jpg --input_size 1024 --out demo.jpg
    python tools/demo.py weights/cv_test_WEPDTOF.pth frames/*.jpg --out vis/ --heatmap
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arpd.boxes import box_vertices, deg2rad_boxes  # noqa: E402
from arpd.data.datasets import read_rgb  # noqa: E402
from arpd.data.transforms import TestTransform  # noqa: E402
from arpd.decode import decode, to_image_coords  # noqa: E402
from arpd.models.arpd import build_model  # noqa: E402


WIDTH_BGR, HEIGHT_BGR = (0, 230, 0), (0, 230, 255)  # green, yellow


def draw(img_bgr, dets):
    t = max(2, round(max(img_bgr.shape[:2]) / 400))
    for d in dets:
        v = np.round(box_vertices(deg2rad_boxes([d['bbox']]))[0]).astype(np.int32)  # tl, tr, br, bl
        for i in range(4):  # tl-tr and br-bl run along the width, tr-br and bl-tl along the height
            color = WIDTH_BGR if i % 2 == 0 else HEIGHT_BGR
            cv2.line(img_bgr, tuple(map(int, v[i])), tuple(map(int, v[(i + 1) % 4])), color, t, cv2.LINE_AA)
    return img_bgr


def heatmap_overlay(img_bgr, hm, Minv, stride=4):
    """Sigmoid heatmap (H/stride, W/stride) of the letterboxed input -> overlay on the original image."""
    h, w = img_bgr.shape[:2]
    up = cv2.resize(hm, None, fx=stride, fy=stride, interpolation=cv2.INTER_LINEAR)
    m = np.clip(cv2.warpAffine(up, Minv, (w, h), flags=cv2.INTER_LINEAR), 0, 1)
    m = np.sqrt(m / max(float(m.max()), 1e-6))  # normalised to the frame's strongest peak; sqrt widens the blobs
    color = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    base = (cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.55).astype(np.float32)
    a = np.clip(m * 1.1, 0, 0.85)[..., None]
    return (base * (1 - a) + color * a).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('images', nargs='+')
    ap.add_argument('--input_size', type=int, default=None, help='default: the checkpoint config')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--out', default='demo.jpg', help='output file (one image) or folder')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--heatmap', action='store_true', help='also save the centre heatmap overlay')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    size = args.input_size or cfg['input_size']
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ck.get('ema') or ck['model'])
    model.to(args.device).eval()
    tf = TestTransform({'input_size': size})

    for path in args.images:
        img = read_rgb(path)
        x, Minv = tf(img)
        with torch.no_grad():
            out = model(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(args.device))
        d = decode(out, stride=cfg['stride'], K=100, angle_repr=cfg['model']['angle_repr'])
        dets = to_image_coords(d[0].cpu().numpy().astype(np.float64), Minv, args.conf)
        vis = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), dets)
        dst = (os.path.join(args.out, os.path.basename(path)) if len(args.images) > 1
               or os.path.isdir(args.out) else args.out)
        os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
        cv2.imwrite(dst, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'{path}: {len(dets)} people -> {dst}')
        if args.heatmap:
            hm = torch.sigmoid(out['hm'][0, 0].float()).cpu().numpy()
            hdst = os.path.splitext(dst)[0] + '_heatmap.jpg'
            cv2.imwrite(hdst, heatmap_overlay(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), hm, Minv, cfg['stride']),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f'  heatmap -> {hdst}')


if __name__ == '__main__':
    main()
