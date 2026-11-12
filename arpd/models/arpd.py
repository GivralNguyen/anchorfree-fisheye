"""ARPD: anchor-free rotating people detector (CenterNet + angle head)."""
import math

import timm
import torch
import torch.nn as nn

from .neck import DLASegNeck


def _pick_stride_indices(name):
    """Indices of a timm features_only model whose reductions are 4, 8, 16, 32."""
    probe = timm.create_model(name, pretrained=False, features_only=True)
    red = probe.feature_info.reduction()
    # reduction() is aligned with the *default* out_indices, not with 0..n-1
    module_idx = list(probe.feature_info.out_indices)
    idx = []
    for s in (4, 8, 16, 32):
        cands = [module_idx[i] for i, r in enumerate(red) if r == s]
        if not cands:
            raise ValueError(f'{name} has no stride-{s} feature map (reductions={red})')
        idx.append(cands[-1])
    return tuple(idx)


class Head(nn.Sequential):
    def __init__(self, cin, cout, head_conv=256, bias=0.0):
        super().__init__(nn.Conv2d(cin, head_conv, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(head_conv, cout, 1))
        nn.init.constant_(self[2].bias, bias)


class ARPD(nn.Module):
    def __init__(self, backbone='dla34', pretrained=True, dcn=True, head_conv=256,
                 num_classes=1, angle_repr='tanh', angle_range=math.pi):
        super().__init__()
        idx = _pick_stride_indices(backbone)
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          features_only=True, out_indices=idx)
        chs = self.backbone.feature_info.channels()
        self.neck = DLASegNeck(chs, dcn=dcn)
        c = self.neck.out_channels
        self.angle_repr = angle_repr
        self.angle_range = angle_range
        self.heads = nn.ModuleDict({
            'hm': Head(c, num_classes, head_conv, bias=-2.19),  # prior p ~= 0.1
            'reg': Head(c, 2, head_conv),
            'wh': Head(c, 2, head_conv),
            'ang': Head(c, 2 if angle_repr == 'sincos' else 1, head_conv),
        })

    def angle_from_raw(self, raw):
        """Map raw angle-head output to the quantity used by the loss/decoder.

        tanh   : theta_hat = range * tanh(t)  (default range = pi, i.e. [-pi, pi))
                 range <= 0 means unbounded (identity).
        sincos : (sin 2theta, cos 2theta), returned as-is.
        """
        if self.angle_repr == 'sincos':
            return raw
        if self.angle_range and self.angle_range > 0:
            return self.angle_range * torch.tanh(raw)
        return raw

    def forward(self, x):
        f = self.neck(self.backbone(x))
        out = {k: h(f) for k, h in self.heads.items()}
        out['ang'] = self.angle_from_raw(out['ang'].float())
        return out


def build_model(cfg, pretrained=True):
    m = cfg['model']
    return ARPD(backbone=m['backbone'], pretrained=pretrained and m.get('pretrained', True),
                dcn=m.get('dcn', True), head_conv=m.get('head_conv', 256),
                angle_repr=m.get('angle_repr', 'tanh'),
                angle_range=m.get('angle_range', math.pi))
