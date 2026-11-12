"""DLA-Up / IDA-Up decoder from CenterNet ("Objects as Points"), re-implemented.

Differences from the original code:
* modulated deformable conv needs no custom CUDA build. Two implementations of
  the same operation are provided (``model.dcn``): ``grid_sample`` (default, used
  for all reported results; runs under bf16 autocast) and ``torchvision``
  (``torchvision.ops.deform_conv2d``, fp32 only). Weights are interchangeable.
* the backbone is any timm ``features_only`` model with stride-4/8/16/32
  feature maps (DLA-34 by default), so backbones can be swapped from config.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def deform_conv2d_gs(x, offset, weight, bias, padding=1, mask=None):
    """Modulated deformable conv (stride 1, dilation 1) via grid_sample + einsum.

    Numerically equivalent to ``torchvision.ops.deform_conv2d`` (same offset
    layout: channel 2k = dy, 2k+1 = dx for kernel tap k = i*kw + j; zero
    padding outside the image), written with standard ops so that it runs
    under bf16 autocast (torchvision's kernel is fp32 only).
    """
    B, C, H, W = x.shape
    O, _, kh, kw = weight.shape
    k = kh * kw
    off = offset.view(B, k, 2, H, W)
    ky = (torch.arange(kh, device=x.device, dtype=offset.dtype) - padding).repeat_interleave(kw)
    kx = (torch.arange(kw, device=x.device, dtype=offset.dtype) - padding).repeat(kh)
    ys = torch.arange(H, device=x.device, dtype=offset.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=x.device, dtype=offset.dtype).view(1, 1, 1, W)
    py = ys + ky.view(1, k, 1, 1) + off[:, :, 0]
    px = xs + kx.view(1, k, 1, 1) + off[:, :, 1]
    grid = torch.stack([px * (2.0 / max(W - 1, 1)) - 1, py * (2.0 / max(H - 1, 1)) - 1], dim=-1)
    s = F.grid_sample(x, grid.view(B, k * H, W, 2), mode='bilinear',
                      padding_mode='zeros', align_corners=True).view(B, C, k, H, W)
    if mask is not None:
        s = s * mask.view(B, 1, k, H, W)
    out = torch.einsum('bckhw,ock->bohw', s.to(weight.dtype) if not torch.is_autocast_enabled()
                       else s, weight.view(O, C, k))
    return out + bias.view(1, -1, 1, 1).to(out.dtype)


class ModulatedDeformConv(nn.Module):
    """DCNv2: 3x3 deformable conv with learned offsets and modulation mask.

    impl='torchvision': torchvision.ops.deform_conv2d in fp32 (reference).
    impl='grid_sample' : deform_conv2d_gs, same maths, autocast-friendly (default).
    """

    def __init__(self, cin, cout, k=3, padding=1, impl='grid_sample'):
        super().__init__()
        self.k, self.padding, self.impl = k, padding, impl
        self.weight = nn.Parameter(torch.empty(cout, cin, k, k))
        self.bias = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.offset_mask = nn.Conv2d(cin, 3 * k * k, k, padding=padding)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    def forward(self, x):
        if self.impl == 'grid_sample':
            o1, o2, m = torch.chunk(self.offset_mask(x), 3, dim=1)
            offset = torch.cat([o1, o2], dim=1).float()  # sampling positions in fp32
            return deform_conv2d_gs(x, offset, self.weight, self.bias,
                                    padding=self.padding, mask=torch.sigmoid(m))
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            o1, o2, m = torch.chunk(self.offset_mask(x), 3, dim=1)
            offset = torch.cat([o1, o2], dim=1)
            return deform_conv2d(x, offset, self.weight, self.bias,
                                 padding=self.padding, mask=torch.sigmoid(m))


class ConvBlock(nn.Module):
    """(Deformable) 3x3 conv + BN + ReLU."""

    def __init__(self, cin, cout, dcn=True):
        super().__init__()
        # dcn: False -> plain conv; True / 'grid_sample' / 'torchvision' -> DCNv2 implementation
        impl = dcn if isinstance(dcn, str) else 'grid_sample'
        self.conv = (ModulatedDeformConv(cin, cout, impl=impl) if dcn
                     else nn.Conv2d(cin, cout, 3, padding=1))
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def _bilinear_up(channels, factor):
    up = nn.ConvTranspose2d(channels, channels, factor * 2, stride=factor,
                            padding=factor // 2, output_padding=0, groups=channels, bias=False)
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - abs(i / f - c)) * (1 - abs(j / f - c))
    for ch in range(1, w.size(0)):
        w[ch, 0] = w[0, 0]
    return up


class IDAUp(nn.Module):
    def __init__(self, out_ch, channels, up_f, dcn=True):
        super().__init__()
        self.proj = nn.ModuleList()
        self.up = nn.ModuleList()
        self.node = nn.ModuleList()
        for i in range(1, len(channels)):
            self.proj.append(ConvBlock(channels[i], out_ch, dcn))
            self.up.append(_bilinear_up(out_ch, int(up_f[i])))
            self.node.append(ConvBlock(out_ch, out_ch, dcn))

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            j = i - startp - 1
            layers[i] = self.up[j](self.proj[j](layers[i]))
            layers[i] = self.node[j](layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, channels, scales, dcn=True):
        super().__init__()
        channels = list(channels)
        in_channels = list(channels)
        scales = np.array(scales, dtype=int)
        self.ida = nn.ModuleList()
        for i in range(len(channels) - 1):
            j = -i - 2
            self.ida.append(IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j], dcn))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        out = [layers[-1]]
        for i, ida in enumerate(self.ida):
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class DLASegNeck(nn.Module):
    """Takes [C2, C3, C4, C5] (strides 4..32), returns one stride-4 feature map."""

    def __init__(self, channels, dcn=True):
        super().__init__()
        n = len(channels)
        self.dla_up = DLAUp(channels, [2 ** i for i in range(n)], dcn)
        self.ida_up = IDAUp(channels[0], channels[:n - 1], [2 ** i for i in range(n - 1)], dcn)
        self.out_channels = channels[0]

    def forward(self, feats):
        x = self.dla_up(feats)
        y = [x[i].clone() for i in range(len(x) - 1)]
        self.ida_up(y, 0, len(y))
        return y[-1]
