"""Detection losses: CornerNet focal loss, L1 regressions and angle losses."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(pred_logits, gt, alpha=2.0, beta=4.0):
    """Penalty-reduced pixel-wise focal loss (eq. 1), normalised by #objects."""
    p = torch.sigmoid(pred_logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - gt) ** beta * neg
    num_pos = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos.clamp(min=1.0)


def gather(feat, ind):
    """feat (B, C, H, W), ind (B, K) -> (B, K, C)."""
    B, C = feat.shape[:2]
    feat = feat.view(B, C, -1).permute(0, 2, 1)
    return feat.gather(1, ind.unsqueeze(2).expand(-1, -1, C))


def masked_l1(pred, target, mask):
    """CenterNet's RegL1Loss normalisation: sum |diff| / (#objects * dims)."""
    m = mask.unsqueeze(2).expand_as(pred)
    return (F.l1_loss(pred * m, target * m, reduction='sum')) / (m.sum() + 1e-4)


def periodic_diff(pred, target, period=math.pi):
    """Smallest signed difference modulo ``period``, in [-period/2, period/2).

    For period = pi this equals arctan(sin d / cos d) = arctan(tan d), with the
    single-argument arctan. Same value and gradient (1) everywhere except at the
    wrap points. NB: atan2(sin d, cos d) would be 2*pi-periodic, i.e. wrong here.
    """
    d = pred - target
    return torch.remainder(d + period / 2, period) - period / 2


def angle_loss(pred, target, mask, kind='periodic_smooth_l1', beta=1.0):
    """Angle losses (periodic and plain regression, + a continuous variant).

    kind:
      l1, smooth_l1                 plain regression on theta
      periodic_l1, periodic_l2      on the pi-periodic difference
      periodic_smooth_l1            default, threshold ``beta``
      sincos_l1                     L1 on (sin 2theta, cos 2theta)  [angle_repr=sincos]
    """
    m = mask.unsqueeze(2).expand_as(pred)
    n = m.sum() + 1e-4
    if kind == 'sincos_l1':
        return (F.l1_loss(pred, target, reduction='none') * m).sum() / n
    if kind in ('l1', 'smooth_l1'):
        d = pred - target
    else:
        d = periodic_diff(pred, target)
    if kind in ('l1', 'periodic_l1'):
        l = d.abs()
    elif kind == 'periodic_l2':
        l = d ** 2
    elif kind in ('smooth_l1', 'periodic_smooth_l1'):
        l = F.smooth_l1_loss(d, torch.zeros_like(d), reduction='none', beta=beta)
    else:
        raise ValueError(kind)
    return (l * m).sum() / n


class DetLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        lc = cfg['loss']
        self.w = {k: lc[f'{k}_weight'] for k in ('hm', 'off', 'size', 'angle')}
        self.angle_kind = lc['angle_loss']
        self.angle_beta = lc.get('angle_beta', 1.0)
        self.alpha, self.beta = lc.get('focal_alpha', 2.0), lc.get('focal_beta', 4.0)

    def forward(self, out, batch):
        ind, mask = batch['ind'], batch['mask']
        hm = focal_loss(out['hm'], batch['hm'], self.alpha, self.beta)
        off = masked_l1(gather(out['reg'].float(), ind), batch['reg'], mask)
        size = masked_l1(gather(out['wh'].float(), ind), batch['wh'], mask)
        ang = angle_loss(gather(out['ang'].float(), ind), batch['ang'], mask,
                         self.angle_kind, self.angle_beta)
        total = (self.w['hm'] * hm + self.w['off'] * off + self.w['size'] * size
                 + self.w['angle'] * ang)
        return total, {'hm': hm.detach(), 'off': off.detach(), 'size': size.detach(),
                       'angle': ang.detach(), 'total': total.detach()}
