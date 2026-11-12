"""Shared training / inference utilities."""
import copy
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .data.datasets import TestDataset
from .decode import decode, to_image_coords


# ----------------------------------------------------------------- distributed
def init_distributed():
    if 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ['LOCAL_RANK'])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


# ------------------------------------------------------------------------ EMA
class ModelEMA:
    """Exponential moving average of weights with warm-up (as in YOLOX)."""

    def __init__(self, model, decay=0.9998, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ------------------------------------------------------------------- schedule
def lr_factor(it, total, cfg, iters_per_epoch):
    warm = cfg.get('warmup_iters', 0)
    if it < warm:
        return (it + 1) / warm
    if cfg['schedule'] == 'cosine':
        t = (it - warm) / max(1, total - warm)
        lo = cfg.get('min_lr_ratio', 0.01)
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * t))
    if cfg['schedule'] == 'step':  # x0.1 at given epochs
        epoch = it / iters_per_epoch
        return cfg.get('gamma', 0.1) ** sum(epoch >= m for m in cfg['milestones'])
    raise ValueError(cfg['schedule'])


# ------------------------------------------------------------------ inference
@torch.no_grad()
def predict(model, records, cfg, batch_size=16, workers=8, conf=0.01, K=100,
            device='cuda', amp=True):
    """Run the detector over ``records``. Returns {video: [det dicts]}."""
    model.eval()
    ds = TestDataset(records, cfg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=True)
    out = {}
    angle_repr = cfg['model'].get('angle_repr', 'tanh')
    for batch in dl:
        x = batch['input'].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            o = model(x)
        dets = decode(o, stride=cfg['stride'], K=K, angle_repr=angle_repr).cpu().numpy()
        for d, Minv, i in zip(dets, batch['Minv'].numpy(), batch['index'].tolist()):
            for det in to_image_coords(d.astype(np.float64), Minv, conf):
                det['image_id'] = records.ids[i]
                out.setdefault(records.videos[i], []).append(det)
    return out


def unwrap(model):
    return model.module if hasattr(model, 'module') else model
