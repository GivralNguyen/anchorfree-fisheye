"""Train ARPD (single GPU or DDP via torchrun).

    python tools/train.py configs/coco_pretrain.yaml [key=value ...]
    torchrun --nproc_per_node=2 tools/train.py configs/coco_pretrain.yaml
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arpd.config import load_config  # noqa: E402
from arpd.data.datasets import EpochTaggedSampler, Records, TrainDataset, build_source  # noqa: E402
from arpd.engine import ModelEMA, init_distributed, is_main, lr_factor, predict, unwrap  # noqa: E402
from arpd.eval.rotated_eval import evaluate_videos  # noqa: E402
from arpd.losses import DetLoss  # noqa: E402
from arpd.models.arpd import build_model  # noqa: E402


def build_records(specs, data_cfg):
    recs, gts = Records(), {}
    for s in specs:
        r, g = build_source(s, data_cfg)
        recs.extend(r)
        gts.update(g)
    return recs.freeze(), gts


def load_init(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('ema') or sd.get('model') or sd
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    skipped = sorted(set(own) - set(ok))
    return len(ok), skipped


def run_eval(model, test_recs, test_gts, cfg, out_dir, tag):
    ec = cfg['eval']
    dets = predict(model, test_recs, cfg, batch_size=ec.get('batch_size', 16),
                   workers=cfg['train']['workers'], conf=ec.get('ap_conf', 0.01))
    res = evaluate_videos(test_gts, dets, conf=ec.get('conf', 0.3))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'metrics_{tag}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    if ec.get('save_dets', True):
        with open(os.path.join(out_dir, f'dets_{tag}.json'), 'w') as f:
            json.dump(dets, f)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('overrides', nargs='*')
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    rank, world, local = init_distributed()
    device = torch.device('cuda', local)
    tc = cfg['train']
    torch.manual_seed(cfg.get('seed', 0) + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = os.path.join(cfg.get('output_root', 'runs'), cfg['name'])
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    # data
    train_recs, _ = build_records(cfg['data']['train'], cfg['data'])
    test_recs, test_gts = (build_records(cfg['data']['test'], cfg['data'])
                           if cfg['data'].get('test') else (None, {}))
    ds = TrainDataset(train_recs, cfg, seed=cfg.get('seed', 0))
    per_gpu = tc['batch_size'] // world
    base = (DistributedSampler(ds, shuffle=True, drop_last=True, seed=cfg.get('seed', 0))
            if world > 1 else RandomSampler(ds))
    sampler = EpochTaggedSampler(base)  # carries the epoch to (persistent) workers
    dl = DataLoader(ds, batch_size=per_gpu, sampler=sampler,
                    num_workers=tc['workers'], pin_memory=True, drop_last=True,
                    persistent_workers=tc['workers'] > 0, prefetch_factor=4 if tc['workers'] else None)
    if is_main():
        print(f'train images: {len(ds)}  | test images: {len(test_recs) if test_recs else 0}'
              f' | global batch {tc["batch_size"]} on {world} GPU(s)', flush=True)

    # model
    model = build_model(cfg, pretrained=not tc.get('init_from')).to(device)
    if tc.get('init_from'):
        n, skipped = load_init(model, tc['init_from'])
        if is_main():
            print(f'init_from {tc["init_from"]}: loaded {n} tensors, skipped {skipped}', flush=True)
    model = model.to(memory_format=torch.channels_last)
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if tc.get('sync_bn') else model
        model = DDP(model, device_ids=[local], find_unused_parameters=False)
    criterion = DetLoss(cfg)
    lr = tc['lr'] * tc['batch_size'] / tc.get('lr_ref_batch', tc['batch_size'])
    opt_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[tc.get('optimizer', 'adamw')]
    opt = opt_cls(model.parameters(), lr=lr, weight_decay=tc.get('weight_decay', 1e-4))
    ema = ModelEMA(unwrap(model), tc.get('ema_decay', 0.9998)) if tc.get('ema', True) else None

    epochs = tc['epochs']
    ipe = len(dl)
    total = epochs * ipe
    start_epoch = 0
    ckpt_path = os.path.join(out_dir, 'last.pth')
    if tc.get('resume', True) and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        unwrap(model).load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if ema and ck.get('ema'):
            ema.ema.load_state_dict(ck['ema'])
            ema.updates = ck['ema_updates']
        start_epoch = ck['epoch'] + 1
        if is_main():
            print(f'resumed from epoch {start_epoch}', flush=True)

    writer = None
    if is_main():
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out_dir)

    amp = tc.get('amp', True)
    for epoch in range(start_epoch, epochs):
        model.train()
        ds.set_epoch(epoch)
        if sampler:
            sampler.set_epoch(epoch)
        t0, seen = time.time(), 0
        agg = {}
        for i, batch in enumerate(dl):
            it = epoch * ipe + i
            for g in opt.param_groups:
                g['lr'] = lr * lr_factor(it, total, tc, ipe)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            x = batch['input'].contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                out = model(x)
            loss, parts = criterion(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])
            opt.step()
            if ema:
                ema.update(unwrap(model))
            seen += x.shape[0] * world
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            if is_main() and (i % tc.get('log_every', 50) == 0 or i == ipe - 1):
                ips = seen / (time.time() - t0)
                msg = ' '.join(f'{k}={float(v):.4f}' for k, v in parts.items())
                print(f'ep {epoch + 1}/{epochs} it {i}/{ipe} lr={opt.param_groups[0]["lr"]:.2e} '
                      f'{msg} | {ips:.0f} img/s', flush=True)
                for k, v in parts.items():
                    writer.add_scalar(f'train/{k}', float(v), it)
                writer.add_scalar('train/lr', opt.param_groups[0]['lr'], it)
        if is_main():
            print(f'epoch {epoch + 1} done in {time.time() - t0:.0f}s | mean '
                  + ' '.join(f'{k}={v / ipe:.4f}' for k, v in agg.items()), flush=True)
            ck = {'model': unwrap(model).state_dict(), 'opt': opt.state_dict(), 'epoch': epoch,
                  'ema': ema.ema.state_dict() if ema else None,
                  'ema_updates': ema.updates if ema else 0, 'cfg': cfg}
            # a full disk must not kill a long run: keep the previous
            # checkpoint, warn, and try again next epoch
            try:
                torch.save(ck, ckpt_path + '.tmp')
                os.replace(ckpt_path + '.tmp', ckpt_path)
                if (epoch + 1) in tc.get('save_epochs', []):
                    torch.save({'model': ck['model'], 'ema': ck['ema'], 'cfg': cfg},
                               os.path.join(out_dir, f'epoch{epoch + 1}.pth'))
            except (OSError, RuntimeError) as e:
                print(f'WARNING: checkpoint save failed at epoch {epoch + 1} ({e}); '
                      f'previous last.pth kept', flush=True)
                if os.path.exists(ckpt_path + '.tmp'):
                    os.remove(ckpt_path + '.tmp')
            # monitoring only: model selection always uses the final epoch
            ev = cfg['eval'].get('interval', 0)
            if test_recs is not None and ev and (epoch + 1) % ev == 0 and epoch + 1 < epochs:
                res = run_eval(ema.ema if ema else unwrap(model), test_recs, test_gts, cfg,
                               out_dir, f'epoch{epoch + 1}')
                m = res['mean']
                print(f'[eval ep{epoch + 1}] AP50={m["AP50"]:.4f} AP={m["AP"]:.4f} '
                      f'P={m["P"]:.3f} R={m["R"]:.3f} F1={m["F1"]:.3f}', flush=True)
                for k, v in m.items():
                    writer.add_scalar(f'eval/{k}', v, epoch + 1)
                if len(res['per_video']) <= 4:  # e.g. separate monitor sets: show each one
                    for vid, pv in res['per_video'].items():
                        print(f'[eval ep{epoch + 1}]   {vid}: AP50={pv["AP50"]:.4f} '
                              f'AP={pv["AP"]:.4f} P={pv["P"]:.3f} R={pv["R"]:.3f}', flush=True)
                        for k in ('AP50', 'AP', 'P', 'R'):
                            writer.add_scalar(f'eval_{vid}/{k}', pv[k], epoch + 1)
        if world > 1:
            dist.barrier()

    if is_main():
        final = ema.ema if ema else unwrap(model)
        torch.save({'model': unwrap(model).state_dict(), 'ema': ema.ema.state_dict() if ema else None,
                    'cfg': cfg}, os.path.join(out_dir, 'final.pth'))
        if test_recs is not None:
            res = run_eval(final, test_recs, test_gts, cfg, out_dir, 'final')
            print('FINAL (per-video mean):', json.dumps(res['mean']), flush=True)
            print('FINAL (pooled):', json.dumps(res['pooled']), flush=True)
        writer.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
