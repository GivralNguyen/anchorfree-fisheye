"""Two-fold cross-validated model selection over test videos (WEPDTOF benchmark protocol).

Videos are split into two folds (sorted names, alternating). For each fold, the candidate
with the best mean AP50 on the *other* fold is chosen and scored on this fold; the reported
number is the per-video mean over all videos of those held-out scores. Candidates and the
selection metric are fixed; nothing is tuned on the fold being scored.

    python tools/select_2fold.py runs/cv_WEPDTOF_lr*/metrics_ep*_in1024.json \
        --ref runs/rapid_baseline/metrics_HBCP608_in1024_WEPDTOF.json
"""
import argparse
import json
import os

import numpy as np

KEYS = ('AP50', 'AP', 'AP75', 'P', 'R')


def load(p):
    return {v: m for v, m in json.load(open(p))['per_video'].items()}


def name(p):
    return f'{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)[8:-5]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('metrics', nargs='+')
    ap.add_argument('--ref', nargs='*', default=[], help='fixed models to score on the same folds')
    ap.add_argument('--by', default='AP50')
    a = ap.parse_args()
    cands = {name(p): load(p) for p in a.metrics}
    videos = sorted(next(iter(cands.values())))
    assert all(sorted(c) == videos for c in cands.values()), 'candidates cover different videos'
    folds = [videos[0::2], videos[1::2]]

    def mean(c, vids, k):
        return float(np.mean([cands[c][v][k] if c in cands else refs[c][v][k] for v in vids]))

    refs = {name(p): load(p) for p in a.ref}
    per_video = {}
    for i, test in enumerate(folds):
        sel = folds[1 - i]
        best = max(cands, key=lambda c: mean(c, sel, a.by))
        print(f'fold {i}: chosen on {len(sel)} videos -> {best} '
              f'(sel {a.by} {100 * mean(best, sel, a.by):.1f}), scored on {len(test)} videos: '
              + ' '.join(f'{k} {100 * mean(best, test, k) if k.startswith("AP") else mean(best, test, k):.3g}'
                         for k in KEYS))
        for v in test:
            per_video[v] = cands[best][v]
    cv = {k: float(np.mean([per_video[v][k] for v in videos])) for k in KEYS}
    print('\n2-fold CV (per-video mean over all videos): '
          + ' '.join(f'{k} {100 * cv[k] if k.startswith("AP") else cv[k]:.3g}' for k in KEYS))
    for r in refs:
        print(f'{r:40s} ' + ' '.join(f'{k} {100 * mean(r, videos, k) if k.startswith("AP") else mean(r, videos, k):.3g}'
                                    for k in KEYS))
    print('\nall candidates (full-set mean AP50 / AP):')
    for c in sorted(cands, key=lambda c: -mean(c, videos, 'AP50')):
        print(f'  {c:40s} {100 * mean(c, videos, "AP50"):5.1f} {100 * mean(c, videos, "AP"):5.1f}')


if __name__ == '__main__':
    main()
