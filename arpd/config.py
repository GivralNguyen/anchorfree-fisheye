"""YAML configs with ``_base_`` inheritance and ``a.b.c=value`` CLI overrides."""
import copy
import os

import yaml


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop('_base_', [])
    bases = [bases] if isinstance(bases, str) else bases
    merged = {}
    for b in bases:
        merged = _merge(merged, load_config(os.path.join(os.path.dirname(path), b)))
    cfg = _merge(merged, cfg)
    for o in overrides:
        key, val = o.split('=', 1)
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg
