"""Đọc configs/eN.yaml kèm ghi đè dạng key=value (vd. lora.r=64, train_fraction=0.1)."""
import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "name": None,
    "model_id": "Qwen/Qwen3-VL-4B-Instruct",
    "max_new_tokens": 256,
    "eval_batch_size": 4,
    "lora": None,                 # None = zero-shot (E0)
    "train_projector": False,
    "lr": 2.0e-4,
    "batch_size": 2,
    "grad_accum": 4,
    "max_steps": 300,
    "save_every": 100,
    "seed": 42,
    "train_fraction": 1.0,        # E4 data_size
    "coord_scale": "norm1000",    # E4 coord_scale
}


def _set_dotted(cfg, key, value):
    *head, last = key.split(".")
    node = cfg
    for k in head:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    node[last] = value


def load_config(path, overrides=()):
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    keys = set()
    for ov in overrides or ():
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"ghi đè phải có dạng key=value: {ov!r}")
        _set_dotted(cfg, key.strip(), yaml.safe_load(val))
        keys.add(key.strip())
    # E4 rank: GUIDE yêu cầu alpha = 2·r
    if "lora.r" in keys and "lora.alpha" not in keys:
        cfg["lora"]["alpha"] = 2 * cfg["lora"]["r"]
    if cfg["coord_scale"] not in ("norm1000", "pixel"):
        raise ValueError(f"coord_scale không hợp lệ: {cfg['coord_scale']}")
    if not 0 < float(cfg["train_fraction"]) <= 1:
        raise ValueError("train_fraction phải trong (0, 1]")
    return cfg
