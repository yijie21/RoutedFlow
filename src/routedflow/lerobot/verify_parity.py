"""Parity + benchmark: Stage2LeRobotDataset vs legacy Stage2Dataset.

Parity: for N random (task, demo, offset) keys in the intersection of both
datasets' window sets, every tuple element must match EXACTLY (videos are
lossless, track grid-sampling is deterministic; float64->float32 casts happen
identically on both paths). This is the acceptance gate for swapping backends.

Benchmark: items/s over K random items, cached and streaming modes, both
backends. Acceptance (grill 2026-08-04): lerobot must not regress.

Run via `run_stage2.py verify-lerobot [--n 64] [--bench 200]`.
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

DATASET_KW = dict(img_size=128, frame_stack=10, num_track_ts=16, num_track_ids=32,
                  track_obs_fs=1, augment_track=False,
                  extra_state_keys=["joint_states", "gripper_states"])


def old_key_map(ds):
    """legacy Stage2Dataset: position in _keep -> (task, demo, offset)."""
    out = {}
    for pos, index in enumerate(ds._keep):
        demo_id = ds._index_to_demo_id[index]
        off = index - ds._demo_id_to_start_indices[demo_id]
        path = ds._demo_id_to_path[demo_id]
        task = os.path.basename(os.path.dirname(os.path.dirname(path)))
        demo = os.path.basename(path).replace(".hdf5", "")
        out[(task, demo, off)] = pos
    return out


def new_key_map(ds):
    return {(*ds._data[li]["key"], off): pos for pos, (li, off) in enumerate(ds._keep)}


def compare(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {set(a)} != {set(b)}"
        for k in a:
            compare(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
        return
    ta, tb = torch.as_tensor(a), torch.as_tensor(b)
    assert ta.shape == tb.shape, f"{path}: shape {ta.shape} != {tb.shape}"
    d = (ta.float() - tb.float()).abs().max().item() if ta.numel() else 0.0
    assert d == 0.0, f"{path}: max abs diff {d}"


def bench(ds, idxs, tag):
    t0 = time.time()
    for i in idxs:
        _ = ds[i]
    dt = time.time() - t0
    print(f"  {tag}: {len(idxs) / dt:.1f} items/s ({dt:.1f}s / {len(idxs)})", flush=True)
    return len(idxs) / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="parity sample count")
    ap.add_argument("--bench", type=int, default=200, help="benchmark item count")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    from routedflow.stage2.dataset import Stage2Dataset
    from routedflow.stage2.dataset_lerobot import Stage2LeRobotDataset

    results = {}
    for mode, kw in (("cached", dict(cache_all=True, cache_image=True)),
                     ("stream", dict(cache_all=False, cache_image=False))):
        print(f"== mode: {mode} ==", flush=True)
        old = Stage2Dataset(fold=0, split=args.split, **kw, **DATASET_KW)
        new = Stage2LeRobotDataset(fold=0, split=args.split, **kw, **DATASET_KW)
        om, nm = old_key_map(old), new_key_map(new)
        inter = sorted(set(om) & set(nm))
        print(f"windows: old {len(om)} new {len(nm)} intersection {len(inter)}", flush=True)
        assert inter, "no overlapping windows — did the conversion cover the split?"
        if mode == "cached":  # parity once (data identical across modes by construction)
            for key in [inter[i] for i in rng.choice(len(inter), min(args.n, len(inter)),
                                                     replace=False)]:
                compare(old[om[key]], new[nm[key]], str(key))
            print(f"PARITY OK on {min(args.n, len(inter))} random windows", flush=True)
        idxs_old = [om[inter[i]] for i in rng.integers(0, len(inter), args.bench)]
        idxs_new = [nm[inter[i]] for i in rng.integers(0, len(inter), args.bench)]
        results[f"{mode}/h5"] = bench(old, idxs_old, f"h5/{mode}")
        results[f"{mode}/lerobot"] = bench(new, idxs_new, f"lerobot/{mode}")
        del old, new
    print("SUMMARY:", {k: round(v, 1) for k, v in results.items()}, flush=True)
    for mode in ("cached", "stream"):
        ratio = results[f"{mode}/lerobot"] / results[f"{mode}/h5"]
        print(f"{mode}: lerobot/h5 speed ratio {ratio:.2f}x", flush=True)


if __name__ == "__main__":
    main()
