"""Stage-2 joint training engine (one-step joint, warm start; grill 2026-07-31).

L = lam0*L_C + lam1*L_flow + lam2*L_action, fully differentiable
(L_action -> L4 -> predicted flow -> L3 -> z -> L1 fusion).
Warm starts: L1/CHead from stage-1 fold ckpt, L3 from ATM track transformer.
Run via `run_stage2.py train [...]` (atm5090, GPU).
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from routedflow.stage2.dataset import Stage2Dataset
from routedflow.stage2.joint_model import JointApproachModel

EXP = os.path.join(REPO, "experiments", "stage2_approach_joint")
STAGE0_CFG = os.path.join(REPO, "experiments", "stage0_routing_causal_test", "configs", "stage0_vilt.yaml")
TRACK_FN = os.path.join(REPO, "third_party", "ATM", "results", "track_transformer",
                        "libero_track_transformer_libero-spatial")

DATASET_KW = dict(img_size=128, frame_stack=10, num_track_ts=16, num_track_ids=32,
                  track_obs_fs=1, augment_track=False,
                  extra_state_keys=["joint_states", "gripper_states"],
                  cache_all=True, cache_image=True)


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_dev(v, dev) for v in x)
    return x


def run_epoch(model, loader, dev, opt=None, accum=1):
    train = opt is not None
    model.train(train)
    agg, n = {}, 0
    with torch.set_grad_enabled(train):
        for i, batch in enumerate(loader):
            batch = to_dev(batch, dev)
            loss, parts = model.forward_loss(batch)
            if train:
                (loss / accum).backward()
                if (i + 1) % accum == 0 or (i + 1) == len(loader):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
    return {k: v / n for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lam", type=float, nargs=3, default=[0.5, 1.0, 0.1])
    ap.add_argument("--flow-noise", type=float, default=0.01)
    ap.add_argument("--l1-ckpt", default=os.path.join(EXP, "..", "stage1_l1_training",
                                                      "runs", "fold0_seed0", "ckpt_best.pt"))
    ap.add_argument("--use-cross-attn", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    name = args.name or f"joint_fold{args.fold}_seed{args.seed}"
    out = os.path.join(EXP, "runs", name)
    os.makedirs(out, exist_ok=True)

    ds_tr = Stage2Dataset(fold=args.fold, split="train", **DATASET_KW)
    ds_id = Stage2Dataset(fold=args.fold, split="val_id", **DATASET_KW)
    print(f"windows: train {len(ds_tr)} / val_id {len(ds_id)}", flush=True)
    # persistent_workers + few workers: per-epoch worker respawn over an ~8G
    # in-RAM cache triggers CoW refcount copies -> silent OOM kill (seen at ep24)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=2,
                       drop_last=True, persistent_workers=True)
    dl_id = DataLoader(ds_id, batch_size=args.bs, num_workers=0)

    cfg = OmegaConf.load(STAGE0_CFG).model_cfg
    cfg.pop("track_gate_cfg", None)
    model = JointApproachModel(TRACK_FN, cfg, l1_ckpt=args.l1_ckpt, lam=tuple(args.lam),
                               flow_noise=args.flow_noise,
                               use_cross_attn=args.use_cross_attn).to(dev)
    groups = [
        {"params": [p for p in list(model.l1.parameters()) + list(model.chead.parameters())], "lr": 1e-4},
        {"params": list(model.l3.parameters()), "lr": 1e-4},
        {"params": list(model.l4.parameters()), "lr": 3e-4},
    ]
    n_par = sum(p.numel() for g in groups for p in g["params"] if p.requires_grad) / 1e6
    print(f"trainable params: {n_par:.1f}M", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    meta = vars(args)
    with open(os.path.join(out, "metrics.jsonl"), "a") as mf:
        for ep in range(args.epochs):
            t0 = time.time()
            tr = run_epoch(model, dl_tr, dev, opt, args.accum)
            row = {"epoch": ep, "train": tr, "sec": round(time.time() - t0, 1)}
            if ep % 5 == 0 or ep == args.epochs - 1:
                row["val_id"] = run_epoch(model, dl_id, dev)
            sched.step()
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            msg = f"ep {ep}: train loss {tr['loss']:.4f} (C {tr['l_c']:.3f} flow {tr['l_flow']:.5f} act {tr['l_action']:.4f}) {row['sec']}s"
            if "val_id" in row:
                msg += f" | val {row['val_id']['loss']:.4f} act {row['val_id']['l_action']:.4f}"
            print(msg, flush=True)
            state = {"l1": model.l1.state_dict(), "chead": model.chead.state_dict(),
                     "l3": model.l3.state_dict(), "l4": model.l4.state_dict(),
                     "cfg": meta, "epoch": ep}
            if "val_id" in row and row["val_id"]["l_action"] < best:
                best = row["val_id"]["l_action"]
                torch.save(state, os.path.join(out, "ckpt_best.pt"))
            torch.cuda.empty_cache()
    torch.save(state, os.path.join(out, "ckpt_final.pt"))
    print(f"done. best val action loss {best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
