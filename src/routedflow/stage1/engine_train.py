"""Stage-1 training engine (curriculum ①: L1+L2, loss = L_C only).

Tiny model over cached features => plain single-GPU torch loop, seconds/epoch.
Outputs under experiments/stage1_l1_training/runs/<name>/:
    ckpt_best.pt / ckpt_final.pt   {l1, chead, cfg, epoch}
    metrics.jsonl                  one line per epoch (train/val_id losses)
Run via `run_stage1.py train-l1 [--fold 0] [--epochs 200] ...` (atm5090).
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from routedflow.stage1.dataset import Stage1Dataset
from routedflow.stage1.model import CHead, L1FrontEnd, stage1_loss

EXP = os.path.join(REPO, "experiments", "stage1_l1_training")


def run_split(l1, chead, loader, dev, opt=None):
    train = opt is not None
    l1.train(train), chead.train(train)
    agg, n = {}, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            hm, z, fm = l1(batch["dino"], batch["prior"], batch["text"])
            yl, pl, wp = chead(z, fm, batch["contact_rowcol"])
            loss, parts = stage1_loss(hm, z, yl, pl, wp, batch)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            b = batch["dino"].shape[0]
            agg["loss"] = agg.get("loss", 0.0) + loss.item() * b
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v * b
            n += b
    return {k: v / n for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--prior-dropout", type=float, default=0.3)
    ap.add_argument("--no-prior", action="store_true", help="train with zeroed AFUN channel")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    name = args.name or f"fold{args.fold}_seed{args.seed}" + ("_noprior" if args.no_prior else "")
    out_dir = os.path.join(EXP, "runs", name)
    os.makedirs(out_dir, exist_ok=True)

    ds_tr = Stage1Dataset(fold=args.fold, split="train", use_prior=not args.no_prior)
    ds_id = Stage1Dataset(fold=args.fold, split="val_id", use_prior=not args.no_prior)
    print(f"train {len(ds_tr)} / val_id {len(ds_id)} · ood tasks: {ds_tr.ood_tasks}", flush=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    dl_id = DataLoader(ds_id, batch_size=args.bs, num_workers=2)

    l1 = L1FrontEnd(prior_dropout=args.prior_dropout).to(dev)
    chead = CHead().to(dev)
    params = list(l1.parameters()) + list(chead.parameters())
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    cfg = vars(args) | {"ood_tasks": ds_tr.ood_tasks}
    best = float("inf")
    with open(os.path.join(out_dir, "metrics.jsonl"), "a") as mf:
        for ep in range(args.epochs):
            t0 = time.time()
            tr = run_split(l1, chead, dl_tr, dev, opt)
            va = run_split(l1, chead, dl_id, dev)
            sched.step()
            row = {"epoch": ep, "train": tr, "val_id": va, "sec": round(time.time() - t0, 1)}
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(f"ep {ep}: train {tr['loss']:.4f} val {va['loss']:.4f} "
                      f"(hm {va['hm']:.4f} ori {va['ori']:.3f} w {va['w']:.4f}) "
                      f"{row['sec']}s", flush=True)
            state = {"l1": l1.state_dict(), "chead": chead.state_dict(), "cfg": cfg, "epoch": ep}
            if va["loss"] < best:
                best = va["loss"]
                torch.save(state, os.path.join(out_dir, "ckpt_best.pt"))
    torch.save(state, os.path.join(out_dir, "ckpt_final.pt"))
    print(f"done. best val {best:.4f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
