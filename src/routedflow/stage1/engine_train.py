"""Stage-1 training engine (curriculum ①: L1+L2, loss = L_C only).

Tiny model over cached features => plain single-GPU torch loop, seconds/epoch.
Outputs under experiments/stage1_l1_training/runs/<name>/:
    ckpt_best.pt / ckpt_final.pt   {l1, chead, cfg, step}
    metrics.jsonl                  train row per --log-every steps, val row per --val-every
Training unit is the OPTIMIZER STEP (VLA convention — dataset size varies, steps
align compute across runs); LR cosine over --steps.
Run via `run_stage1.py train-l1 [--fold 0] [--steps 1200] ...` (atm5090).
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
from routedflow.wandb_util import WandbLogger, flatten

EXP = os.path.join(REPO, "experiments", "stage1_l1_training")


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def make_prep(dev, dino_net=None):
    """Move batch to device; with a dino_net, compute features online from
    batch['rgb'] (raw/augmented mode) instead of using cached ones."""
    mean, std = MEAN.to(dev), STD.to(dev)

    def prep(batch):
        batch = {k: v.to(dev) for k, v in batch.items()}
        if dino_net is not None and "rgb" in batch:
            x = torch.nn.functional.interpolate(batch["rgb"], size=518,
                                                mode="bilinear", align_corners=False)
            with torch.no_grad():
                batch["dino"] = dino_net.forward_features((x - mean) / std)["x_norm_patchtokens"]
        return batch
    return prep


def run_split(l1, chead, loader, dev, opt=None, prep=None):
    train = opt is not None
    l1.train(train), chead.train(train)
    agg, n = {}, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = prep(batch) if prep else {k: v.to(dev) for k, v in batch.items()}
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
    ap.add_argument("--steps", type=int, default=1200, help="optimizer steps (unit of training)")
    ap.add_argument("--log-every", type=int, default=20, help="steps per train-metric window")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--prior-dropout", type=float, default=0.3)
    ap.add_argument("--no-prior", action="store_true", help="train with zeroed AFUN channel")
    ap.add_argument("--extra-suites", default="libero_object,libero_goal",
                    help="comma list of suites whose demos all join the train split "
                         "(C-head 开小灶); '' disables")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--cached-feats", action="store_true",
                    help="legacy path: precomputed DINO cache, no aug, primary suite only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    name = args.name or f"fold{args.fold}_seed{args.seed}" + ("_noprior" if args.no_prior else "")
    out_dir = os.path.join(EXP, "runs", name)
    os.makedirs(out_dir, exist_ok=True)

    raw = not args.cached_feats
    extra = [s for s in args.extra_suites.split(",") if s] if raw else None
    ds_tr = Stage1Dataset(fold=args.fold, split="train", use_prior=not args.no_prior,
                          extra_suites=extra, raw=raw,
                          augment=raw and not args.no_augment, seed=args.seed)
    ds_id = Stage1Dataset(fold=args.fold, split="val_id", use_prior=not args.no_prior,
                          raw=raw, augment=False)
    print(f"train {len(ds_tr)} (extra: {extra}) / val_id {len(ds_id)} · "
          f"ood tasks: {ds_tr.ood_tasks}", flush=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    dl_id = DataLoader(ds_id, batch_size=args.bs, num_workers=2)

    dino_net = None
    if raw:
        dino_net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(dev).eval()
    prep = make_prep(dev, dino_net)

    l1 = L1FrontEnd(prior_dropout=args.prior_dropout).to(dev)
    chead = CHead().to(dev)
    params = list(l1.parameters()) + list(chead.parameters())
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    cfg = vars(args) | {"ood_tasks": ds_tr.ood_tasks}
    wb = WandbLogger(name, "stage1", config=cfg, enabled=not args.no_wandb)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    best = float("inf")

    def cycle(dl):
        while True:
            yield from dl

    def snapshot(step):
        return {"l1": l1.state_dict(), "chead": chead.state_dict(), "cfg": cfg, "step": step}

    it = cycle(dl_tr)
    l1.train(), chead.train()
    win, wn, t0 = {}, 0, time.time()
    with open(os.path.join(out_dir, "metrics.jsonl"), "a") as mf:
        for step in range(1, args.steps + 1):
            lr_now = opt.param_groups[0]["lr"]
            batch = prep(next(it))
            hm, z, fm = l1(batch["dino"], batch["prior"], batch["text"])
            yl, pl, wp = chead(z, fm, batch["contact_rowcol"])
            loss, parts = stage1_loss(hm, z, yl, pl, wp, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            win["loss"] = win.get("loss", 0.0) + loss.item()
            for k, v in parts.items():
                win[k] = win.get(k, 0.0) + v
            wn += 1
            if step % args.log_every == 0:
                tr = {k: v / wn for k, v in win.items()}
                row = {"step": step, "train": tr, "sec": round(time.time() - t0, 1)}
                mf.write(json.dumps(row) + "\n")
                mf.flush()
                wb.log(flatten(row) | {"lr": lr_now}, step=step)
                win, wn, t0 = {}, 0, time.time()
            if step % args.val_every == 0 or step == args.steps:
                va = run_split(l1, chead, dl_id, dev, prep=prep)
                mf.write(json.dumps({"step": step, "val_id": va}) + "\n")
                mf.flush()
                wb.log(flatten({"val_id": va}), step=step)
                print(f"step {step}: val {va['loss']:.4f} (hm {va['hm']:.4f} "
                      f"ori {va['ori']:.3f} w {va['w']:.4f})", flush=True)
                if va["loss"] < best:
                    best = va["loss"]
                    torch.save(snapshot(step), os.path.join(out_dir, "ckpt_best.pt"))
                l1.train(), chead.train()
    torch.save(snapshot(args.steps), os.path.join(out_dir, "ckpt_final.pt"))
    wb.summary(best_val_loss=best)
    wb.finish()
    print(f"done. best val {best:.4f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
