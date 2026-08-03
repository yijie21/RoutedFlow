"""Stage-2 joint training engine (one-step joint, warm start; grill 2026-07-31).

L = lam0*L_C + lam1*L_flow + lam2*L_action, fully differentiable
(L_action -> L4 -> predicted flow -> L3 -> z -> L1 fusion).
Warm starts: L1/CHead from stage-1 fold ckpt, L3 from ATM track transformer.
Training unit is the OPTIMIZER STEP (VLA convention — dataset size varies, steps
align compute across runs): dataloader cycles, LR cosine over --steps, train
metrics logged per --log-every window, val + ckpt_best per --val-every.
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
from routedflow.wandb_util import WandbLogger, flatten

EXP = os.path.join(REPO, "experiments", "stage2_approach_joint")
STAGE0_CFG = os.path.join(REPO, "experiments", "stage0_routing_causal_test", "configs", "stage0_vilt.yaml")
TRACK_FN = os.path.join(REPO, "third_party", "ATM", "results", "track_transformer",
                        "libero_track_transformer_libero-spatial")

DATASET_KW = dict(img_size=128, frame_stack=10, num_track_ts=16, num_track_ids=32,
                  track_obs_fs=1, augment_track=False,
                  extra_state_keys=["joint_states", "gripper_states"])


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_dev(v, dev) for v in x)
    return x


def run_val(model, loader, dev):
    model.eval()
    agg, n = {}, 0
    with torch.no_grad():
        for batch in loader:
            batch = to_dev(batch, dev)
            _, parts = model.forward_loss(batch)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
    return {k: v / n for k, v in agg.items()}


def cycle(dl):
    while True:
        yield from dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--steps", type=int, default=25000, help="optimizer steps (unit of training)")
    ap.add_argument("--log-every", type=int, default=20, help="steps per train-metric window")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lam", type=float, nargs=3, default=[0.5, 1.0, 0.1])
    ap.add_argument("--flow-noise", type=float, default=0.01)
    ap.add_argument("--l1-ckpt", default=os.path.join(EXP, "..", "stage1_l1_training",
                                                      "runs", "fold0_seed0", "ckpt_best.pt"))
    ap.add_argument("--stream-train", action="store_true",
                    help="no in-RAM train cache (~8G instead of ~14-25G); use when "
                         "neighbor jobs crowd the 85 GiB cgroup")
    ap.add_argument("--extra-suites", default="",
                    help="comma list of extra suites mixed into TRAIN windows (e.g. "
                         "'libero_goal': one scene, ten goals — visual ambiguity forces "
                         "the gradient through z; spatial fold/val/rollout unchanged)")
    ap.add_argument("--freeze-l1", action="store_true",
                    help="freeze L1+CHead at their warm-start weights and set lam_C=0 — "
                         "two-stage posture: joint fine-tuning on 360 spatial samples "
                         "erodes a good z (A1 ood 0.70) faster than the policy can learn it "
                         "(probe evidence 2026-08-03); frozen L1 also runs in eval mode so "
                         "train-time z == rollout z (no dropout noise)")
    ap.add_argument("--use-cross-attn", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="ckpt path: restore model (+opt if saved) and continue from its step")
    ap.add_argument("--probe-every", type=int, default=2500,
                    help="closed-loop probe cadence in steps (0 = off): subprocess rollout "
                         "on the first 4 tasks (2 ood + 2 train), 5 eps each — robomimic-style "
                         "rollout-based selection (val BC loss is blind to closed-loop SR)")
    ap.add_argument("--probe-tasks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    name = args.name or f"joint_fold{args.fold}_seed{args.seed}"
    out = os.path.join(EXP, "runs", name)
    os.makedirs(out, exist_ok=True)

    # val always streams; train caches by default but can stream too (shared box:
    # neighbor experiments can eat 50G+ of the 85 GiB cgroup — cached train (~14G
    # with goal mix) then gets OOM-killed; streaming drops us to ~8G)
    extra = tuple(s for s in args.extra_suites.split(",") if s)
    ds_tr = Stage2Dataset(fold=args.fold, split="train",
                          cache_all=not args.stream_train,
                          cache_image=not args.stream_train,
                          extra_suites=extra, **DATASET_KW)
    ds_id = Stage2Dataset(fold=args.fold, split="val_id", cache_all=False,
                          cache_image=False, **DATASET_KW)
    print(f"windows: train {len(ds_tr)} / val_id {len(ds_id)}"
          f"{' (streaming)' if args.stream_train else ''}", flush=True)
    # cached mode: few persistent workers (CoW over the cache killed a run once);
    # streaming mode: more workers to hide disk latency
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       num_workers=6 if args.stream_train else 2,
                       drop_last=True, persistent_workers=True,
                       prefetch_factor=4 if args.stream_train else 2)
    # val streams from disk now (no in-RAM cache to CoW) -> workers are cheap again
    dl_id = DataLoader(ds_id, batch_size=args.bs, num_workers=2)

    cfg = OmegaConf.load(STAGE0_CFG).model_cfg
    cfg.pop("track_gate_cfg", None)
    lam = (0.0, args.lam[1], args.lam[2]) if args.freeze_l1 else tuple(args.lam)
    model = JointApproachModel(TRACK_FN, cfg, l1_ckpt=args.l1_ckpt, lam=lam,
                               flow_noise=args.flow_noise,
                               use_cross_attn=args.use_cross_attn).to(dev)
    groups = [
        {"params": list(model.l3.parameters()), "lr": 1e-4},
        {"params": list(model.l4.parameters()), "lr": 3e-4},
    ]
    if args.freeze_l1:
        for p in list(model.l1.parameters()) + list(model.chead.parameters()):
            p.requires_grad = False
    else:
        groups.insert(0, {"params": list(model.l1.parameters()) + list(model.chead.parameters()),
                          "lr": 1e-4})

    def set_train():
        model.train()
        if args.freeze_l1:
            model.l1.eval()
            model.chead.eval()
    n_par = sum(p.numel() for g in groups for p in g["params"] if p.requires_grad) / 1e6
    print(f"trainable params: {n_par:.1f}M", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    best, start_step = float("inf"), 1
    if args.resume:
        rs = torch.load(args.resume, map_location=dev, weights_only=False)
        for part in ("l1", "chead", "l3", "l4"):
            getattr(model, part).load_state_dict(rs[part])
        if "opt" in rs:
            opt.load_state_dict(rs["opt"])
        best = rs.get("best", best)
        start_step = rs["step"] + 1
        for _ in range(start_step - 1):        # fast-forward the cosine schedule
            sched.step()
        print(f"resumed from {args.resume} at step {rs['step']} "
              f"(opt state: {'yes' if 'opt' in rs else 'NO — moments rebuild'})", flush=True)

    meta = vars(args)
    wb = WandbLogger(name, "stage2", config=meta, enabled=not args.no_wandb)

    def snapshot(step):
        return {"l1": model.l1.state_dict(), "chead": model.chead.state_dict(),
                "l3": model.l3.state_dict(), "l4": model.l4.state_dict(),
                "opt": opt.state_dict(), "best": best,
                "cfg": meta, "step": step}

    it = cycle(dl_tr)
    set_train()
    win, t0 = {}, time.time()
    with open(os.path.join(out, "metrics.jsonl"), "a") as mf:
        for step in range(start_step, args.steps + 1):
            lr_now = opt.param_groups[0]["lr"]
            opt.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                batch = to_dev(next(it), dev)
                loss, parts = model.forward_loss(batch)
                (loss / args.accum).backward()
                for k, v in parts.items():
                    win[k] = win.get(k, 0.0) + v / args.accum
            opt.step()
            sched.step()
            if step % args.log_every == 0:
                tr = {k: v / args.log_every for k, v in win.items()}
                row = {"step": step, "train": tr, "sec": round(time.time() - t0, 1)}
                mf.write(json.dumps(row) + "\n")
                mf.flush()
                wb.log(flatten(row) | {"lr": lr_now}, step=step)
                print(f"step {step}: loss {tr['loss']:.4f} (C {tr['l_c']:.3f} "
                      f"flow {tr['l_flow']:.5f} act {tr['l_action']:.4f}) "
                      f"{row['sec']}s/{args.log_every}", flush=True)
                win, t0 = {}, time.time()
            if step % args.val_every == 0 or step == args.steps:
                va = run_val(model, dl_id, dev)
                mf.write(json.dumps({"step": step, "val_id": va}) + "\n")
                mf.flush()
                wb.log(flatten({"val_id": va}), step=step)
                print(f"step {step}: VAL loss {va['loss']:.4f} act {va['l_action']:.4f}",
                      flush=True)
                torch.save(snapshot(step), os.path.join(out, "ckpt_last.pt"))
                if va["l_action"] < best:
                    best = va["l_action"]
                    torch.save(snapshot(step), os.path.join(out, "ckpt_best.pt"))
                set_train()
                torch.cuda.empty_cache()
            if args.probe_every and (step % args.probe_every == 0 or step == args.steps):
                torch.save(snapshot(step), os.path.join(out, f"ckpt_step{step}.pt"))
                torch.cuda.empty_cache()
                import subprocess
                r = subprocess.run(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "eval_rollout.py"),
                     "--run", name, "--ckpt", f"ckpt_step{step}.pt",
                     "--tasks-limit", str(args.probe_tasks), "--vec", "5", "--rounds", "1",
                     "--n-video", "0", "--no-wandb", "--out-tag", f"probe{step}"],
                    capture_output=True, text=True)
                try:
                    sm = json.load(open(os.path.join(EXP, f"rollout_{name}_probe{step}",
                                                     "summary.json")))
                    pr = {f"probe/{'ood' if v['split']=='ood' else 'train'}_"
                          f"{k.split('_bowl_')[-1][:24]}": v["sr"]
                          for k, v in sm["per_task"].items()}
                    pr["probe/mean"] = float(np.mean([v["sr"] for v in sm["per_task"].values()]))
                    mf.write(json.dumps({"step": step, "probe": sm["per_task"],
                                         "probe_mean": pr["probe/mean"]}) + "\n")
                    mf.flush()
                    wb.log(pr, step=step)
                    print(f"step {step}: PROBE mean SR {pr['probe/mean']:.3f} "
                          f"({ {k.split('/')[-1]: round(v,2) for k,v in pr.items()} })", flush=True)
                except Exception as e:
                    print(f"step {step}: probe failed ({e}); tail: {r.stdout[-300:]}", flush=True)
                set_train()
    torch.save(snapshot(args.steps), os.path.join(out, "ckpt_final.pt"))
    wb.summary(best_val_l_action=best)
    wb.finish()
    print(f"done. best val action loss {best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
