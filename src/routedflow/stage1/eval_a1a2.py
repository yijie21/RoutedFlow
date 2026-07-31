"""Stage-1 acceptance evaluation: A1 (heatmap peaks) + A2 (latent probes).

A1 (plan §2.4): a sample is a hit if any of the top-5 heatmap peaks lies within
10px@512 of the GT contact point. Reported for val_id, val_ood, per task,
the two AFUN-spike-failure tasks, and stratified by AFUN-prior correctness
(prior_qc.json, <=20px criterion).

A2: fit linear (ridge) probes on TRAIN features, evaluate on val_id:
targets = contact (px err), yaw/pitch bin (acc), w (L1 err). Three feature sets:
    z_trained  — the [C] token of the trained model      (the claim)
    dino_pool  — mean DINO patch feature                 (no-training baseline)
    z_random   — [C] token of an UNTRAINED same-arch L1  (probe-capacity control)
Run via `run_stage1.py eval-l1 --run <name>` (atm5090, GPU).
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from routedflow.stage1.dataset import Stage1Dataset
from routedflow.stage1.model import CHead, L1FrontEnd

EXP = os.path.join(REPO, "experiments", "stage1_l1_training")
SPIKE_FAIL_TASKS = ("pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo",
                    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo")
A1_RADIUS_512 = 10.0


def top5_peaks(hm, k=5, suppress=5):
    """hm (128,128) prob -> list of (row512, col512), greedy NMS."""
    h = hm.copy()
    peaks = []
    for _ in range(k):
        r, c = np.unravel_index(h.argmax(), h.shape)
        peaks.append((r * 4.0, c * 4.0))
        h[max(r - suppress, 0):r + suppress + 1, max(c - suppress, 0):c + suppress + 1] = -1
    return peaks


@torch.no_grad()
def collect(l1, ds, dev, bs=64):
    """Forward a dataset -> heatmaps, z, plus targets (numpy)."""
    dl = DataLoader(ds, batch_size=bs, num_workers=2)
    out = {k: [] for k in ("hm", "z", "contact", "yaw", "pitch", "w", "dino_pool")}
    for b in dl:
        hm, z, _ = l1(b["dino"].to(dev), b["prior"].to(dev), b["text"].to(dev))
        B = hm.shape[0]
        out["hm"].append(torch.softmax(hm.reshape(B, -1), -1).reshape(B, 128, 128).cpu().numpy())
        out["z"].append(z.cpu().numpy())
        out["dino_pool"].append(b["dino"].mean(1).numpy())
        out["contact"].append(b["contact_rowcol"].numpy())
        out["yaw"].append(b["yaw_bin"].numpy())
        out["pitch"].append(b["pitch_bin"].numpy())
        out["w"].append(b["w"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def a1_hits(hms, contacts):
    hits = []
    for hm, gt in zip(hms, contacts):
        d = min(np.hypot(pr - gt[0], pc - gt[1]) for pr, pc in top5_peaks(hm))
        hits.append(d <= A1_RADIUS_512)
    return np.array(hits)


def prior_correct_map(suite="libero_spatial", thresh=20.0):
    """{(task, demo): prior-correct bool} recomputed from the mask h5 by NAME —
    immune to the list-ordering pitfall of prior_qc.json's rows."""
    import h5py
    path = os.path.join(REPO, "data", "stage1_cache", suite, "afun_prior.h5")
    if not os.path.exists(path):
        return {}
    out = {}
    with h5py.File(path, "r") as pf:
        for task in pf:
            with h5py.File(os.path.join(REPO, "data", "c_labels", suite, f"{task}.h5"), "r") as lf:
                for demo in pf[task]:
                    mask = np.array(pf[task][demo])
                    r, c = np.array(lf[demo]["contact_rowcol"])
                    ys, xs = np.nonzero(mask)
                    ok = len(ys) > 0 and float(np.min(np.hypot(ys - r, xs - c))) <= thresh
                    out[(task, demo)] = bool(ok)
    return out


def ridge_fit(X, Y, lam=1e-2):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)
    return lambda Xq: np.concatenate([Xq, np.ones((len(Xq), 1))], 1) @ W


def probe_report(Xtr, Xte, tr, te):
    z = lambda X: (X - Xtr.mean(0)) / (Xtr.std(0) + 1e-6)
    Xtr_, Xte_ = z(Xtr), z(Xte)
    rep = {}
    f = ridge_fit(Xtr_, tr["contact"])
    rep["contact_px"] = float(np.hypot(*(f(Xte_) - te["contact"]).T).mean())
    for key, n in (("yaw", 36), ("pitch", 12)):
        f = ridge_fit(Xtr_, np.eye(n)[tr[key]])
        rep[f"{key}_acc"] = float((f(Xte_).argmax(1) == te[key]).mean())
    f = ridge_fit(Xtr_, tr["w"][:, None])
    rep["w_l1"] = float(np.abs(f(Xte_)[:, 0] - te["w"]).mean())
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    args = ap.parse_args()

    dev = "cuda"
    run_dir = os.path.join(EXP, "runs", args.run)
    state = torch.load(os.path.join(run_dir, args.ckpt), map_location=dev, weights_only=False)
    cfg = state["cfg"]
    fold, use_prior = cfg["fold"], not cfg.get("no_prior", False)

    l1 = L1FrontEnd().to(dev).eval()
    l1.load_state_dict(state["l1"])
    torch.manual_seed(1234)
    l1_rand = L1FrontEnd().to(dev).eval()  # untrained control, same arch/dim

    ds = {s: Stage1Dataset(fold=fold, split=s, use_prior=use_prior)
          for s in ("train", "val_id", "val_ood")}
    col = {s: collect(l1, d, dev) for s, d in ds.items()}
    col_rand = {s: collect(l1_rand, ds[s], dev) for s in ("train", "val_id")}

    # ---- A1 ----
    report = {"run": args.run, "fold": fold, "ood_tasks": ds["train"].ood_tasks}
    prior_ok = prior_correct_map()  # {(task, demo): bool} keyed by name, order-safe
    for split in ("val_id", "val_ood"):
        hits = a1_hits(col[split]["hm"], col[split]["contact"])
        rep = {"overall": float(hits.mean()), "n": int(len(hits)), "per_task": {}}
        metas = [(s["task"], s["demo"]) for s in ds[split].samples]
        for i, (task, demo) in enumerate(metas):
            rep["per_task"].setdefault(task, []).append(bool(hits[i]))
        rep["per_task"] = {t: round(float(np.mean(v)), 3) for t, v in rep["per_task"].items()}
        spike = [v for t, v in rep["per_task"].items() if t in SPIKE_FAIL_TASKS]
        if spike:
            rep["spike_fail_tasks_mean"] = round(float(np.mean(spike)), 3)
        if prior_ok:
            strata = {True: [], False: []}
            for i, (task, demo) in enumerate(metas):
                if (task, demo) in prior_ok:
                    strata[prior_ok[(task, demo)]].append(bool(hits[i]))
            rep["a1_prior_correct"] = round(float(np.mean(strata[True])), 3) if strata[True] else None
            rep["a1_prior_wrong"] = round(float(np.mean(strata[False])), 3) if strata[False] else None
        report[f"A1_{split}"] = rep

    # ---- A2 ----
    report["A2"] = {
        "z_trained": probe_report(col["train"]["z"], col["val_id"]["z"],
                                  col["train"], col["val_id"]),
        "dino_pool": probe_report(col["train"]["dino_pool"], col["val_id"]["dino_pool"],
                                  col["train"], col["val_id"]),
        "z_random": probe_report(col_rand["train"]["z"], col_rand["val_id"]["z"],
                                 col["train"], col["val_id"]),
    }

    out = os.path.join(run_dir, f"eval_{args.ckpt.replace('.pt', '')}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("A")}, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
