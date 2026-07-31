"""Offline job #2: batch AFUN inference -> prior masks for all 500 demos.

Runs in the `afun` conda env (NOT atm5090). Model loads ONCE, then loops all
demos; incremental h5 writes + skip-existing => crash-safe and resumable.

Input : data/c_labels/<suite>/<task>.h5  (rgb0 512^2, depth0 meters, K, contact)
Output: data/stage1_cache/<suite>/afun_prior.h5
            <task>/<demo>: (512,512) uint8 mask (resized to 512 if needed)
            attrs: confidence; task-level attrs: query
        data/stage1_cache/<suite>/prior_qc.json   # pre-registered prior-accuracy QC

Uses GT sim depth with --no-refine semantics (refine=False path: raw depth
straight to the model, as validated in the AFUN spike).
Run via `run_stage1.py afun-prior` (2-3h GPU, background).
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AFUN = os.path.join(REPO, "third_party", "AFUN")
DATA = os.path.join(REPO, "data", "c_labels")
OUT_DIR = os.path.join(REPO, "data", "stage1_cache")

sys.path.insert(0, AFUN)
os.environ.setdefault("AFUN_ROOT", AFUN)
os.chdir(AFUN)  # hydra config paths are relative to the AFUN repo

import h5py  # noqa: E402
from PIL import Image  # noqa: E402


def to_512_mask(mask):
    m = np.asarray(mask).squeeze().astype(np.uint8)
    if m.shape != (512, 512):
        m = np.array(Image.fromarray(m * 255).resize((512, 512), Image.NEAREST)) > 127
        m = m.astype(np.uint8)
    return m


def qc_row(mask, contact_rowcol):
    r, c = contact_rowcol
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return {"hit": False, "dist_px": None}
    d = float(np.min(np.hypot(ys - r, xs - c)))
    return {"hit": bool(mask[int(round(r)), int(round(c))]), "dist_px": round(d, 1)}


def main(suite="libero_spatial"):
    from src.inference import load_model, infer_image

    out_dir = os.path.join(OUT_DIR, suite)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "afun_prior.h5")

    print("==> loading AFUN model (once)...", flush=True)
    model = load_model(os.path.join(AFUN, "checkpoints", "afun.pt"), "inference", device="cuda:0")

    t0 = time.time()
    n_done = 0
    suite_dir = os.path.join(DATA, suite)
    with h5py.File(out_path, "a") as out:
        for tf in sorted(f for f in os.listdir(suite_dir) if f.endswith(".h5")):
            task = tf[:-3]
            with h5py.File(os.path.join(suite_dir, tf), "r") as f:
                query = f.attrs["task_language"]
                K = np.array(f["K"])
                cam = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                       "cx": float(K[0, 2]), "cy": float(K[1, 2])}
                demos = sorted(k for k in f.keys() if k.startswith("demo"))
                g = out.require_group(task)
                g.attrs["query"] = query
                for k in demos:
                    if k in g:
                        continue  # resumable
                    rgb = Image.fromarray(np.array(f[k]["rgb0"]))
                    depth_mm = (np.array(f[k]["depth0"]) * 1000.0).astype(np.float32)
                    try:
                        res = infer_image(model, image=rgb, query=str(query),
                                          device="cuda:0", depth=depth_mm, cam=cam)
                        mask = to_512_mask(res["mask"])
                        conf = float(np.asarray(res.get("score", np.nan)).squeeze())
                    except Exception as e:  # record failure, keep going
                        print(f"!! {task}/{k} FAILED: {e}", flush=True)
                        mask, conf = np.zeros((512, 512), np.uint8), float("nan")
                    d = g.create_dataset(k, data=mask, compression="gzip")
                    d.attrs["confidence"] = conf
                    n_done += 1
                    out.flush()
                print(f"[{task}] done (+{n_done} this run, {(time.time()-t0)/60:.0f} min)", flush=True)

    # pre-registered prior-accuracy table (plan §2.2) — full pass over the
    # finished h5 so resumed runs still produce a complete table.
    qc, summary = {}, {}
    with h5py.File(out_path, "r") as out:
        for tf in sorted(f for f in os.listdir(suite_dir) if f.endswith(".h5")):
            task = tf[:-3]
            with h5py.File(os.path.join(suite_dir, tf), "r") as f:
                for k in sorted(x for x in out.get(task, {}) ):
                    row = qc_row(np.array(out[task][k]), np.array(f[k]["contact_rowcol"]))
                    qc.setdefault(task, []).append(row)
    for task, rows in qc.items():
        hits = sum(r["hit"] for r in rows)
        near = sum(1 for r in rows if r["dist_px"] is not None and r["dist_px"] <= 20)
        summary[task] = {"n": len(rows), "in_mask": hits, "within_20px": near}
    tot = {"n": sum(s["n"] for s in summary.values()),
           "in_mask": sum(s["in_mask"] for s in summary.values()),
           "within_20px": sum(s["within_20px"] for s in summary.values())}
    json.dump({"global": tot, "per_task": summary, "rows": qc},
              open(os.path.join(out_dir, "prior_qc.json"), "w"), indent=2)
    print("PRIOR QC:", json.dumps(tot), flush=True)
    print("AFUN PRIOR DONE ->", out_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "libero_spatial")
