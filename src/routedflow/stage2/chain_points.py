"""FK chain points (D11): 32 semantic points along the robot kinematic chain.

Point identities are FIXED interpolation weights over link-origin segments,
chosen by arc length at t=0 — the same convex combination is applied at every
frame, so a point is "the same piece of robot" across time. 24 points on the
main chain (link0..link7 -> hand), 4 on each finger branch.

Per demo, appends to data/c_labels/<suite>/<task>.h5:
    chain_uv (T,32,2) float32   normalized (x=col/512, y=row/512) — ATM track convention
    chain_z  (T,32)   float32   camera-frame depth (m)

These give: L3 query points + query depth at any frame t, and zero-noise flow GT
(chain_uv[t:t+tl]). At deployment the same points come from proprio+FK — no
segmentation model needed.

QA (mask-bug lesson: numeric + visual anchor): for the first demo of each task,
reports the fraction of t=0 points inside the robot mask (seg0) and saves an
overlay PNG under experiments/stage2_approach_joint/qa/.
Run via `run_stage2.py chain-prep` (atm5090, CPU).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

import h5py
import numpy as np
from PIL import Image, ImageDraw

C_LABELS = os.path.join(REPO, "data", "c_labels")
QA_DIR = os.path.join(REPO, "experiments", "stage2_approach_joint", "qa")

MAIN = ["robot0_link0", "robot0_link1", "robot0_link2", "robot0_link3", "robot0_link4",
        "robot0_link5", "robot0_link6", "robot0_link7", "gripper0_right_gripper"]
BRANCHES = [["gripper0_right_gripper", "gripper0_leftfinger", "gripper0_finger_joint1_tip"],
            ["gripper0_right_gripper", "gripper0_rightfinger", "gripper0_finger_joint2_tip"]]
N_MAIN, N_BRANCH = 24, 4  # 24 + 2*4 = 32


def polyline_weights(P, n_pts):
    """P (L,3) polyline vertices -> [(seg_idx, frac)] for n_pts arc-length-even points."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, cum[-1], n_pts)
    out = []
    for s in targets:
        i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
        frac = (s - cum[i]) / max(seg[i], 1e-9)
        out.append((i, float(np.clip(frac, 0, 1))))
    return out


def chain_points_3d(link_pos_t, idx_main, idx_br, w_main, w_brs):
    pts = []
    Pm = link_pos_t[idx_main]
    for i, f in w_main:
        pts.append((1 - f) * Pm[i] + f * Pm[i + 1])
    for idx_b, w_b in zip(idx_br, w_brs):
        Pb = link_pos_t[idx_b]
        for i, f in w_b:
            pts.append((1 - f) * Pb[i] + f * Pb[i + 1])
    return np.stack(pts)  # (32,3)


def process_task(path):
    with h5py.File(path, "r+") as f:
        names = json.loads(f["link_names"][()])
        idx = {n: i for i, n in enumerate(names)}
        idx_main = [idx[n] for n in MAIN]
        idx_br = [[idx[n] for n in b] for b in BRANCHES]
        w2p = np.array(f["world_to_pix"])
        ext_inv = np.linalg.inv(np.array(f["extrinsic"]))
        demos = sorted(k for k in f.keys() if k.startswith("demo"))
        if not demos:
            return None, 0   # all demos skipped at extraction (e.g. no-grasp goal tasks)
        n_new = 0
        for k in demos:
            g = f[k]
            if "chain_uv" in g:
                continue
            lp = np.array(g["link_pos"])  # (T,L,3)
            T = lp.shape[0]
            wm = polyline_weights(lp[0][idx_main], N_MAIN)
            wbs = [polyline_weights(lp[0][b], N_BRANCH + 1)[1:] for b in idx_br]  # skip dup at hand
            uv = np.zeros((T, 32, 2), np.float32)
            zz = np.zeros((T, 32), np.float32)
            for t in range(T):
                P = chain_points_3d(lp[t], idx_main, idx_br, wm, wbs)
                homo = np.concatenate([P, np.ones((32, 1))], 1)
                pix = (w2p @ homo.T).T
                col, row = pix[:, 0] / pix[:, 2], pix[:, 1] / pix[:, 2]
                uv[t] = np.stack([col / 512.0, row / 512.0], 1)
                zz[t] = (ext_inv @ homo.T).T[:, 2]
            g.create_dataset("chain_uv", data=uv)
            g.create_dataset("chain_z", data=zz)
            n_new += 1
        return demos[0], n_new


def qa(path, demo):
    os.makedirs(QA_DIR, exist_ok=True)
    with h5py.File(path, "r") as f:
        g = f[demo]
        rgb = np.array(g["rgb0"]); seg = np.array(g["seg0"])
        robot_m = np.isin(seg, np.array(f["robot_geom_ids"]))
        uv = np.array(g["chain_uv"]); tg = int(g.attrs["t_g"])
    cols, rows = (uv[0, :, 0] * 512).astype(int), (uv[0, :, 1] * 512).astype(int)
    ok = (rows >= 0) & (rows < 512) & (cols >= 0) & (cols < 512)
    inside = float(robot_m[rows[ok], cols[ok]].mean()) if ok.any() else 0.0
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    for t, color in ((0, (60, 220, 90)), (tg, (255, 160, 40))):
        for x, y in uv[t]:
            d.ellipse([x * 512 - 4, y * 512 - 4, x * 512 + 4, y * 512 + 4], outline=color, width=2)
    d.text((8, 8), f"chain pts t=0 (green, in-mask {inside:.0%}) vs t_g={tg} (orange)",
           fill=(255, 255, 255))
    name = os.path.basename(path).replace(".h5", "")
    img.save(os.path.join(QA_DIR, f"chain_{name[:40]}.png"))
    return inside


def main(suite="libero_spatial"):
    suite_dir = os.path.join(C_LABELS, suite)
    fracs = []
    for tf in sorted(f for f in os.listdir(suite_dir) if f.endswith(".h5")):
        path = os.path.join(suite_dir, tf)
        demo0, n_new = process_task(path)
        if demo0 is None:
            print(f"[{tf[:-3][:40]}] no demos (all skipped at extraction) — chain skipped",
                  flush=True)
            continue
        frac = qa(path, demo0)
        fracs.append(frac)
        print(f"[{tf[:-3][:40]}] +{n_new} demos, t0 in-mask {frac:.0%}", flush=True)
    print(f"CHAIN PREP DONE · in-mask mean {np.mean(fracs):.0%} min {np.min(fracs):.0%}", flush=True)
    assert np.mean(fracs) > 0.6, "chain projection suspect — check link ordering / convention"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "libero_spatial")
