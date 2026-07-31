"""Quality control for extracted C labels (data/c_labels/<suite>/*.h5).

Outputs into the experiment folder (experiments/stage1_c_labels/qc/):
    qc_stats.json                 per-task + global stats
    overlay_<task>_<demo>.png     contact point on t=0 RGB (sampled demos)
    spread_<task>.png             ALL demos' contact points on demo_0's RGB
                                  (multimodality view, plan §2.3 #2)
    qc_montage.png                grid of the sampled overlays

Pure CPU/h5 reading — no sim. Run via `run_stage1.py qc`.
"""
import argparse
import json
import os

import h5py
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "c_labels")
QC_DIR = os.path.join(REPO_ROOT, "experiments", "stage1_c_labels", "qc")


def dot(draw, row, col, color, r=5):
    draw.ellipse([col - r, row - r, col + r, row + r], outline=color, width=3)


def robust_lift_gap(g, K, extrinsic, win=4):
    """Lift with the MIN depth in a (2*win+1)^2 window around the contact pixel
    (nearest-surface heuristic vs single-pixel lift, which falls off the object
    at depth discontinuities — elevated grasps)."""
    row, col = np.array(g["contact_rowcol"])
    depth = np.array(g["depth0"])
    R = depth.shape[0]
    r_i, c_i = int(round(row)), int(round(col))
    if not (0 <= r_i < R and 0 <= c_i < R):
        return float("nan")
    d = depth[max(r_i - win, 0):r_i + win + 1, max(c_i - win, 0):c_i + win + 1].min()
    x_lift = d * (np.linalg.inv(K) @ np.array([col, row, 1.0]))
    tg = int(g.attrs["t_g"])
    x_true = (np.linalg.inv(extrinsic) @ np.append(np.array(g["ee_pos"])[tg], 1.0))[:3]
    return float(np.linalg.norm(x_lift - x_true))


def grasped_by_motion(g, obj_names, min_move=0.03):
    """The grasped object is the one that MOVES after t_g (transport displaces
    it; distractors stay put). Far more robust than nearest-body-at-t_g."""
    tg = int(g.attrs["t_g"])
    obj_pos = np.array(g["obj_pos"])
    disp = np.linalg.norm(obj_pos[-1] - obj_pos[tg], axis=-1)
    i = int(np.argmax(disp))
    return (obj_names[i] if disp[i] >= min_move else None), float(disp[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--per-task", type=int, default=2, help="sampled overlay demos per task")
    args = ap.parse_args()

    os.makedirs(QC_DIR, exist_ok=True)
    task_files = sorted(f for f in os.listdir(os.path.join(DATA_ROOT, args.suite)) if f.endswith(".h5"))
    stats, overlays = {}, []

    for tf in task_files:
        name = tf[:-3]
        with h5py.File(os.path.join(DATA_ROOT, args.suite, tf), "r") as f:
            demos = sorted(k for k in f.keys() if k.startswith("demo"))
            obj_names = json.loads(f["obj_names"][()])
            K, extrinsic = np.array(f["K"]), np.array(f["extrinsic"])
            tg = [f[k].attrs["t_g"] for k in demos]
            T = [f[k].attrs["T"] for k in demos]
            gd = [f[k].attrs["grasp_dist_m"] for k in demos]
            lg = [f[k].attrs["lift_gap_m"] for k in demos]
            drift = [f[k].attrs["replay_ee_maxdiff_m"] for k in demos]
            lg_rob = [robust_lift_gap(f[k], K, extrinsic) for k in demos]
            motion = [grasped_by_motion(f[k], obj_names) for k in demos]
            mismatch = [k for k, (mb, _) in zip(demos, motion)
                        if mb is not None and mb != f[k].attrs["grasped_body"]]
            stats[name] = {
                "n_demos": len(demos),
                "skipped": json.loads(f.attrs["skipped_demos"]),
                "t_g_median": float(np.median(tg)), "T_median": float(np.median(T)),
                "grasp_dist_m": [float(np.median(gd)), float(np.max(gd))],
                "lift_gap_m": [float(np.nanmedian(lg)), float(np.nanmax(lg))],
                "lift_gap_robust_m": [float(np.nanmedian(lg_rob)), float(np.nanmax(lg_rob))],
                "replay_ee_maxdiff_m": float(np.max(drift)),
                "grasped_by_motion": sorted(set(mb for mb, _ in motion if mb)),
                "no_motion_demos": [k for k, (mb, _) in zip(demos, motion) if mb is None],
                "nearest_vs_motion_mismatch": mismatch,
            }

            # sampled overlays
            step = max(len(demos) // args.per_task, 1)
            for k in demos[::step][: args.per_task]:
                img = Image.fromarray(np.array(f[k]["rgb0"]))
                draw = ImageDraw.Draw(img)
                row, col = np.array(f[k]["contact_rowcol"])
                dot(draw, row, col, (255, 40, 40))
                draw.text((8, 8), f"{name[:40]}\n{k} tg={f[k].attrs['t_g']} "
                          f"{f[k].attrs['grasped_body']}", fill=(255, 255, 0))
                p = os.path.join(QC_DIR, f"overlay_{name}_{k}.png")
                img.save(p)
                overlays.append(p)

            # spread image: all contact points on demo_0's frame
            img = Image.fromarray(np.array(f[demos[0]]["rgb0"]))
            draw = ImageDraw.Draw(img)
            for k in demos:
                row, col = np.array(f[k]["contact_rowcol"])
                dot(draw, row, col, (40, 255, 40), r=3)
            img.save(os.path.join(QC_DIR, f"spread_{name}.png"))

    # montage of sampled overlays (rows of 5)
    thumbs = [Image.open(p).resize((256, 256)) for p in overlays]
    cols = 5
    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * 256, rows * 256))
    for i, t in enumerate(thumbs):
        canvas.paste(t, ((i % cols) * 256, (i // cols) * 256))
    canvas.save(os.path.join(QC_DIR, "qc_montage.png"))

    glo = {
        "n_tasks": len(task_files),
        "n_demos": int(sum(s["n_demos"] for s in stats.values())),
        "n_skipped": int(sum(len(s["skipped"]) for s in stats.values())),
        "lift_gap_m_median": float(np.nanmedian([s["lift_gap_m"][0] for s in stats.values()])),
        "lift_gap_robust_m_median": float(np.nanmedian([s["lift_gap_robust_m"][0] for s in stats.values()])),
        "grasp_dist_m_max": float(np.max([s["grasp_dist_m"][1] for s in stats.values()])),
        "replay_ee_maxdiff_m_max": float(np.max([s["replay_ee_maxdiff_m"] for s in stats.values()])),
        "n_no_motion": int(sum(len(s["no_motion_demos"]) for s in stats.values())),
        "n_nearest_vs_motion_mismatch": int(sum(len(s["nearest_vs_motion_mismatch"]) for s in stats.values())),
    }
    json.dump({"global": glo, "per_task": stats},
              open(os.path.join(QC_DIR, "qc_stats.json"), "w"), indent=2)
    print(json.dumps(glo, indent=2))
    for name, s in stats.items():
        print(f"{name}: n={s['n_demos']} skip={len(s['skipped'])} tg~{s['t_g_median']:.0f} "
              f"graspdist~{s['grasp_dist_m'][0]*100:.1f}cm liftgap~{s['lift_gap_m'][0]*100:.1f}cm "
              f"robust~{s['lift_gap_robust_m'][0]*100:.1f}cm mism={len(s['nearest_vs_motion_mismatch'])} "
              f"grasped={s['grasped_by_motion']}")


if __name__ == "__main__":
    main()
