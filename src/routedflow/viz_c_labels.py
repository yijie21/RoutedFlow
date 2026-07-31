"""Rich visualization of the extracted labels (all modalities, one demo per task).

For each task h5 renders a 4-panel figure into experiments/stage1_c_labels/viz/:
    A  t=0 RGB + contact pixel + E(t_g) orientation axes (projected) + full EE
       trajectory projected, colored by phase (approach green / transport orange)
    B  metric depth + contact pixel (single-pixel vs robust lift readout in title)
    C  segmentation categories: robot / grasped object (by motion) / other movable
       objects / background
    D  aperture (gripper_q) timeline with latched phase shading and t_g marker

Pure offline h5 reading — no sim. Run via `run_stage1.py viz`.
"""
import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "c_labels")
VIZ_DIR = os.path.join(REPO_ROOT, "experiments", "stage1_c_labels", "viz")

from routedflow.qc_c_labels import grasped_by_motion, robust_lift_gap  # noqa: E402


def quat_to_mat(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project(world_to_pix, pts_w):
    """(N,3) world points -> (N,2) [row, col]."""
    homo = np.concatenate([pts_w, np.ones((len(pts_w), 1))], axis=1)
    uvw = (world_to_pix @ homo.T).T
    return np.stack([uvw[:, 1] / uvw[:, 2], uvw[:, 0] / uvw[:, 2]], axis=1)


def seg_categories(seg, geom_map, robot_gids, obj_names, grasped):
    """0 bg / 1 robot / 2 grasped / 3 other movable object."""
    cat = np.zeros_like(seg, dtype=np.uint8)
    robot = np.isin(seg, robot_gids)
    cat[robot] = 1
    body_of = np.array([geom_map.get(str(g), geom_map.get(g, "")) or "" for g in range(seg.max() + 1)])
    body_img = body_of[np.clip(seg, 0, len(body_of) - 1)]
    for name in obj_names:
        m = (body_img == name) & ~robot
        cat[m] = 2 if name == grasped else 3
    return cat


def viz_task(path, demo_key, out_png):
    with h5py.File(path, "r") as f:
        g = f[demo_key]
        rgb = np.array(g["rgb0"]); depth = np.array(g["depth0"]); seg = np.array(g["seg0"])
        w2p = np.array(f["world_to_pix"]); K = np.array(f["K"]); ext = np.array(f["extrinsic"])
        obj_names = json.loads(f["obj_names"][()])
        geom_map = json.loads(f["geom_id_to_body"][()])
        robot_gids = np.array(f["robot_geom_ids"])
        ee_pos = np.array(g["ee_pos"]); ee_quat = np.array(g["ee_quat"])
        gq = np.array(g["gripper_q"]); phase = np.array(g["phase"])
        row, col = np.array(g["contact_rowcol"])
        tg = int(g.attrs["t_g"])
        grasped, disp = grasped_by_motion(g, obj_names)
        lg = float(g.attrs["lift_gap_m"]); lgr = robust_lift_gap(g, K, ext)
        task_lang = f.attrs["task_language"]

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.6))
    fig.suptitle(f"{task_lang}   [{demo_key}]", fontsize=13)

    # A: RGB + trajectory + contact + orientation axes
    ax = axes[0]
    ax.imshow(rgb)
    traj = project(w2p, ee_pos)
    app, tra = phase == 0, phase == 1
    ax.plot(traj[app, 1], traj[app, 0], "-", color="lime", lw=2, label="EE approach")
    ax.plot(traj[tra, 1], traj[tra, 0], "-", color="orange", lw=2, label="EE transport")
    Rm = quat_to_mat(ee_quat[tg])
    axis_colors = [("red", "x"), ("green", "y"), ("blue", "z")]
    for k, (c, lab) in enumerate(axis_colors):
        tip = project(w2p, (ee_pos[tg] + 0.07 * Rm[:, k])[None])[0]
        ax.annotate("", xy=(tip[1], tip[0]), xytext=(col, row),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.2))
    ax.plot(col, row, "*", color="yellow", ms=16, mec="black", label="contact p̂")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"A · RGB + EE traj + R(t_g) axes   t_g={tg}/{len(phase)}")
    ax.axis("off")

    # B: depth
    ax = axes[1]
    im = ax.imshow(depth, cmap="turbo")
    ax.plot(col, row, "*", color="white", ms=14, mec="black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"B · depth (m)   lift gap {lg*100:.1f}cm / robust {lgr*100:.1f}cm")
    ax.axis("off")

    # C: segmentation categories
    ax = axes[2]
    palette = np.array([[235, 235, 235], [220, 60, 60], [60, 180, 60], [70, 110, 220]], np.uint8)
    ax.imshow(palette[seg_categories(seg, geom_map, robot_gids, obj_names, grasped)])
    ax.plot(col, row, "*", color="black", ms=14)
    ax.set_title(f"C · seg: robot(red) grasped(green)={grasped}\nothers(blue), moved {disp*100:.0f}cm")
    ax.axis("off")

    # D: aperture + phase
    ax = axes[3]
    t = np.arange(len(gq))
    ax.plot(t, gq[:, 0], label="finger q1")
    ax.plot(t, gq[:, 1], label="finger q2")
    ax.fill_between(t, *ax.get_ylim(), where=phase == 1, alpha=0.15, color="orange",
                    label="transport (latched)")
    ax.axvline(tg, color="red", ls="--", lw=1.5, label=f"t_g={tg}")
    ax.set_title("D · aperture w(t) + phase")
    ax.set_xlabel("frame")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--demo", default="demo_0")
    args = ap.parse_args()

    os.makedirs(VIZ_DIR, exist_ok=True)
    for tf in sorted(os.listdir(os.path.join(DATA_ROOT, args.suite))):
        if not tf.endswith(".h5"):
            continue
        out = os.path.join(VIZ_DIR, f"{tf[:-3]}_{args.demo}.png")
        viz_task(os.path.join(DATA_ROOT, args.suite, tf), args.demo, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
