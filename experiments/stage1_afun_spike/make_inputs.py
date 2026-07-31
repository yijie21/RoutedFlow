"""Render high-res LIBERO spike inputs for AFUN.

For each of the first N libero_spatial tasks: restore demo_0's t=0 mujoco state,
render agentview at 512x512 (RGB + metric depth + intrinsics), and record the
GT grasp proxy = the demo's EE position at t_g projected into the same camera.

Outputs per task under inputs/<task>/:
    rgb.png          upright 512x512 RGB (fed to AFUN --rgb)
    depth/depth.npy  float32 depth in MILLIMETERS (AFUN --depth-dir format)
    depth/cam_K.txt  3x3 intrinsics
    gt.json          {query, tg, gt_pixel_rowcol}   (512-res, upright convention)
    gt_overlay.png   rgb with the projected grasp point marked (eyeball check)

Run with the atm5090 interpreter via env vars set as in run_stage0.py.
"""
import json
import os
import sys

os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/code/ATM/.libero")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO = "/workspace/code/RoutedFlow"
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")]

import h5py
import numpy as np
from PIL import Image

from routedflow.convert_libero_raw import build_env, get_task_name_from_file_name
from routedflow.phase import latched_phase

RAW = "/workspace/datasets/libero/hdf5/libero_spatial"
OUT = os.path.join(REPO, "experiments", "stage1_afun_spike", "inputs")
RES = 512
CAM = "agentview"


def main(n_tasks=5):
    from robosuite.utils.camera_utils import (get_camera_intrinsic_matrix,
                                              get_camera_transform_matrix, get_real_depth_map)

    tasks = sorted(f for f in os.listdir(RAW) if f.endswith(".hdf5"))[:n_tasks]
    for tf in tasks:
        name = tf.split(".")[0]
        query = get_task_name_from_file_name(name)
        out_dir = os.path.join(OUT, name)
        os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)

        with h5py.File(os.path.join(RAW, tf), "r") as f:
            demo = f["data/demo_0"]
            states = np.array(demo["states"])
            actions = np.array(demo["actions"])
            ee_pos = np.array(demo["obs"]["ee_pos"])  # world frame

        env, sim = build_env("libero_spatial", name, RES)
        sim.set_state_from_flattened(states[0])
        sim.forward()

        rgb = sim.render(camera_name=CAM, height=RES, width=RES)[::-1]  # upright
        zbuf = sim.render(camera_name=CAM, height=RES, width=RES, depth=True)[1][::-1]
        depth_m = get_real_depth_map(sim, zbuf)  # meters
        K = get_camera_intrinsic_matrix(sim, CAM, RES, RES)

        tg = int(np.argmax(latched_phase(actions)))
        world_to_pix = get_camera_transform_matrix(sim, CAM, RES, RES)
        p = np.append(ee_pos[tg], 1.0)
        uvw = world_to_pix @ p
        col, row = uvw[0] / uvw[2], uvw[1] / uvw[2]
        # verified 2026-07-30 (gt_overlay_both.png): robosuite's transform already
        # yields rows in the upright convention matching rgb[::-1] — no extra flip.

        Image.fromarray(rgb).save(os.path.join(out_dir, "rgb.png"))
        np.save(os.path.join(out_dir, "depth", "depth.npy"),
                (depth_m * 1000.0).astype(np.float32).squeeze())
        np.savetxt(os.path.join(out_dir, "depth", "cam_K.txt"), K)
        json.dump({"query": query, "tg": tg, "gt_pixel_rowcol": [float(row), float(col)]},
                  open(os.path.join(out_dir, "gt.json"), "w"), indent=2)

        vis = rgb.copy()
        r, c = int(round(row)), int(round(col))
        if 0 <= r < RES and 0 <= c < RES:
            vis[max(r-4, 0):r+5, max(c-4, 0):c+5] = (255, 0, 0)
        Image.fromarray(vis).save(os.path.join(out_dir, "gt_overlay.png"))

        env.close()
        print(f"[{name}] tg={tg} gt_px=({row:.0f},{col:.0f}) depth_range="
              f"{depth_m.min():.2f}-{depth_m.max():.2f}m query={query!r}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
