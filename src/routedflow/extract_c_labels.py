"""Unified label extraction for Stage 1: one zero-drift replay pass per demo.

For every raw LIBERO demo this records everything needed to derive ALL v1
supervision labels offline (C labels now; robot/object flow labels later by
pure projection of the stored pose trajectories — no further sim work):

    per task h5 (data/c_labels/<suite>/<task>.h5):
        attrs: task_language, camera, res, K (3x3), world_to_pix (4x4),
               extrinsic (4x4), robot_base_pos/quat, skipped_demos
        geom_id_to_body: json string dataset (int geom id -> body name)
        robot_geom_ids:  (G,) int32
        demo_<k>/
            attrs: t_g, T, grasped_body, grasp_dist_m, lift_gap_m,
                   replay_ee_maxdiff_m
            rgb0        (R,R,3) uint8   upright t=0 agentview RGB
            depth0      (R,R)  float32  metric depth (m), upright
            seg0        (R,R)  int32    geom ids, upright
            contact_rowcol (2,) float64 E(t_g) position projected into t=0 frame
            contact_depth0 ()   float64 depth0 at the contact pixel
            ee_pos      (T,3) float32   gripper site, world frame (replayed)
            ee_quat     (T,4) float32   gripper site orientation (xyzw)
            obj_pos     (T,n,3) float32 free-joint object bodies, world frame
            obj_quat    (T,n,4) float32
            obj_names   json string dataset
            gripper_q   (T,2) float32   finger joint positions (aperture)
            phase       (T,)  uint8     latched contact phase

C label consumption (L1/L2 training): input = rgb0 + task_language; targets =
contact_rowcol (heatmap), ee_quat[t_g] (orientation bins after frame transform),
gripper_q[t_g] (aperture). lift_gap_m quantifies the systematic error of the
inference-time translation-by-depth-lift against the true E(t_g) position.

Projection convention verified 2026-07-30 (AFUN spike gt_overlay_both.png):
upright image = render[::-1]; robosuite's world_to_pix already yields rows in
the upright convention — no extra flip.

Run via `run_stage1.py extract` (sets MUJOCO_GL / LIBERO_CONFIG_PATH / PYTHONPATH).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/code/ATM/.libero")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ATM_ROOT = os.path.join(REPO_ROOT, "third_party", "ATM")
for p in (os.path.join(REPO_ROOT, "src"), ATM_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
from natsort import natsorted

from routedflow.convert_libero_raw import build_env, get_task_name_from_file_name, robot_geom_id_set
from routedflow.phase import latched_phase

RAW_ROOT = "/workspace/datasets/libero/hdf5"
OUT_ROOT = os.path.join(REPO_ROOT, "data", "c_labels")
RES = 512
CAM = "agentview"
ROBOT_BODY_PREFIXES = ("robot0", "gripper0", "mount0")
MJ_JNT_FREE = 0


def find_grip_site(sim):
    names = [sim.model.site_id2name(i) for i in range(sim.model.nsite)]
    for cand in ("gripper0_grip_site",):
        if cand in names:
            return sim.model.site_name2id(cand)
    hits = [n for n in names if n and "grip_site" in n]
    assert hits, f"no grip site found among {names}"
    return sim.model.site_name2id(hits[0])


def free_object_bodies(sim):
    """Bodies with a free joint that are not part of the robot (movable objects)."""
    m = sim.model
    ids = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != MJ_JNT_FREE:
            continue
        bid = int(m.jnt_bodyid[j])
        name = m.body_id2name(bid)
        if name and not name.startswith(ROBOT_BODY_PREFIXES):
            ids.append(bid)
    assert ids, "no free-joint object bodies found"
    return ids, [m.body_id2name(b) for b in ids]


def render_t0(sim):
    from robosuite.utils.camera_utils import get_camera_segmentation, get_real_depth_map

    rgb = sim.render(camera_name=CAM, height=RES, width=RES)[::-1]
    zbuf = sim.render(camera_name=CAM, height=RES, width=RES, depth=True)[1][::-1]
    depth_m = np.asarray(get_real_depth_map(sim, zbuf)).squeeze()
    # get_camera_segmentation already returns UPRIGHT (it applies [::-1] internally,
    # unlike raw sim.render) — verified empirically 2026-07-30, do NOT flip again.
    seg = get_camera_segmentation(sim, CAM, RES, RES)[..., -1].astype(np.int32)
    return rgb, depth_m.astype(np.float32), seg


def extract_demo(sim, demo_grp, site_id, obj_bids, world_to_pix, K, extrinsic):
    from robosuite.utils.transform_utils import mat2quat

    actions = np.array(demo_grp["actions"])
    states = np.array(demo_grp["states"])
    obs_ee_pos = np.array(demo_grp["obs"]["ee_pos"])
    gripper_q = np.array(demo_grp["obs"]["gripper_states"]).astype(np.float32)
    T = actions.shape[0]

    phase = latched_phase(actions)
    if not phase.any():
        return None  # demo never closes the gripper — cannot define t_g
    t_g = int(np.argmax(phase))

    ee_pos = np.zeros((T, 3), np.float32)
    ee_quat = np.zeros((T, 4), np.float32)
    obj_pos = np.zeros((T, len(obj_bids), 3), np.float32)
    obj_quat = np.zeros((T, len(obj_bids), 4), np.float32)
    rgb0 = depth0 = seg0 = None
    for t in range(T):
        sim.set_state_from_flattened(states[t])
        sim.forward()
        if t == 0:
            rgb0, depth0, seg0 = render_t0(sim)
        ee_pos[t] = sim.data.site_xpos[site_id]
        ee_quat[t] = mat2quat(np.array(sim.data.site_xmat[site_id]).reshape(3, 3))
        for i, bid in enumerate(obj_bids):
            obj_pos[t, i] = sim.data.body_xpos[bid]
            obj_quat[t, i] = sim.data.body_xquat[bid][[1, 2, 3, 0]]  # wxyz -> xyzw

    # contact pixel: E(t_g) position into the (static) t=0 camera
    uvw = world_to_pix @ np.append(ee_pos[t_g], 1.0)
    col, row = uvw[0] / uvw[2], uvw[1] / uvw[2]
    r_i, c_i = int(round(row)), int(round(col))
    in_view = 0 <= r_i < RES and 0 <= c_i < RES
    contact_depth0 = float(depth0[r_i, c_i]) if in_view else float("nan")

    # lift-gap QC: reconstruct 3D from (pixel, t=0 depth) and compare with true E(t_g)
    lift_gap = float("nan")
    if in_view:
        x_cam_lift = contact_depth0 * (np.linalg.inv(K) @ np.array([col, row, 1.0]))
        world_to_cam = np.linalg.inv(extrinsic)
        x_cam_true = (world_to_cam @ np.append(ee_pos[t_g], 1.0))[:3]
        lift_gap = float(np.linalg.norm(x_cam_lift - x_cam_true))

    # grasped object: nearest free body at t_g
    dists = np.linalg.norm(obj_pos[t_g] - ee_pos[t_g], axis=-1)
    grasped_i = int(np.argmin(dists))

    return {
        "t_g": t_g, "T": T, "grasped_i": grasped_i,
        "grasp_dist_m": float(dists[grasped_i]),
        "lift_gap_m": lift_gap,
        "replay_ee_maxdiff_m": float(np.abs(ee_pos - obs_ee_pos).max()),
        "rgb0": rgb0, "depth0": depth0, "seg0": seg0,
        "contact_rowcol": np.array([row, col]), "contact_depth0": contact_depth0,
        "ee_pos": ee_pos, "ee_quat": ee_quat,
        "obj_pos": obj_pos, "obj_quat": obj_quat,
        "gripper_q": gripper_q, "phase": phase.astype(np.uint8),
    }


def write_demo(task_f, key, d, obj_names):
    g = task_f.create_group(key)
    for k in ("t_g", "T", "grasp_dist_m", "lift_gap_m", "replay_ee_maxdiff_m"):
        g.attrs[k] = d[k]
    g.attrs["grasped_body"] = obj_names[d["grasped_i"]]
    g.create_dataset("rgb0", data=d["rgb0"], compression="gzip")
    g.create_dataset("depth0", data=d["depth0"], compression="gzip")
    g.create_dataset("seg0", data=d["seg0"], compression="gzip")
    for k in ("contact_rowcol", "contact_depth0", "ee_pos", "ee_quat",
              "obj_pos", "obj_quat", "gripper_q", "phase"):
        g.create_dataset(k, data=d[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks-limit", type=int, default=None)
    ap.add_argument("--demos-limit", type=int, default=None)
    args = ap.parse_args()

    from robosuite.utils.camera_utils import (get_camera_extrinsic_matrix,
                                              get_camera_intrinsic_matrix,
                                              get_camera_transform_matrix)

    suite_dir = os.path.join(RAW_ROOT, args.suite)
    out_dir = os.path.join(OUT_ROOT, args.suite)
    os.makedirs(out_dir, exist_ok=True)

    task_files = natsorted(f for f in os.listdir(suite_dir) if f.endswith(".hdf5"))
    if args.tasks_limit:
        task_files = task_files[: args.tasks_limit]

    for tf in task_files:
        name = tf.split(".")[0]
        out_path = os.path.join(out_dir, f"{name}.h5")
        if os.path.exists(out_path):
            print(f"skip existing {out_path}")
            continue

        env, sim = build_env(args.suite, name, RES)
        site_id = find_grip_site(sim)
        obj_bids, obj_names = free_object_bodies(sim)
        robot_ids = robot_geom_id_set(sim)
        K = get_camera_intrinsic_matrix(sim, CAM, RES, RES)
        world_to_pix = get_camera_transform_matrix(sim, CAM, RES, RES)
        extrinsic = get_camera_extrinsic_matrix(sim, CAM)
        base_bid = sim.model.body_name2id("robot0_base")
        geom_map = {int(g): sim.model.body_id2name(int(sim.model.geom_bodyid[g]))
                    for g in range(sim.model.ngeom)}

        skipped = []
        with h5py.File(os.path.join(suite_dir, tf), "r") as raw, \
                h5py.File(out_path + ".tmp", "w") as out:
            out.attrs.update({
                "task_language": get_task_name_from_file_name(name),
                "camera": CAM, "res": RES,
                "robot_base_pos": np.array(sim.data.body_xpos[base_bid]),
                "robot_base_quat": np.array(sim.data.body_xquat[base_bid])[[1, 2, 3, 0]],
            })
            for k, v in (("K", K), ("world_to_pix", world_to_pix), ("extrinsic", extrinsic)):
                out.create_dataset(k, data=v)
            out.create_dataset("geom_id_to_body", data=json.dumps(geom_map))
            out.create_dataset("robot_geom_ids", data=robot_ids.astype(np.int32))
            out.create_dataset("obj_names", data=json.dumps(obj_names))

            keys = natsorted(list(raw["data"].keys()))
            if args.demos_limit:
                keys = keys[: args.demos_limit]
            for key in keys:
                d = extract_demo(sim, raw["data"][key], site_id, obj_bids,
                                 world_to_pix, K, extrinsic)
                if d is None:
                    skipped.append(key)
                    continue
                write_demo(out, key, d, obj_names)
            out.attrs["skipped_demos"] = json.dumps(skipped)

        os.replace(out_path + ".tmp", out_path)
        env.close()
        print(f"[{name}] demos={len(keys) - len(skipped)} skipped={skipped} objs={obj_names}")

    print("extract done.")


if __name__ == "__main__":
    main()
