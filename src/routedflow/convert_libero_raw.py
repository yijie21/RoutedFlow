"""Convert raw LIBERO hdf5 demos to ATM-format per-demo h5 + RoutedFlow extras.

Skips CoTracker entirely (BC training never reads GT tracks — zero dummies are
stored to keep ATM's dataloader machinery untouched). On top of the ATM format
(`third_party/ATM/scripts/preprocess_libero.py`) each demo h5 gains:

    root/phase:              (T,) uint8    latched contact phase (routedflow.phase)
    root/<view>/grid_labels: (T, 32) uint8 robot membership of the fixed query grid
    root/<view>/robot_seg:   (T, H, W) uint8 full robot mask

Robot masks come from a zero-drift state replay: for every frame we restore the
recorded mujoco state (`states` in the raw h5) and render a segmentation image,
mapping geoms to the robot via body-name ancestry (robot0*/gripper0*/mount0*).

Orientation handling: raw LIBERO images are stored upside-down; ATM's preprocess
flips them (`rgb[:, ::-1]`). We must give masks the SAME final orientation as the
stored video. Instead of trusting conventions, we auto-detect per camera: render
RGB through the same path as the segmentation, compare both flip options against
the final (flipped) stored frame, pick the better one, and apply that choice to
the segmentation. The residual RGB difference doubles as a replay-fidelity check.

Run via `run_stage0.py prep` (sets MUJOCO_GL / LIBERO_CONFIG_PATH / PYTHONPATH).
"""
import argparse
import json
import os
import sys

# Must be set before any libero import
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

from routedflow.grid import grid_pixel_indices, grid_robot_labels
from routedflow.phase import latched_phase

EXTRA_STATES_KEYS = ["gripper_states", "joint_states", "ee_ori", "ee_pos", "ee_states"]
VIEW_TO_CAMERA = {"agentview": "agentview", "eye_in_hand": "robot0_eye_in_hand"}
ROBOT_BODY_PREFIXES = ("robot0", "gripper0", "mount0")

RAW_ROOT = "/workspace/datasets/libero/hdf5"
OUT_ROOT = os.path.join(REPO_ROOT, "data", "atm_libero_gated")
TASK_EMB_CACHE = os.path.join(ATM_ROOT, "libero", "task_embedding_caches", "task_emb_bert.npy")


def get_task_name_from_file_name(file_name):
    """Copied from third_party/ATM/scripts/preprocess_libero.py (keep in sync)."""
    name = file_name.replace("_demo", "")
    if name[0].isupper():  # LIBERO-10 and LIBERO-90
        if "SCENE10" in name:
            language = " ".join(name[name.find("SCENE") + 8:].split("_"))
        else:
            language = " ".join(name[name.find("SCENE") + 7:].split("_"))
    else:
        language = " ".join(name.split("_"))
    return language


def build_env(suite, task_file_name, img_hw):
    from libero.envs import OffScreenRenderEnv

    bddl_root = "/workspace/code/ATM/libero/bddl_files"
    task_name = task_file_name.replace("_demo", "")
    bddl = os.path.join(bddl_root, suite, f"{task_name}.bddl")
    assert os.path.exists(bddl), f"bddl not found: {bddl}"
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=img_hw, camera_widths=img_hw)
    env.reset()
    sim = getattr(env, "sim", None)
    if sim is None:
        sim = env.env.sim
    return env, sim


def robot_geom_id_set(sim):
    """Geom ids whose body-name ancestry starts with a robot prefix."""
    model = sim.model
    robot_bodies = set()
    for bid in range(model.nbody):
        name = model.body_id2name(bid)
        if name and name.startswith(ROBOT_BODY_PREFIXES):
            robot_bodies.add(bid)
    ids = [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in robot_bodies]
    assert len(ids) > 0, "no robot geoms found — body naming assumption broken"
    return np.array(ids)


def render_seg_and_rgb(sim, camera, hw):
    from robosuite.utils.camera_utils import get_camera_segmentation

    # NOTE (2026-07-30): get_camera_segmentation returns UPRIGHT (it applies [::-1]
    # internally), while raw sim.render returns upside-down. The two are NOT in the
    # same orientation — never apply the rgb flip choice to the segmentation.
    seg = get_camera_segmentation(sim, camera, hw, hw)  # (H, W, 2); last channel = geom ids
    geom_ids = seg[..., -1]
    rgb = sim.render(camera_name=camera, height=hw, width=hw)  # raw path: upside-down
    return geom_ids, rgb


def pick_flip(rendered_rgb, target_rgb):
    """Return (flip_needed, diff_chosen, diff_other): whether flipping rendered
    vertically matches the target frame better."""
    d_plain = np.abs(rendered_rgb.astype(np.int16) - target_rgb.astype(np.int16)).mean()
    d_flip = np.abs(rendered_rgb[::-1].astype(np.int16) - target_rgb.astype(np.int16)).mean()
    return (d_flip < d_plain), float(min(d_flip, d_plain)), float(max(d_flip, d_plain))


def convert_demo(demo_grp, out_path, task_emb, env_sim, robot_ids, hw, diag):
    actions = np.array(demo_grp["actions"])
    states = np.array(demo_grp["states"])
    T = actions.shape[0]

    videos = {}
    for view, _ in VIEW_TO_CAMERA.items():
        raw = np.array(demo_grp["obs"][f"{view}_rgb"])  # (T, H, W, 3) upside-down
        flipped = raw[:, ::-1, :, :].copy()  # ATM preprocess convention
        videos[view] = np.transpose(flipped, (0, 3, 1, 2))  # (T, C, H, W)

    # --- replay & render segmentation ---
    # Bug fixed 2026-07-30: the rgb flip choice used to be applied to the mask too,
    # but the segmentation is ALREADY upright (unlike the raw rgb render) — applying
    # the flip inverted every mask. Verified: IoU(stored, upright)≈0 pre-fix, ≈0.95
    # after. pick_flip stays as a replay-fidelity diagnostic only.
    robot_seg = {view: np.zeros((T, hw, hw), dtype=np.uint8) for view in VIEW_TO_CAMERA}
    for t in range(T):
        env_sim.set_state_from_flattened(states[t])
        env_sim.forward()
        for view, camera in VIEW_TO_CAMERA.items():
            geom_ids, rgb = render_seg_and_rgb(env_sim, camera, hw)
            mask = np.isin(geom_ids, robot_ids).astype(np.uint8)
            if t == 0:
                target = np.transpose(videos[view][0], (1, 2, 0))  # final orientation
                _, d_min, d_max = pick_flip(rgb, target)
                diag.setdefault("replay_rgb_diff", {}).setdefault(view, []).append(d_min)
                diag.setdefault("flip_margin", {}).setdefault(view, []).append(d_max - d_min)
            robot_seg[view][t] = mask

    phase = latched_phase(actions)

    with h5py.File(out_path, "w") as f:
        root = f.create_group("root")
        root.create_dataset("actions", data=actions)
        root.create_dataset("task_emb_bert", data=task_emb)
        root.create_dataset("phase", data=phase)
        es = root.create_group("extra_states")
        for k in EXTRA_STATES_KEYS:
            es.create_dataset(k, data=np.array(demo_grp["obs"][k]))
        for view in VIEW_TO_CAMERA:
            vg = root.create_group(view)
            vg.create_dataset("video", data=videos[view][None].astype(np.uint8))
            # zero dummies: BC training never reads GT tracks (vilt.py forward docstring)
            vg.create_dataset("tracks", data=np.zeros((1, T, 32, 2), dtype=np.float32))
            vg.create_dataset("vis", data=np.ones((1, T, 32), dtype=np.float32))
            vg.create_dataset("robot_seg", data=robot_seg[view])
            vg.create_dataset("grid_labels", data=grid_robot_labels(robot_seg[view]).astype(np.uint8))

    return {view: robot_seg[view].mean() for view in VIEW_TO_CAMERA}


def save_overlay(out_dir, tag, video_frame, seg_frame, labels):
    """Sanity-check PNG: grid points (green=object, red=robot) over RGB and mask."""
    from PIL import Image

    img = np.transpose(video_frame, (1, 2, 0)).astype(np.uint8).copy()
    segv = np.stack([seg_frame * 255] * 3, axis=-1).astype(np.uint8)
    h, w = img.shape[:2]
    rows, cols = grid_pixel_indices(h, w)
    for canvas in (img, segv):
        for i, (r, c) in enumerate(zip(rows, cols)):
            color = (255, 40, 40) if labels[i] else (40, 255, 40)
            r0, r1 = max(r - 2, 0), min(r + 3, h)
            c0, c1 = max(c - 2, 0), min(c + 3, w)
            canvas[r0:r1, c0:c1] = color
    combo = np.concatenate([img, segv], axis=1)
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(combo).save(os.path.join(out_dir, f"{tag}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--train-demos", type=int, default=10, help="first N demos -> bc_train_10")
    ap.add_argument("--val-demos", type=int, default=5, help="last N demos -> val")
    ap.add_argument("--tasks-limit", type=int, default=None, help="only convert first N tasks (debug)")
    ap.add_argument("--overlay-dir", default=os.path.join(REPO_ROOT, "experiments", "stage0_routing_causal_test", "sanity_overlays"))
    args = ap.parse_args()

    hw = 128
    task_emb_map = np.load(TASK_EMB_CACHE, allow_pickle=True).item()

    suite_dir = os.path.join(RAW_ROOT, args.suite)
    task_files = natsorted(os.listdir(suite_dir))
    if args.tasks_limit:
        task_files = task_files[: args.tasks_limit]

    summary = {}
    for task_file in task_files:
        task_file_name = task_file.split(".")[0]
        task_language = get_task_name_from_file_name(task_file_name)
        task_emb = task_emb_map[task_language]

        out_task_dir = os.path.join(OUT_ROOT, args.suite, task_file_name)
        env, sim = build_env(args.suite, task_file_name, hw)
        robot_ids = robot_geom_id_set(sim)
        diag = {}

        with h5py.File(os.path.join(suite_dir, task_file), "r") as raw:
            demos = raw["data"]
            keys = natsorted(list(demos.keys()))
            picks = [(k, "bc_train_10") for k in keys[: args.train_demos]] + \
                    [(k, "val") for k in keys[-args.val_demos:]]
            for demo_k, split in picks:
                split_dir = os.path.join(out_task_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                out_path = os.path.join(split_dir, f"{demo_k}.hdf5")
                if os.path.exists(out_path):
                    print(f"skip existing {out_path}")
                    continue
                fracs = convert_demo(demos[demo_k], out_path, task_emb, sim, robot_ids, hw, diag)
                print(f"{task_file_name}/{split}/{demo_k}: robot-pixel fraction {fracs}")

            # env_meta for later eval (same content ATM's preprocess writes)
            attrs = json.loads(demos.attrs["env_args"])
            with open(os.path.join(out_task_dir, "env_meta.json"), "w") as fp:
                json.dump(attrs, fp)

            # overlay sanity images from the first train demo
            first = keys[0]
            with h5py.File(os.path.join(out_task_dir, "bc_train_10", f"{first}.hdf5"), "r") as f:
                for view in VIEW_TO_CAMERA:
                    vid = np.array(f[f"root/{view}/video"])[0]
                    seg = np.array(f[f"root/{view}/robot_seg"])
                    gl = np.array(f[f"root/{view}/grid_labels"])
                    for t in (0, vid.shape[0] // 2):
                        save_overlay(args.overlay_dir, f"{task_file_name}_{view}_t{t}", vid[t], seg[t], gl[t])

        env.close()
        summary[task_file_name] = {
            "replay_rgb_diff": {v: float(np.mean(d)) for v, d in diag.get("replay_rgb_diff", {}).items()},
            "flip_margin": {v: float(np.mean(d)) for v, d in diag.get("flip_margin", {}).items()},
        }
        print(f"[{task_file_name}] replay diagnostics: {summary[task_file_name]}")

    with open(os.path.join(OUT_ROOT, args.suite, "conversion_diagnostics.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
