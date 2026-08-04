"""Hindsight C-label frames: K extra agentview renders per demo, before contact.

Rationale (2026-08-04, user decision): C is a FUTURE event of the episode —
where the gripper will close is fixed the moment the demo is recorded — so every
frame t before contact is a valid input for "(frame_t, instruction) -> C".
Supervision density goes from 1 pair/episode to ~K pairs/episode while the label
stays untouched; the network is forced to learn that C depends on scene+language,
NOT on the current arm configuration (the invariance we want z to carry).

Leak guard: frames are sampled uniformly over [0, t_first_close - guard]. The
bound is the FIRST debounced closure (not attrs t_g = last closure): in fumble
demos frames between first and last closure show the gripper interacting with
the object — near-contact frames let the net shortcut "predict just ahead of the
gripper" (same leak family as the goalmix arm-motion leak, 2026-08-04).

Output: data/stage1_cache/<suite>/hindsight_frames.h5
    attrs: k_frames, guard, res
    <task>/<demo>/rgb (K,512,512,3) uint8 gzip   upright agentview renders
    <task>/<demo>/ts  (K,) int32                 source frame indices

Camera is static, so contact_rowcol / orientation / w labels (all t_g
quantities) remain valid for every stored frame. Consumed by
Stage1Dataset(hindsight=True). Run via `run_stage1.py extract-hindsight`.
"""
import argparse
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

from routedflow.convert_libero_raw import build_env

RAW_ROOT = "/workspace/datasets/libero/hdf5"
C_LABELS = os.path.join(REPO_ROOT, "data", "c_labels")
CACHE = os.path.join(REPO_ROOT, "data", "stage1_cache")
RES = 512
CAM = "agentview"


def hindsight_ts(t_first_close, k, guard):
    """Frame indices to store: k unique ints uniform over [0, t_first_close-guard]."""
    hi = max(0, int(t_first_close) - int(guard))
    return np.unique(np.linspace(0, hi, k).round().astype(np.int64))


def first_close_t(label_grp):
    """FIRST debounced closure (stage-2 window-bound semantics), not attrs t_g."""
    ph = np.asarray(label_grp["phase"])
    return int(np.argmax(ph)) if ph.any() else int(label_grp.attrs["t_g"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--k-frames", type=int, default=8)
    ap.add_argument("--guard", type=int, default=10)
    ap.add_argument("--tasks-limit", type=int, default=None)
    ap.add_argument("--demos-limit", type=int, default=None)
    args = ap.parse_args()

    label_dir = os.path.join(C_LABELS, args.suite)
    tasks = natsorted(f[:-3] for f in os.listdir(label_dir) if f.endswith(".h5"))
    if args.tasks_limit:
        tasks = tasks[: args.tasks_limit]

    os.makedirs(os.path.join(CACHE, args.suite), exist_ok=True)
    out_path = os.path.join(CACHE, args.suite, "hindsight_frames.h5")
    with h5py.File(out_path, "a") as out:
        for key, val in (("k_frames", args.k_frames), ("guard", args.guard), ("res", RES)):
            if key in out.attrs:
                assert int(out.attrs[key]) == val, \
                    f"{out_path} was built with {key}={out.attrs[key]}, got {val}; " \
                    f"delete the file to rebuild with new params"
            else:
                out.attrs[key] = val

        for task in tasks:
            raw_path = os.path.join(RAW_ROOT, args.suite, f"{task}.hdf5")
            env = sim = None
            with h5py.File(os.path.join(label_dir, f"{task}.h5"), "r") as lab, \
                    h5py.File(raw_path, "r") as raw:
                demos = natsorted(k for k in lab.keys() if k.startswith("demo"))
                if args.demos_limit:
                    demos = demos[: args.demos_limit]
                done = 0
                for demo in demos:
                    if task in out and demo in out[task]:
                        continue  # resume-safe
                    if sim is None:  # lazy: fully-extracted tasks never build an env
                        env, sim = build_env(args.suite, task, RES)
                    ts = hindsight_ts(first_close_t(lab[demo]), args.k_frames, args.guard)
                    states = np.array(raw["data"][demo]["states"])
                    rgbs = np.zeros((len(ts), RES, RES, 3), np.uint8)
                    for j, t in enumerate(ts):
                        sim.set_state_from_flattened(states[t])
                        sim.forward()
                        rgbs[j] = sim.render(camera_name=CAM, height=RES, width=RES)[::-1]
                    g = out.require_group(task).create_group(demo)
                    g.create_dataset("rgb", data=rgbs, compression="gzip")
                    g.create_dataset("ts", data=ts.astype(np.int32))
                    done += 1
                if env is not None:
                    env.close()
            print(f"[{task}] new={done} total={len(out[task]) if task in out else 0}",
                  flush=True)
    print("hindsight extract done.", flush=True)


if __name__ == "__main__":
    main()
