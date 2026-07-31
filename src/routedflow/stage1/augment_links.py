"""Supplemental pass: add per-frame robot LINK poses to existing c_labels h5.

Fills the L3 robot-flow GT gap (plan §2.8): extract_c_labels stored only the EE
site trajectory; robot-flow labels need every robot body so surface points can
be transformed per frame. Pure state replay, NO rendering => fast (~10 min for
500 demos) and safe to run next to a GPU-heavy job.

Adds to each demo group (skips demos that already have them => resumable):
    link_pos  (T, L, 3) float32   world frame
    link_quat (T, L, 4) float32   xyzw
Task-level dataset `link_names` (json list, order matches L axis).
Link set = robot bodies that own >=1 geom (arm links + hand + fingers + mount).
Run via `run_stage1.py augment-links` (atm5090).
"""
import json
import os
import sys

os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/code/ATM/.libero")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
from natsort import natsorted

from routedflow.convert_libero_raw import build_env

RAW_ROOT = "/workspace/datasets/libero/hdf5"
DATA = os.path.join(REPO, "data", "c_labels")
ROBOT_BODY_PREFIXES = ("robot0", "gripper0", "mount0")


def robot_link_bodies(sim):
    m = sim.model
    ids = []
    for bid in range(m.nbody):
        name = m.body_id2name(bid)
        if not (name and name.startswith(ROBOT_BODY_PREFIXES)):
            continue
        if not np.any(np.asarray(m.geom_bodyid) == bid):
            continue  # bodies without geoms have no surface points to track
        ids.append(bid)
    assert ids, "no robot link bodies found"
    return ids, [m.body_id2name(b) for b in ids]


def main(suite="libero_spatial"):
    suite_dir = os.path.join(RAW_ROOT, suite)
    for tf in natsorted(f for f in os.listdir(suite_dir) if f.endswith(".hdf5")):
        name = tf.split(".")[0]
        lab_path = os.path.join(DATA, suite, f"{name}.h5")
        if not os.path.exists(lab_path):
            print(f"skip {name}: no c_labels file")
            continue

        env, sim = build_env(suite, name, 128)  # small render size; we never render
        bids, names = robot_link_bodies(sim)

        with h5py.File(os.path.join(suite_dir, tf), "r") as raw, h5py.File(lab_path, "r+") as lab:
            if "link_names" not in lab:
                lab.create_dataset("link_names", data=json.dumps(names))
            n_new = 0
            for k in natsorted(list(raw["data"].keys())):
                if k not in lab or "link_pos" in lab[k]:
                    continue
                states = np.array(raw["data"][k]["states"])
                T = int(lab[k].attrs["T"])
                assert states.shape[0] == T, f"{name}/{k}: states {states.shape[0]} != T {T}"
                lp = np.zeros((T, len(bids), 3), np.float32)
                lq = np.zeros((T, len(bids), 4), np.float32)
                for t in range(T):
                    sim.set_state_from_flattened(states[t])
                    sim.forward()
                    for i, bid in enumerate(bids):
                        lp[t, i] = sim.data.body_xpos[bid]
                        lq[t, i] = sim.data.body_xquat[bid][[1, 2, 3, 0]]  # wxyz -> xyzw
                lab[k].create_dataset("link_pos", data=lp)
                lab[k].create_dataset("link_quat", data=lq)
                n_new += 1
        env.close()
        print(f"[{name}] links={len(bids)} augmented {n_new} demos", flush=True)
    print("LINK AUGMENT DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "libero_spatial")
