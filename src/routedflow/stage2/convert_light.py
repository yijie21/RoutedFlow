"""Light LIBERO -> ATM-format conversion: NO replay, NO segmentation, pure IO.

The FK chain-point decision (D11) removed the need for per-frame robot masks, so
stage-2 frame data only needs: flipped obs videos (both views), actions,
extra_states, phase, and zero-dummy tracks (BC never reads GT tracks). This
makes converting ALL 50 demos per task a minutes-scale job (the stage-0
converter spent its time on replay+seg rendering).

Output: data/atm_libero_light/<suite>/<task>/all/demo_X.hdf5  (+ env_meta.json)
Run via `run_stage2.py convert` (atm5090; no GPU needed).
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "third_party", "ATM")):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
from natsort import natsorted

from routedflow.convert_libero_raw import (EXTRA_STATES_KEYS, VIEW_TO_CAMERA,
                                           get_task_name_from_file_name)
from routedflow.phase import latched_phase

RAW_ROOT = "/workspace/datasets/libero/hdf5"
OUT_ROOT = os.path.join(REPO_ROOT, "data", "atm_libero_light")
TASK_EMB_CACHE = os.path.join(REPO_ROOT, "third_party", "ATM", "libero",
                              "task_embedding_caches", "task_emb_bert.npy")


def convert_demo(demo_grp, out_path, task_emb):
    actions = np.array(demo_grp["actions"])
    T = actions.shape[0]
    with h5py.File(out_path, "w") as f:
        root = f.create_group("root")
        root.create_dataset("actions", data=actions)
        root.create_dataset("task_emb_bert", data=task_emb)
        root.create_dataset("phase", data=latched_phase(actions).astype(np.uint8))
        es = root.create_group("extra_states")
        for k in EXTRA_STATES_KEYS:
            es.create_dataset(k, data=np.array(demo_grp["obs"][k]))
        for view in VIEW_TO_CAMERA:
            raw = np.array(demo_grp["obs"][f"{view}_rgb"])          # (T,H,W,3) upside-down
            vid = np.transpose(raw[:, ::-1], (0, 3, 1, 2)).copy()   # (T,C,H,W) upright
            vg = root.create_group(view)
            vg.create_dataset("video", data=vid[None].astype(np.uint8))
            vg.create_dataset("tracks", data=np.zeros((1, T, 32, 2), np.float32))
            vg.create_dataset("vis", data=np.ones((1, T, 32), np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    args = ap.parse_args()

    emb_map = np.load(TASK_EMB_CACHE, allow_pickle=True).item()
    suite_dir = os.path.join(RAW_ROOT, args.suite)
    for tf in natsorted(f for f in os.listdir(suite_dir) if f.endswith(".hdf5")):
        name = tf.split(".")[0]
        emb = emb_map[get_task_name_from_file_name(name)]
        out_dir = os.path.join(OUT_ROOT, args.suite, name, "all")
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        with h5py.File(os.path.join(suite_dir, tf), "r") as raw:
            for k in natsorted(list(raw["data"].keys())):
                out_path = os.path.join(out_dir, f"{k}.hdf5")
                if not os.path.exists(out_path):
                    convert_demo(raw["data"][k], out_path, emb)
                    n += 1
            attrs = json.loads(raw["data"].attrs["env_args"])
            with open(os.path.join(os.path.dirname(out_dir), "env_meta.json"), "w") as fp:
                json.dump(attrs, fp)
        print(f"[{name}] +{n} demos", flush=True)
    print("LIGHT CONVERT DONE", flush=True)


if __name__ == "__main__":
    main()
