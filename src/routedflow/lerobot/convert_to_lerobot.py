"""Convert RoutedFlow h5 data to a LeRobot-v2.1-layout dataset (per suite).

Layout mirrors the starVLA reference
(/workspace/code/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA/...):
    data/lerobot/<suite>/
        data/chunk-000/episode_XXXXXX.parquet     per-frame table
        videos/chunk-000/<key>/episode_XXXXXX.mp4 lossless libx264rgb qp0 (NOT av1)
        meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl,
              modality.json}

Video keys:
    observation.images.agentview      (128) from atm_libero_light (bitwise equal)
    observation.images.eye_in_hand    (128) from atm_libero_light (bitwise equal)
    observation.images.agentview_512  APPROACH SEGMENT ONLY, frames [0, t_first_close],
                                      re-rendered at 512 from raw states — feeds L1
                                      (rgb0 = frame 0) and hindsight sampling (any
                                      frame ≤ t_close - guard; guard applied by the
                                      reader). Documented deviation from LeRobot:
                                      this key is SHORTER than the episode; its
                                      length is `len_512` in episodes.jsonl.

Per-frame parquet columns (fixed_size_list float32 unless noted): observation.state
(8 = ee_pos+ee_ori+gripper), observation.extra.* (the 5 ATM groups), action (7),
phase (int64), routedflow.chain_uv (64 = 32x2 flat), routedflow.chain_z (32),
routedflow.tracks.<view> (64), routedflow.vis.<view> (32), timestamp, frame_index,
episode_index, index, task_index.

Per-episode extras in episodes.jsonl["routedflow"]: task_dir, demo, t_g (last
debounced closure, C-label rule v2), t_first_close (window-bound rule), n_cycles,
grasped_body, contact_rowcol@512, yaw/pitch bins+cont, w, len_512 — everything
Stage-1 labels need, so training no longer touches c_labels h5 at runtime.

NOT migrated (regenerable derived caches, stay as h5 sidecars): DINO feats,
AFUN prior, depth0/seg0, ee/obj pose trajectories (c_labels stays source of truth).

Resume-safe: per-episode sidecar json in meta/.episodes/; finished episodes are
skipped; meta files are (re)assembled from sidecars at the end of every run.
Run via `run_stage2.py convert-lerobot [--suite ...]` (needs sim for 512 render).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/code/ATM/.libero")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from natsort import natsorted

from routedflow.lerobot.video import FPS, encode_video
from routedflow.stage1.dataset import orientation_targets

RAW_ROOT = "/workspace/datasets/libero/hdf5"
LIGHT_ROOT = os.path.join(REPO, "data", "atm_libero_light")
C_LABELS = os.path.join(REPO, "data", "c_labels")
OUT_ROOT = os.path.join(REPO, "data", "lerobot")
RES512 = 512
CAM = "agentview"

VIDEO_KEYS = ("observation.images.agentview", "observation.images.eye_in_hand",
              "observation.images.agentview_512")
EXTRA_GROUPS = ("ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states")


def video_feature(h, codec="libx264rgb", pix_fmt="rgb24"):
    return {"dtype": "video", "shape": [h, h, 3], "names": ["height", "width", "rgb"],
            "info": {"video.height": h, "video.width": h, "video.codec": codec,
                     "video.pix_fmt": pix_fmt, "video.is_depth_map": False,
                     "video.fps": FPS, "video.channels": 3, "has_audio": False}}


def features_schema():
    f = {k: video_feature(512 if k.endswith("_512") else 128) for k in VIDEO_KEYS}
    f["observation.state"] = {"dtype": "float32", "shape": [8],
                              "names": ["x", "y", "z", "roll", "pitch", "yaw",
                                        "gripper_0", "gripper_1"]}
    dims = {"ee_ori": 3, "ee_pos": 3, "ee_states": 6, "gripper_states": 2, "joint_states": 7}
    for g, d in dims.items():
        f[f"observation.extra.{g}"] = {"dtype": "float32", "shape": [d], "names": None}
    f["action"] = {"dtype": "float32", "shape": [7], "names": ["dx", "dy", "dz",
                   "droll", "dpitch", "dyaw", "gripper"]}
    f["phase"] = {"dtype": "int64", "shape": [1], "names": None}
    f["routedflow.chain_uv"] = {"dtype": "float32", "shape": [64],
                                "names": ["32 chain points x (u,v), row-major"]}
    f["routedflow.chain_z"] = {"dtype": "float32", "shape": [32], "names": None}
    for v in ("agentview", "eye_in_hand"):
        f[f"routedflow.tracks.{v}"] = {"dtype": "float32", "shape": [64],
                                       "names": ["32 tracks x (x,y), row-major"]}
        f[f"routedflow.vis.{v}"] = {"dtype": "float32", "shape": [32], "names": None}
    for k in ("timestamp",):
        f[k] = {"dtype": "float32", "shape": [1], "names": None}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        f[k] = {"dtype": "int64", "shape": [1], "names": None}
    return f


def img_stats(frames_thwc_u8):
    """Per-channel stats over normalized frames, in the v2.1 nested [[[v]]] shape."""
    x = frames_thwc_u8.astype(np.float64) / 255.0
    return {
        "min": [[[float(x[..., c].min())]] for c in range(3)],
        "max": [[[float(x[..., c].max())]] for c in range(3)],
        "mean": [[[float(x[..., c].mean())]] for c in range(3)],
        "std": [[[float(x[..., c].std())]] for c in range(3)],
        "count": [int(len(x))],
    }


def vec_stats(arr):
    a = np.asarray(arr, np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "count": [int(len(a))]}


def first_close_t(phase, t_g):
    ph = np.asarray(phase)
    return int(np.argmax(ph)) if ph.any() else int(t_g)


def load_light(path):
    with h5py.File(path, "r") as f:
        return {
            "actions": np.asarray(f["root/actions"], np.float32),
            "task_emb": np.asarray(f["root/task_emb_bert"], np.float32),
            "phase": np.asarray(f["root/phase"]),
            "extra": {g: np.asarray(f[f"root/extra_states/{g}"], np.float32)
                      for g in EXTRA_GROUPS},
            "views": {v: {"video": np.asarray(f[f"root/{v}/video"])[0],       # (T,3,128,128) u8
                          "tracks": np.asarray(f[f"root/{v}/tracks"], np.float32)[0],
                          "vis": np.asarray(f[f"root/{v}/vis"], np.float32)[0]}
                      for v in ("agentview", "eye_in_hand")},
        }


def build_table(light, lab_demo, ep_idx, task_idx, global_index0):
    T = light["actions"].shape[0]
    uv = np.asarray(lab_demo["chain_uv"], np.float32)[:T]
    zz = np.asarray(lab_demo["chain_z"], np.float32)[:T]
    assert uv.shape == (T, 32, 2) and zz.shape == (T, 32), (uv.shape, zz.shape, T)
    ex = light["extra"]
    state = np.concatenate([ex["ee_pos"], ex["ee_ori"], ex["gripper_states"]], 1)

    def fl(a, d):
        return pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(a, np.float32).reshape(-1), pa.float32()), d)

    cols = {
        "observation.state": fl(state, 8),
        "action": fl(light["actions"], 7),
        "phase": pa.array(light["phase"].astype(np.int64)),
        "routedflow.chain_uv": fl(uv, 64),
        "routedflow.chain_z": fl(zz, 32),
        "timestamp": pa.array((np.arange(T) / FPS).astype(np.float32)),
        "frame_index": pa.array(np.arange(T, dtype=np.int64)),
        "episode_index": pa.array(np.full(T, ep_idx, np.int64)),
        "index": pa.array(np.arange(global_index0, global_index0 + T, dtype=np.int64)),
        "task_index": pa.array(np.full(T, task_idx, np.int64)),
    }
    for g in EXTRA_GROUPS:
        cols[f"observation.extra.{g}"] = fl(ex[g], ex[g].shape[1])
    for v in ("agentview", "eye_in_hand"):
        cols[f"routedflow.tracks.{v}"] = fl(light["views"][v]["tracks"], 64)
        cols[f"routedflow.vis.{v}"] = fl(light["views"][v]["vis"], 32)
    meta = {"huggingface": json.dumps({"info": {"features": {
        k: v for k, v in features_schema().items() if v["dtype"] != "video"}}})}
    return pa.table(cols).replace_schema_metadata(meta), state


def render_512(sim, states, t_hi):
    frames = np.zeros((t_hi + 1, RES512, RES512, 3), np.uint8)
    for t in range(t_hi + 1):
        sim.set_state_from_flattened(states[t])
        sim.forward()
        frames[t] = sim.render(camera_name=CAM, height=RES512, width=RES512)[::-1]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks-limit", type=int, default=None)
    ap.add_argument("--demos-limit", type=int, default=None)
    ap.add_argument("--no-render", action="store_true",
                    help="skip the 512 approach-segment video (no sim needed)")
    ap.add_argument("--labels-only", action="store_true",
                    help="suite has c_labels but NO atm_libero_light (e.g. libero_object, "
                         "which only feeds stage-1 C 开小灶): write agentview_512 video + "
                         "episode meta only — no parquet, no 128 videos. Enough for "
                         "Stage1Dataset hindsight; such a suite must not be passed to "
                         "stage-2 --extra-suites.")
    args = ap.parse_args()

    from routedflow.convert_libero_raw import build_env, get_task_name_from_file_name

    out = os.path.join(OUT_ROOT, args.suite)
    for d in ("data/chunk-000", "meta/.episodes",
              *[f"videos/chunk-000/{k}" for k in VIDEO_KEYS]):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    if args.labels_only:
        assert not args.no_render, "--labels-only without render produces nothing"
        task_dirs = sorted(f[:-3] for f in os.listdir(os.path.join(C_LABELS, args.suite))
                           if f.endswith(".h5"))
    else:
        task_dirs = sorted(os.listdir(os.path.join(LIGHT_ROOT, args.suite)))
    if args.tasks_limit:
        task_dirs = task_dirs[: args.tasks_limit]

    # deterministic episode enumeration: sorted task dirs x natsorted demo files,
    # keeping only demos present in c_labels (zero-closure demos/tasks drop out)
    episodes, languages = [], []
    for td in task_dirs:
        lab_path = os.path.join(C_LABELS, args.suite, f"{td}.h5")
        if not os.path.exists(lab_path):
            print(f"[{td}] no c_labels -> skip task", flush=True)
            continue
        with h5py.File(lab_path, "r") as lab:
            demo_keys = {k for k in lab.keys() if k.startswith("demo")}
        if args.labels_only:
            kept = natsorted(demo_keys)
            if args.demos_limit:
                kept = kept[: args.demos_limit]
        else:
            files = natsorted(os.listdir(os.path.join(LIGHT_ROOT, args.suite, td, "all")))
            if args.demos_limit:
                files = files[: args.demos_limit]
            kept = [f[:-5] for f in files if f.endswith(".hdf5") and f[:-5] in demo_keys]
        if not kept:
            print(f"[{td}] no grasp-bearing demos -> skip task", flush=True)
            continue
        lang = get_task_name_from_file_name(td)
        if lang not in languages:
            languages.append(lang)
        episodes += [(td, demo, lang) for demo in kept]

    env = sim = cur_task = None
    global_index = 0
    for ep_idx, (td, demo, lang) in enumerate(episodes):
        side = os.path.join(out, "meta", ".episodes", f"{ep_idx:06d}.json")
        pq_path = os.path.join(out, "data", "chunk-000", f"episode_{ep_idx:06d}.parquet")
        light = None
        if not args.labels_only:
            light = load_light(os.path.join(LIGHT_ROOT, args.suite, td, "all", f"{demo}.hdf5"))
            T = light["actions"].shape[0]
        else:
            with h5py.File(os.path.join(C_LABELS, args.suite, f"{td}.h5"), "r") as lab:
                T = int(lab[demo].attrs["T"])
        if os.path.exists(side):
            global_index += T
            continue

        with h5py.File(os.path.join(C_LABELS, args.suite, f"{td}.h5"), "r") as lab:
            g = lab[demo]
            base_quat = np.array(lab.attrs["robot_base_quat"])
            t_g = int(g.attrs["t_g"])
            t_close = first_close_t(g["phase"], t_g)
            yb, pb, yc, pc = orientation_targets(np.array(g["ee_quat"])[t_g], base_quat)
            gq = np.array(g["gripper_q"])[t_g]
            ep_meta = {
                "task_dir": td, "demo": demo,
                "t_g": t_g, "t_first_close": t_close,
                "n_cycles": int(g.attrs.get("n_cycles", 1)),
                "grasped_body": str(g.attrs["grasped_body"]),
                "contact_rowcol": np.array(g["contact_rowcol"]).tolist(),
                "yaw_bin": yb, "pitch_bin": pb, "yaw": yc, "pitch": pc,
                "w": float(gq[0] - gq[1]),
            }
            table = state = None
            if not args.labels_only:
                table, state = build_table(light, g, ep_idx, languages.index(lang),
                                           global_index)

        stats = {}
        if not args.labels_only:
            stats = {"observation.state": vec_stats(state),
                     "action": vec_stats(light["actions"])}
            for v, key in (("agentview", VIDEO_KEYS[0]), ("eye_in_hand", VIDEO_KEYS[1])):
                frames = light["views"][v]["video"].transpose(0, 2, 3, 1)  # t h w c
                encode_video(frames, os.path.join(out, "videos", "chunk-000", key,
                                                  f"episode_{ep_idx:06d}.mp4"))
                stats[key] = img_stats(frames)

        len_512 = 0
        if not args.no_render:
            if td != cur_task:
                if env is not None:
                    env.close()
                env, sim = build_env(args.suite, td, RES512)
                cur_task = td
            with h5py.File(os.path.join(RAW_ROOT, args.suite, f"{td}.hdf5"), "r") as raw:
                states = np.array(raw["data"][demo]["states"])
            t_hi = min(t_close, T - 1, len(states) - 1)
            fr512 = render_512(sim, states, t_hi)
            encode_video(fr512, os.path.join(out, "videos", "chunk-000", VIDEO_KEYS[2],
                                             f"episode_{ep_idx:06d}.mp4"))
            stats[VIDEO_KEYS[2]] = img_stats(fr512)
            len_512 = t_hi + 1
        ep_meta["len_512"] = len_512

        if table is not None:
            pq.write_table(table, pq_path)
        json.dump({"episode_index": ep_idx, "task": lang, "length": T,
                   "routedflow": ep_meta, "stats": stats}, open(side + ".tmp", "w"))
        os.replace(side + ".tmp", side)
        global_index += T
        if ep_idx % 20 == 0:
            print(f"[{ep_idx + 1}/{len(episodes)}] {td}/{demo} T={T} len512={len_512}",
                  flush=True)
    if env is not None:
        env.close()

    # ---- assemble meta from sidecars (idempotent, cheap) ----
    sides = []
    for i in range(len(episodes)):
        p = os.path.join(out, "meta", ".episodes", f"{i:06d}.json")
        assert os.path.exists(p), f"missing episode sidecar {p} — rerun to fill"
        sides.append(json.load(open(p)))
    with open(os.path.join(out, "meta", "tasks.jsonl"), "w") as f:
        for i, lang in enumerate(languages):
            f.write(json.dumps({"task_index": i, "task": lang}) + "\n")
    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as f:
        for s in sides:
            f.write(json.dumps({"episode_index": s["episode_index"], "tasks": [s["task"]],
                                "length": s["length"], "routedflow": s["routedflow"]}) + "\n")
    with open(os.path.join(out, "meta", "episodes_stats.jsonl"), "w") as f:
        for s in sides:
            f.write(json.dumps({"episode_index": s["episode_index"], "stats": s["stats"]}) + "\n")
    total_frames = sum(s["length"] for s in sides)
    n_vid = 1 if args.labels_only else (3 if not args.no_render else 2)
    feats = features_schema()
    if args.labels_only:
        feats = {VIDEO_KEYS[2]: feats[VIDEO_KEYS[2]]}
    info = {
        "codebase_version": "v2.1", "robot_type": "franka",
        "total_episodes": len(sides), "total_frames": total_frames,
        "total_tasks": len(languages), "total_videos": n_vid * len(sides),
        "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
        "splits": {"train": f"0:{len(sides)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feats,
        "routedflow": {
            "source": "atm_libero_light + c_labels (lossless: decoded frames are "
                      "bitwise-equal to the h5 uint8)",
            "agentview_512_note": "approach segment only, frames [0, t_first_close]; "
                                  "length per episode = episodes.jsonl routedflow.len_512",
            "labels_only": bool(args.labels_only),
        },
    }
    json.dump(info, open(os.path.join(out, "meta", "info.json"), "w"), indent=4)
    modality = {
        "state": {n: {"start": i, "end": i + 1} for i, n in enumerate(
            ["x", "y", "z", "roll", "pitch", "yaw", "gripper_0", "gripper_1"])},
        "action": {n: {"start": i, "end": i + 1} for i, n in enumerate(
            ["x", "y", "z", "roll", "pitch", "yaw", "gripper"])},
        "video": {k.split(".")[-1]: {"original_key": k} for k in VIDEO_KEYS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    json.dump(modality, open(os.path.join(out, "meta", "modality.json"), "w"), indent=4)
    print(f"done: {len(sides)} episodes, {total_frames} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
