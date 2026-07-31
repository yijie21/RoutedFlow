"""Stage-1 (curriculum ①) dataset: cached features -> L_C training samples.

Sources (all precomputed, this module does NO sim/model work):
    data/c_labels/<suite>/<task>.h5          labels (contact/ee_quat[t_g]/gripper_q)
    data/stage1_cache/<suite>/dino_feats.h5  (1369,768) f16 per demo
    data/stage1_cache/<suite>/afun_prior.h5  (512,512) uint8 per demo (optional:
                                             zeros + has_prior=0 while job #2 runs)
    ATM BERT task-emb cache                  text embedding per task language

Targets:
    heatmap  (128,128) normalized Gaussian, sigma = 8px@512 = 2px@128
    yaw_bin  36 bins over [0, pi)  — finger-axis azimuth in BASE frame, +-pi folded
             (parallel gripper symmetry)
    pitch_bin 12 bins over [0, pi/2] — approach-axis tilt from vertical (base frame)
    w        finger opening q[0]-q[1] at t_g

Axis conventions (verified empirically in tests/test_stage1_units.py against all
500 demos): grip-site z-axis = approach direction (points down in top-down
grasps); site y-axis = finger-opening line.

Split protocol (plan §2.2): fold k in 0..4 holds out tasks [2k, 2k+1] entirely
(val_ood); remaining 8 tasks give train (demo 0..44) and val_id (demo 45..49).
"""
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
C_LABELS = os.path.join(REPO, "data", "c_labels")
CACHE = os.path.join(REPO, "data", "stage1_cache")
BERT_CACHE = os.path.join(REPO, "third_party", "ATM", "libero",
                          "task_embedding_caches", "task_emb_bert.npy")

N_YAW, N_PITCH = 36, 12
HEATMAP_RES, SIGMA = 128, 2.0  # sigma 2px@128 == 8px@512
GRID = 37


def quat_to_mat(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def orientation_targets(ee_quat_tg, base_quat):
    """(yaw_bin, pitch_bin, yaw_cont, pitch_cont) in the robot base frame."""
    R_wb = quat_to_mat(base_quat)
    R_b = R_wb.T @ quat_to_mat(ee_quat_tg)
    a = R_b @ np.array([0.0, 0.0, 1.0])   # approach axis (down for top-down grasps)
    f = R_b @ np.array([0.0, 1.0, 0.0])   # finger-opening axis
    pitch = float(np.arccos(np.clip(-a[2], -1.0, 1.0)))          # 0 = straight down
    yaw = float(np.arctan2(f[1], f[0]) % np.pi)                  # +-pi fold
    yaw_bin = min(int(yaw / (np.pi / N_YAW)), N_YAW - 1)
    pitch_bin = min(int(np.clip(pitch, 0, np.pi / 2 - 1e-6) / ((np.pi / 2) / N_PITCH)), N_PITCH - 1)
    return yaw_bin, pitch_bin, yaw, pitch


def gaussian_heatmap(row512, col512, res=HEATMAP_RES, sigma=SIGMA):
    r, c = row512 * res / 512.0, col512 * res / 512.0
    yy, xx = np.mgrid[0:res, 0:res]
    h = np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma ** 2))
    s = h.sum()
    return (h / s).astype(np.float32) if s > 0 else np.full((res, res), 1.0 / res ** 2, np.float32)


def pool_mask_to_grid(mask512, grid=GRID):
    """(512,512) {0,1} -> (grid,grid) float coverage via area interpolation."""
    t = torch.from_numpy(mask512.astype(np.float32))[None, None]
    return torch.nn.functional.interpolate(t, size=grid, mode="area")[0, 0].numpy()


def fold_split(tasks, fold):
    tasks = sorted(tasks)
    assert len(tasks) == 10 and 0 <= fold <= 4
    held = set(tasks[2 * fold: 2 * fold + 2])
    return [t for t in tasks if t not in held], sorted(held)


class Stage1Dataset(Dataset):
    """split: 'train' | 'val_id' | 'val_ood'. Fully in-memory (~1.1G for train)."""

    def __init__(self, suite="libero_spatial", fold=0, split="train", use_prior=True):
        self.samples = []
        suite_dir = os.path.join(C_LABELS, suite)
        tasks = sorted(f[:-3] for f in os.listdir(suite_dir) if f.endswith(".h5"))
        train_tasks, ood_tasks = fold_split(tasks, fold)
        sel_tasks = ood_tasks if split == "val_ood" else train_tasks

        emb_map = np.load(BERT_CACHE, allow_pickle=True).item()
        prior_path = os.path.join(CACHE, suite, "afun_prior.h5")
        prior = h5py.File(prior_path, "r") if (use_prior and os.path.exists(prior_path)) else None
        dino = h5py.File(os.path.join(CACHE, suite, "dino_feats.h5"), "r")

        for task in sel_tasks:
            with h5py.File(os.path.join(suite_dir, f"{task}.h5"), "r") as f:
                base_quat = np.array(f.attrs["robot_base_quat"])
                lang = str(f.attrs["task_language"])
                emb = np.asarray(emb_map[lang], np.float32)
                if emb.ndim > 1:
                    emb = emb.mean(0)
                demos = sorted(k for k in f.keys() if k.startswith("demo"))
                if split == "train":
                    demos = demos[:45]
                elif split == "val_id":
                    demos = demos[45:]
                for k in demos:
                    g = f[k]
                    row, col = np.array(g["contact_rowcol"])
                    yb, pb, yc, pc = orientation_targets(
                        np.array(g["ee_quat"])[int(g.attrs["t_g"])], base_quat)
                    gq = np.array(g["gripper_q"])[int(g.attrs["t_g"])]
                    has_prior = prior is not None and task in prior and k in prior[task]
                    mask = (pool_mask_to_grid(np.array(prior[task][k])) if has_prior
                            else np.zeros((GRID, GRID), np.float32))
                    self.samples.append({
                        "dino": np.asarray(dino[task][k]),          # (1369,768) f16
                        "prior": mask.astype(np.float32),           # (37,37)
                        "has_prior": np.float32(has_prior),
                        "text": emb,                                # (768,)
                        "heatmap": gaussian_heatmap(row, col),      # (128,128)
                        "yaw_bin": np.int64(yb), "pitch_bin": np.int64(pb),
                        "w": np.float32(gq[0] - gq[1]),
                        "contact_rowcol": np.array([row, col], np.float32),  # @512
                        "task": task, "demo": k,
                    })
        dino.close()
        if prior is not None:
            prior.close()
        self.train_tasks, self.ood_tasks = train_tasks, ood_tasks
        assert self.samples, f"empty split {split} fold {fold}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "dino": torch.from_numpy(np.asarray(s["dino"], np.float32)),
            "prior": torch.from_numpy(s["prior"]),
            "has_prior": torch.tensor(s["has_prior"]),
            "text": torch.from_numpy(s["text"]),
            "heatmap": torch.from_numpy(s["heatmap"]),
            "yaw_bin": torch.tensor(s["yaw_bin"]),
            "pitch_bin": torch.tensor(s["pitch_bin"]),
            "w": torch.tensor(s["w"]),
            "contact_rowcol": torch.from_numpy(s["contact_rowcol"]),
        }
