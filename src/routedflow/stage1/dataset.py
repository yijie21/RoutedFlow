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
    """split: 'train' | 'val_id' | 'val_ood'. Fully in-memory (~1.1G for train).

    Modes:
      raw=False (default): samples carry cached DINO features — the original path,
        also consumed by Stage2Dataset (do not change its sample keys).
      raw=True: samples additionally carry rgb0 (512^2 uint8) + full-res prior
        mask; __getitem__ applies augmentation (augment=True: contact-preserving
        random resized crop + color jitter) and recomputes heatmap/prior-grid
        after the transform. The engine computes DINO features online.
      extra_suites: additional suites whose demos ALL go into the train split
        (C-head 开小灶 2026-08-02 — semantic diversity for the C supervision only;
        fold/ood semantics remain defined on the primary suite).
      hindsight (requires raw, train split only): per __getitem__, the input frame
        is drawn uniformly from the demo's pre-contact frames — C is a future
        event, so every frame before contact is a valid input for the same label
        (supervision density 1 -> ~t_close per episode). Source: the LeRobot
        agentview_512 approach-segment video (data/lerobot/<suite>/, lossless;
        frames [0, t_first_close]); a guard of `hindsight_guard` frames before
        closure is excluded (arm-position leak, goalmix lesson). Labels/augment/
        prior are untouched (static camera; prior stays the t=0 AFUN mask).
        Demos missing from the LeRobot set silently fall back to rgb0; val
        splits always use rgb0 (matches rollout: z computed once at phase start).
    """

    def __init__(self, suite="libero_spatial", fold=0, split="train", use_prior=True,
                 extra_suites=None, raw=False, augment=False, hindsight=False,
                 hindsight_guard=10, seed=0):
        self.samples = []
        self.raw, self.augment = raw, augment
        self.split = split
        self.hindsight = hindsight and split == "train"
        self.hindsight_guard = hindsight_guard
        assert not (hindsight and not raw), "hindsight needs raw=True (online DINO)"
        self._hs_meta = {}    # (suite, task, demo) -> (video_path, max_t exclusive)
        self._hs_cover = set()
        self._rng = np.random.default_rng(seed)
        suite_dir = os.path.join(C_LABELS, suite)
        tasks = sorted(f[:-3] for f in os.listdir(suite_dir) if f.endswith(".h5"))
        train_tasks, ood_tasks = fold_split(tasks, fold)
        sel_tasks = ood_tasks if split == "val_ood" else train_tasks

        emb_map = np.load(BERT_CACHE, allow_pickle=True).item()
        plan = [(suite, sel_tasks, split)]
        if extra_suites and split == "train":
            for s in extra_suites:
                s_tasks = sorted(f[:-3] for f in os.listdir(os.path.join(C_LABELS, s))
                                 if f.endswith(".h5"))
                plan.append((s, s_tasks, "all"))
        for cur_suite, cur_tasks, cur_split in plan:
            self._load_suite(cur_suite, cur_tasks, cur_split, emb_map, use_prior)
        self.train_tasks, self.ood_tasks = train_tasks, ood_tasks
        assert self.samples, f"empty split {split} fold {fold}"
        if self.hindsight:
            self._index_hindsight()

    def _index_hindsight(self):
        """Index LeRobot 512 approach videos once -> (path, sampling bound) map."""
        from routedflow.lerobot.reader import V_512, LeRobotSuite
        for st in sorted({s["suite"] for s in self.samples}):
            try:
                suite = LeRobotSuite(st)
            except AssertionError:
                continue
            for e in suite.episodes:
                rf = e["routedflow"]
                if not rf.get("len_512"):
                    continue
                # sample t in [0, t_close - guard]; degenerate demos keep frame 0
                max_t = max(1, rf["len_512"] - self.hindsight_guard)
                key = (st, rf["task_dir"], rf["demo"])
                self._hs_meta[key] = (suite.video_path(V_512, e["episode_index"]), max_t)
                self._hs_cover.add(key)
        n_cov = sum(1 for s in self.samples
                    if (s["suite"], s["task"], s["demo"]) in self._hs_cover)
        print(f"hindsight (lerobot 512) covers {n_cov}/{len(self.samples)} train samples "
              f"(missing ones fall back to rgb0)", flush=True)

    def _hs_frame(self, s):
        """Random pre-contact frame for sample s, decoded from the 512 video."""
        from routedflow.lerobot.video import decode_frame
        path, max_t = self._hs_meta[(s["suite"], s["task"], s["demo"])]
        return decode_frame(path, int(self._rng.integers(max_t)))

    def _load_suite(self, suite, sel_tasks, split, emb_map, use_prior):
        suite_dir = os.path.join(C_LABELS, suite)
        prior_path = os.path.join(CACHE, suite, "afun_prior.h5")
        prior = h5py.File(prior_path, "r") if (use_prior and os.path.exists(prior_path)) else None
        dino_path = os.path.join(CACHE, suite, "dino_feats.h5")
        dino = None if self.raw else h5py.File(dino_path, "r")

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
                    s = {
                        "has_prior": np.float32(has_prior),
                        "text": emb,                                # (768,)
                        "yaw_bin": np.int64(yb), "pitch_bin": np.int64(pb),
                        "w": np.float32(gq[0] - gq[1]),
                        "contact_rowcol": np.array([row, col], np.float32),  # @512
                        "task": task, "demo": k, "suite": suite,
                    }
                    if self.raw:
                        s["rgb"] = np.array(g["rgb0"])              # (512,512,3) u8
                        s["prior_full"] = (np.array(prior[task][k]) if has_prior
                                           else np.zeros((512, 512), np.uint8))
                    else:
                        s["dino"] = np.asarray(dino[task][k])       # (1369,768) f16
                        mask = (pool_mask_to_grid(np.array(prior[task][k])) if has_prior
                                else np.zeros((GRID, GRID), np.float32))
                        s["prior"] = mask.astype(np.float32)        # (37,37)
                        s["heatmap"] = gaussian_heatmap(row, col)   # (128,128)
                    self.samples.append(s)
        if dino is not None:
            dino.close()
        if prior is not None:
            prior.close()

    def __len__(self):
        return len(self.samples)

    def _augment_geo(self, rgb, prior, row, col):
        """Contact-preserving random resized crop (no flip/rotation: spatial
        language and base-frame orientation targets must stay valid)."""
        rng = self._rng
        r0 = c0 = 0
        size = 512
        for _ in range(10):
            s = int(512 * rng.uniform(0.7, 1.0))
            rr = int(rng.integers(0, 512 - s + 1))
            cc = int(rng.integers(0, 512 - s + 1))
            if rr + 16 <= row < rr + s - 16 and cc + 16 <= col < cc + s - 16:
                r0, c0, size = rr, cc, s
                break
        rgb = rgb[r0:r0 + size, c0:c0 + size]
        prior = prior[r0:r0 + size, c0:c0 + size]
        sc = 512.0 / size
        return rgb, prior, (row - r0) * sc, (col - c0) * sc

    def __getitem__(self, i):
        s = self.samples[i]
        common = {
            "has_prior": torch.tensor(s["has_prior"]),
            "text": torch.from_numpy(s["text"]),
            "yaw_bin": torch.tensor(s["yaw_bin"]),
            "pitch_bin": torch.tensor(s["pitch_bin"]),
            "w": torch.tensor(s["w"]),
        }
        if not self.raw:
            return common | {
                "dino": torch.from_numpy(np.asarray(s["dino"], np.float32)),
                "prior": torch.from_numpy(s["prior"]),
                "heatmap": torch.from_numpy(s["heatmap"]),
                "contact_rowcol": torch.from_numpy(s["contact_rowcol"]),
            }
        rgb, prior_m = s["rgb"], s["prior_full"]
        if self.hindsight and (s["suite"], s["task"], s["demo"]) in self._hs_cover:
            rgb = self._hs_frame(s)
        row, col = float(s["contact_rowcol"][0]), float(s["contact_rowcol"][1])
        if self.augment:
            rgb, prior_m, row, col = self._augment_geo(rgb, prior_m, row, col)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        pm = torch.from_numpy(np.ascontiguousarray(prior_m)).float()[None, None]
        if x.shape[-1] != 512:
            x = torch.nn.functional.interpolate(x[None], size=512, mode="bilinear",
                                                align_corners=False)[0]
            pm = torch.nn.functional.interpolate(pm, size=512, mode="area")
        if self.augment:
            b, c_, sat = [float(v) for v in self._rng.uniform(0.8, 1.2, 3)]
            x = (x * b).clamp(0, 1)
            x = ((x - x.mean()) * c_ + x.mean()).clamp(0, 1)
            grey = x.mean(0, keepdim=True)
            x = (grey + (x - grey) * sat).clamp(0, 1)
        prior_grid = torch.nn.functional.interpolate(pm, size=GRID, mode="area")[0, 0]
        return common | {
            "rgb": x,                                                    # (3,512,512) 0-1
            "prior": prior_grid,
            "heatmap": torch.from_numpy(gaussian_heatmap(row, col)),
            "contact_rowcol": torch.tensor([row, col], dtype=torch.float32),
        }
