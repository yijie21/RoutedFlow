"""Stage-2 joint-training dataset: ATM BC windows + chain flow GT + stage-1 sample.

Extends ATM's BCDataset (windows of frame_stack=10 steps over the light-converted
demos). Each item adds:
    chain_query (t, 32, 2)      L3 query points at each window step (normalized xy)
    chain_depth (t, 32)         camera-frame depth of the queries (D8a)
    chain_gt    (t, tl, 32, 2)  zero-noise flow GT windows
    s1          dict            the demo's stage-1 sample (dino/prior/text + L_C targets)

Split protocol = EXACTLY stage-1 fold0's demo partition (lexicographic sorted,
train = demos[:45], val_id = demos[45:] of the 8 train tasks) so the pretrained
C-VLM never saw val demos. Windows are approach-only: window must END at or
before t_g + margin.
"""
import os

import numpy as np
import torch

from atm.dataloader.bc_dataloader import BCDataset

from routedflow.stage1.dataset import Stage1Dataset, fold_split

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
C_LABELS = os.path.join(REPO, "data", "c_labels", "libero_spatial")
LIGHT = os.path.join(REPO, "data", "atm_libero_light", "libero_spatial")


def light_dirs(fold, split):
    tasks = sorted(d for d in os.listdir(LIGHT))
    train_tasks, ood = fold_split(tasks, fold)
    sel = ood if split == "val_ood" else train_tasks
    return [os.path.join(LIGHT, t, "all") for t in sel]


class Stage2Dataset(BCDataset):
    def __init__(self, *args, fold=0, split="train", tg_margin=2, use_prior=True, **kwargs):
        self.fold, self.split, self.tg_margin = fold, split, tg_margin
        kwargs.setdefault("views", ["agentview", "eye_in_hand"])
        super().__init__(*args, dataset_dir=light_dirs(fold, split), **kwargs)
        assert self.cache_all, "Stage2Dataset requires cache_all=True"

        # stage-1 samples keyed by (task, demo) — same fold/split semantics
        s1 = Stage1Dataset(fold=fold, split=split, use_prior=use_prior)
        self._s1 = {(s["task"], s["demo"]): s for s in s1.samples}

        # chain data + t_g per demo_id; then build the filtered index:
        # keep (a) demos belonging to this split's demo partition, (b) approach windows
        self._chain, keep = {}, []
        import h5py
        allowed = {(t, d) for (t, d) in self._s1.keys()}
        for demo_id, path in self._demo_id_to_path.items():
            task = os.path.basename(os.path.dirname(os.path.dirname(path)))
            demo = os.path.basename(path).replace(".hdf5", "")
            if (task, demo) not in allowed:
                continue
            with h5py.File(os.path.join(C_LABELS, f"{task}.h5"), "r") as f:
                g = f[demo]
                uv = np.asarray(g["chain_uv"], np.float32)
                zz = np.asarray(g["chain_z"], np.float32)
                tg = int(g.attrs["t_g"])
            pad = self.frame_stack + self.num_track_ts
            uv = np.concatenate([uv, np.repeat(uv[-1:], pad, 0)])
            zz = np.concatenate([zz, np.repeat(zz[-1:], pad, 0)])
            self._chain[demo_id] = (torch.from_numpy(uv), torch.from_numpy(zz), tg, (task, demo))
            start = self._demo_id_to_start_indices[demo_id]
            length = self._demo_id_to_demo_length[demo_id]
            for off in range(length):
                if off + self.frame_stack <= tg + self.tg_margin:
                    keep.append(start + off)
        self._keep = keep
        assert keep, f"no approach windows for fold {fold} split {split}"

    def __len__(self):
        return len(self._keep)

    def __getitem__(self, i):
        index = self._keep[i]
        obs, track_obs, track, task_emb, actions, extra_states = super().__getitem__(index)

        demo_id = self._index_to_demo_id[index]
        off = index - self._demo_id_to_start_indices[demo_id]
        uv, zz, tg, (task, demo) = self._chain[demo_id]

        t, tl = self.frame_stack, self.num_track_ts
        chain_query = uv[off:off + t]                       # (t, 32, 2)
        chain_depth = zz[off:off + t]                       # (t, 32)
        chain_gt = torch.stack([uv[off + k: off + k + tl] for k in range(t)])  # (t, tl, 32, 2)

        s = self._s1[(task, demo)]
        s1 = {
            "dino": torch.from_numpy(np.asarray(s["dino"], np.float32)),
            "prior": torch.from_numpy(s["prior"]),
            "text": torch.from_numpy(s["text"]),
            "heatmap": torch.from_numpy(s["heatmap"]),
            "yaw_bin": torch.tensor(s["yaw_bin"]),
            "pitch_bin": torch.tensor(s["pitch_bin"]),
            "w": torch.tensor(s["w"]),
            "contact_rowcol": torch.from_numpy(s["contact_rowcol"]),
        }
        return obs, track_obs, track, task_emb, actions, extra_states, \
            chain_query, chain_depth, chain_gt, s1
