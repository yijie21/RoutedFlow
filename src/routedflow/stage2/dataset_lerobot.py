"""Stage-2 dataset on the LeRobot backend — drop-in replacement for Stage2Dataset.

Emits the EXACT same 10-tuple as Stage2Dataset (parity-tested item-for-item:
videos are lossless so tensors match bitwise; track grid-sampling is
deterministic). Differences are purely operational:

  - streaming mode reads ONLY the 10-frame window from mp4 (the old h5 path
    loaded + tensorized the ENTIRE demo per __getitem__ — the real cause of the
    35% streaming slowdown);
  - cached mode holds uint8 frames (~4x smaller than the old float32 cache);
  - non-video columns (actions/extras/tracks/vis/chain) are ALWAYS cached in
    RAM (a few hundred MB), read once from parquet at init;
  - t_first_close comes from episodes.jsonl (no c_labels h5 at runtime here;
    Stage1Dataset still provides the s1 dict exactly as before).

Split protocol is unchanged: fold_split on sorted task dirs; demo partition by
LEXICOGRAPHIC demo name, train=[:45], val_id=[45:]; extra suites train-only;
approach windows end at or before t_first_close + tg_margin.
"""
import os

import numpy as np
import torch

from atm.utils.flow_utils import sample_tracks_nearest_to_grids

from routedflow.lerobot.reader import V_AGENT, V_WRIST, LeRobotSuite
from routedflow.lerobot.video import decode_all, decode_window
from routedflow.stage1.dataset import Stage1Dataset, fold_split

PRIMARY = "libero_spatial"
BERT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "third_party", "ATM",
    "libero", "task_embedding_caches", "task_emb_bert.npy")


class Stage2LeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, img_size=128, frame_stack=10, num_track_ts=16, num_track_ids=32,
                 track_obs_fs=1, augment_track=False, extra_state_keys=None,
                 fold=0, split="train", tg_margin=2, use_prior=True,
                 extra_suites=(), cache_all=True, cache_image=True, views=None):
        assert track_obs_fs == 1 and img_size == 128 and not augment_track
        self.frame_stack, self.num_track_ts = frame_stack, num_track_ts
        self.num_track_ids = num_track_ids
        self.extra_state_keys = extra_state_keys or ["joint_states", "gripper_states"]
        self.views = sorted(views or ["agentview", "eye_in_hand"])
        self.cache_video = bool(cache_all and cache_image)
        self.fold, self.split, self.tg_margin = fold, split, tg_margin

        emb_map = np.load(BERT_CACHE, allow_pickle=True).item()

        primary = LeRobotSuite(PRIMARY)
        train_tasks, ood = fold_split(primary.task_dirs, fold)
        plan = [(primary, ood if split == "val_ood" else train_tasks, split)]
        if split == "train":
            for s in extra_suites:
                st = LeRobotSuite(s)
                plan.append((st, st.task_dirs, "all"))

        # episode selection == Stage1Dataset demo partition semantics
        self._eps = []          # (suite_obj, ep_record)
        for st, tasks, sp in plan:
            for td in tasks:
                demos = st.demos_of(td)
                if sp == "train":
                    demos = demos[:45]
                elif sp == "val_id":
                    demos = demos[45:]
                self._eps += [(st, e) for e in demos]

        # per-episode arrays (non-video: always in RAM) + window keep-list
        pad = frame_stack + num_track_ts
        self._data, self._keep = [], []
        for li, (st, e) in enumerate(self._eps):
            rf = e["routedflow"]
            tab = st.read_table(e["episode_index"])
            T = e["length"]
            emb = np.asarray(emb_map[e["tasks"][0]], np.float32)
            if emb.ndim > 1:
                emb = emb.mean(0)

            def rep_pad(a, n=pad):
                return np.concatenate([a, np.repeat(a[-1:], n, 0)])

            d = {
                "T": T, "suite": st, "ep": e["episode_index"],
                "key": (rf["task_dir"], rf["demo"]),
                "actions": np.concatenate([tab["action"],
                                           np.zeros((pad, 7), np.float32)]),
                "extra": {k: rep_pad(tab[f"observation.extra.{k}"])
                          for k in self.extra_state_keys},
                "tracks": {v: rep_pad(tab[f"routedflow.tracks.{v}"]) for v in self.views},
                "vis": {v: rep_pad(tab[f"routedflow.vis.{v}"]) for v in self.views},
                "chain_uv": rep_pad(tab["routedflow.chain_uv"]),
                "chain_z": rep_pad(tab["routedflow.chain_z"]),
                "emb": emb,
                "video": None,
            }
            if self.cache_video:
                d["video"] = {v: decode_all(st.video_path(k, e["episode_index"]))
                              for v, k in (("agentview", V_AGENT), ("eye_in_hand", V_WRIST))}
            self._data.append(d)
            tg = int(rf["t_first_close"])
            for off in range(T):
                if off + frame_stack <= tg + tg_margin:
                    self._keep.append((li, off))
        assert self._keep, f"no approach windows for fold {fold} split {split}"

        # s1 dict source — unchanged path (cached DINO + labels via Stage1Dataset)
        s1 = Stage1Dataset(fold=fold, split=split, use_prior=use_prior,
                           extra_suites=[s for s in extra_suites] or None)
        self._s1 = {(s["task"], s["demo"]): s for s in s1.samples}

    def __len__(self):
        return len(self._keep)

    def _frames(self, d, off):
        """(t_pad-clamped) window frames per view as float tensors (t c h w)."""
        t, T = self.frame_stack, d["T"]
        n_real = max(0, min(t, T - off))
        out = {}
        for v in self.views:
            if d["video"] is not None:
                fr = d["video"][v][off:off + n_real]
            else:
                key = V_AGENT if v == "agentview" else V_WRIST
                fr = decode_window(d["suite"].video_path(key, d["ep"]), off, n_real)
            if n_real < t:  # repeat-last pad (matches BaseDataset.process_demo)
                fr = np.concatenate([fr, np.repeat(fr[-1:], t - n_real, 0)])
            out[v] = torch.from_numpy(fr.transpose(0, 3, 1, 2).copy()).float()
        return out

    def __getitem__(self, i):
        li, off = self._keep[i]
        d = self._data[li]
        t, tl = self.frame_stack, self.num_track_ts

        fr = self._frames(d, off)
        obs = torch.stack([fr[v] for v in self.views], 0)               # v t c h w
        track_obs = obs[:, :, None]                                     # v t 1 c h w

        track, vi = [], []
        for v in self.views:
            tr = torch.from_numpy(np.stack(
                [d["tracks"][v][off + k: off + k + tl] for k in range(t)]))  # t tl 32 2
            vv = torch.from_numpy(np.stack(
                [d["vis"][v][off + k: off + k + tl] for k in range(t)]))
            st_tr, st_vi = [], []
            for k in range(t):
                a, b = sample_tracks_nearest_to_grids(tr[k], vv[k], self.num_track_ids)
                st_tr.append(a)
                st_vi.append(b)
            track.append(torch.stack(st_tr))
            vi.append(torch.stack(st_vi))
        track = torch.stack(track, 0)                                   # v t tl n 2

        actions = torch.from_numpy(d["actions"][off:off + t].copy())
        task_emb = torch.from_numpy(d["emb"].copy())
        extra = {k: torch.from_numpy(d["extra"][k][off:off + t].copy())
                 for k in self.extra_state_keys}

        chain_query = torch.from_numpy(d["chain_uv"][off:off + t].copy())
        chain_depth = torch.from_numpy(d["chain_z"][off:off + t].copy())
        chain_gt = torch.stack([torch.from_numpy(d["chain_uv"][off + k: off + k + tl].copy())
                                for k in range(t)])

        s = self._s1[d["key"]]
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
        return obs, track_obs, track, task_emb, actions, extra, \
            chain_query, chain_depth, chain_gt, s1
