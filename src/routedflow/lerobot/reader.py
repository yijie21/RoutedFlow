"""Hand-written LeRobot v2.1 reader (pyarrow + pyav; no lerobot-pip dependency).

Grill decision 2026-08-04: the official lerobot package pins torch versions that
would fight our hand-built 5090 cu128 torch, so the training env reads the
format directly. Scope: exactly what convert_to_lerobot.py writes (single
chunk, per-episode parquet, lossless mp4 videos, routedflow episode extras).
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LEROBOT_ROOT = os.path.join(REPO, "data", "lerobot")

V_AGENT = "observation.images.agentview"
V_WRIST = "observation.images.eye_in_hand"
V_512 = "observation.images.agentview_512"


class LeRobotSuite:
    def __init__(self, suite, root=LEROBOT_ROOT):
        self.suite, self.dir = suite, os.path.join(root, suite)
        meta = os.path.join(self.dir, "meta")
        assert os.path.exists(os.path.join(meta, "info.json")), \
            f"no LeRobot dataset for {suite} — run `run_stage2.py convert-lerobot --suite {suite}`"
        self.info = json.load(open(os.path.join(meta, "info.json")))
        self.episodes = [json.loads(ln) for ln in open(os.path.join(meta, "episodes.jsonl"))]
        self.tasks = {r["task_index"]: r["task"] for r in
                      (json.loads(ln) for ln in open(os.path.join(meta, "tasks.jsonl")))}
        self._by_key = {(e["routedflow"]["task_dir"], e["routedflow"]["demo"]): e
                        for e in self.episodes}
        self.task_dirs = sorted({e["routedflow"]["task_dir"] for e in self.episodes})

    def lookup(self, task_dir, demo):
        return self._by_key.get((task_dir, demo))

    def demos_of(self, task_dir):
        """Episodes of a task, demo-name LEXICOGRAPHIC order (= Stage1Dataset's
        sorted(f.keys()) split semantics — NOT natsort)."""
        eps = [e for e in self.episodes if e["routedflow"]["task_dir"] == task_dir]
        return sorted(eps, key=lambda e: e["routedflow"]["demo"])

    def parquet_path(self, ep_index):
        return os.path.join(self.dir, "data", "chunk-000", f"episode_{ep_index:06d}.parquet")

    def video_path(self, key, ep_index):
        return os.path.join(self.dir, "videos", "chunk-000", key, f"episode_{ep_index:06d}.mp4")

    def read_table(self, ep_index, columns=None):
        """Parquet -> dict of numpy arrays with routedflow columns reshaped."""
        t = pq.read_table(self.parquet_path(ep_index), columns=columns)
        out = {}
        for name in t.column_names:
            col = t[name].combine_chunks()
            arr = (np.asarray(col.flatten()).reshape(t.num_rows, -1)
                   if hasattr(col.type, "list_size") else np.asarray(col))
            if name.endswith("chain_uv") or ".tracks." in name:
                arr = arr.reshape(t.num_rows, 32, 2)
            out[name] = arr
        return out
