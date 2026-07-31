"""GatedBCDataset: ATM's BCDataset + per-frame grid-point robot labels and phase.

Expects demo h5 files produced by `routedflow.convert_libero_raw` which add, on
top of the ATM format:
    root/phase:               (T,) uint8   — latched contact phase
    root/<view>/grid_labels:  (T, 32) uint8 — robot membership of the 32 grid points
    root/<view>/robot_seg:    (T, H, W) uint8 — full robot mask (kept for later use)
(GT tracks are zero dummies — BC training never reads them, confirmed in
vilt.py forward() docstring.)

The labels index the frames the track transformer actually sees (`track_obs`,
which is NOT augmented — bc_dataloader.py only augments `obs` and the GT track),
so no augmentation correction is needed.
"""
import torch

from atm.dataloader.bc_dataloader import BCDataset


class GatedBCDataset(BCDataset):
    def __init__(self, *args, views=None, **kwargs):
        if views is None:
            views = ["agentview", "eye_in_hand"]
        # BaseDataset auto-detects views from h5 root keys when views is None,
        # which would wrongly pick up our extra "phase" key — always pass explicitly.
        views = list(views)
        super().__init__(*args, views=views, **kwargs)
        assert self.cache_all, "GatedBCDataset requires cache_all=True (label slicing reads from the demo cache)"

    def process_demo(self, demo):
        demo = super().process_demo(demo)
        pad_length = self.frame_stack + self.num_track_ts

        phase = torch.as_tensor(demo["root"]["phase"]).flatten().to(torch.bool)
        phase = torch.cat([phase, phase[-1:].expand(pad_length)], dim=0)
        demo["root"]["phase"] = phase

        for v in self.views:
            gl = torch.as_tensor(demo["root"][v]["grid_labels"]).to(torch.bool)  # (T, 32)
            gl = torch.cat([gl, gl[-1:].expand(pad_length, -1)], dim=0)
            demo["root"][v]["grid_labels"] = gl
            # robot_seg is not needed at train time; drop it from the RAM cache
            demo["root"][v].pop("robot_seg", None)
        return demo

    def __getitem__(self, index):
        obs, track_transformer_obs, track, task_embs, actions, extra_states = super().__getitem__(index)

        demo_id = self._index_to_demo_id[index]
        time_offset = index - self._demo_id_to_start_indices[demo_id]
        demo = self._cache[demo_id]

        sl = slice(time_offset, time_offset + self.frame_stack)
        phase = demo["root"]["phase"][sl]  # (t,)
        robot_labels = torch.stack([demo["root"][v]["grid_labels"][sl] for v in self.views], dim=0)  # (v, t, 32)

        return obs, track_transformer_obs, track, task_embs, actions, extra_states, robot_labels, phase
