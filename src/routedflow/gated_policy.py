"""BCViLTPolicyGated: ATM's BC policy + track gating.

Subclasses `atm.policy.vilt.BCViLTPolicy` and overrides `track_encode` only. The
body is a verbatim copy of the original (vilt.py L240-274) with the gate inserted
right after the frozen track transformer's `reconstruct` — before both use sites
of the predicted tracks (the track token pathway AND the raw-coordinate concat
into the policy head input), so gating covers everything the policy sees.

Gate context (per-frame robot labels + latched phase) is stashed on `self` right
before each forward because `track_encode`'s signature is fixed by the caller
chain (forward -> spatial_encode -> track_encode).
"""
import torch
from einops import rearrange, repeat

from atm.policy.vilt import BCViLTPolicy
from atm.utils.flow_utils import sample_double_grid

from routedflow.track_gate import VALID_MODES, gate_tracks


class BCViLTPolicyGated(BCViLTPolicy):
    def __init__(self, track_gate_cfg=None, **kwargs):
        super().__init__(**kwargs)
        cfg = dict(track_gate_cfg or {})
        self.gate_mode = cfg.get("mode", "none")
        assert self.gate_mode in VALID_MODES, f"gate mode {self.gate_mode!r} not in {VALID_MODES}"
        self._gate_ctx = None

    def set_gate_ctx(self, robot_labels, phase):
        """robot_labels: (b, v, t, 32) bool-like; phase: (b, t) bool-like."""
        self._gate_ctx = (robot_labels, phase)

    def clear_gate_ctx(self):
        self._gate_ctx = None

    def track_encode(self, track_obs, task_emb):
        """Copy of BCViLTPolicy.track_encode + gating. Keep in sync with upstream."""
        assert self.num_track_ids == 32
        b, v, t, *_ = track_obs.shape

        if self.use_zero_track:
            recon_tr = torch.zeros((b, v, t, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
        else:
            track_obs_to_pred = rearrange(track_obs, "b v t fs c h w -> (b v t) fs c h w")

            grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
            grid_sampled_track = repeat(grid_points, "n d -> b v t tl n d", b=b, v=v, t=t, tl=self.num_track_ts)
            grid_sampled_track = rearrange(grid_sampled_track, "b v t tl n d -> (b v t) tl n d")

            expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
            expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
            with torch.no_grad():
                pred_tr, _ = self.track.reconstruct(track_obs_to_pred, grid_sampled_track, expand_task_emb, p_img=0)  # (b v t) tl n d
                recon_tr = rearrange(pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t)

            # === RoutedFlow gate (the only change vs upstream) ===
            if self.gate_mode != "none":
                assert self._gate_ctx is not None, "gate ctx not set — call set_gate_ctx() before forward"
                robot_labels, phase = self._gate_ctx
                assert robot_labels.shape[:3] == (b, v, t) and robot_labels.shape[-1] == self.num_track_ids, \
                    f"robot_labels shape {tuple(robot_labels.shape)} vs expected ({b},{v},{t},{self.num_track_ids})"
                recon_tr = gate_tracks(
                    recon_tr,
                    grid_points,
                    robot_labels.to(track_obs.device),
                    phase.to(track_obs.device),
                    self.gate_mode,
                )

        recon_tr = recon_tr[:, :, :, :self.policy_num_track_ts, :, :]  # truncate the track to a shorter one
        _recon_tr = recon_tr.clone()  # b v t tl n 2
        with torch.no_grad():
            tr_view = self._get_view_one_hot(recon_tr)  # b v t tl n c

        tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
        tr = self.track_proj_encoder(tr_view)  # (b v t) track_patch_num n d
        tr = rearrange(tr, "(b v t) pn n d -> (b t n) (v pn) d", b=b, v=v, t=t, n=self.num_track_ids)  # (b t n) (v patch_num) d

        return tr, _recon_tr

    def forward_loss_gated(self, obs, track_obs, track, task_emb, extra_states, action, robot_labels, phase):
        self.set_gate_ctx(robot_labels, phase)
        try:
            return self.forward_loss(obs, track_obs, track, task_emb, extra_states, action)
        finally:
            self.clear_gate_ctx()

    def act_gated(self, obs, task_emb, extra_states, robot_labels, phase):
        """Eval-time entry. robot_labels: (b, v, 1, 32), phase: (b, 1) for the current step."""
        self.set_gate_ctx(robot_labels, phase)
        try:
            return self.act(obs, task_emb, extra_states)
        finally:
            self.clear_gate_ctx()
