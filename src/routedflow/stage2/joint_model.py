"""Joint approach-branch model: C-VLM (L1) -> z -> L3 flow -> L4 action expert.

Fully differentiable: L_action's gradient flows through L4 -> predicted tracks
-> L3 -> z -> L1's trainable fusion. Per D10, L4 consumes L3's PREDICTIONS
(with gradient) plus gaussian flow noise at train time — never GT teacher
forcing. Wrist-view flow is zeroed (grill decision: no wrist camera poses
stored); wrist IMAGES still feed the policy.

ApproachPolicy = BCViLTPolicy with track_encode consuming an external flow
context instead of its internal frozen track transformer (which is deleted).
"""
import torch
import torch.nn as nn
from einops import rearrange

from atm.policy.vilt import BCViLTPolicy

from routedflow.stage1.model import CHead, L1FrontEnd, stage1_loss
from routedflow.stage2.flow_l3 import L3RobotFlow

AGENTVIEW = 0  # view order in the dataset: ["agentview", "eye_in_hand"]


class ApproachPolicy(BCViLTPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # External flow replaces the internal frozen track transformer. Keep a
        # stub (not `del`): vilt's train() override calls self.track.eval().
        self.track = nn.Identity()
        self._flow_ctx = None

    def set_flow(self, flow_agentview):
        """flow_agentview: (b, t, tl, n, 2) — may carry gradient (D10)."""
        self._flow_ctx = flow_agentview

    def track_encode(self, track_obs, task_emb):
        """External-flow version of BCViLTPolicy.track_encode (same tail as upstream)."""
        assert self.num_track_ids == 32 and self._flow_ctx is not None
        b, v, t, *_ = track_obs.shape
        flow = self._flow_ctx
        recon_tr = torch.zeros(b, v, t, flow.shape[2], flow.shape[3], 2,
                               device=flow.device, dtype=flow.dtype)
        recon_tr[:, AGENTVIEW] = flow  # wrist view stays zeros (no wrist flow GT)

        recon_tr = recon_tr[:, :, :, : self.policy_num_track_ts, :, :]
        _recon_tr = recon_tr.clone()
        tr_view = self._get_view_one_hot(recon_tr)  # b v t tl n c
        tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
        tr = self.track_proj_encoder(tr_view)
        tr = rearrange(tr, "(b v t) pn n d -> (b t n) (v pn) d", b=b, v=v, t=t, n=self.num_track_ids)
        return tr, _recon_tr


class JointApproachModel(nn.Module):
    def __init__(self, track_fn, policy_kwargs, l1_ckpt=None, lam=(0.5, 1.0, 0.1),
                 flow_noise=0.01, use_cross_attn=False):
        super().__init__()
        self.l1 = L1FrontEnd()
        self.chead = CHead()
        if l1_ckpt:
            state = torch.load(l1_ckpt, map_location="cpu", weights_only=False)
            self.l1.load_state_dict(state["l1"])
            self.chead.load_state_dict(state["chead"])
        self.l3 = L3RobotFlow(track_fn, cond_dim=384, use_cross_attn=use_cross_attn)
        self.l4 = ApproachPolicy(**policy_kwargs)
        self.lam, self.flow_noise = lam, flow_noise

    def forward_loss(self, batch):
        obs, track_obs, track, task_emb, actions, extra_states, \
            chain_query, chain_depth, chain_gt, s1 = batch
        b, v, t = track_obs.shape[:3]

        # ---- L1 + C head (demo-level, one forward per window) ----
        hm_l, z, fm = self.l1(s1["dino"], s1["prior"], s1["text"])
        yl, pl, wp = self.chead(z, fm, s1["contact_rowcol"])
        l_c, c_parts = stage1_loss(hm_l, z, yl, pl, wp, s1)

        # ---- L3 flow on agentview frames of every window step ----
        frames = rearrange(track_obs[:, AGENTVIEW], "b t fs c h w -> (b t) fs c h w")
        q = rearrange(chain_query, "b t n d -> (b t) n d")
        qd = rearrange(chain_depth, "b t n -> (b t) n")
        z_rep = z[:, None].expand(-1, t, -1).reshape(b * t, -1)
        pred = self.l3(frames, q, z_rep, qd)                       # (b t) tl n 2
        gt = rearrange(chain_gt, "b t tl n d -> (b t) tl n d")
        l_flow = torch.nn.functional.mse_loss(pred, gt)

        # ---- L4 on predicted flow (grad flows; D10 noise) ----
        flow = rearrange(pred, "(b t) tl n d -> b t tl n d", b=b, t=t)
        if self.training and self.flow_noise > 0:
            flow = flow + torch.randn_like(flow) * self.flow_noise
        self.l4.set_flow(flow)
        l_action, act_dict = self.l4.forward_loss(obs, track_obs, track, task_emb,
                                                  extra_states, actions)
        self.l4._flow_ctx = None

        lam = self.lam
        loss = lam[0] * l_c + lam[1] * l_flow + lam[2] * l_action
        parts = {"loss": loss.item(), "l_c": l_c.item(), "l_flow": l_flow.item(),
                 "l_action": float(act_dict["bc_loss"]),
                 "hm": c_parts["hm"], "ori": c_parts["ori"], "w": c_parts["w"]}
        return loss, parts
