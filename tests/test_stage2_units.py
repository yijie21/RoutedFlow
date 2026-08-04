"""Unit tests for stage-2 (joint approach branch). Run via `run_stage2.py test`."""
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")):
    sys.path.insert(0, p)

TRACK_FN = os.path.join(REPO, "third_party", "ATM", "results", "track_transformer",
                        "libero_track_transformer_libero-spatial")


def test_chain_weights_identity_fixed():
    from routedflow.stage2.chain_points import chain_points_3d, polyline_weights
    P0 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float)
    w = polyline_weights(P0, 5)
    pts0 = chain_points_3d(P0, [0, 1, 2], [], w, [])
    assert np.allclose(pts0[0], [0, 0, 0]) and np.allclose(pts0[-1], [1, 1, 0])
    # same weights applied to a MOVED polyline -> points move rigidly with it
    P1 = P0 + np.array([2.0, 0.0, 1.0])
    pts1 = chain_points_3d(P1, [0, 1, 2], [], w, [])
    assert np.allclose(pts1 - pts0, [2.0, 0.0, 1.0])


def test_l3_shapes_grad_and_depth_noop():
    from routedflow.stage2.flow_l3 import L3RobotFlow
    torch.manual_seed(0)
    l3 = L3RobotFlow(TRACK_FN)
    B, n = 2, 32
    vid = torch.randint(0, 255, (B, 1, 3, 128, 128)).float()
    q = torch.rand(B, n, 2)
    z = torch.randn(B, 384, requires_grad=True)
    d = torch.rand(B, n)
    out = l3(vid, q, z, d)
    assert out.shape == (B, l3.num_track_ts, n, 2)
    # z-slot conditioning must carry gradient (the causal artery)
    out.sum().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0
    # depth MLP is zero-init: depth changes nothing at init (eval: kill dropout noise)
    l3.eval()
    with torch.no_grad():
        o1 = l3(vid, q, z.detach(), d)
        o2 = l3(vid, q, z.detach(), torch.zeros_like(d))
    assert torch.allclose(o1, o2, atol=1e-6)


def test_l4_flow_ctx_gradient():
    from omegaconf import OmegaConf

    from routedflow.stage2.joint_model import ApproachPolicy
    cfg = OmegaConf.load(os.path.join(REPO, "experiments", "stage0_routing_causal_test",
                                      "configs", "stage0_vilt.yaml")).model_cfg
    cfg.pop("track_gate_cfg", None)
    pol = ApproachPolicy(**cfg)
    b, v, t, tl, n = 1, 2, 2, 16, 32
    track_obs = torch.randint(0, 255, (b, v, t, 1, 3, 128, 128)).float()
    task_emb = torch.randn(b, 768)
    flow = torch.rand(b, t, tl, n, 2, requires_grad=True)
    pol.set_flow(flow)
    tr, recon = pol.track_encode(track_obs, task_emb)
    assert recon.shape[:3] == (b, v, t)
    tr.sum().backward()
    assert flow.grad is not None and float(flow.grad.abs().sum()) > 0
    # wrist view slot must be zeros
    assert float(recon[:, 1].abs().sum()) == 0.0
    # task_emb removed from the approach-branch interface (2026-08-03):
    # act takes (obs, extra_states) only and runs on the zero-text plumbing
    import inspect
    assert "task_emb" not in inspect.signature(pol.act).parameters
    assert "task_emb" not in inspect.signature(pol.forward_loss).parameters
    pol.reset()
    pol.set_flow(torch.rand(b, 1, tl, n, 2))
    obs_np = np.random.randint(0, 255, (b, v, 128, 128, 3)).astype(np.uint8)
    extra = {"joint_states": np.random.randn(b, 7).astype(np.float32),
             "gripper_states": np.random.randn(b, 2).astype(np.float32)}
    a, _ = pol.act(obs_np, extra)
    assert a.shape == (b, 7)


def test_z_to_l4_language_slot():
    """方案A: z rides L4's language slots — gradient reaches z, output depends on z."""
    from omegaconf import OmegaConf

    from routedflow.stage2.joint_model import ApproachPolicy
    cfg = OmegaConf.load(os.path.join(REPO, "experiments", "stage0_routing_causal_test",
                                      "configs", "stage0_vilt.yaml")).model_cfg
    cfg.pop("track_gate_cfg", None)
    cfg.language_encoder_cfg.input_size = 384
    cfg.spatial_transformer_cfg.use_language_token = True
    cfg.temporal_transformer_cfg.use_language_token = True
    torch.manual_seed(0)
    pol = ApproachPolicy(use_z=True, **cfg)
    b, v, t, tl, n = 1, 2, 2, 16, 32
    obs = torch.rand(b, v, t, 3, 128, 128)
    track_obs = torch.randint(0, 255, (b, v, t, 1, 3, 128, 128)).float()
    extra = {"joint_states": torch.randn(b, t, 7),
             "gripper_states": torch.randn(b, t, 2)}
    # gradient artery: L_action-side output must backprop into z
    pol.set_flow(torch.rand(b, t, tl, n, 2))
    z = torch.randn(b, 384, requires_grad=True)
    out = pol.spatial_encode(obs, track_obs, z, extra, return_recon=False)
    out.sum().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0
    # structural consumption: with everything else fixed, changing z changes the action
    pol.eval()
    flow = torch.rand(b, 1, tl, n, 2)
    obs_np = np.random.randint(0, 255, (b, v, 128, 128, 3)).astype(np.uint8)
    extra_np = {"joint_states": np.random.randn(b, 7).astype(np.float32),
                "gripper_states": np.random.randn(b, 2).astype(np.float32)}
    pol.reset()
    pol.set_flow(flow)
    a1, _ = pol.act(obs_np, extra_np, z=np.zeros((b, 384), np.float32))
    pol.reset()
    pol.set_flow(flow)
    a2, _ = pol.act(obs_np, extra_np, z=np.full((b, 384), 3.0, np.float32))
    assert not np.allclose(a1, a2), "z must be load-bearing in the 方案A wiring"
    # interface: still no task_emb anywhere in the approach-branch API
    import inspect
    assert "task_emb" not in inspect.signature(pol.act).parameters
    assert "z" in inspect.signature(pol.forward_loss).parameters


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} stage-2 unit tests passed")
