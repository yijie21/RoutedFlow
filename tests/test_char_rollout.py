"""Characterization tests: rollout eval chain (retrofit target #1).

Pin the CURRENT behavior of eval_env2/eval_rollout — no judgement, only "any
change moves a pinned number". Values recorded 2026-08-02 on this box (task 0,
first init state, deterministic). Spec: .scratch/retrofit-rollout-chain/spec.md.
Slow (~1 min: builds one 512^2 LIBERO env, needs GPU/EGL). Run via
`run_stage2.py char-env`; the pure-helper tests run without an env.
"""
import os
import sys

os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/code/ATM/.libero")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")):
    sys.path.insert(0, p)

TASK0 = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"


def test_pure_helpers():
    from routedflow.stage2.eval_rollout import LIFT_ACTION, short, watermark
    assert np.allclose(LIFT_ACTION, [0, 0, 0.9, 0, 0, 0, 1.0])  # +z, gripper close
    assert short("pick_up_the_black_bowl_between_the_plate_and_the_ramekin"
                 "_and_place_it_on_the_plate_demo") == "between_the_plate_and_the_ramekin"
    fr = np.full((256, 256, 3), 200, np.uint8)
    out = watermark(fr, "POLICY ROLLOUT", "approach")
    assert out.shape == fr.shape and out.dtype == np.uint8
    assert out[:26].mean() < 60 < 190 < out[30:].mean()   # burned-in top banner


def test_env_contract():
    """Reset obs contract of the rollout env stack (vec=1, first init state)."""
    from routedflow.stage2.eval_env2 import build_env_chain
    env, task_emb = build_env_chain("libero_spatial", TASK0, img_size=512,
                                    gpu_id=0, vec_env_num=1)
    try:
        obs = env.reset()
        img = obs["agentview_image"]
        assert img.shape == (1, 512, 512, 3) and img.dtype == np.uint8
        assert obs["robot0_eye_in_hand_image"].shape == (1, 512, 512, 3)
        # upright anchor: table (bright) at the BOTTOM of the frame
        assert img[0, -100:].mean() > img[0, :100].mean() + 20
        assert obs["robot0_joint_pos"].shape == (1, 7)
        assert obs["robot0_gripper_qpos"].shape == (1, 2)
        assert np.allclose(np.abs(obs["robot0_gripper_qpos"][0]), 0.0339, atol=2e-3)
        # FK chain block (ChainStateWrapper)
        uv, zz, eez = obs["chain_uv"], obs["chain_z"], obs["ee_z"]
        assert uv.shape == (1, 32, 2) and zz.shape == (1, 32) and eez.shape == (1, 1)
        assert np.isfinite(uv).all() and np.isfinite(zz).all()
        assert abs(float(eez[0, 0]) - 1.1743) < 5e-3                  # home-pose EE height
        assert float(zz.min()) > 0.5 and float(zz.max()) < 2.0        # camera-frame depth (m)
        # cross-anchor vs the OFFLINE label pipeline (c_labels chain_uv t=0 stats
        # 0.4988/-0.0288, 0.0217/0.2182): two independent projection paths agree
        assert np.allclose(uv[0].mean(0), [0.4988, -0.0288], atol=0.02)
        assert np.allclose(uv[0].std(0), [0.0217, 0.2182], atol=0.02)
        # zero action -> chain barely moves
        obs2, _, _, _ = env.step(np.zeros((1, 7), np.float32))
        assert float(np.abs(obs2["chain_uv"] - uv).max()) < 0.02
    finally:
        env.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} rollout characterization tests passed")
