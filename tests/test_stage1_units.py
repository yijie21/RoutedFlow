"""Unit tests for stage-1 (L1+L2) components. Pure CPU. Run via `run_stage1.py test`."""
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from routedflow.stage1.dataset import (GRID, HEATMAP_RES, N_PITCH, N_YAW, fold_split,
                                       gaussian_heatmap, orientation_targets,
                                       pool_mask_to_grid, quat_to_mat)
from routedflow.stage1.model import CHead, L1FrontEnd, stage1_loss


def test_heatmap_normalized_and_centered():
    h = gaussian_heatmap(300.0, 100.0)
    assert h.shape == (HEATMAP_RES, HEATMAP_RES)
    assert abs(h.sum() - 1.0) < 1e-5
    r, c = np.unravel_index(h.argmax(), h.shape)
    assert abs(r - 300 / 4) <= 1 and abs(c - 100 / 4) <= 1


def test_orientation_topdown_identity():
    # gripper pointing straight down (180 deg about x), base = identity
    q_down = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw
    base = np.array([0.0, 0.0, 0.0, 1.0])
    yb, pb, yaw, pitch = orientation_targets(q_down, base)
    assert pitch < 1e-6 and pb == 0


def test_yaw_pi_symmetry_fold():
    base = np.array([0.0, 0.0, 0.0, 1.0])
    q_down = np.array([1.0, 0.0, 0.0, 0.0])
    R = quat_to_mat(q_down)
    # rotate pi about the approach (site z) axis -> physically identical gripper
    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    R2 = R @ Rz
    # mat -> quat (xyzw) via trace-free branch helper
    def mat2quat(M):
        w = np.sqrt(max(0, 1 + M[0, 0] + M[1, 1] + M[2, 2])) / 2
        if w > 1e-6:
            return np.array([(M[2, 1] - M[1, 2]) / (4 * w), (M[0, 2] - M[2, 0]) / (4 * w),
                             (M[1, 0] - M[0, 1]) / (4 * w), w])
        # w ~ 0: pick largest diagonal (enough for this test's matrices)
        x = np.sqrt(max(0, 1 + M[0, 0] - M[1, 1] - M[2, 2])) / 2
        return np.array([x, (M[0, 1] + M[1, 0]) / (4 * x + 1e-12),
                         (M[0, 2] + M[2, 0]) / (4 * x + 1e-12),
                         (M[2, 1] - M[1, 2]) / (4 * x + 1e-12)])
    yb1, _, _, _ = orientation_targets(mat2quat(R), base)
    yb2, _, _, _ = orientation_targets(mat2quat(R2), base)
    assert yb1 == yb2, f"pi-fold broken: {yb1} vs {yb2}"


def test_pool_mask_full_coverage():
    m = np.ones((512, 512), np.uint8)
    g = pool_mask_to_grid(m)
    assert g.shape == (GRID, GRID) and np.allclose(g, 1.0, atol=1e-5)


def test_fold_split_partition():
    tasks = [f"t{i:02d}" for i in range(10)]
    seen = set()
    for k in range(5):
        tr, ood = fold_split(tasks, k)
        assert len(tr) == 8 and len(ood) == 2 and not set(tr) & set(ood)
        seen |= set(ood)
    assert seen == set(tasks)


def _rand_batch(B=2):
    return {
        "dino": torch.randn(B, GRID * GRID, 768),
        "prior": torch.rand(B, GRID, GRID),
        "text": torch.randn(B, 768),
        "heatmap": torch.softmax(torch.randn(B, HEATMAP_RES * HEATMAP_RES), -1)
                        .reshape(B, HEATMAP_RES, HEATMAP_RES),
        "yaw_bin": torch.randint(0, N_YAW, (B,)),
        "pitch_bin": torch.randint(0, N_PITCH, (B,)),
        "w": torch.rand(B),
        "contact_rowcol": torch.rand(B, 2) * 511,
    }


def test_model_shapes_and_loss():
    torch.manual_seed(0)
    l1, ch = L1FrontEnd(), CHead()
    b = _rand_batch()
    hm, z, fm = l1(b["dino"], b["prior"], b["text"])
    assert hm.shape == (2, 128, 128) and z.shape == (2, 384) and fm.shape == (2, 96, 128, 128)
    yl, pl, wp = ch(z, fm, b["contact_rowcol"])
    assert yl.shape == (2, N_YAW) and pl.shape == (2, N_PITCH) and wp.shape == (2,)
    loss, parts = stage1_loss(hm, z, yl, pl, wp, b)
    assert torch.isfinite(loss)
    loss.backward()  # gradient path intact through both modules


def test_prior_dropout_zeroes_channel():
    torch.manual_seed(0)
    l1 = L1FrontEnd(prior_dropout=1.0).train()
    b = _rand_batch()
    with torch.no_grad():
        hm1, _, _ = l1(b["dino"], b["prior"], b["text"])
        hm2, _, _ = l1(b["dino"], torch.zeros_like(b["prior"]), b["text"])
    assert torch.allclose(hm1, hm2, atol=1e-5), "p=1.0 dropout must equal zeroed prior"


def test_gather_feat_delta():
    fm = torch.zeros(1, 3, 128, 128)
    fm[0, :, 40, 100] = torch.tensor([1.0, 2.0, 3.0])
    # pixel (40,100)@128 corresponds to (~160,~400)@512 under align_corners=True mapping
    rc512 = torch.tensor([[40 * 511.0 / 127.0, 100 * 511.0 / 127.0]])
    f = CHead.gather_feat(fm, rc512)
    assert torch.allclose(f, torch.tensor([[1.0, 2.0, 3.0]]), atol=1e-4)


def test_axis_convention_against_data():
    """Data-anchored check (mask-bug lesson): LIBERO grasps are mostly top-down,
    so the derived pitch must be small for most demos. Skips if labels absent/locked."""
    import glob

    import h5py
    files = sorted(glob.glob(os.path.join(REPO, "data", "c_labels", "libero_spatial", "*.h5")))
    if not files:
        return
    try:
        with h5py.File(files[0], "r") as f:
            base = np.array(f.attrs["robot_base_quat"])
            pitches = []
            for k in sorted(x for x in f.keys() if x.startswith("demo"))[:50]:
                q = np.array(f[k]["ee_quat"])[int(f[k].attrs["t_g"])]
                _, _, _, pitch = orientation_targets(q, base)
                pitches.append(pitch)
    except (OSError, BlockingIOError):
        return  # file locked by a running job — covered when run standalone
    frac_topdown = float(np.mean(np.array(pitches) < np.deg2rad(30)))
    assert frac_topdown > 0.8, f"axis convention suspect: only {frac_topdown:.0%} top-down"


def test_flow_skeleton_injection_blocks():
    """L3/L5 skeleton: injection blocks are shape-correct and identity at init."""
    from routedflow.flow_models import (CondAdaLN, CondCrossAttn,
                                        ConditionedTrackTransformer, QueryDepthEmbed)
    torch.manual_seed(0)
    tokens, cond = torch.randn(2, 50, 384), torch.randn(2, 384)
    for blk in (CondCrossAttn(384, 384), ):
        out = blk(tokens, cond)
        assert out.shape == tokens.shape
        assert torch.allclose(out, tokens, atol=1e-5), "zero-init must be identity"
    ada = CondAdaLN(384, 384)
    out = ada(tokens, cond)
    assert out.shape == tokens.shape
    assert torch.allclose(out, torch.nn.functional.layer_norm(tokens, (384,)), atol=1e-5)
    qe = QueryDepthEmbed(384)
    q = torch.randn(2, 32, 384)
    assert torch.allclose(qe(q, torch.rand(2, 32, 1)), q, atol=1e-6)
    # role/text asymmetry is enforced at the interface
    l3 = ConditionedTrackTransformer(role="robot")
    try:
        l3(None, None, None, cond=None, text_emb=torch.randn(2, 768))
        assert False, "L3 must reject text"
    except AssertionError as e:
        assert "MUST NOT see text" in str(e)


def test_grasp_cycles_debounce_and_fumble():
    from routedflow.phase import grasp_cycles

    def acts(g):
        return np.stack([np.zeros(len(g)), np.array(g, float)], 1)

    assert grasp_cycles(acts([-1] * 5 + [1] * 10 + [-1] * 5)) == [(5, 15)]
    # 2-frame close blip (dither) is ignored
    assert grasp_cycles(acts([-1] * 5 + [1, 1] + [-1] * 5 + [1] * 10)) == [(12, 22)]
    # 2-frame open dip inside a grasp is ignored
    assert grasp_cycles(acts([-1] * 5 + [1] * 6 + [-1, -1] + [1] * 8)) == [(5, 21)]
    # fumble: close-open-close -> two cycles; t_g rule v2 takes the LAST close
    c = grasp_cycles(acts([-1] * 5 + [1] * 6 + [-1] * 6 + [1] * 10))
    assert c == [(5, 11), (17, 27)]
    assert grasp_cycles(acts([-1] * 10)) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} stage-1 unit tests passed")
