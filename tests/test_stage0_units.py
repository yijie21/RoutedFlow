"""Unit tests for the Stage-0 building blocks (pure CPU, no data needed)."""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "third_party", "ATM"))

from routedflow.grid import double_grid_32, grid_robot_labels
from routedflow.phase import latched_phase
from routedflow.track_gate import gate_tracks, keep_mask


def test_grid_matches_atm():
    from atm.utils.flow_utils import sample_double_grid
    ours = double_grid_32()
    theirs = sample_double_grid(4, device="cpu")
    assert torch.allclose(ours, theirs), "grid must replicate ATM's sample_double_grid(4) exactly"


def test_grid_labels_lookup():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[:, 64:] = 1  # right half is "robot"
    labels = grid_robot_labels(mask)
    g = double_grid_32().numpy()
    # convention: dim0 = x (width). Right-half points must be labeled robot.
    assert labels.shape == (32,)
    assert (labels == (np.round(g[:, 0] * 127) >= 64)).all()


def test_phase_latch():
    a = np.zeros((10, 7)); a[:, -1] = -1
    assert latched_phase(a).sum() == 0                       # never closes
    a[4:, -1] = 1
    assert (latched_phase(a) == [0]*4 + [1]*6).all()         # latch at first close
    a[7:, -1] = -1                                           # opens again (release)
    assert (latched_phase(a) == [0]*4 + [1]*6).all()         # stays latched


def _rand_inputs(b=2, v=2, t=3, tl=16, n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    recon = torch.rand((b, v, t, tl, n, 2), generator=g)
    labels = torch.rand((b, v, t, n), generator=g) > 0.7
    phase = torch.tensor([[False, False, True], [True, True, True]])
    grid = double_grid_32()
    return recon, labels, phase, grid


def test_gate_none_is_identity():
    recon, labels, phase, grid = _rand_inputs()
    out = gate_tracks(recon, grid, labels, phase, "none")
    assert torch.equal(out, recon)


def test_gate_complementarity():
    recon, labels, phase, grid = _rand_inputs()
    ko = keep_mask(labels, phase, "object_only")
    kr = keep_mask(labels, phase, "robot_only")
    assert torch.equal(ko, ~kr), "object_only and robot_only must partition the 32 points"


def test_gate_phase_switched_semantics():
    recon, labels, phase, grid = _rand_inputs()
    kps = keep_mask(labels, phase, "phase_switched")
    kr = keep_mask(labels, phase, "robot_only")
    ko = keep_mask(labels, phase, "object_only")
    for bi in range(phase.shape[0]):
        for ti in range(phase.shape[1]):
            expected = ko[bi, :, ti] if phase[bi, ti] else kr[bi, :, ti]
            assert torch.equal(kps[bi, :, ti], expected)


def test_gate_replaces_excluded_with_stationary_grid():
    recon, labels, phase, grid = _rand_inputs()
    out = gate_tracks(recon, grid, labels, phase, "robot_only")
    keep = keep_mask(labels, phase, "robot_only")
    stationary = grid[None, None, None, None, :, :].expand_as(recon)
    kept = keep[:, :, :, None, :, None].expand_as(recon)
    assert torch.equal(out[kept], recon[kept]), "kept tracks must be untouched"
    assert torch.equal(out[~kept], stationary[~kept]), "excluded tracks must sit at the grid point"
    # stationarity: zero displacement across the track-length dim
    excl = ~keep
    if excl.any():
        b, v, t, n = torch.nonzero(excl, as_tuple=True)
        tr = out[b, v, t, :, n, :]  # (k, tl, 2)
        assert (tr - tr[:, :1]).abs().max() == 0


def test_gate_shape_and_dtype_preserved():
    recon, labels, phase, grid = _rand_inputs()
    for mode in ("object_only", "robot_only", "phase_switched"):
        out = gate_tracks(recon, grid, labels, phase, mode)
        assert out.shape == recon.shape and out.dtype == recon.dtype
