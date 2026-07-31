"""Track gating: the single intervention that defines the Stage-0 variants.

Operates on the predicted tracks returned by the frozen track transformer inside
`BCViLTPolicy.track_encode` (third_party/ATM/atm/policy/vilt.py, after the
`reconstruct` call). Excluded points are flattened to "stationary at the query
grid location" (zero displacement) — closer to in-distribution than zeroing the
coordinates (stationary background points occur naturally), and applied
identically at train and eval time (lesson from the freeze-intervention failure:
never create a train/eval mismatch).

Pure tensor ops, no ATM imports — unit-testable standalone.
"""
import torch

VALID_MODES = ("none", "object_only", "robot_only", "phase_switched")


def keep_mask(robot_labels, phase, mode):
    """Which of the 32 tracks keep their information.

    robot_labels: (b, v, t, n) bool — grid point lies on the robot at that frame.
    phase:        (b, t) bool — False = approach, True = transport (latched).
    Returns (b, v, t, n) bool.
    """
    if mode == "none":
        return torch.ones_like(robot_labels, dtype=torch.bool)
    if mode == "robot_only":
        return robot_labels.bool()
    if mode == "object_only":
        return ~robot_labels.bool()
    if mode == "phase_switched":
        ph = phase.bool()[:, None, :, None]  # (b, 1, t, 1)
        return torch.where(ph, ~robot_labels.bool(), robot_labels.bool())
    raise ValueError(f"unknown gate mode {mode!r}, valid: {VALID_MODES}")


def gate_tracks(recon_tr, grid_points, robot_labels, phase, mode):
    """Apply the gate to predicted tracks.

    recon_tr:    (b, v, t, tl, n, 2) predicted tracks in normalized coords.
    grid_points: (n, 2) the fixed query grid.
    robot_labels: (b, v, t, n) bool. phase: (b, t) bool.
    Returns a gated copy; excluded tracks become the grid point repeated tl times.
    """
    if mode == "none":
        return recon_tr
    keep = keep_mask(robot_labels, phase, mode)  # (b, v, t, n)
    stationary = grid_points.to(dtype=recon_tr.dtype, device=recon_tr.device)
    stationary = stationary[None, None, None, None, :, :].expand_as(recon_tr)
    keep = keep[:, :, :, None, :, None]  # (b, v, t, 1, n, 1)
    return torch.where(keep, recon_tr, stationary)
