"""The fixed 32-point query grid used by ATM's BC policy, and robot-label lookup.

Replicates `atm.utils.flow_utils.sample_double_grid(4)` exactly (two offset 4x4
grids, normalized coords). Coordinate convention follows ATM's stored tracks
(`scripts/preprocess_libero.py` L171-172): dim0 = x / W (width), dim1 = y / H
(height). The sanity-check overlay produced by the converter is the ground truth
for this convention — if the overlay shows transposed labels, flip `coord_order`.
"""
import numpy as np
import torch


def double_grid_32(dtype=torch.float32):
    """Return the (32, 2) normalized query grid, identical to ATM's sample_double_grid(4)."""

    def _grid(left, right):
        u = torch.linspace(left, right, 4, dtype=dtype)
        v = torch.linspace(left, right, 4, dtype=dtype)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)

    return torch.cat([_grid(0.05, 0.85), _grid(0.15, 0.95)], dim=0)


def grid_pixel_indices(height, width, coord_order="xy", grid=None):
    """Map the 32 grid points to integer pixel (row, col) indices for an HxW image."""
    g = double_grid_32().numpy() if grid is None else np.asarray(grid)
    if coord_order == "xy":
        cols = np.clip(np.round(g[:, 0] * (width - 1)).astype(int), 0, width - 1)
        rows = np.clip(np.round(g[:, 1] * (height - 1)).astype(int), 0, height - 1)
    elif coord_order == "yx":
        rows = np.clip(np.round(g[:, 0] * (height - 1)).astype(int), 0, height - 1)
        cols = np.clip(np.round(g[:, 1] * (width - 1)).astype(int), 0, width - 1)
    else:
        raise ValueError(f"unknown coord_order {coord_order}")
    return rows, cols


def grid_robot_labels(robot_mask, coord_order="xy"):
    """Label each of the 32 grid points as robot (True) / non-robot (False).

    robot_mask: (..., H, W) bool/uint8 numpy array. Returns (..., 32) bool.
    """
    robot_mask = np.asarray(robot_mask)
    h, w = robot_mask.shape[-2:]
    rows, cols = grid_pixel_indices(h, w, coord_order=coord_order)
    return robot_mask[..., rows, cols].astype(bool)
