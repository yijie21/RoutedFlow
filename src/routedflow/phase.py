"""Latched contact-phase signal from gripper actions.

Phase semantics (design doc `PHASE_GATED_FLOW_ROUTING.md`): approach (0) before the
gripper first closes, transport (1) from the first closing command onward. The
signal is LATCHED — it never returns to 0 after the first closure, even if the
gripper later opens to release (post-release frames are treated as transport;
LIBERO demos end shortly after release, so this tail is small). This matches how
C freezes at t_g.
"""
import numpy as np


def latched_phase(actions):
    """actions: (T, A) with gripper command in the last dim (>0 = close).

    Returns (T,) uint8: 0 = approach, 1 = transport (latched at first closure).
    """
    actions = np.asarray(actions)
    close = actions[:, -1] > 0
    phase = np.zeros(len(close), dtype=np.uint8)
    if close.any():
        phase[np.argmax(close):] = 1
    return phase
