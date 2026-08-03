"""Latched contact-phase signal from gripper actions.

Phase semantics (design doc `PHASE_GATED_FLOW_ROUTING.md`): approach (0) before the
gripper first closes, transport (1) from the first closing command onward. The
signal is LATCHED — it never returns to 0 after the first closure, even if the
gripper later opens to release (post-release frames are treated as transport;
LIBERO demos end shortly after release, so this tail is small). This matches how
C freezes at t_g.
"""
import numpy as np


def grasp_cycles(actions, debounce=3):
    """Debounced gripper close/open cycles: [(t_close, t_open), ...] (t_open=T if unreleased).

    A state transition only counts when the new gripper state persists >= debounce
    frames (the same robust-latch lesson as the rollout D6 fix — raw >0 edges count
    dither blips as cycles). Multi-cycle demos = regrasp fumbles (single-atom tasks)
    or genuine multi-grasp long-horizon demos (libero_10). Measured 2026-08-02:
    spatial 50/500, object 58/500, goal 20/500 multi-cycle; libero_10 356/500.
    """
    a = np.asarray(actions)[:, -1] > 0
    T = len(a)
    runs, s = [], 0
    for t in range(1, T + 1):
        if t == T or a[t] != a[s]:
            runs.append((bool(a[s]), s, t - s))
            s = t
    cycles, state = [], False
    for val, start, ln in runs:
        if ln < debounce and start + ln < T:
            continue  # blip — no state change
        if val and not state:
            cycles.append([start, T])
            state = True
        elif (not val) and state:
            cycles[-1][1] = start
            state = False
    return [(int(c), int(o)) for c, o in cycles]


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
