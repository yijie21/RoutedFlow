"""Characterization tests: label-extraction chain (retrofit target #2).

Pin the CURRENT contents of data/c_labels + data/atm_libero_light — no judgement
about correctness, only "any change to these numbers must be noticed". Snapshot
values recorded 2026-08-02 from the data that trained joint_fold0_seed0.
Spec: .scratch/retrofit-label-extraction/spec.md. Run via `run_stage2.py char`.
Pure CPU / h5py; skips (rc 0, SKIP printed) if the data dirs are absent.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
C_LABELS = os.path.join(REPO, "data", "c_labels", "libero_spatial")
LIGHT = os.path.join(REPO, "data", "atm_libero_light", "libero_spatial")

TASK0 = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo"


def _h5py():
    import h5py
    return h5py


def test_sweep_contact_always_in_view_and_finite():
    """Spec §5-1: the unguarded out-of-view path is never taken in current data."""
    h5py = _h5py()
    tasks = sorted(f for f in os.listdir(C_LABELS) if f.endswith(".h5"))
    assert len(tasks) == 10
    total = 0
    for tf in tasks:
        with h5py.File(os.path.join(C_LABELS, tf), "r") as f:
            for k in (x for x in f.keys() if x.startswith("demo")):
                total += 1
                rc = np.array(f[k]["contact_rowcol"])
                assert 0 <= rc[0] < 512 and 0 <= rc[1] < 512, (tf, k, rc)
                assert np.isfinite(np.array(f[k]["contact_depth0"])), (tf, k)
    assert total == 500, total


def test_task_file_contract():
    h5py = _h5py()
    with h5py.File(os.path.join(C_LABELS, TASK0 + ".h5"), "r") as f:
        assert int(f.attrs["res"]) == 512 and f.attrs["camera"] == "agentview"
        K = np.array(f["K"])
        assert np.allclose(K, [[618.04, 0, 256], [0, 618.04, 256], [0, 0, 1]], atol=0.01)
        assert np.array(f["robot_geom_ids"]).dtype == np.int32
        assert len(json.loads(f["link_names"][()])) == 17


def test_demo0_snapshot():
    h5py = _h5py()
    with h5py.File(os.path.join(C_LABELS, TASK0 + ".h5"), "r") as f:
        g = f["demo_0"]
        assert (int(g.attrs["T"]), int(g.attrs["t_g"])) == (98, 36)
        assert np.allclose(np.array(g["contact_rowcol"]), [304.203, 369.189], atol=0.01)
        assert abs(float(np.array(g["contact_depth0"])) - 1.010859) < 1e-4
        assert np.allclose(np.array(g["ee_pos"])[36], [-0.0378, 0.1750, 0.9522], atol=1e-3)
        assert np.allclose(np.array(g["gripper_q"])[0], [0.0362, -0.0362], atol=1e-3)
        eqn = np.linalg.norm(np.array(g["ee_quat"]), axis=1)
        assert np.allclose(eqn, 1.0, atol=1e-4)              # xyzw, unit
        ph = np.array(g["phase"])
        assert ph.dtype == np.uint8 and np.all(np.diff(ph.astype(int)) >= 0)  # latched
        assert ph[35] == 0 and ph[36] == 1                    # flips exactly at t_g
        assert g["rgb0"].shape == (512, 512, 3) and g["rgb0"].dtype == np.uint8
        assert g["link_pos"].shape == (98, 17, 3)


def test_demo0_chain_snapshot():
    """chain_uv is NOT clamped to [0,1]: at t=0 roughly half the arm is above the
    frame (y<0). Pin the actual measured range + t=0 stats + robot-mask anchor
    (the numeric-anchor lesson from the 2026-07-30 mask-orientation bug)."""
    h5py = _h5py()
    with h5py.File(os.path.join(C_LABELS, TASK0 + ".h5"), "r") as f:
        g = f["demo_0"]
        uv, zz = np.array(g["chain_uv"]), np.array(g["chain_z"])
        assert uv.shape == (98, 32, 2) and zz.shape == (98, 32)
        assert np.allclose(uv[0].mean(0), [0.4988, -0.0288], atol=1e-3)
        assert np.allclose(uv[0].std(0), [0.0217, 0.2182], atol=1e-3)
        assert abs(uv.min() - (-0.4061)) < 1e-2 and abs(uv.max() - 0.7844) < 1e-2
        assert abs(zz.min() - 0.6901) < 1e-2 and abs(zz.max() - 1.4646) < 1e-2
        rob = np.isin(np.array(g["seg0"]), np.array(f["robot_geom_ids"]))
        cols, rows = (uv[0, :, 0] * 512).astype(int), (uv[0, :, 1] * 512).astype(int)
        ok = (rows >= 0) & (rows < 512) & (cols >= 0) & (cols < 512)
        assert ok.sum() >= 10                                 # some points on-screen at t=0
        assert rob[rows[ok], cols[ok]].mean() >= 0.95         # projection/orientation anchor


def test_light_file_consistency():
    h5py = _h5py()
    lt = os.path.join(LIGHT, TASK0, "all", "demo_0.hdf5")
    with h5py.File(lt, "r") as f, h5py.File(os.path.join(C_LABELS, TASK0 + ".h5"), "r") as c:
        v = f["root/agentview/video"]
        assert v.shape == (1, 98, 3, 128, 128) and v.dtype == np.uint8
        assert f["root/actions"].shape == (98, 7)
        assert np.array_equal(np.array(f["root/phase"]), np.array(c["demo_0"]["phase"]))
        assert set(f["root/extra_states"].keys()) == \
            {"ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"}
        assert abs(float(np.array(v[0, 36]).mean()) - 118.376) < 0.5  # pixel-level pin @ t_g


if __name__ == "__main__":
    if not (os.path.isdir(C_LABELS) and os.path.isdir(LIGHT)):
        print("SKIP: data/c_labels or data/atm_libero_light not present")
        sys.exit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} label characterization tests passed")
