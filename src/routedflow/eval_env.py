"""Eval-side env building with online robot-mask grid labels.

The label computation must happen INSIDE each sub-env (subprocess) where the
mujoco sim lives. `StackSubprocVectorEnv.merge_dict` stacks every obs key across
sub-envs, so a `robot_grid_labels` key added here arrives in the policy loop as
(b, v, 32) for free.

Orientation (fixed 2026-07-30): the policy sees images flipped by ATM's
LiberoImageUpsideDownWrapper (final = raw[::-1], upright). robosuite's
get_camera_segmentation ALREADY returns upright (it applies [::-1] internally,
unlike raw sim.render) — so the segmentation is in the final orientation as-is
and must NOT be flipped. The old auto-detect flipped it (assuming seg shared the
raw render orientation), inverting every eval-time mask.

`make_libero_env_masked` / `build_env_masked` are copies of
`libero.utils.env_utils.make_libero_env` / `atm.utils.env_utils.build_env`
(libero branch) with the wrapper inserted — keep in sync with upstream.
"""
import os
import time
from functools import partial

import numpy as np
import torch
from robosuite.wrappers import Wrapper

from routedflow.grid import grid_robot_labels

CAMERA_TO_OBS_KEY = {"agentview": "agentview_image", "robot0_eye_in_hand": "robot0_eye_in_hand_image"}
ROBOT_BODY_PREFIXES = ("robot0", "gripper0", "mount0")


class RobotGridLabelWrapper(Wrapper):
    """Sub-env wrapper: adds obs["robot_grid_labels"] (v, 32) uint8, cameras sorted."""

    def __init__(self, env, img_hw=128, cameras=("agentview", "robot0_eye_in_hand")):
        super().__init__(env)
        self.img_hw = img_hw
        self.cameras = sorted(cameras)
        self._robot_ids = None

    def _robot_geom_ids(self):
        model = self.env.sim.model
        robot_bodies = {bid for bid in range(model.nbody)
                        if (model.body_id2name(bid) or "").startswith(ROBOT_BODY_PREFIXES)}
        ids = [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in robot_bodies]
        assert ids, "no robot geoms found"
        return np.array(ids)

    def _labels(self, obs):
        from robosuite.utils.camera_utils import get_camera_segmentation

        sim = self.env.sim
        if self._robot_ids is None:
            self._robot_ids = self._robot_geom_ids()
        out = []
        for cam in self.cameras:
            # already upright == final orientation; no flip (see module docstring)
            seg = get_camera_segmentation(sim, cam, self.img_hw, self.img_hw)
            mask = np.isin(seg[..., -1], self._robot_ids)
            out.append(grid_robot_labels(mask))
        return np.stack(out, axis=0).astype(np.uint8)  # (v, 32)

    def reset(self):
        obs = self.env.reset()
        self._robot_ids = None  # model can change across resets; recompute lazily
        obs["robot_grid_labels"] = self._labels(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["robot_grid_labels"] = self._labels(obs)
        return obs, reward, done, info


def make_libero_env_masked(task_suite_name, task_name, img_h, img_w, task_embedding=None,
                           gpu_id=-1, vec_env_num=1, seed=0):
    """Copy of libero.utils.env_utils.make_libero_env + RobotGridLabelWrapper (outermost)."""
    from easydict import EasyDict  # noqa: F401
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv
    from libero.utils.env_utils import (LiberoResetWrapper, LiberoTaskEmbWrapper,
                                        StackDummyVectorEnv, StackSubprocVectorEnv)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    descriptions = [task.language for task in task_suite.tasks]
    if task_embedding is None:
        task_embedding_map = np.load(os.path.join(get_libero_path("task_embeddings"), "task_emb_bert.npy"),
                                     allow_pickle=True).item()
        task_embs = torch.from_numpy(np.stack([task_embedding_map[des] for des in descriptions]))
    else:
        task_embs = task_embedding
    task_suite.set_task_embs(task_embs)

    task_id = task_suite.get_task_id(task_name)
    task = task_suite.get_task_from_name(task_name)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": img_h,
        "camera_widths": img_w,
        "render_gpu_device_id": gpu_id,
    }

    init_states = task_suite.get_task_init_states(task_id)
    assert len(init_states) % vec_env_num == 0, "number of initial states must be divisible by vec_env_num"
    num_states_per_env = len(init_states) // vec_env_num

    def env_func(env_idx):
        base_env = OffScreenRenderEnv(**env_args)
        base_env = LiberoResetWrapper(base_env, init_states=init_states[env_idx * num_states_per_env:(env_idx + 1) * num_states_per_env])
        base_env = LiberoTaskEmbWrapper(base_env, task_emb=task_suite.get_task_emb(task_id))
        base_env = RobotGridLabelWrapper(base_env, img_hw=img_h)  # outermost: labels on the post-init-state obs
        base_env.seed(seed)
        return base_env

    env, count = None, 0
    while env is None and count < 5:
        try:
            cls = StackDummyVectorEnv if vec_env_num == 1 else StackSubprocVectorEnv
            env = cls([partial(env_func, env_idx=i) for i in range(vec_env_num)])
        except Exception:
            time.sleep(5)
            count += 1
    if env is None:
        raise Exception("Failed to create environment")
    return env, task_embs


def build_env_masked(suite_name, task_name, img_size=128, gpu_id=0, vec_env_num=10, seed=0):
    """One task -> the full wrapper chain ATM's rollout expects, with grid labels in obs."""
    from atm.utils.env_utils import (LiberoImageUpsideDownWrapper, LiberoObservationWrapper,
                                     LiberoSuccessWrapper)

    env, _ = make_libero_env_masked(suite_name, task_name, img_size, img_size,
                                    gpu_id=gpu_id, vec_env_num=vec_env_num, seed=seed)
    env = LiberoImageUpsideDownWrapper(env)
    env = LiberoSuccessWrapper(env)
    cameras = sorted(CAMERA_TO_OBS_KEY.keys())
    env = LiberoObservationWrapper(env, masks=None, cameras=cameras)
    return env
