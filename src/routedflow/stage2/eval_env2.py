"""Rollout env for the approach branch: FK chain points computed INSIDE sub-envs.

ChainStateWrapper (sub-env level, lives in the subprocess with the sim) adds:
    obs["chain_uv"] (32,2)  normalized xy of the FK chain points (D11)
    obs["chain_z"]  (32,)   camera-frame depth
    obs["ee_z"]     (1,)    EE height (lift verification)
(No extra rgb key: the C-VLM reset frame comes from the regular obs stream —
the wrapper itself never renders; see the viewport note in reset().)

make_libero_env_chain / build_env_chain mirror routedflow.eval_env's copies of
the upstream builders (keep in sync), swapping in this wrapper.
"""
import os
import time
from functools import partial

import numpy as np
import torch
from robosuite.wrappers import Wrapper

from routedflow.stage2.chain_points import BRANCHES, MAIN, N_BRANCH, N_MAIN, \
    chain_points_3d, polyline_weights

CAM = "agentview"


class ChainStateWrapper(Wrapper):
    def __init__(self, env, img_hw=128):
        super().__init__(env)
        self.img_hw = img_hw
        self._ids = None

    def _setup(self):
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, \
            get_camera_transform_matrix
        sim = self.env.sim
        m = sim.model
        name2id = {m.body_id2name(i): i for i in range(m.nbody)}
        self._main = [name2id[n] for n in MAIN]
        self._brs = [[name2id[n] for n in b] for b in BRANCHES]
        self._site = m.site_name2id("gripper0_grip_site")
        S = self.img_hw
        self._w2p = get_camera_transform_matrix(sim, CAM, S, S)
        self._ext_inv = np.linalg.inv(get_camera_extrinsic_matrix(sim, CAM))
        self._S = S
        self._w = None  # chain weights, set on reset

    def _chain_obs(self, obs):
        sim = self.env.sim
        lp = np.stack([sim.data.body_xpos[i] for i in self._main +
                       [b for br in self._brs for b in br]])
        # body positions indexed per MAIN/BRANCH structure
        Pm = np.stack([sim.data.body_xpos[i] for i in self._main])
        if self._w is None:
            wm = polyline_weights(Pm, N_MAIN)
            wbs = []
            for br in self._brs:
                Pb = np.stack([sim.data.body_xpos[i] for i in br])
                wbs.append(polyline_weights(Pb, N_BRANCH + 1)[1:])
            self._w = (wm, wbs)
        wm, wbs = self._w
        pts = []
        for i, f in wm:
            pts.append((1 - f) * Pm[i] + f * Pm[i + 1])
        for br, wb in zip(self._brs, wbs):
            Pb = np.stack([sim.data.body_xpos[i] for i in br])
            for i, f in wb:
                pts.append((1 - f) * Pb[i] + f * Pb[i + 1])
        P = np.stack(pts)  # (32,3)
        homo = np.concatenate([P, np.ones((32, 1))], 1)
        pix = (self._w2p @ homo.T).T
        obs["chain_uv"] = np.stack([pix[:, 0] / pix[:, 2] / self._S,
                                    pix[:, 1] / pix[:, 2] / self._S], 1).astype(np.float32)
        obs["chain_z"] = (self._ext_inv @ homo.T).T[:, 2].astype(np.float32)
        obs["ee_z"] = np.array([sim.data.site_xpos[self._site][2]], np.float32)
        return obs

    def reset(self):
        # NOTE: never sim.render() at a size != the env's camera buffer here —
        # it corrupts the offscreen viewport for the WHOLE episode (all obs
        # renders become center crops). Build the env at 512 instead; the reset
        # frame for C-VLM is taken from the regular obs stream.
        obs = self.env.reset()
        self._setup()
        self._w = None
        return self._chain_obs(obs)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        obs = self._chain_obs(obs)
        return obs, r, done, info


def make_libero_env_chain(task_suite_name, task_name, img_h, img_w, gpu_id=-1,
                          vec_env_num=1, seed=0):
    """Copy of libero.utils.env_utils.make_libero_env + ChainStateWrapper (keep in sync)."""
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv
    from libero.utils.env_utils import (LiberoResetWrapper, LiberoTaskEmbWrapper,
                                        StackDummyVectorEnv, StackSubprocVectorEnv)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    descriptions = [task.language for task in task_suite.tasks]
    task_embedding_map = np.load(os.path.join(get_libero_path("task_embeddings"),
                                              "task_emb_bert.npy"), allow_pickle=True).item()
    task_embs = torch.from_numpy(np.stack([task_embedding_map[d] for d in descriptions]))
    task_suite.set_task_embs(task_embs)

    task_id = task_suite.get_task_id(task_name)
    task = task_suite.get_task_from_name(task_name)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": img_h,
                "camera_widths": img_w, "render_gpu_device_id": gpu_id}

    init_states = task_suite.get_task_init_states(task_id)
    assert len(init_states) % vec_env_num == 0
    per = len(init_states) // vec_env_num

    def env_func(env_idx):
        e = OffScreenRenderEnv(**env_args)
        e = LiberoResetWrapper(e, init_states=init_states[env_idx * per:(env_idx + 1) * per])
        e = LiberoTaskEmbWrapper(e, task_emb=task_suite.get_task_emb(task_id))
        e = ChainStateWrapper(e, img_hw=img_h)
        e.seed(seed)
        return e

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
    return env, task_suite.get_task_emb(task_id)


def build_env_chain(suite_name, task_name, img_size=128, gpu_id=0, vec_env_num=10, seed=0):
    from atm.utils.env_utils import (LiberoImageUpsideDownWrapper,
                                     LiberoObservationWrapper, LiberoSuccessWrapper)
    env, task_emb = make_libero_env_chain(suite_name, task_name, img_size, img_size,
                                          gpu_id=gpu_id, vec_env_num=vec_env_num, seed=seed)
    env = LiberoImageUpsideDownWrapper(env)
    env = LiberoSuccessWrapper(env)
    env = LiberoObservationWrapper(env, masks=None,
                                   cameras=["agentview", "robot0_eye_in_hand"])
    return env, task_emb
