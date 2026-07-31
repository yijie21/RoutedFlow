"""Stage-0 evaluation: rollouts with online robot-mask gating.

Phase at eval time is latched from the policy's OWN gripper output (per env,
reset each episode). Known one-frame approximation: at the exact closing step the
policy still sees approach-phase gating (training labels mark that step as
transport) — a 1-frame boundary discrepancy out of ~150 steps, accepted and
documented in the experiment README.

Results per task: per-episode successes (for paired stats later), success rate,
phase-latch step stats. Written to <run_dir>/eval_<ckpt>_<nroll>.json.

Launched via `run_stage0.py eval --mode <gate_mode>`.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "third_party", "ATM")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from omegaconf import OmegaConf

from engine.utils import obs_key_mapping
from routedflow.eval_env import build_env_masked
from routedflow.gated_policy import BCViLTPolicyGated


@torch.no_grad()
def rollout_gated_task(env, policy, num_batches, horizon=600):
    """One task. Each batch = vec_env_num parallel episodes. Returns per-episode records."""
    records = []
    for _ in range(num_batches):
        obs = env.reset()
        policy.reset()
        b = obs["image"].shape[0]
        phase = np.zeros(b, dtype=bool)
        latch_step = np.full(b, -1, dtype=int)
        done, step_i, info = False, 0, None
        while not done and step_i < horizon:
            rgb = obs["image"]  # (b, v, h, w, c)
            task_emb = obs.get("task_emb", None)
            extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
            labels = torch.from_numpy(obs["robot_grid_labels"]).bool()[:, :, None, :]  # (b, v, 1, 32)
            phase_t = torch.from_numpy(phase).bool()[:, None]  # (b, 1)
            a, _ = policy.act_gated(rgb, task_emb, extra_states, labels, phase_t)
            obs, r, dones, info = env.step(a)
            newly = (~phase) & (a[:, -1] > 0)
            latch_step[newly] = step_i
            phase |= a[:, -1] > 0
            done = all(dones)
            step_i += 1
        success = list(info["success"]) if info is not None else [False] * b
        for i in range(b):
            records.append({"success": bool(success[i]), "latch_step": int(latch_step[i]),
                            "steps": step_i})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="model_final.ckpt")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--nroll", type=int, default=40, help="episodes per task")
    ap.add_argument("--vec", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--tasks-limit", type=int, default=None, help="only first N tasks (smoke)")
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    ckpt_path = os.path.join(args.run_dir, args.ckpt)
    # keep DictConfig: ATM's __init__ uses attribute access (track_cfg.track_fn)
    cfg.model_cfg.load_path = ckpt_path
    gate_cfg = OmegaConf.to_container(cfg.track_gate_cfg, resolve=True)
    policy = BCViLTPolicyGated(track_gate_cfg=gate_cfg, **cfg.model_cfg).cuda()
    policy.eval()
    print(f"loaded {ckpt_path} gate_mode={policy.gate_mode}")

    assert args.nroll % args.vec == 0
    data_root = os.path.join(REPO_ROOT, "data", "atm_libero_gated", args.suite)
    task_names = sorted(d[:-len("_demo")] for d in os.listdir(data_root) if d.endswith("_demo"))
    assert task_names, f"no tasks under {data_root}"
    if args.tasks_limit:
        task_names = task_names[: args.tasks_limit]

    results = {"ckpt": ckpt_path, "gate_mode": policy.gate_mode, "nroll": args.nroll, "tasks": {}}
    for tname in task_names:
        env = build_env_masked(args.suite, tname, img_size=cfg.img_size, gpu_id=0,
                               vec_env_num=args.vec, seed=cfg.seed)
        recs = rollout_gated_task(env, policy, args.nroll // args.vec, horizon=args.horizon)
        env.close()
        sr = float(np.mean([r["success"] for r in recs]))
        latched = [r["latch_step"] for r in recs if r["latch_step"] >= 0]
        results["tasks"][tname] = {
            "success_rate": sr,
            "episodes": recs,
            "latch_rate": len(latched) / len(recs),
            "latch_step_mean": float(np.mean(latched)) if latched else None,
        }
        print(f"[{tname}] SR={sr:.3f} latch_rate={len(latched)/len(recs):.2f}")

    srs = [t["success_rate"] for t in results["tasks"].values()]
    results["success_avg"] = float(np.mean(srs))
    out = os.path.join(args.run_dir, f"eval_{os.path.splitext(args.ckpt)[0]}_n{args.nroll}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"AVG SR over {len(srs)} tasks: {results['success_avg']:.4f} -> {out}")


if __name__ == "__main__":
    main()
