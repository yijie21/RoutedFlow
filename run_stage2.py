#!/usr/bin/env python3
"""Stage-2 orchestrator (approach branch joint training). Any python3; delegates
to the atm5090 interpreter with the right env vars.

    python3 run_stage2.py convert          # light ATM-format conversion (all 500 demos)
    python3 run_stage2.py chain-prep       # FK chain points into c_labels + QA overlays
    python3 run_stage2.py test             # unit tests
    python3 run_stage2.py smoke            # 1-epoch tiny joint run (pipeline + OOM check)
    python3 run_stage2.py train [args...]  # joint training (engine args pass through)
    python3 run_stage2.py status

Experiment home: experiments/stage2_approach_joint/
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PY = "/workspace/miniconda3/envs/atm5090/bin/python"
S2 = os.path.join(REPO, "src", "routedflow", "stage2")

ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
    "LIBERO_CONFIG_PATH": "/workspace/code/ATM/.libero",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONPATH": os.pathsep.join([os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")]),
}


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=ENV, cwd=REPO).returncode


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    cmd, rest = sys.argv[1], sys.argv[2:]
    table = {
        "convert": [PY, os.path.join(S2, "convert_light.py")],
        "chain-prep": [PY, os.path.join(S2, "chain_points.py")],
        "test": [PY, os.path.join(REPO, "tests", "test_stage2_units.py")],
        "char": [PY, os.path.join(REPO, "tests", "test_char_labels.py")],
        "char-env": [PY, os.path.join(REPO, "tests", "test_char_rollout.py")],
        "train": [PY, os.path.join(S2, "engine_train.py")],
        "smoke": [PY, os.path.join(S2, "engine_train.py"), "--name", "smoke",
                  "--steps", "8", "--log-every", "4", "--val-every", "8",
                  "--bs", "4", "--accum", "2", "--no-wandb"],
        "eval": [PY, os.path.join(S2, "eval_rollout.py")],
    }
    if cmd == "status":
        runs = os.path.join(REPO, "experiments", "stage2_approach_joint", "runs")
        for r in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
            m = os.path.join(runs, r, "metrics.jsonl")
            n = sum(1 for _ in open(m)) if os.path.exists(m) else 0
            print(f"{r}: {n} metric rows logged")
        raise SystemExit(0)
    if cmd not in table:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(sh(table[cmd] + rest))


if __name__ == "__main__":
    main()
