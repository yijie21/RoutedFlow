#!/usr/bin/env python3
"""Stage-0 orchestrator (routing causal test). Run with any python3; heavy work is
delegated to the atm5090 conda interpreter with the right env vars.

    python3 run_stage0.py prep   [--tasks-limit N]      # convert raw LIBERO -> gated h5 (+ masks)
    python3 run_stage0.py test                          # unit tests (pure CPU, fast)
    python3 run_stage0.py smoke                         # 2-epoch mode=none run: wall-clock + pipeline check
    python3 run_stage0.py train --mode phase_switched   # one 101-epoch variant
    python3 run_stage0.py status                        # what is running / what exists

Experiment home: experiments/stage0_routing_causal_test/ (symlinked as experiments/CURRENT)
"""
import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PY = "/workspace/miniconda3/envs/atm5090/bin/python"
EXP_DIR = os.path.join(REPO, "experiments", "stage0_routing_causal_test")
DATA_ROOT = os.path.join(REPO, "data", "atm_libero_gated", "libero_spatial")

ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "LIBERO_CONFIG_PATH": "/workspace/code/ATM/.libero",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": "/workspace/.hf_home",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONPATH": os.pathsep.join([os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")]),
}

MODES = ("none", "object_only", "robot_only", "phase_switched")


def sh(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, env=ENV, cwd=REPO, **kw).returncode


def dataset_lists():
    tasks = sorted(glob.glob(os.path.join(DATA_ROOT, "*_demo")))
    assert tasks, f"no converted tasks under {DATA_ROOT} — run `prep` first"
    train = [os.path.join(t, "bc_train_10") for t in tasks]
    val = [os.path.join(t, "val") for t in tasks]
    return train, val


def cmd_prep(args):
    cmd = [PY, os.path.join(REPO, "src", "routedflow", "convert_libero_raw.py"), "--suite", args.suite]
    if args.tasks_limit:
        cmd += ["--tasks-limit", str(args.tasks_limit)]
    return sh(cmd)


def cmd_test(args):
    return sh([PY, "-m", "pytest", os.path.join(REPO, "tests"), "-q"])


def launch_train(mode, seed, epochs, experiment=None):
    assert mode in MODES, f"mode must be one of {MODES}"
    train, val = dataset_lists()
    experiment = experiment or f"{mode}"
    overrides = [
        "--config-name=stage0_vilt",
        f"experiment={experiment}",
        f"seed={seed}",
        f"epochs={epochs}",
        f"track_gate_cfg.mode={mode}",
        "train_dataset=[" + ",".join(train) + "]",
        "val_dataset=[" + ",".join(val) + "]",
    ]
    return sh([PY, "-m", "routedflow.engine_train_stage0"] + overrides)


def cmd_train(args):
    return launch_train(args.mode, args.seed, args.epochs)


def cmd_smoke(args):
    return launch_train("none", seed=0, epochs=2, experiment="smoke_none")


def cmd_eval(args):
    run_dir = os.path.join(EXP_DIR, "runs", f"{args.mode}_seed{args.seed}")
    assert os.path.isdir(run_dir), f"run dir not found: {run_dir}"
    cmd = [PY, os.path.join(REPO, "src", "routedflow", "engine_eval_stage0.py"),
           "--run-dir", run_dir, "--ckpt", args.ckpt,
           "--nroll", str(args.nroll), "--vec", str(args.vec)]
    return sh(cmd)


def cmd_status(args):
    runs = sorted(glob.glob(os.path.join(EXP_DIR, "runs", "*")))
    print(f"experiment: {EXP_DIR}")
    print(f"converted tasks: {len(glob.glob(os.path.join(DATA_ROOT, '*_demo')))}")
    for r in runs:
        line = "(no metrics yet)"
        mp = os.path.join(r, "metrics.jsonl")
        if os.path.exists(mp):
            with open(mp) as f:
                lines = f.read().strip().splitlines()
            if lines:
                rec = json.loads(lines[-1])
                line = f"epoch {rec.get('epoch')} | {rec.get('epoch_seconds')}s/epoch | train loss {rec.get('train/loss', float('nan')):.4f}"
        print(f"  {os.path.basename(r)}: {line}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep"); p.add_argument("--suite", default="libero_spatial"); p.add_argument("--tasks-limit", type=int, default=None); p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("test"); p.set_defaults(fn=cmd_test)
    p = sub.add_parser("smoke"); p.set_defaults(fn=cmd_smoke)
    p = sub.add_parser("train"); p.add_argument("--mode", required=True, choices=MODES); p.add_argument("--seed", type=int, default=0); p.add_argument("--epochs", type=int, default=101); p.set_defaults(fn=cmd_train)
    p = sub.add_parser("eval"); p.add_argument("--mode", required=True, choices=MODES); p.add_argument("--seed", type=int, default=0); p.add_argument("--ckpt", default="model_final.ckpt"); p.add_argument("--nroll", type=int, default=40); p.add_argument("--vec", type=int, default=10); p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
