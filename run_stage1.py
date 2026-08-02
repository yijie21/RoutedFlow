#!/usr/bin/env python3
"""Stage-1 orchestrator (C distribution prediction). Run with any python3; heavy
work is delegated to the right conda interpreter with the right env vars.

  data:      extract / qc / viz / augment-links        (labels + visualization)
  caches:    dino-feats / afun-prior                   (offline jobs #3 / #2)
  training:  train-l1 [engine args...] / eval-l1 --run <name> / test
  status

Experiment homes: experiments/stage1_c_labels/ (data), experiments/stage1_l1_training/
"""
import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PY = "/workspace/miniconda3/envs/atm5090/bin/python"
PY_AFUN = "/workspace/miniconda3/envs/afun/bin/python"
DATA_ROOT = os.path.join(REPO, "data", "c_labels", "libero_spatial")
QC_DIR = os.path.join(REPO, "experiments", "stage1_c_labels", "qc")
S1 = os.path.join(REPO, "src", "routedflow", "stage1")

ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "LIBERO_CONFIG_PATH": "/workspace/code/ATM/.libero",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": "/workspace/.hf_home",
    "PYTHONPATH": os.pathsep.join([os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")]),
}


def sh(cmd):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, env=ENV, cwd=REPO).returncode


def cmd_extract(args):
    cmd = [PY, os.path.join(REPO, "src", "routedflow", "extract_c_labels.py"), "--suite", args.suite]
    if args.tasks_limit:
        cmd += ["--tasks-limit", str(args.tasks_limit)]
    if args.demos_limit:
        cmd += ["--demos-limit", str(args.demos_limit)]
    return sh(cmd)


def cmd_qc(args):
    return sh([PY, os.path.join(REPO, "src", "routedflow", "qc_c_labels.py"), "--suite", args.suite])


def cmd_viz(args):
    return sh([PY, os.path.join(REPO, "src", "routedflow", "viz_c_labels.py"), "--suite", args.suite])


def cmd_augment_links(args):
    return sh([PY, os.path.join(S1, "augment_links.py"), args.suite])


def cmd_dino_feats(args):
    return sh([PY, os.path.join(S1, "dino_feats.py"), "--suite", args.suite])


def cmd_afun_prior(args):
    return sh([PY_AFUN, os.path.join(S1, "afun_prior.py"), args.suite])


def passthrough(script):
    def fn(args):
        return sh([PY, os.path.join(S1, script)] + args.rest)
    return fn


def cmd_test(args):
    return sh([PY, os.path.join(REPO, "tests", "test_stage1_units.py")])


def cmd_status(args):
    h5s = sorted(glob.glob(os.path.join(DATA_ROOT, "*.h5")))
    print(f"extracted task files: {len(h5s)}")
    for p in h5s:
        print("  ", os.path.basename(p), f"{os.path.getsize(p) / 1e6:.0f}MB")
    sp = os.path.join(QC_DIR, "qc_stats.json")
    if os.path.exists(sp):
        print("qc global:", json.dumps(json.load(open(sp))["global"]))
    else:
        print("qc not run yet")
    return 0


def main():
    # pass-through subcommands: hand everything after the verb to the engine verbatim
    if len(sys.argv) > 1 and sys.argv[1] in ("train-l1", "eval-l1", "eval-viz"):
        script = {"train-l1": "engine_train.py", "eval-l1": "eval_a1a2.py",
                  "eval-viz": "eval_viz.py"}[sys.argv[1]]
        raise SystemExit(sh([PY, os.path.join(S1, script)] + sys.argv[2:]))

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = (("extract", cmd_extract), ("qc", cmd_qc), ("viz", cmd_viz), ("status", cmd_status),
            ("augment-links", cmd_augment_links), ("dino-feats", cmd_dino_feats),
            ("afun-prior", cmd_afun_prior), ("test", cmd_test))
    for name, fn in cmds:
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--suite", default="libero_spatial")
        if name == "extract":
            p.add_argument("--tasks-limit", type=int, default=None)
            p.add_argument("--demos-limit", type=int, default=None)
    args = ap.parse_args()
    raise SystemExit(args.fn(args))


if __name__ == "__main__":
    main()
