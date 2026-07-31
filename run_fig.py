#!/usr/bin/env python3
"""Pipeline-figure orchestrator. Run with any python3; delegates to atm5090.

    python3 run_fig.py assets    # regenerate doc/fig_assets/ from data/c_labels (choose demo in src)
    python3 run_fig.py build     # rebuild doc/pipeline_fig_v01.svg + doc/pipeline_fig.html
    python3 run_fig.py all

Editing guide (coordinate map, QA loop, publish flow): doc/PIPELINE_FIG_HOWTO.md
"""
import argparse
import os
import subprocess

REPO = os.path.dirname(os.path.abspath(__file__))
PY = "/workspace/miniconda3/envs/atm5090/bin/python"
ENV = {**os.environ,
       "PYTHONPATH": os.pathsep.join([os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")])}


def sh(script):
    cmd = [PY, os.path.join(REPO, "src", "routedflow", script)]
    print("+", " ".join(cmd))
    return subprocess.run(cmd, env=ENV, cwd=REPO).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["assets", "build", "all"])
    args = ap.parse_args()
    rc = 0
    if args.cmd in ("assets", "all"):
        rc = rc or sh("fig_assets.py")
    if args.cmd in ("build", "all"):
        rc = rc or sh("fig_build.py")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
