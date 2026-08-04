"""Generate universally-playable preview copies of the lossless LeRobot videos.

Why: the training masters are H.264 High 4:4:4 RGB (bitwise lossless — the
whole point of the codec choice), but mainstream players (browsers, VS Code
preview, many system players) only decode yuv420p and render 4:4:4 RGB as
psychedelic pink/green. There is NO codec that is both truly lossless for RGB
and universally playable (yuv420p subsamples chroma by definition), so the
resolution is dual-track: masters stay lossless for training; this script
mirrors them to data/lerobot/<suite>/videos_preview/... as yuv420p crf18 —
HUMAN EYES ONLY, never read by any loader.

Idempotent (skips existing previews); parallel over CPU. Run via
`run_stage2.py preview-lerobot [--suite all]`.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LEROBOT_ROOT = os.path.join(REPO, "data", "lerobot")


def convert_one(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
         dst + ".tmp.mp4"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL {src}: {r.stderr[-200:]}", file=sys.stderr, flush=True)
        return 0
    os.replace(dst + ".tmp.mp4", dst)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    suites = (sorted(os.listdir(LEROBOT_ROOT)) if args.suite == "all" else [args.suite])
    jobs = []
    for s in suites:
        vdir = os.path.join(LEROBOT_ROOT, s, "videos")
        if not os.path.isdir(vdir):
            continue
        for root, _, files in os.walk(vdir):
            for f in files:
                if not f.endswith(".mp4"):
                    continue
                src = os.path.join(root, f)
                dst = src.replace(os.sep + "videos" + os.sep,
                                  os.sep + "videos_preview" + os.sep)
                if not os.path.exists(dst):
                    jobs.append((src, dst))
    print(f"{len(jobs)} previews to generate ({args.jobs} workers)", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        done = sum(ex.map(lambda a: convert_one(*a), jobs))
    print(f"previews done: {done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
