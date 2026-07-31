"""Offline job #3: precompute frozen DINOv2 patch features for all rgb0 images.

Input : data/c_labels/<suite>/<task>.h5  (demo_*/rgb0, 512x512)
Output: data/stage1_cache/<suite>/dino_feats.h5
            <task>/<demo>: (1369, 768) float16   # 37x37 patch tokens, ViT-B/14 @ 518
        attrs: model, input_res, grid (37)

Frozen backbone == deterministic features => cache once, train L1 in seconds/epoch.
Cost of this choice: no image augmentation in v1 (recorded in plan §2.2).
Run via `run_stage1.py dino-feats` (atm5090 interpreter, GPU, ~5 min).
"""
import argparse
import os

import h5py
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(REPO, "data", "c_labels")
OUT_DIR = os.path.join(REPO, "data", "stage1_cache")

INPUT_RES = 518  # 37 * 14
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    dev = "cuda"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(dev).eval()

    suite_dir = os.path.join(DATA, args.suite)
    out_dir = os.path.join(OUT_DIR, args.suite)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dino_feats.h5")

    with h5py.File(out_path, "w") as out:
        out.attrs.update({"model": "dinov2_vitb14", "input_res": INPUT_RES, "grid": INPUT_RES // 14})
        for tf in sorted(f for f in os.listdir(suite_dir) if f.endswith(".h5")):
            with h5py.File(os.path.join(suite_dir, tf), "r") as f:
                demos = sorted(k for k in f.keys() if k.startswith("demo"))
                imgs = np.stack([np.array(f[k]["rgb0"]) for k in demos])  # (N,512,512,3)
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
            x = torch.nn.functional.interpolate(x, size=INPUT_RES, mode="bilinear", align_corners=False)
            x = (x - MEAN) / STD
            feats = []
            for i in range(0, len(x), args.batch):
                r = model.forward_features(x[i:i + args.batch].to(dev))
                feats.append(r["x_norm_patchtokens"].half().cpu())  # (b, 1369, 768)
            feats = torch.cat(feats).numpy()
            g = out.create_group(tf[:-3])
            for k, fe in zip(demos, feats):
                g.create_dataset(k, data=fe)
            print(f"[{tf[:-3]}] {feats.shape}", flush=True)
    print("dino feats ->", out_path)


if __name__ == "__main__":
    main()
