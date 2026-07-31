"""Score AFUN spike outputs against the projected GT grasp point.

Metrics per task: GT-pixel-in-mask hit, distance from GT pixel to the nearest
mask pixel (px @512), mask area fraction, model confidence. Writes results.json
and prints a table. Run with the afun env python (needs only numpy).
"""
import json
import os
import sys

import numpy as np

SPIKE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = []
    for task in sorted(os.listdir(os.path.join(SPIKE, "outputs"))):
        out_dir = os.path.join(SPIKE, "outputs", task)
        pred_path = os.path.join(out_dir, "pred.npz")
        if not os.path.exists(pred_path):
            rows.append({"task": task, "error": "no pred.npz"})
            continue
        pred = np.load(pred_path, allow_pickle=True)
        mask = pred["mask"].astype(bool).squeeze()
        gt = json.load(open(os.path.join(SPIKE, "inputs", task, "gt.json")))
        r, c = gt["gt_pixel_rowcol"]
        if mask.shape != (512, 512):  # model may predict at another res
            sr, sc = mask.shape[0] / 512.0, mask.shape[1] / 512.0
            r, c = r * sr, c * sc
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            rows.append({"task": task, "hit": False, "dist_px": None, "area": 0.0})
            continue
        d = float(np.min(np.hypot(ys - r, xs - c)) / mask.shape[0] * 512)  # in 512-px units
        conf = float(np.asarray(pred["confidence"]).squeeze()) if "confidence" in pred else None
        rows.append({
            "task": task,
            "hit": bool(mask[int(round(r)), int(round(c))]),
            "dist_px": round(d, 1),
            "area": round(float(mask.mean()), 4),
            "confidence": conf,
        })

    json.dump(rows, open(os.path.join(SPIKE, "results.json"), "w"), indent=2)
    for row in rows:
        print(row)
    hits = [r for r in rows if r.get("hit")]
    near = [r for r in rows if r.get("dist_px") is not None and r["dist_px"] <= 20]
    print(f"\nGT-in-mask: {len(hits)}/{len(rows)} | within 20px: {len(near)}/{len(rows)}")


if __name__ == "__main__":
    main()
