"""Per-episode visualization sets for a trained stage-1 (C-VLM + C head) run.

For every val_id demo of the given run's fold, writes a folder
experiments/stage1_l1_training/eval_viz_<run>/<task_short>__<demo>/ with:

    01_afun_mask.png    AFUN prompt mask over rgb0 + GT contact + hit verdict
    02_c_pred.png       model prediction: heatmap overlay, top-5 peaks, predicted
                        contact + finger-axis line (yaw) + pitch/w readout, GT triad
    03_robot_flow.png   FROZEN ATM track transformer's predicted tracks for 32
                        robot-mask query points (L3 is an untrained skeleton — this
                        is the base flow model WITHOUT z conditioning), plus the GT
                        EE approach path as reference
    04_episode.mp4      demonstration replay (raw LIBERO agentview), border colored
                        by phase (green approach / orange transport). NOT a policy
                        rollout — no trained action expert exists yet.
    info.json           per-episode numbers

Honesty notes are baked into the filenames/labels: (3) is the pretrained flow
basis, (4) is the demo, both clearly marked.
Run via `run_stage1.py eval-viz --run fold0_seed0` (atm5090, GPU).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from routedflow.stage1.dataset import Stage1Dataset, quat_to_mat
from routedflow.stage1.eval_a1a2 import top5_peaks
from routedflow.stage1.model import CHead, L1FrontEnd

EXP = os.path.join(REPO, "experiments", "stage1_l1_training")
C_LABELS = os.path.join(REPO, "data", "c_labels", "libero_spatial")
CACHE = os.path.join(REPO, "data", "stage1_cache", "libero_spatial")
RAW = "/workspace/datasets/libero/hdf5/libero_spatial"
TRACK_FN = os.path.join(REPO, "third_party", "ATM", "results", "track_transformer",
                        "libero_track_transformer_libero-spatial")

GREEN, ORANGE, YELLOW = (60, 200, 90), (235, 150, 50), (255, 220, 40)


def short(task):
    return task.replace("pick_up_the_black_bowl_", "").replace("_and_place_it_on_the_plate_demo", "")


def load_track_transformer(dev):
    from atm.model.track_transformer import TrackTransformer
    cfg = OmegaConf.load(f"{TRACK_FN}/config.yaml")
    cfg.model_cfg.load_path = f"{TRACK_FN}/model_best.ckpt"
    m = TrackTransformer(**cfg.model_cfg).to(dev).eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    return m


def overlay_mask(rgb, mask, color, alpha=0.45):
    img = rgb.astype(np.float32).copy()
    img[mask > 0] = (1 - alpha) * img[mask > 0] + alpha * np.array(color, np.float32)
    return img.astype(np.uint8)


def draw_star(draw, row, col, color, r=9, w=4):
    draw.line([col - r, row, col + r, row], fill=color, width=w)
    draw.line([col, row - r, col, row + r], fill=color, width=w)


def project(w2p, pts_w):
    homo = np.concatenate([np.atleast_2d(pts_w), np.ones((len(np.atleast_2d(pts_w)), 1))], 1)
    uvw = (w2p @ homo.T).T
    return np.stack([uvw[:, 1] / uvw[:, 2], uvw[:, 0] / uvw[:, 2]], 1)  # (row, col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    args = ap.parse_args()

    dev = "cuda"
    run_dir = os.path.join(EXP, "runs", args.run)
    state = torch.load(os.path.join(run_dir, args.ckpt), map_location=dev, weights_only=False)
    fold, use_prior = state["cfg"]["fold"], not state["cfg"].get("no_prior", False)
    l1 = L1FrontEnd().to(dev).eval(); l1.load_state_dict(state["l1"])
    chead = CHead().to(dev).eval(); chead.load_state_dict(state["chead"])
    tracker = load_track_transformer(dev)
    tl = tracker.num_track_ts

    ds = Stage1Dataset(fold=fold, split="val_id", use_prior=use_prior)
    prior_h5 = h5py.File(os.path.join(CACHE, "afun_prior.h5"), "r")
    out_root = os.path.join(EXP, f"eval_viz_{args.run}")
    os.makedirs(out_root, exist_ok=True)

    summary = []
    for i in range(len(ds)):
        meta, t = ds.samples[i], ds[i]
        task, demo = meta["task"], meta["demo"]
        out = os.path.join(out_root, f"{short(task)}__{demo}")
        os.makedirs(out, exist_ok=True)

        with h5py.File(os.path.join(C_LABELS, f"{task}.h5"), "r") as f:
            g = f[demo]
            rgb0 = np.array(g["rgb0"]); depth0 = np.array(g["depth0"]); seg0 = np.array(g["seg0"])
            w2p = np.array(f["world_to_pix"]); K = np.array(f["K"]); ext = np.array(f["extrinsic"])
            base_q = np.array(f.attrs["robot_base_quat"])
            robot_gids = np.array(f["robot_geom_ids"])
            ee_pos = np.array(g["ee_pos"]); ee_quat = np.array(g["ee_quat"])
            phase = np.array(g["phase"]); tg = int(g.attrs["t_g"])
            gt_rc = np.array(g["contact_rowcol"]); gt_w = float(t["w"])

        # ---------- forward ----------
        with torch.no_grad():
            hm_l, z, fm = l1(t["dino"][None].to(dev), t["prior"][None].to(dev), t["text"][None].to(dev))
            hm = torch.softmax(hm_l.reshape(1, -1), -1).reshape(128, 128).cpu().numpy()
            peaks = top5_peaks(hm)                      # [(row512, col512)] top-5
            pr, pc = peaks[0]
            yl, pl, wp = chead(z, fm, torch.tensor([[pr, pc]], dtype=torch.float32, device=dev))
            yaw = (int(yl.argmax()) + 0.5) * np.pi / 36
            pitch = (int(pl.argmax()) + 0.5) * (np.pi / 2) / 12
            w_pred = float(wp)

        # ---------- 01: AFUN mask ----------
        mask = np.array(prior_h5[task][demo]) if (task in prior_h5 and demo in prior_h5[task]) else np.zeros((512, 512), np.uint8)
        ys, xs = np.nonzero(mask)
        d_prior = float(np.min(np.hypot(ys - gt_rc[0], xs - gt_rc[1]))) if len(ys) else float("inf")
        img = Image.fromarray(overlay_mask(rgb0, mask, (80, 160, 255)))
        d = ImageDraw.Draw(img)
        draw_star(d, *gt_rc, YELLOW)
        d.text((8, 8), f"AFUN prompt mask · GT dist {d_prior:.0f}px "
                       f"{'HIT' if d_prior <= 20 else 'MISS'}", fill=(255, 255, 255))
        img.save(os.path.join(out, "01_afun_mask.png"))

        # ---------- 02: C prediction ----------
        import matplotlib.cm as cm
        hm512 = np.array(Image.fromarray((hm / hm.max() * 255).astype(np.uint8)).resize((512, 512)))
        heat = (cm.get_cmap("hot")(hm512 / 255.0)[..., :3] * 255)
        a = (0.55 * (hm512 / 255.0) + 0.05)[..., None]
        img = Image.fromarray((rgb0 * (1 - a) + heat * a).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for k, (rr, cc) in enumerate(peaks):
            d.ellipse([cc - 6, rr - 6, cc + 6, rr + 6], outline=(120, 255, 120) if k else (0, 255, 0), width=3)
        # predicted 3D position via robust lift + finger-axis (yaw) line
        r_i, c_i = int(round(pr)), int(round(pc))
        win = depth0[max(r_i - 4, 0):r_i + 5, max(c_i - 4, 0):c_i + 5]
        dep = float(win.min()) if win.size else float("nan")
        x_cam = dep * (np.linalg.inv(K) @ np.array([pc, pr, 1.0]))
        pos_w = (ext @ np.append(x_cam, 1.0))[:3]
        R_wb = quat_to_mat(base_q)
        f_w = R_wb @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
        seg_px = project(w2p, np.stack([pos_w - 0.05 * f_w, pos_w + 0.05 * f_w]))
        d.line([tuple(seg_px[0][::-1]), tuple(seg_px[1][::-1])], fill=(0, 120, 255), width=5)
        # GT triad (thin) at GT contact
        Rg = quat_to_mat(ee_quat[tg])
        for k, c in enumerate(((235, 60, 60), (60, 190, 60), (70, 110, 235))):
            tip = project(w2p, ee_pos[tg] + 0.06 * Rg[:, k])[0]
            d.line([tuple(gt_rc[::-1]), tuple(tip[::-1])], fill=c, width=2)
        draw_star(d, *gt_rc, YELLOW, r=8, w=3)
        err = float(np.hypot(pr - gt_rc[0], pc - gt_rc[1]))
        d.text((8, 8), f"pred peak err {err:.0f}px · yaw {np.degrees(yaw):.0f}deg "
                       f"pitch {np.degrees(pitch):.0f}deg · w {w_pred:.3f} (gt {gt_w:.3f})",
               fill=(255, 255, 255))
        d.text((8, 24), "blue line = predicted finger axis @ lifted 3D pos · thin RGB triad = GT pose",
               fill=(200, 200, 200))
        img.save(os.path.join(out, "02_c_pred.png"))

        # ---------- 03: robot flow (frozen ATM base, no z conditioning) ----------
        with h5py.File(os.path.join(RAW, f"{task}.hdf5"), "r") as rf:
            vid = np.array(rf[f"data/{demo}/obs/agentview_rgb"])[:, ::-1].copy()  # (T,128,128,3) upright
        robot_m512 = np.isin(seg0, robot_gids)
        rm = np.array(Image.fromarray(robot_m512.astype(np.uint8) * 255).resize((128, 128), Image.NEAREST)) > 127
        rr_, cc_ = np.nonzero(rm)
        rng = np.random.RandomState(0)
        sel = rng.choice(len(rr_), size=min(32, len(rr_)), replace=False)
        q = np.stack([cc_[sel] / 128.0, rr_[sel] / 128.0], 1)  # (n,2) x,y in [0,1]
        q_t = torch.from_numpy(np.repeat(q[None, None], tl, axis=1).reshape(1, tl, -1, 2)).float().to(dev)
        vid_t = torch.from_numpy(vid[0].transpose(2, 0, 1)[None, None]).float().to(dev)
        with torch.no_grad():
            pred_tr, _ = tracker.reconstruct(vid_t, q_t, t["text"][None].to(dev), p_img=0)
        tr = pred_tr[0].cpu().numpy()  # (tl, n, 2) x,y in [0,1]
        S = 3
        img = Image.fromarray(vid[0]).resize((128 * S, 128 * S))
        d = ImageDraw.Draw(img)
        for n in range(tr.shape[1]):
            pts = [(float(tr[s, n, 0] * 128 * S), float(tr[s, n, 1] * 128 * S)) for s in range(tl)]
            d.line(pts, fill=GREEN, width=2)
            d.ellipse([pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2], fill=(255, 255, 255))
        ee_px = project(w2p, ee_pos[:tg + 1])[::2] / 512.0 * 128 * S  # GT approach path
        d.line([(p[1], p[0]) for p in ee_px], fill=YELLOW, width=3)
        d.text((6, 6), "green: FROZEN ATM track pred (32 robot-mask pts, no z cond — L3 untrained)",
               fill=(255, 255, 255))
        d.text((6, 20), "yellow: GT EE approach path (reference)", fill=YELLOW)
        img.save(os.path.join(out, "03_robot_flow.png"))

        # ---------- 04: episode video (demo replay) ----------
        wr = imageio.get_writer(os.path.join(out, "04_episode_DEMO_REPLAY.mp4"), fps=20,
                                codec="libx264", quality=7, macro_block_size=1)
        for ti in range(vid.shape[0]):
            fim = Image.fromarray(vid[ti]).resize((256, 256))
            dd = ImageDraw.Draw(fim)
            # watermark burned into EVERY frame: this is the recorded demonstration,
            # no trained policy exists yet (stage-1 model predicts no actions at all)
            dd.rectangle([0, 0, 256, 30], fill=(0, 0, 0))
            dd.text((6, 2), "DEMO REPLAY - not a policy rollout", fill=(255, 80, 80))
            dd.text((6, 16), f"phase: {'approach' if phase[ti] == 0 else 'transport'}"
                             f"  (t_g={tg})", fill=(200, 200, 200))
            fr = np.array(fim)
            col = GREEN if phase[ti] == 0 else ORANGE
            fr[-6:] = col; fr[:, :6] = col; fr[:, -6:] = col
            wr.append_data(fr)
        wr.close()

        info = {"task": task, "demo": demo, "t_g": tg, "T": int(len(phase)),
                "a1_top1_err_px": round(err, 1),
                "a1_top5_hit": bool(min(np.hypot(p[0] - gt_rc[0], p[1] - gt_rc[1]) for p in peaks) <= 10),
                "yaw_pred_deg": round(float(np.degrees(yaw)), 1),
                "pitch_pred_deg": round(float(np.degrees(pitch)), 1),
                "w_pred": round(w_pred, 4), "w_gt": round(gt_w, 4),
                "prior_hit_20px": bool(d_prior <= 20),
                "note": "03=frozen-ATM base flow (L3 untrained); 04=demo replay (no policy yet)"}
        json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=2)
        summary.append(info)
        print(f"[{short(task)}/{demo}] top5_hit={info['a1_top5_hit']} err={err:.0f}px", flush=True)

    prior_h5.close()
    agg = {"n": len(summary),
           "a1_top5": float(np.mean([s["a1_top5_hit"] for s in summary])),
           "a1_top1_err_px_median": float(np.median([s["a1_top1_err_px"] for s in summary])),
           "w_l1": float(np.mean([abs(s["w_pred"] - s["w_gt"]) for s in summary]))}
    json.dump({"aggregate": agg, "episodes": summary},
              open(os.path.join(out_root, "summary.json"), "w"), indent=2)
    print("AGGREGATE:", json.dumps(agg), flush=True)
    print("->", out_root, flush=True)


if __name__ == "__main__":
    main()
