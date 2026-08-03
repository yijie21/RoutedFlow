"""Approach-branch policy rollout: the first REAL policy execution of the pipeline.

Per episode: C-VLM runs once on the 512^2 t=0 frame (AFUN prior zeroed at
rollout — noted), z is fixed; every step L3 predicts flow for the FK chain
points (computed in the sub-env), L4 acts. When the policy closes the gripper
(latch), a SCRIPTED lift (+z, gripper closed) runs; success = EE rose >= 3cm
AND gripper width says something is still between the fingers.

Outputs under experiments/stage2_approach_joint/rollout_<run>/:
    summary.json                        per-task approach-SR (train-8 vs ood-2)
    <task_short>/ep<k>.mp4              first N_VIDEO episodes per task, watermarked
                                        "POLICY ROLLOUT (approach) + scripted lift"
Run via `run_stage2.py eval --run <name>` (atm5090, GPU).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("LIBERO_CONFIG_PATH", "/workspace/code/ATM/.libero")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "third_party", "ATM")):
    if p not in sys.path:
        sys.path.insert(0, p)

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from routedflow.stage1.dataset import fold_split
from routedflow.stage2.joint_model import JointApproachModel
from routedflow.wandb_util import WandbLogger

EXP = os.path.join(REPO, "experiments", "stage2_approach_joint")
STAGE0_CFG = os.path.join(EXP, "..", "stage0_routing_causal_test", "configs", "stage0_vilt.yaml")
TRACK_FN = os.path.join(REPO, "third_party", "ATM", "results", "track_transformer",
                        "libero_track_transformer_libero-spatial")
BERT_CACHE = os.path.join(REPO, "third_party", "ATM", "libero",
                          "task_embedding_caches", "task_emb_bert.npy")

APPROACH_TIMEOUT, LIFT_STEPS = 120, 25
LIFT_ACTION = np.array([0, 0, 0.9, 0, 0, 0, 1.0], np.float32)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def short(task):
    return task.replace("pick_up_the_black_bowl_", "").replace("_and_place_it_on_the_plate_demo", "")


@torch.no_grad()
def compute_z(model, dino_net, rgb512, text_emb, dev):
    x = torch.from_numpy(rgb512.copy()).permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=518, mode="bilinear", align_corners=False)
    x = ((x - MEAN) / STD).to(dev)
    feats = dino_net.forward_features(x)["x_norm_patchtokens"]        # (b,1369,768)
    prior = torch.zeros(len(x), 37, 37, device=dev)                    # rollout: no AFUN prior
    _, z, _ = model.l1(feats, prior, text_emb.expand(len(x), -1))
    return z


def watermark(fr, text, phase_txt):
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    d.text((5, 1), text, fill=(90, 220, 120))
    d.text((5, 13), phase_txt, fill=(220, 220, 220))
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--vec", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2, help="episodes per task = vec * rounds")
    ap.add_argument("--n-video", type=int, default=3)
    ap.add_argument("--tasks-limit", type=int, default=None)
    ap.add_argument("--out-tag", default="", help="suffix for output dir + wandb name (e.g. 'final')")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    state = torch.load(os.path.join(EXP, "runs", args.run, args.ckpt),
                       map_location=dev, weights_only=False)
    cfg = OmegaConf.load(STAGE0_CFG).model_cfg
    cfg.pop("track_gate_cfg", None)
    model = JointApproachModel(TRACK_FN, cfg, l1_ckpt=None).to(dev)
    for part in ("l1", "chead", "l3", "l4"):
        getattr(model, part).load_state_dict(state[part])
    model.eval()
    dino_net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(dev).eval()
    emb_map = np.load(BERT_CACHE, allow_pickle=True).item()

    from routedflow.convert_libero_raw import get_task_name_from_file_name
    from routedflow.stage2.eval_env2 import build_env_chain

    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_root = os.path.join(EXP, f"rollout_{args.run}{tag}")
    os.makedirs(out_root, exist_ok=True)
    tasks = sorted(os.listdir(os.path.join(REPO, "data", "atm_libero_light", "libero_spatial")))
    train_tasks, ood_tasks = fold_split(tasks, state["cfg"]["fold"])
    if args.tasks_limit:
        tasks = tasks[: args.tasks_limit]

    summary = {}
    for task in tasks:
        lang = get_task_name_from_file_name(task)
        text_emb = torch.from_numpy(np.asarray(emb_map[lang], np.float32)).to(dev)
        if text_emb.ndim > 1:
            text_emb = text_emb.mean(0)
        text_emb = text_emb[None]
        env, _ = build_env_chain("libero_spatial", task.replace("_demo", ""),
                                 img_size=512, gpu_id=0, vec_env_num=args.vec)
        # NOTE: no task_emb to the policy — text enters only via compute_z (L1).

        succ, saved = [], 0
        tdir = os.path.join(out_root, short(task))
        os.makedirs(tdir, exist_ok=True)
        for rnd in range(args.rounds):
            obs = env.reset()
            model.l4.reset()
            z = compute_z(model, dino_net, obs["agentview_image"], text_emb, dev)  # 512 native
            B = args.vec
            mode = np.zeros(B, int)     # 0 approach, 1 lifting, 2 done
            lift_left = np.full(B, LIFT_STEPS)
            ee_z_latch = np.zeros(B)
            ok = np.zeros(B, bool)
            close_streak = np.zeros(B, int)  # robust latch: 3 consecutive steps > 0.5
                                             # (hair-trigger >0 latched at ~step 26 mid-air
                                             #  — the D6 switching-signal fragility, again)
            frames = [[] for _ in range(B)]
            for t in range(APPROACH_TIMEOUT + LIFT_STEPS + 5):
                # env renders 512 native; downscale to the policy/L3 resolution once
                img_a = np.stack([np.array(Image.fromarray(f).resize((128, 128)))
                                  for f in obs["agentview_image"]])
                img_w = np.stack([np.array(Image.fromarray(f).resize((128, 128)))
                                  for f in obs["robot0_eye_in_hand_image"]])
                with torch.no_grad():
                    fr = torch.from_numpy(img_a.copy()).permute(0, 3, 1, 2)[:, None].float().to(dev)
                    q = torch.from_numpy(obs["chain_uv"]).float().to(dev)
                    qd = torch.from_numpy(obs["chain_z"]).float().to(dev)
                    pred = model.l3(fr, q, z, qd)        # (b, tl, 32, 2)
                    model.l4.set_flow(pred[:, None])
                    obs_v = np.stack([img_a, img_w], 1)  # (b, v, h, w, c)
                    extra = {"joint_states": obs["robot0_joint_pos"],
                             "gripper_states": obs["robot0_gripper_qpos"]}
                    act, _ = model.l4.act(obs_v, extra)
                    model.l4._flow_ctx = None
                for i in range(B):
                    if mode[i] == 0:
                        close_streak[i] = close_streak[i] + 1 if act[i, -1] > 0.5 else 0
                        if close_streak[i] >= 3:          # robust latch
                            mode[i] = 1
                            ee_z_latch[i] = float(obs["ee_z"][i, 0])
                    if mode[i] == 0 and t >= APPROACH_TIMEOUT:
                        mode[i] = 2                        # timeout, fail
                step_act = act.copy()
                for i in range(B):
                    if mode[i] == 1:
                        step_act[i] = LIFT_ACTION
                        lift_left[i] -= 1
                        if lift_left[i] <= 0:
                            gw = float(obs["robot0_gripper_qpos"][i, 0] - obs["robot0_gripper_qpos"][i, 1])
                            rose = float(obs["ee_z"][i, 0]) - ee_z_latch[i]
                            ok[i] = (rose >= 0.03) and (0.004 < gw < 0.075)
                            mode[i] = 2
                    elif mode[i] == 2:
                        step_act[i] = 0 * step_act[i]
                if saved < args.n_video:
                    for i in range(min(B, args.n_video)):
                        ph = ["approach (policy)", "scripted lift", "done"][mode[i]]
                        frames[i].append(watermark(
                            np.array(Image.fromarray(img_a[i]).resize((256, 256))),
                            "POLICY ROLLOUT - approach branch", ph))
                if (mode == 2).all():
                    break
                obs, _, _, _ = env.step(step_act)
            succ.extend(ok.tolist())
            if saved < args.n_video:
                for i in range(min(B, args.n_video - saved)):
                    wr = imageio.get_writer(os.path.join(tdir, f"ep{rnd}_{i}_{'ok' if ok[i] else 'fail'}.mp4"),
                                            fps=20, codec="libx264", quality=7, macro_block_size=1)
                    for f in frames[i]:
                        wr.append_data(f)
                    wr.close()
                saved += min(B, args.n_video - saved)
        env.close()
        sr = float(np.mean(succ))
        summary[task] = {"sr": sr, "n": len(succ),
                         "split": "ood" if task in ood_tasks else "train"}
        print(f"[{short(task)}] approach-SR {sr:.2f} ({'ood' if task in ood_tasks else 'train'})",
              flush=True)

    agg = {
        "train8": float(np.mean([s["sr"] for s in summary.values() if s["split"] == "train"])),
        "ood2": float(np.mean([s["sr"] for s in summary.values() if s["split"] == "ood"])),
    }
    json.dump({"aggregate": agg, "per_task": summary, "note":
               "z computed with AFUN prior zeroed (rollout); lift is scripted post-latch"},
              open(os.path.join(out_root, "summary.json"), "w"), indent=2)
    print("AGGREGATE:", json.dumps(agg), flush=True)

    wb = WandbLogger(f"rollout_{args.run}{tag}", "rollout", enabled=not args.no_wandb,
                     config=vars(args) | {"fold": state["cfg"]["fold"],
                                          "ckpt_step": state.get("step", state.get("epoch"))})
    wb.log({"sr/train8": agg["train8"], "sr/ood2": agg["ood2"]}
           | {f"sr_task/{short(t)}": s["sr"] for t, s in summary.items()})
    wb.log_table("per_task_sr", ["task", "split", "sr", "n"],
                 [[short(t), s["split"], s["sr"], s["n"]] for t, s in summary.items()])
    wb.summary(train8=agg["train8"], ood2=agg["ood2"])
    wb.finish()


if __name__ == "__main__":
    main()
