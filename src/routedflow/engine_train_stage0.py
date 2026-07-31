"""Stage-0 BC training engine.

Mirror of third_party/ATM/engine/train_bc.py with the deltas RoutedFlow needs:
  - GatedBCDataset / BCViLTPolicyGated instead of BCDataset / eval(model_name)
  - batch is an 8-tuple (adds robot_labels, phase); loss via forward_loss_gated
  - Fabric strategy "auto" (deepspeed is not installed in atm5090), single GPU
  - no wandb, no vis dataloaders (no GT tracks), no in-training env rollouts
  - per-epoch wall-clock seconds logged to metrics.jsonl in the run dir

Launch through run_stage0.py, which sets PYTHONPATH / CUDA_VISIBLE_DEVICES and
passes the hydra overrides.
"""
import json
import os
import time

import hydra
import lightning
import torch
from lightning.fabric import Fabric
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from atm.dataloader import get_dataloader
from atm.utils.train_utils import setup_optimizer, setup_lr_scheduler
from atm.utils.log_utils import BestAvgLoss

from routedflow.gated_dataset import GatedBCDataset
from routedflow.gated_policy import BCViLTPolicyGated


def move_batch(batch, device):
    obs, track_obs, track, task_emb, actions, extra_states, robot_labels, phase = batch
    obs, track_obs, track, task_emb, actions = (
        obs.to(device), track_obs.to(device), track.to(device), task_emb.to(device), actions.to(device))
    extra_states = {k: v.to(device) for k, v in extra_states.items()}
    robot_labels, phase = robot_labels.to(device), phase.to(device)
    return obs, track_obs, track, task_emb, actions, extra_states, robot_labels, phase


def run_epoch(fabric, model, loader, optimizer=None, clip_grad=100.0, scheduler=None, grad_accum=1):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, n = {}, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    pending = False
    if train:
        optimizer.zero_grad()
    with ctx:
        for i, batch in enumerate(tqdm(loader, disable=None)):
            obs, track_obs, track, task_emb, actions, extra_states, robot_labels, phase = \
                move_batch(batch, fabric.device)
            loss, ret = model.forward_loss_gated(
                obs, track_obs, track, task_emb, extra_states, actions, robot_labels, phase)
            if train:
                # micro-batches: effective batch = batch_size * grad_accum (ATM recipe = 128)
                fabric.backward(loss / grad_accum)
                pending = True
                if (i + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()
                    pending = False
            for k, v in ret.items():
                tot[k] = tot.get(k, 0.0) + v
            n += 1
    if train and pending:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        optimizer.zero_grad()
    if train and scheduler is not None:
        scheduler.step()
    tag = "train" if train else "val"
    return {f"{tag}/{k}": v / max(n, 1) for k, v in tot.items()}


@hydra.main(config_path="../../experiments/stage0_routing_causal_test/configs", version_base="1.3")
def main(cfg: DictConfig):
    import warnings
    warnings.simplefilter("ignore")
    lightning.seed_everything(cfg.seed)

    work_dir = HydraConfig.get().runtime.output_dir
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))

    train_dataset = GatedBCDataset(dataset_dir=list(cfg.train_dataset), **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_loader = get_dataloader(train_dataset, mode="train", num_workers=cfg.num_workers, batch_size=cfg.batch_size)
    val_dataset = GatedBCDataset(dataset_dir=list(cfg.val_dataset), num_demos=cfg.val_num_demos, **cfg.dataset_cfg, aug_prob=0.)
    val_loader = get_dataloader(val_dataset, mode="val", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), strategy="auto")
    fabric.launch()

    model = BCViLTPolicyGated(track_gate_cfg=OmegaConf.to_container(cfg.track_gate_cfg), **cfg.model_cfg)
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    best_loss_logger = BestAvgLoss(window_size=5)
    metrics_path = os.path.join(work_dir, "metrics.jsonl")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        torch.cuda.empty_cache()
        train_metrics = run_epoch(fabric, model, train_loader, optimizer=optimizer,
                                  clip_grad=cfg.clip_grad, scheduler=scheduler,
                                  grad_accum=cfg.get("grad_accum", 1))
        record = {"epoch": epoch, "epoch_seconds": round(time.time() - t0, 1),
                  "lr": optimizer.param_groups[0]["lr"], **train_metrics}

        if epoch % cfg.val_freq == 0:
            val_metrics = run_epoch(fabric, model, val_loader)
            record.update(val_metrics)
            if best_loss_logger.update_best(val_metrics["val/loss"], epoch):
                model.save(os.path.join(work_dir, "model_best.ckpt"))
                with open(os.path.join(work_dir, "best_epoch.txt"), "w") as f:
                    f.write(f"Best epoch: {epoch}, Best loss: {best_loss_logger.best_loss:.4f}")

        if epoch % cfg.save_freq == 0:
            model.save(os.path.join(work_dir, f"model_{epoch}.ckpt"))

        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                             for k, v in record.items()))

    model.save(os.path.join(work_dir, "model_final.ckpt"))
    print(f"finished training in {work_dir}")


if __name__ == "__main__":
    main()
