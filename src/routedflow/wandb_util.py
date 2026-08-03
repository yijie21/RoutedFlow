"""Fail-safe wandb wrapper (grill 2026-08-02): default ON, --no-wandb to opt out.

Contract shared by all entry points: wandb being unavailable (no package, no
network, bad key, init timeout) degrades to a printed warning and no-ops —
training/eval NEVER crashes or hangs because of logging, and the local
metrics.jsonl / summary.json remain the source of truth.

Organization: single project "routedflow"; run name = local run name;
job_type = stage1 | stage2 | rollout; per-epoch granularity mirroring jsonl.
"""

PROJECT = "routedflow"


def flatten(row):
    """{"train": {"loss": x}, "sec": s} -> {"train/loss": x, "sec": s}"""
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}/{k2}"] = v2
        else:
            out[k] = v
    return out


class WandbLogger:
    def __init__(self, name, job_type, config=None, enabled=True):
        self._run = None
        if not enabled:
            return
        try:
            import wandb
            self._run = wandb.init(project=PROJECT, name=name, job_type=job_type,
                                   config=config,
                                   settings=wandb.Settings(init_timeout=30))
            print(f"[wandb] {self._run.url}", flush=True)
        except Exception as e:  # noqa: BLE001 — degrade, never break the caller
            print(f"[wandb] logging disabled ({type(e).__name__}: {e})", flush=True)

    def log(self, metrics, step=None):
        if self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception:
            pass

    def log_table(self, key, columns, rows):
        if self._run is None:
            return
        try:
            import wandb
            self._run.log({key: wandb.Table(columns=columns, data=rows)})
        except Exception:
            pass

    def summary(self, **kv):
        if self._run is None:
            return
        try:
            self._run.summary.update(kv)
        except Exception:
            pass

    def finish(self):
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:
            pass
