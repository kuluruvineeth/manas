import json
import os
import time

import torch


class MetricsLogger:
    def __init__(self, path, use_wandb=False, project="manas", run_name=None, config=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")  # noqa: SIM115
        self.started = time.time()
        self.wandb = None
        if use_wandb:
            import wandb

            self.wandb = wandb.init(project=project, name=run_name, config=config)

    def log(self, step, **values):
        record = {"step": step, "elapsed": round(time.time() - self.started, 2), **values}
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def close(self):
        self.file.close()
        if self.wandb is not None:
            self.wandb.finish()


@torch.no_grad()
def evaluate(model, loader, compute_loss, autocast, max_batches=None):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        with autocast:
            total += compute_loss(model, batch).item()
        count += 1
    if was_training:
        model.train()
    return total / max(count, 1)


def read_metrics(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
