import os

import torch

from manas.training.utils import unwrap


def resume_path(save_dir, name, config):
    suffix = "_moe" if getattr(config, "use_moe", False) else ""
    return os.path.join(save_dir, f"{name}_{config.hidden_size}{suffix}_resume.pth")


def save_resume(path, model, optimizer, scaler, epoch, step, world_size=1, wandb_id=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bundle = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "world_size": world_size,
        "wandb_id": wandb_id,
    }
    temporary = path + ".tmp"
    torch.save(bundle, temporary)
    os.replace(temporary, path)
    return path


def load_resume(path, world_size=1, map_location="cpu"):
    if not os.path.exists(path):
        return None
    bundle = torch.load(path, map_location=map_location, weights_only=False)
    saved_world_size = bundle.get("world_size", 1)
    if saved_world_size != world_size:
        bundle["step"] = bundle.get("step", 0) * saved_world_size // world_size
    return bundle


def restore(bundle, model, optimizer, scaler, strict=True):
    unwrap(model).load_state_dict(bundle["model"], strict=strict)
    optimizer.load_state_dict(bundle["optimizer"])
    scaler.load_state_dict(bundle["scaler"])
    return bundle["epoch"], bundle.get("step", 0)


class SkipBatchSampler:
    def __init__(self, indices, batch_size, skip_batches=0):
        self.indices = indices
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        indices = list(self.indices)
        batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        yield from batches[self.skip_batches :]

    def __len__(self):
        total = (len(self.indices) + self.batch_size - 1) // self.batch_size
        return max(total - self.skip_batches, 0)
