import math
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from manas.model.causal_lm import ManasForCausalLM

TOKENIZER_DIR = "tokenizer"


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(message):
    if is_main_process():
        print(message, flush=True)


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def weight_path(save_dir, name, config):
    return os.path.join(save_dir, f"{name}_{config.hidden_size}.pth")


def unwrap(model):
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def save_weights(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {k: v.detach().half().cpu() for k, v in unwrap(model).state_dict().items()}
    torch.save(state, path)


def init_model(config, from_weight="none", save_dir="out", device="cpu", tokenizer_dir=TOKENIZER_DIR):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = ManasForCausalLM(config)
    if from_weight != "none":
        state = torch.load(weight_path(save_dir, from_weight, config), map_location=device)
        model.load_state_dict(state, strict=False)
    log(f"model params: {count_params(model):.2f}M")
    return model.to(device), tokenizer


def autocast_context(device, dtype):
    if not str(device).startswith("cuda"):
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16)
