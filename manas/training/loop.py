import argparse
import time

import torch
from torch.utils.data import DataLoader

from manas.training.utils import autocast_context, get_lr, log, save_weights, weight_path

COMMON_DEFAULTS = {
    "save_dir": "out",
    "epochs": 2,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "dtype": "bfloat16",
    "num_workers": 8,
    "grad_clip": 1.0,
    "log_interval": 100,
    "save_interval": 1000,
    "hidden_size": 768,
    "num_hidden_layers": 8,
    "seed": 42,
    "tokenizer_dir": "tokenizer",
    "max_steps": 0,
}


def build_parser(description, **stage_defaults):
    defaults = {**COMMON_DEFAULTS, **stage_defaults}
    parser = argparse.ArgumentParser(description=description)
    for name, value in defaults.items():
        kind = type(value) if not isinstance(value, bool) else int
        parser.add_argument(f"--{name}", type=kind, default=value)
    return parser


def train_epoch(epoch, args, model, loader, optimizer, scaler, autocast, compute_loss):
    iters = len(loader)
    started = time.time()
    loss_value = None
    for step, batch in enumerate(loader, start=1):
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast:
            loss = compute_loss(model, batch) / args.accumulation_steps
        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_value = loss.item() * args.accumulation_steps
        if step % args.log_interval == 0 or step == iters:
            eta = (time.time() - started) / step * (iters - step) / 60
            log(
                f"epoch {epoch + 1}/{args.epochs} step {step}/{iters} "
                f"loss {loss_value:.4f} lr {lr:.2e} eta {eta:.1f}min"
            )
        if step % args.save_interval == 0 or step == iters or (args.max_steps and step >= args.max_steps):
            save_weights(model, weight_path(args.save_dir, args.save_weight, model.config))
        if args.max_steps and step >= args.max_steps:
            break
    return loss_value


def fit(args, model, dataset, compute_loss):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    on_cuda = str(args.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16" and on_cuda)
    autocast = autocast_context(args.device, args.dtype)
    last_loss = None
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
            num_workers=args.num_workers, pin_memory=on_cuda,
        )
        last_loss = train_epoch(epoch, args, model, loader, optimizer, scaler, autocast, compute_loss)
        if args.max_steps:
            break
    return last_loss


def language_model_loss(model, batch):
    input_ids, labels = batch
    device = next(model.parameters()).device
    return model(input_ids.to(device), labels=labels.to(device)).loss
