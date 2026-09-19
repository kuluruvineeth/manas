import argparse
import time

import torch
from torch.utils.data import DataLoader, random_split

from manas.training.checkpoint import SkipBatchSampler, load_resume, restore, resume_path, save_resume
from manas.training.metrics import MetricsLogger, evaluate
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
    "eval_interval": 500,
    "val_samples": 256,
    "hidden_size": 768,
    "num_hidden_layers": 8,
    "seed": 42,
    "tokenizer_dir": "tokenizer",
    "max_steps": 0,
    "from_resume": 0,
    "use_wandb": 0,
    "wandb_project": "manas",
    "run_name": "",
    "use_moe": 0,
}


def build_parser(description, **stage_defaults):
    defaults = {**COMMON_DEFAULTS, **stage_defaults}
    parser = argparse.ArgumentParser(description=description)
    for name, value in defaults.items():
        parser.add_argument(f"--{name}", type=type(value), default=value)
    return parser


def batch_tokens(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0].numel()
    return sum(v.numel() for v in batch.values() if torch.is_tensor(v))


def save_full_weights(args, model):
    save_weights(model, weight_path(args.save_dir, args.save_weight, model.config))


def train_epoch(
    epoch, args, model, loader, val_loader, optimizer, scaler, autocast, compute_loss, metrics, save, skip=0
):
    iters = len(loader) + skip
    started = time.time()
    tokens_seen = 0
    loss_value = None
    grad_norm = None
    for step, batch in enumerate(loader, start=skip + 1):
        global_step = epoch * iters + step
        lr = get_lr(global_step, args.epochs * iters, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast:
            loss = compute_loss(model, batch) / args.accumulation_steps
        scaler.scale(loss).backward()
        tokens_seen += batch_tokens(batch)
        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_value = loss.item() * args.accumulation_steps
        if step % args.log_interval == 0 or step == iters:
            elapsed = time.time() - started
            eta = elapsed / max(step - skip, 1) * (iters - step) / 60
            metrics.log(global_step, loss=loss_value, lr=lr, grad_norm=grad_norm, tokens_per_sec=tokens_seen / elapsed)
            log(
                f"epoch {epoch + 1}/{args.epochs} step {step}/{iters} loss {loss_value:.4f} "
                f"lr {lr:.2e} grad {grad_norm or 0:.2f} tok/s {tokens_seen / elapsed:,.0f} eta {eta:.1f}min"
            )
        stopping = bool(args.max_steps) and step >= args.max_steps
        if val_loader is not None and (step % args.eval_interval == 0 or step == iters or stopping):
            val_loss = evaluate(model, val_loader, compute_loss, autocast)
            metrics.log(global_step, val_loss=val_loss)
            log(f"epoch {epoch + 1}/{args.epochs} step {step}/{iters} val_loss {val_loss:.4f}")
        if step % args.save_interval == 0 or step == iters or stopping:
            save(args, model)
            save_resume(
                resume_path(args.save_dir, args.save_weight, model.config),
                model, optimizer, scaler, epoch, step,
                wandb_id=metrics.run_id(),
            )
        if stopping:
            break
    return loss_value


def split_validation(dataset, val_samples, seed):
    val_size = min(val_samples, len(dataset) // 10)
    if val_size == 0:
        return dataset, None
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [len(dataset) - val_size, val_size], generator=generator)


def fit(args, model, dataset, compute_loss, parameters=None, save=save_full_weights):
    optimizer = torch.optim.AdamW(parameters or model.parameters(), lr=args.learning_rate)
    on_cuda = str(args.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16" and on_cuda)
    autocast = autocast_context(args.device, args.dtype)
    train_set, val_set = split_validation(dataset, args.val_samples, args.seed)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    start_epoch, start_step, wandb_id = 0, 0, None
    if args.from_resume:
        bundle = load_resume(resume_path(args.save_dir, args.save_weight, model.config), map_location=args.device)
        if bundle is not None:
            start_epoch, start_step = restore(bundle, model, optimizer, scaler)
            wandb_id = bundle.get("wandb_id")
            log(f"resuming from epoch {start_epoch + 1}, step {start_step}")

    metrics = MetricsLogger(
        weight_path(args.save_dir, args.save_weight, model.config).replace(".pth", "_metrics.jsonl"),
        use_wandb=bool(args.use_wandb),
        project=args.wandb_project,
        run_name=args.run_name or f"{args.save_weight}-{model.config.hidden_size}",
        config=vars(args),
        resume_id=wandb_id,
        append=bool(args.from_resume),
    )
    last_loss = None
    try:
        for epoch in range(start_epoch, args.epochs):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            skip = start_step if epoch == start_epoch else 0
            indices = torch.randperm(len(train_set), generator=generator).tolist()
            loader = DataLoader(
                train_set,
                batch_sampler=SkipBatchSampler(indices, args.batch_size, skip),
                num_workers=args.num_workers,
                pin_memory=on_cuda,
            )
            last_loss = train_epoch(
                epoch, args, model, loader, val_loader, optimizer, scaler, autocast, compute_loss, metrics, save, skip
            )
            if args.max_steps:
                break
    finally:
        metrics.close()
    return last_loss


def language_model_loss(model, batch):
    input_ids, labels = batch
    device = next(model.parameters()).device
    output = model(input_ids.to(device), labels=labels.to(device))
    return output.loss + output.aux_loss


def model_config(args):
    from manas.config import ManasConfig

    return ManasConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)
    )
