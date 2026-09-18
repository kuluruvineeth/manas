import argparse
import time

import torch
from torch.utils.data import DataLoader

from manas.config import ManasConfig
from manas.data.pretrain import PretrainDataset
from manas.training.utils import autocast_context, get_lr, init_model, log, save_weights, setup_seed, weight_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Manas pretraining")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--save_weight", default="pretrain")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", default="dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--from_weight", default="none")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--max_steps", type=int, default=0)
    return parser.parse_args(argv)


def train_epoch(epoch, args, model, loader, optimizer, scaler, autocast):
    iters = len(loader)
    started = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids, labels = input_ids.to(args.device), labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast:
            loss = model(input_ids, labels=labels).loss / args.accumulation_steps
        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step % args.log_interval == 0 or step == iters:
            eta = (time.time() - started) / step * (iters - step) / 60
            log(
                f"epoch {epoch + 1}/{args.epochs} step {step}/{iters} "
                f"loss {loss.item() * args.accumulation_steps:.4f} lr {lr:.2e} eta {eta:.1f}min"
            )
        if step % args.save_interval == 0 or step == iters:
            save_weights(model, weight_path(args.save_dir, args.save_weight, model.config))
        if args.max_steps and step >= args.max_steps:
            save_weights(model, weight_path(args.save_dir, args.save_weight, model.config))
            break
    return loss.item() * args.accumulation_steps


def train(args):
    setup_seed(args.seed)
    config = ManasConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16" and str(args.device).startswith("cuda"))
    autocast = autocast_context(args.device, args.dtype)
    last_loss = None
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
            num_workers=args.num_workers, pin_memory=str(args.device).startswith("cuda"), drop_last=False,
        )
        last_loss = train_epoch(epoch, args, model, loader, optimizer, scaler, autocast)
        if args.max_steps:
            break
    return last_loss


if __name__ == "__main__":
    train(parse_args())
