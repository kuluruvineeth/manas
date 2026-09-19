import time

import torch
from torch.utils.data import DataLoader

from manas.data.agent import AgentRLDataset, collate_agent
from manas.training.agent import agent_reward, rollout_trajectory
from manas.training.grpo import group_advantages, grpo_loss
from manas.training.loop import build_parser, model_config
from manas.training.metrics import MetricsLogger
from manas.training.ppo import per_token_logps_from_logits
from manas.training.rollout import per_token_logps
from manas.training.utils import (
    autocast_context,
    get_lr,
    init_model,
    log,
    save_weights,
    setup_seed,
    weight_path,
)


def parse_args(argv=None):
    parser = build_parser(
        "Manas agentic reinforcement learning",
        save_weight="agent",
        batch_size=2,
        learning_rate=3e-7,
        accumulation_steps=1,
        max_seq_len=1024,
        data_path="dataset/agent_rl.jsonl",
        from_weight="full_sft",
        num_generations=4,
        max_turns=3,
        max_gen_len=256,
        max_total_len=2500,
        temperature=0.8,
        thinking_ratio=0.1,
        beta=0.1,
        epsilon=0.2,
        epsilon_high=5.0,
        loss_type="cispo",
        max_rows=0,
    )
    parser.set_defaults(epochs=1, log_interval=1, save_interval=10, eval_interval=1000000, val_samples=0)
    return parser.parse_args(argv)


def pack(trajectories, pad_token_id, max_total_len, device):
    sequences, masks, old_logps = [], [], []
    for trajectory, logps in trajectories:
        ids = trajectory["prompt_ids"] + trajectory["response_ids"]
        mask = [0] * len(trajectory["prompt_ids"]) + trajectory["response_mask"]
        if len(ids) > max_total_len:
            ids, mask = ids[-max_total_len:], mask[-max_total_len:]
            logps = logps[-sum(mask):] if sum(mask) else logps
        sequences.append(ids)
        masks.append(mask)
        old_logps.append(logps)
    width = max(len(ids) for ids in sequences)
    padded = torch.full((len(sequences), width), pad_token_id, dtype=torch.long, device=device)
    mask_tensor = torch.zeros((len(sequences), width), dtype=torch.float, device=device)
    for row, (ids, mask) in enumerate(zip(sequences, masks, strict=True)):
        padded[row, : len(ids)] = torch.tensor(ids, device=device)
        mask_tensor[row, : len(mask)] = torch.tensor(mask, dtype=torch.float, device=device)
    return padded, mask_tensor


def train(args):
    setup_seed(args.seed)
    config = model_config(args)
    policy, tokenizer = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    ref_model, _ = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    ref_model.eval().requires_grad_(False)
    autocast = autocast_context(args.device, args.dtype)

    dataset = AgentRLDataset(args.data_path, max_rows=args.max_rows or None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_agent,
                        num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    metrics = MetricsLogger(
        weight_path(args.save_dir, args.save_weight, config).replace(".pth", "_metrics.jsonl"),
        use_wandb=bool(args.use_wandb),
        project=args.wandb_project,
        run_name=args.run_name or "agent",
        config=vars(args),
    )

    iters = len(loader)
    started = time.time()
    last_reward = None
    generator = torch.Generator().manual_seed(args.seed)
    try:
        for epoch in range(args.epochs):
            for step, rows in enumerate(loader, start=1):
                lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                trajectories, rewards, tool_calls = [], [], 0
                for row in rows:
                    open_thinking = torch.rand(1, generator=generator).item() < args.thinking_ratio
                    for _ in range(args.num_generations):
                        trajectory = rollout_trajectory(
                            policy, tokenizer, row["messages"], row["tools"],
                            max_turns=args.max_turns, max_new_tokens=args.max_gen_len,
                            temperature=args.temperature, open_thinking=open_thinking, device=args.device,
                        )
                        reward = agent_reward(
                            trajectory["text"], row["gt"], trajectory["offered"], trajectory["unfinished"]
                        )
                        trajectories.append((trajectory, None))
                        rewards.append(reward)
                        tool_calls += trajectory["text"].count("<tool_call>")

                padded, mask = pack(trajectories, tokenizer.pad_token_id, args.max_total_len, args.device)
                keep = padded.size(1) - 1
                reward_tensor = torch.tensor(rewards, device=args.device, dtype=torch.float)
                advantages = group_advantages(reward_tensor, args.num_generations)
                response_mask = mask[:, 1:]

                with torch.no_grad(), autocast:
                    old_logps = per_token_logps(policy, padded, keep)
                    ref_logps = per_token_logps_from_logits(
                        ref_model(padded, logits_to_keep=keep + 1).logits[:, :-1], padded[:, 1:]
                    )
                with autocast:
                    logits = policy(padded, logits_to_keep=keep + 1).logits[:, :-1]
                    logps = per_token_logps_from_logits(logits, padded[:, 1:])
                    loss, kl = grpo_loss(
                        logps, old_logps, advantages, response_mask, beta=args.beta, epsilon=args.epsilon,
                        ref_logps=ref_logps, loss_type=args.loss_type, epsilon_high=args.epsilon_high,
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()

                last_reward = reward_tensor.mean().item()
                if step % args.log_interval == 0 or step == iters:
                    calls_per_trajectory = tool_calls / max(len(trajectories), 1)
                    unfinished = sum(t["unfinished"] for t, _ in trajectories) / max(len(trajectories), 1)
                    metrics.log(epoch * iters + step, reward=last_reward, loss=loss.item(), kl_to_ref=kl.item(),
                                lr=lr, tool_calls=calls_per_trajectory, unfinished_rate=unfinished)
                    log(
                        f"epoch {epoch + 1}/{args.epochs} step {step}/{iters} reward {last_reward:.3f} "
                        f"tools/traj {calls_per_trajectory:.2f} unfinished {unfinished:.0%} "
                        f"kl {kl.item():.4f} [{(time.time() - started) / 60:.1f}min]"
                    )
                if step % args.save_interval == 0 or step == iters:
                    save_weights(policy, weight_path(args.save_dir, args.save_weight, config))
                if args.max_steps and step >= args.max_steps:
                    save_weights(policy, weight_path(args.save_dir, args.save_weight, config))
                    break
            if args.max_steps:
                break
    finally:
        metrics.close()
    return last_reward


if __name__ == "__main__":
    train(parse_args())
