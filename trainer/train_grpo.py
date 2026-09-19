import time

import torch
from torch.utils.data import DataLoader

from manas.data.rlaif import RLAIFDataset, collate_prompts, tokenize_prompts
from manas.training.grpo import group_advantages, grpo_loss
from manas.training.loop import build_parser, model_config
from manas.training.metrics import MetricsLogger
from manas.training.ppo import per_token_logps_from_logits
from manas.training.reward import RewardModel, calculate_rewards
from manas.training.rollout import TorchRolloutEngine
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
        "Manas GRPO / CISPO",
        save_weight="grpo",
        batch_size=2,
        learning_rate=3e-7,
        accumulation_steps=1,
        max_seq_len=768,
        data_path="dataset/rlaif.jsonl",
        from_weight="full_sft",
        reward_model_path="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        use_reward_model=1,
        max_gen_len=512,
        temperature=0.8,
        num_generations=6,
        beta=0.1,
        epsilon=0.2,
        epsilon_high=5.0,
        loss_type="cispo",
    )
    parser.set_defaults(epochs=1, log_interval=1, save_interval=10, eval_interval=1000000, val_samples=0)
    return parser.parse_args(argv)


def train(args):
    setup_seed(args.seed)
    config = model_config(args)
    policy, tokenizer = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    ref_model, _ = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    ref_model.eval().requires_grad_(False)

    reward_model = None
    if args.use_reward_model:
        log(f"loading reward model {args.reward_model_path}")
        reward_model = RewardModel(args.reward_model_path, device=args.device)

    autocast = autocast_context(args.device, args.dtype)
    engine = TorchRolloutEngine(policy, tokenizer, args.device, autocast)
    dataset = RLAIFDataset(args.data_path, tokenizer, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_prompts,
                        num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    metrics = MetricsLogger(
        weight_path(args.save_dir, args.save_weight, config).replace(".pth", "_metrics.jsonl"),
        use_wandb=bool(args.use_wandb),
        project=args.wandb_project,
        run_name=args.run_name or args.loss_type,
        config=vars(args),
    )

    iters = len(loader)
    started = time.time()
    last_reward = None
    try:
        for epoch in range(args.epochs):
            for step, prompts in enumerate(loader, start=1):
                lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                prompt_ids, prompt_mask = tokenize_prompts(tokenizer, prompts, args.max_seq_len, args.device)
                result = engine.rollout(
                    prompt_ids, prompt_mask, args.num_generations, args.max_gen_len, args.temperature
                )
                expanded = [p for p in prompts for _ in range(args.num_generations)]
                rewards = calculate_rewards(expanded, result.completions, reward_model, device=args.device)
                advantages = group_advantages(rewards, args.num_generations)
                mask = result.completion_mask.float()
                keep = result.completion_ids.size(1)

                with torch.no_grad(), autocast:
                    ref_logps = per_token_logps_from_logits(
                        ref_model(result.output_ids, logits_to_keep=keep + 1).logits[:, :-1], result.completion_ids
                    )
                with autocast:
                    logits = policy(result.output_ids, logits_to_keep=keep + 1).logits[:, :-1]
                    logps = per_token_logps_from_logits(logits, result.completion_ids)
                    loss, kl = grpo_loss(
                        logps, result.per_token_logps, advantages, mask,
                        beta=args.beta, epsilon=args.epsilon, ref_logps=ref_logps,
                        loss_type=args.loss_type, epsilon_high=args.epsilon_high,
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()

                last_reward = rewards.mean().item()
                if step % args.log_interval == 0 or step == iters:
                    length = mask.sum(dim=1).float().mean().item()
                    spread = rewards.view(-1, args.num_generations).std(dim=1).mean().item()
                    metrics.log(epoch * iters + step, reward=last_reward, loss=loss.item(), kl_to_ref=kl.item(),
                                lr=lr, response_len=length, group_reward_std=spread)
                    log(
                        f"epoch {epoch + 1}/{args.epochs} step {step}/{iters} reward {last_reward:.3f} "
                        f"spread {spread:.3f} kl {kl.item():.4f} len {length:.0f} "
                        f"[{(time.time() - started) / 60:.1f}min]"
                    )
                if step % args.save_interval == 0 or step == iters:
                    save_weights(policy, weight_path(args.save_dir, args.save_weight, config))
                    engine.update_policy(policy)
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
