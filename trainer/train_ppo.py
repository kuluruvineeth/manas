import time

import torch
from torch.utils.data import DataLoader

from manas.data.rlaif import RLAIFDataset, collate_prompts, tokenize_prompts
from manas.training.loop import build_parser, model_config
from manas.training.metrics import MetricsLogger
from manas.training.ppo import (
    CriticModel,
    compute_gae,
    per_token_logps_from_logits,
    ppo_objective,
    sparse_token_rewards,
    whiten,
)
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
        "Manas PPO",
        save_weight="ppo_actor",
        batch_size=2,
        learning_rate=3e-7,
        accumulation_steps=1,
        max_seq_len=768,
        data_path="dataset/rlaif.jsonl",
        from_weight="full_sft",
        critic_learning_rate=5e-7,
        reward_model_path="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        use_reward_model=1,
        max_gen_len=512,
        temperature=0.8,
        num_generations=1,
        ppo_epochs=2,
        mini_batch_size=2,
        clip_range=0.2,
        clip_range_value=0.2,
        vf_coef=0.5,
        kl_coef=0.02,
        gamma=1.0,
        lam=0.95,
        early_stop_kl=0.25,
    )
    parser.set_defaults(epochs=1, log_interval=1, save_interval=10, eval_interval=1000000, val_samples=0)
    return parser.parse_args(argv)


def rollout_batch(args, engine, tokenizer, prompts, reward_model):
    prompt_ids, prompt_mask = tokenize_prompts(tokenizer, prompts, args.max_seq_len, args.device)
    result = engine.rollout(prompt_ids, prompt_mask, args.num_generations, args.max_gen_len, args.temperature)
    expanded = [p for p in prompts for _ in range(args.num_generations)]
    rewards = calculate_rewards(expanded, result.completions, reward_model, device=args.device)
    return result, rewards


def ppo_step(args, actor, critic, ref_model, result, rewards, autocast, actor_optimizer, critic_optimizer):
    mask = result.completion_mask.float()
    response_ids = result.completion_ids
    keep = response_ids.size(1)
    with torch.no_grad(), autocast:
        old_values = critic.values(result.output_ids)[:, -keep - 1 : -1] * mask
        ref_logps = per_token_logps_from_logits(
            ref_model(result.output_ids, logits_to_keep=keep + 1).logits[:, :-1], response_ids
        )
        token_rewards = sparse_token_rewards(rewards, mask)
        advantages, returns = compute_gae(token_rewards, old_values, mask, args.gamma, args.lam)
        advantages = whiten(advantages, mask)

    stats = {}
    for _ in range(args.ppo_epochs):
        with autocast:
            total, stats = ppo_objective(
                args, actor, critic, response_ids, result.output_ids, mask,
                result.per_token_logps, ref_logps, advantages, old_values, returns,
            )
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
        actor_optimizer.step()
        critic_optimizer.step()
    return stats


def train(args):
    setup_seed(args.seed)
    config = model_config(args)
    actor, tokenizer = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    ref_model, _ = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    ref_model.eval().requires_grad_(False)
    critic = CriticModel(config).to(args.device)
    critic.load_state_dict(actor.state_dict(), strict=False)

    reward_model = None
    if args.use_reward_model:
        log(f"loading reward model {args.reward_model_path}")
        reward_model = RewardModel(args.reward_model_path, device=args.device)

    autocast = autocast_context(args.device, args.dtype)
    engine = TorchRolloutEngine(actor, tokenizer, args.device, autocast)
    dataset = RLAIFDataset(args.data_path, tokenizer, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_prompts,
                        num_workers=args.num_workers)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_learning_rate)
    metrics = MetricsLogger(
        weight_path(args.save_dir, args.save_weight, config).replace(".pth", "_metrics.jsonl"),
        use_wandb=bool(args.use_wandb),
        project=args.wandb_project,
        run_name=args.run_name or "ppo",
        config=vars(args),
    )

    iters = len(loader)
    started = time.time()
    last_reward = None
    try:
        for epoch in range(args.epochs):
            for step, prompts in enumerate(loader, start=1):
                lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
                for group in actor_optimizer.param_groups:
                    group["lr"] = lr
                result, rewards = rollout_batch(args, engine, tokenizer, prompts, reward_model)
                stats = ppo_step(
                    args, actor, critic, ref_model, result, rewards, autocast, actor_optimizer, critic_optimizer
                )
                last_reward = rewards.mean().item()
                if step % args.log_interval == 0 or step == iters:
                    length = result.completion_mask.sum(dim=1).float().mean().item()
                    metrics.log(epoch * iters + step, reward=last_reward, lr=lr, response_len=length, **stats)
                    elapsed = (time.time() - started) / 60
                    log(
                        f"epoch {epoch + 1}/{args.epochs} step {step}/{iters} reward {last_reward:.3f} "
                        f"kl {stats['kl_to_ref']:.4f} clip {stats['clip_fraction']:.2f} len {length:.0f} "
                        f"[{elapsed:.1f}min]"
                    )
                if step % args.save_interval == 0 or step == iters:
                    save_weights(actor, weight_path(args.save_dir, args.save_weight, config))
                    engine.update_policy(actor)
                if args.max_steps and step >= args.max_steps:
                    save_weights(actor, weight_path(args.save_dir, args.save_weight, config))
                    break
            if args.max_steps:
                break
    finally:
        metrics.close()
    return last_reward


if __name__ == "__main__":
    train(parse_args())
