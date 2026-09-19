import torch
import torch.nn.functional as F
from torch import nn

from manas.model.causal_lm import ManasForCausalLM


class CriticModel(ManasForCausalLM):
    def __init__(self, config=None):
        super().__init__(config)
        self.value_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def values(self, input_ids, attention_mask=None):
        hidden_states, _, _ = self.model(input_ids, attention_mask)
        return self.value_head(hidden_states).squeeze(-1)


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for step in reversed(range(rewards.size(1))):
        next_value = values[:, step + 1] if step + 1 < values.size(1) else torch.zeros_like(values[:, 0])
        delta = rewards[:, step] + gamma * next_value - values[:, step]
        running = delta + gamma * lam * running
        advantages[:, step] = running
    return advantages * mask, (advantages + values) * mask


def whiten(values, mask):
    total = mask.sum().clamp(min=1)
    mean = (values * mask).sum() / total
    variance = (((values - mean) ** 2) * mask).sum() / total
    return ((values - mean) * torch.rsqrt(variance + 1e-8)) * mask


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1)


def policy_loss(logps, old_logps, advantages, mask, clip_range=0.2):
    log_ratio = (logps - old_logps) * mask
    ratio = torch.exp(log_ratio)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    loss = masked_mean(torch.max(unclipped, clipped), mask)
    clip_fraction = masked_mean((torch.abs(ratio - 1.0) > clip_range).float(), mask)
    approx_kl = masked_mean(0.5 * log_ratio**2, mask)
    return loss, clip_fraction, approx_kl


def kl_penalty(logps, ref_logps, mask):
    log_ratio = (ref_logps - logps) * mask
    return masked_mean(torch.exp(log_ratio) - log_ratio - 1, mask)


def value_loss(values, old_values, returns, mask, clip_range=0.2):
    clipped = old_values + torch.clamp(values - old_values, -clip_range, clip_range)
    losses = torch.max((values - returns) ** 2, (clipped - returns) ** 2)
    return 0.5 * masked_mean(losses, mask)


def sparse_token_rewards(sequence_rewards, mask):
    rewards = torch.zeros_like(mask, dtype=sequence_rewards.dtype)
    lengths = mask.sum(dim=1).long().clamp(min=1) - 1
    rewards[torch.arange(rewards.size(0)), lengths] = sequence_rewards
    return rewards * mask


def per_token_logps_from_logits(logits, targets):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def ppo_objective(args, actor, critic, response_ids, output_ids, mask, old_logps, ref_logps, advantages,
                  old_values, returns):
    keep = response_ids.size(1)
    logits = actor(output_ids, logits_to_keep=keep + 1).logits[:, :-1]
    logps = per_token_logps_from_logits(logits, response_ids)
    loss, clip_fraction, approx_kl = policy_loss(logps, old_logps, advantages, mask, args.clip_range)
    kl = kl_penalty(logps, ref_logps, mask)
    values = critic.values(output_ids)[:, -keep - 1 : -1] * mask
    critic_loss = value_loss(values, old_values, returns, mask, args.clip_range_value)
    total = loss + args.kl_coef * kl + args.vf_coef * critic_loss
    if approx_kl.item() > args.early_stop_kl:
        # Zero the loss instead of breaking: every rank must still run backward or DDP deadlocks.
        total = total * 0.0
    stats = {
        "policy_loss": loss.item(),
        "critic_loss": critic_loss.item(),
        "kl_to_ref": kl.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_fraction.item(),
    }
    return total, stats
