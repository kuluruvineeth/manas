import torch

from manas.training.ppo import masked_mean


def group_advantages(rewards, num_generations):
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + 1e-4)).view(-1)


def token_kl(logps, ref_logps):
    log_ratio = ref_logps - logps
    return torch.exp(log_ratio) - log_ratio - 1


def grpo_loss(logps, old_logps, advantages, mask, beta=0.1, epsilon=0.2, ref_logps=None, loss_type="cispo",
              epsilon_high=5.0):
    advantages = advantages.unsqueeze(1)
    ratio = torch.exp(logps - old_logps)
    if loss_type == "cispo":
        weight = torch.clamp(ratio, max=epsilon_high).detach()
        per_token = -(weight * advantages * logps)
    else:
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        per_token = -torch.min(unclipped, clipped)
    if ref_logps is not None:
        per_token = per_token + beta * token_kl(logps, ref_logps)
    per_sequence = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return per_sequence.mean(), masked_mean(token_kl(logps, ref_logps) if ref_logps is not None else
                                            torch.zeros_like(per_token), mask)
