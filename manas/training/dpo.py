import torch
import torch.nn.functional as F


def token_log_probs(logits, labels):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def dpo_loss(policy_log_probs, ref_log_probs, mask, beta):
    policy = (policy_log_probs * mask).sum(dim=1)
    reference = (ref_log_probs * mask).sum(dim=1)
    half = policy.shape[0] // 2
    policy_margin = policy[:half] - policy[half:]
    reference_margin = reference[:half] - reference[half:]
    return -F.logsigmoid(beta * (policy_margin - reference_margin)).mean()


def preference_loss(model, batch, ref_model, beta):
    device = next(model.parameters()).device
    x = torch.cat([batch["x_chosen"], batch["x_rejected"]]).to(device)
    y = torch.cat([batch["y_chosen"], batch["y_rejected"]]).to(device)
    mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]]).to(device)
    with torch.no_grad():
        ref_log_probs = token_log_probs(ref_model(x).logits, y)
    policy = model(x)
    policy_log_probs = token_log_probs(policy.logits, y)
    return dpo_loss(policy_log_probs, ref_log_probs, mask, beta) + policy.aux_loss
