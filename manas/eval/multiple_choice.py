import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def choice_log_likelihood(model, tokenizer, context, choices, device="cpu", normalize=True):
    """Score each choice by how likely the model finds it after the context."""
    scores = []
    context_ids = tokenizer(context, add_special_tokens=False).input_ids
    for choice in choices:
        choice_ids = tokenizer(choice, add_special_tokens=False).input_ids
        if not choice_ids:
            scores.append(-math.inf)
            continue
        ids = torch.tensor([context_ids + choice_ids], device=device)
        logits = model(ids).logits[0, :-1]
        targets = ids[0, 1:]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[-len(choice_ids) :]
        total = picked.sum().item()
        scores.append(total / len(choice_ids) if normalize else total)
    return scores


def predict(model, tokenizer, context, choices, device="cpu", normalize=True):
    scores = choice_log_likelihood(model, tokenizer, context, choices, device, normalize)
    return max(range(len(scores)), key=scores.__getitem__), scores


def accuracy(records, model, tokenizer, device="cpu", normalize=True, limit=None):
    correct = 0
    seen = 0
    for record in records[: limit or len(records)]:
        chosen, _ = predict(model, tokenizer, record["context"], record["choices"], device, normalize)
        correct += int(chosen == record["answer"])
        seen += 1
    return correct / max(seen, 1), seen


def random_baseline(records):
    if not records:
        return 0.0
    return sum(1 / len(record["choices"]) for record in records) / len(records)
