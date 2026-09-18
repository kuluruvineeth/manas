import torch


def apply_repetition_penalty(logits, input_ids, penalty):
    if penalty == 1.0:
        return logits
    for row in range(input_ids.shape[0]):
        seen = torch.unique(input_ids[row])
        score = logits[row, seen]
        logits[row, seen] = torch.where(score > 0, score / penalty, score * penalty)
    return logits


def top_k_filter(logits, top_k):
    if top_k <= 0:
        return logits
    threshold = torch.topk(logits, min(top_k, logits.shape[-1]))[0][..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits, top_p):
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    return logits.masked_fill(remove.scatter(1, sorted_indices, remove), float("-inf"))


def sample_next_token(logits, input_ids, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, do_sample=True):
    logits = logits / temperature
    logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    if do_sample:
        return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)
