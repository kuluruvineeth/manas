import torch

from manas.model.sampling import apply_repetition_penalty, sample_next_token, top_k_filter, top_p_filter


def test_top_k_keeps_exactly_k_candidates():
    logits = torch.tensor([[0.1, 2.0, 0.5, 3.0, -1.0]])
    kept = torch.isfinite(top_k_filter(logits.clone(), 2))
    assert kept.sum() == 2 and kept[0, 1] and kept[0, 3]
    assert torch.isfinite(top_k_filter(logits.clone(), 0)).all()


def test_top_p_keeps_the_smallest_nucleus():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    kept = torch.isfinite(top_p_filter(logits.clone(), 0.7))
    assert kept.tolist() == [[True, True, False, False]]
    assert torch.isfinite(top_p_filter(logits.clone(), 0.45)).sum() == 1
    assert torch.isfinite(top_p_filter(logits.clone(), 1.0)).all()


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    seen = torch.tensor([[0, 1]])
    out = apply_repetition_penalty(logits.clone(), seen, 2.0)
    assert out[0, 0] == 1.0 and out[0, 1] == -4.0 and out[0, 2] == 1.0
    assert torch.equal(apply_repetition_penalty(logits.clone(), seen, 1.0), logits)


def test_greedy_and_low_temperature_agree():
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    ids = torch.zeros(1, 1, dtype=torch.long)
    greedy = sample_next_token(logits.clone(), ids, do_sample=False)
    sampled = sample_next_token(logits.clone(), ids, temperature=0.01, top_k=0, top_p=1.0)
    assert greedy.item() == sampled.item() == 1
