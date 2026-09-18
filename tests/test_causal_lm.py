import math

import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM

CONFIG = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)


def make():
    torch.manual_seed(0)
    return ManasForCausalLM(CONFIG).eval()


def test_logits_shape_and_tied_embeddings():
    model = make()
    ids = torch.randint(0, CONFIG.vocab_size, (2, 7))
    out = model(ids)
    assert out.logits.shape == (2, 7, CONFIG.vocab_size)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert out.loss is None


def test_logits_to_keep_returns_only_the_tail():
    model = make()
    ids = torch.randint(0, CONFIG.vocab_size, (1, 7))
    full = model(ids).logits
    tail = model(ids, logits_to_keep=1).logits
    torch.testing.assert_close(tail, full[:, -1:, :])


def test_loss_is_shifted_cross_entropy_and_ignores_minus_100():
    model = make()
    ids = torch.randint(0, CONFIG.vocab_size, (2, 6))
    labels = ids.clone()
    labels[:, :3] = -100
    out = model(ids, labels=labels)
    logits = model(ids).logits[:, :-1].reshape(-1, CONFIG.vocab_size)
    targets = labels[:, 1:].reshape(-1)
    expected = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
    torch.testing.assert_close(out.loss, expected)
    assert math.isclose(out.loss.item(), expected.item())


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = ManasForCausalLM(CONFIG).train()
    ids = torch.randint(0, CONFIG.vocab_size, (4, 12))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = model(ids, labels=ids).loss
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.3


def test_save_and_load_round_trip(tmp_path):
    model = make()
    model.save_pretrained(tmp_path)
    reloaded = ManasForCausalLM.from_pretrained(tmp_path).eval()
    ids = torch.randint(0, CONFIG.vocab_size, (1, 5))
    torch.testing.assert_close(reloaded(ids).logits, model(ids).logits)
    assert reloaded.lm_head.weight.data_ptr() == reloaded.model.embed_tokens.weight.data_ptr()
