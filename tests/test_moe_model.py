import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.model.feed_forward import MOEFeedForward
from manas.training.utils import count_params

SMALL = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2, use_moe=True)


def test_moe_model_uses_expert_blocks_and_reports_aux_loss():
    torch.manual_seed(0)
    model = ManasForCausalLM(SMALL)
    assert all(isinstance(layer.mlp, MOEFeedForward) for layer in model.model.layers)
    ids = torch.randint(3, SMALL.vocab_size, (2, 8))
    model.train()
    out = model(ids, labels=ids)
    assert out.aux_loss.item() > 0
    (out.loss + out.aux_loss).backward()
    model.eval()
    assert model(ids).aux_loss.item() == 0.0


def test_moe_model_generates_and_round_trips(tmp_path):
    torch.manual_seed(0)
    model = ManasForCausalLM(SMALL).eval()
    prompt = torch.randint(3, SMALL.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=5, do_sample=False, eos_token_id=None)
    assert out.shape == (1, 9)
    model.save_pretrained(tmp_path)
    reloaded = ManasForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(reloaded(prompt).logits, model(prompt).logits)


def test_default_moe_is_198m_total_64m_active():
    total = count_params(ManasForCausalLM(ManasConfig(use_moe=True)))
    assert 196 < total < 200
