import torch

from manas.config import ManasConfig
from manas.model.feed_forward import MOEFeedForward

CONFIG = ManasConfig(hidden_size=32, use_moe=True, num_experts=4, num_experts_per_tok=1)


def make():
    torch.manual_seed(0)
    return MOEFeedForward(CONFIG)


def test_shape_and_expert_count():
    moe = make().eval()
    x = torch.randn(2, 5, 32)
    assert moe(x).shape == x.shape
    assert len(moe.experts) == 4 and moe.gate.weight.shape == (4, 32)
    assert moe.experts[0].gate_proj.weight.shape == (CONFIG.moe_intermediate_size, 32)


def test_top1_routing_weights_are_one_and_each_token_uses_one_expert():
    moe = make()
    x = torch.randn(6, 32)
    scores, weight, idx = moe.route(x)
    torch.testing.assert_close(scores.sum(-1), torch.ones(6))
    torch.testing.assert_close(weight, torch.ones(6, 1))
    assert idx.shape == (6, 1) and idx.max() < 4


def test_output_equals_the_chosen_expert_for_top1():
    moe = make().eval()
    x = torch.randn(1, 4, 32)
    _, _, idx = moe.route(x.view(-1, 32))
    out = moe(x).view(-1, 32)
    for token in range(4):
        expected = moe.experts[int(idx[token])](x.view(-1, 32)[token : token + 1])[0]
        torch.testing.assert_close(out[token], expected, atol=1e-5, rtol=1e-5)


def test_aux_loss_only_in_training_and_rewards_balance():
    moe = make()
    x = torch.randn(4, 8, 32)
    moe.train()
    moe(x)
    training_aux = moe.aux_loss
    assert training_aux.item() > 0 and training_aux.requires_grad
    moe.eval()
    moe(x)
    assert moe.aux_loss.item() == 0.0


def test_unused_experts_still_receive_gradients_in_training():
    moe = make().train()
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0] += 10.0
    x = torch.randn(2, 3, 32)
    (moe(x).sum() + moe.aux_loss).backward()
    assert all(p.grad is not None for expert in moe.experts for p in expert.parameters())
