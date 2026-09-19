import math

import torch

from manas.training.grpo import group_advantages, grpo_loss, token_kl


def test_group_advantages_centre_each_group_on_zero():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    advantages = group_advantages(rewards, num_generations=3)
    assert advantages.shape == (6,)
    assert abs(advantages[:3].mean().item()) < 1e-5
    assert abs(advantages[3:].mean().item()) < 1e-5
    assert advantages[0] < 0 < advantages[2]
    torch.testing.assert_close(advantages[:3], advantages[3:], atol=1e-4, rtol=1e-4)


def test_identical_rewards_in_a_group_give_no_signal():
    advantages = group_advantages(torch.tensor([5.0, 5.0, 5.0, 5.0]), num_generations=4)
    torch.testing.assert_close(advantages, torch.zeros(4), atol=1e-3, rtol=0)


def test_token_kl_is_zero_for_identical_policies():
    logps = torch.randn(2, 4)
    torch.testing.assert_close(token_kl(logps, logps.clone()), torch.zeros(2, 4), atol=1e-6, rtol=0)
    assert token_kl(logps, logps + 0.5).mean() > 0


def test_cispo_weights_log_probs_by_a_detached_clamped_ratio():
    logps = torch.full((1, 2), -1.0, requires_grad=True)
    old = torch.full((1, 2), -1.0)
    advantages = torch.tensor([2.0])
    mask = torch.ones(1, 2)
    loss, _ = grpo_loss(logps, old, advantages, mask, loss_type="cispo", ref_logps=None)
    assert math.isclose(loss.item(), -(1.0 * 2.0 * -1.0), rel_tol=1e-6)
    loss.backward()
    assert logps.grad is not None and torch.all(logps.grad < 0)


def test_cispo_clamps_a_runaway_ratio():
    old = torch.zeros(1, 1)
    huge = torch.full((1, 1), 10.0)
    loss, _ = grpo_loss(huge, old, torch.tensor([1.0]), torch.ones(1, 1), loss_type="cispo", epsilon_high=5.0)
    assert math.isclose(loss.item(), -(5.0 * 1.0 * 10.0), rel_tol=1e-6)


def test_grpo_clipping_matches_the_ppo_style_objective():
    old = torch.zeros(1, 1)
    new = torch.full((1, 1), 1.0)
    loss, _ = grpo_loss(new, old, torch.tensor([1.0]), torch.ones(1, 1), loss_type="grpo", epsilon=0.2,
                        ref_logps=None)
    assert math.isclose(loss.item(), -1.2, rel_tol=1e-6)


def test_kl_term_pulls_the_loss_up_when_the_policy_drifts():
    logps = torch.zeros(1, 3)
    old = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    without, _ = grpo_loss(logps, old, torch.tensor([1.0]), mask, ref_logps=None)
    with_kl, kl = grpo_loss(logps, old, torch.tensor([1.0]), mask, beta=0.5, ref_logps=logps + 0.6)
    assert with_kl > without and kl > 0


def test_masked_tokens_are_ignored():
    logps = torch.tensor([[0.0, 5.0]])
    old = torch.zeros(1, 2)
    mask = torch.tensor([[1.0, 0.0]])
    loss, _ = grpo_loss(logps, old, torch.tensor([1.0]), mask, loss_type="cispo", ref_logps=None)
    assert math.isclose(loss.item(), 0.0, abs_tol=1e-6)
