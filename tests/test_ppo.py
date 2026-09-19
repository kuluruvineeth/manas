import math

import torch

from manas.config import ManasConfig
from manas.training.ppo import (
    CriticModel,
    compute_gae,
    kl_penalty,
    masked_mean,
    policy_loss,
    sparse_token_rewards,
    value_loss,
    whiten,
)

CONFIG = ManasConfig(hidden_size=64, num_hidden_layers=2)


def test_critic_has_a_value_head_and_keeps_the_language_head():
    torch.manual_seed(0)
    critic = CriticModel(CONFIG)
    ids = torch.randint(3, CONFIG.vocab_size, (2, 6))
    values = critic.values(ids)
    assert values.shape == (2, 6)
    assert hasattr(critic, "lm_head"), "lm_head stays attached so DDP sees every parameter"
    assert critic(ids).logits.shape == (2, 6, CONFIG.vocab_size)


def test_gae_with_a_single_terminal_reward_discounts_backwards():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    expected = torch.tensor([[0.95**2, 0.95, 1.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_gae_is_zero_when_values_already_predict_the_reward():
    rewards = torch.tensor([[0.0, 0.0]])
    values = torch.tensor([[0.0, 0.0]])
    advantages, _ = compute_gae(rewards, values, torch.ones(1, 2))
    torch.testing.assert_close(advantages, torch.zeros(1, 2))


def test_whiten_normalizes_only_the_unmasked_entries():
    values = torch.tensor([[1.0, 2.0, 3.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    whitened = whiten(values, mask)
    assert abs(masked_mean(whitened, mask).item()) < 1e-6
    assert whitened[0, 3] == 0.0


def test_policy_loss_is_the_negative_advantage_when_the_policy_has_not_moved():
    logps = torch.zeros(2, 4)
    advantages = torch.full((2, 4), 0.5)
    mask = torch.ones(2, 4)
    loss, clip_fraction, approx_kl = policy_loss(logps, logps.clone(), advantages, mask)
    assert math.isclose(loss.item(), -0.5, rel_tol=1e-6)
    assert clip_fraction.item() == 0.0 and approx_kl.item() == 0.0


def test_policy_loss_clips_a_large_positive_move():
    old = torch.zeros(1, 1)
    new = torch.full((1, 1), 1.0)
    advantages = torch.ones(1, 1)
    mask = torch.ones(1, 1)
    loss, clip_fraction, approx_kl = policy_loss(new, old, advantages, mask, clip_range=0.2)
    assert math.isclose(loss.item(), -1.2, rel_tol=1e-6)
    assert clip_fraction.item() == 1.0 and approx_kl.item() > 0


def test_kl_penalty_is_zero_for_identical_policies_and_positive_otherwise():
    logps = torch.randn(2, 5)
    mask = torch.ones(2, 5)
    assert abs(kl_penalty(logps, logps.clone(), mask).item()) < 1e-6
    assert kl_penalty(logps, logps + 0.4, mask).item() > 0


def test_value_loss_clipping_limits_a_big_update():
    old = torch.zeros(1, 2)
    returns = torch.full((1, 2), 5.0)
    mask = torch.ones(1, 2)
    small = value_loss(torch.full((1, 2), 0.1), old, returns, mask)
    huge = value_loss(torch.full((1, 2), 50.0), old, returns, mask)
    assert huge > small
    clipped_prediction = value_loss(torch.full((1, 2), 0.2), old, returns, mask, clip_range=0.2)
    assert math.isclose(clipped_prediction.item(), 0.5 * (5.0 - 0.2) ** 2, rel_tol=1e-6)


def test_sparse_rewards_land_on_the_last_real_token():
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    rewards = sparse_token_rewards(torch.tensor([2.0, -1.0]), mask)
    assert rewards.tolist() == [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]
