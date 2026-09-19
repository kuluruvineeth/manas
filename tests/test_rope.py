import torch

from manas.config import ManasConfig
from manas.model.rope import apply_rotary_pos_emb, precompute_freqs_cis, rotate_half, yarn_correction_range

YARN = {
    "type": "yarn",
    "factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 2048,
    "attention_factor": 1.0,
}


def test_table_shapes_and_first_position_is_identity():
    cos, sin = precompute_freqs_cis(96, end=64)
    assert cos.shape == sin.shape == (64, 96)
    torch.testing.assert_close(cos[0], torch.ones(96))
    torch.testing.assert_close(sin[0], torch.zeros(96))
    torch.testing.assert_close(cos[:, :48], cos[:, 48:])


def test_rotation_preserves_length():
    torch.manual_seed(0)
    cos, sin = precompute_freqs_cis(32, end=16)
    q = torch.randn(2, 16, 4, 32)
    k = torch.randn(2, 16, 4, 32)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_scores_depend_only_on_relative_position():
    torch.manual_seed(1)
    cos, sin = precompute_freqs_cis(32, end=64)
    q = torch.randn(1, 1, 1, 32)
    k = torch.randn(1, 1, 1, 32)

    def score(pos_q, pos_k):
        q_rot, _ = apply_rotary_pos_emb(q, q, cos[pos_q : pos_q + 1], sin[pos_q : pos_q + 1])
        k_rot, _ = apply_rotary_pos_emb(k, k, cos[pos_k : pos_k + 1], sin[pos_k : pos_k + 1])
        return (q_rot * k_rot).sum().item()

    assert abs(score(5, 2) - score(20, 17)) < 1e-4
    assert abs(score(5, 2) - score(5, 4)) > 1e-3


def test_yarn_is_inert_inside_the_trained_window():
    plain = precompute_freqs_cis(96, end=2048, rope_base=1e6)
    stretched = precompute_freqs_cis(96, end=2048, rope_base=1e6, rope_scaling=YARN)
    torch.testing.assert_close(plain[0], stretched[0])
    torch.testing.assert_close(plain[1], stretched[1])


def test_yarn_leaves_fast_dimensions_alone_and_slows_the_rest():
    dim, end = 96, 32768
    plain_cos, _ = precompute_freqs_cis(dim, end=end, rope_base=1e6)
    yarn_cos, _ = precompute_freqs_cis(dim, end=end, rope_base=1e6, rope_scaling=YARN)
    low, high = yarn_correction_range(dim, 1e6, 2048, 32, 1)
    assert 0 <= low < high <= dim // 2 - 1
    torch.testing.assert_close(yarn_cos[:, :low], plain_cos[:, :low])
    assert not torch.allclose(yarn_cos[:, high + 1 : dim // 2], plain_cos[:, high + 1 : dim // 2])


def test_yarn_keeps_far_positions_distinguishable():
    dim, end = 96, 32768
    for scaling, label in ((None, "plain"), (YARN, "yarn")):
        cos, sin = precompute_freqs_cis(dim, end=end, rope_base=1e6, rope_scaling=scaling)
        drift = (cos[20000] - cos[20001]).abs().max().item()
        assert drift > 0, label
    yarn_cos, _ = precompute_freqs_cis(dim, end=end, rope_base=1e6, rope_scaling=YARN)
    plain_cos, _ = precompute_freqs_cis(dim, end=end, rope_base=1e6)
    slow_dims = slice(dim // 2 - 8, dim // 2)
    yarn_spread = (yarn_cos[2048:, slow_dims] - yarn_cos[:-2048, slow_dims]).abs().mean()
    plain_spread = (plain_cos[2048:, slow_dims] - plain_cos[:-2048, slow_dims]).abs().mean()
    assert yarn_spread < plain_spread


def test_attention_factor_scales_both_tables():
    scaling = {**YARN, "attention_factor": 2.0}
    base_cos, base_sin = precompute_freqs_cis(96, end=32768, rope_base=1e6, rope_scaling=YARN)
    scaled_cos, scaled_sin = precompute_freqs_cis(96, end=32768, rope_base=1e6, rope_scaling=scaling)
    torch.testing.assert_close(scaled_cos, base_cos * 2.0)
    torch.testing.assert_close(scaled_sin, base_sin * 2.0)


def test_config_exposes_yarn_only_when_asked():
    assert ManasConfig().rope_scaling is None
    scaling = ManasConfig(inference_rope_scaling=True).rope_scaling
    assert scaling["type"] == "yarn" and scaling["factor"] == 16
    assert scaling["original_max_position_embeddings"] * scaling["factor"] == 32768


def test_rotate_half_is_a_quarter_turn():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))
    torch.testing.assert_close(rotate_half(rotate_half(x)), -x)
