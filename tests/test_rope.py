import torch

from manas.model.rope import apply_rotary_pos_emb, precompute_freqs_cis, rotate_half


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


def test_rotate_half_is_a_quarter_turn():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))
    torch.testing.assert_close(rotate_half(rotate_half(x)), -x)
