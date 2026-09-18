import torch

from manas.config import ManasConfig
from manas.model.attention import Attention, repeat_kv
from manas.model.rope import apply_rotary_pos_emb, precompute_freqs_cis

CONFIG = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)


def make(seq_len=8, batch=2, seed=0):
    torch.manual_seed(seed)
    attn = Attention(CONFIG).eval()
    x = torch.randn(batch, seq_len, CONFIG.hidden_size)
    cos, sin = precompute_freqs_cis(CONFIG.head_dim, end=seq_len)
    return attn, x, (cos, sin)


def test_repeat_kv_duplicates_heads_in_groups():
    x = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).view(2, 3, 2, 4)
    out = repeat_kv(x, 2)
    assert out.shape == (2, 3, 4, 4)
    torch.testing.assert_close(out[:, :, 0], out[:, :, 1])
    torch.testing.assert_close(out[:, :, 2], x[:, :, 1])
    assert repeat_kv(x, 1) is x


def test_output_shape_and_grouped_projections():
    attn, x, pos = make()
    out, past = attn(x, pos)
    assert out.shape == x.shape and past is None
    assert attn.k_proj.weight.shape == (2 * CONFIG.head_dim, 64)
    assert attn.q_proj.weight.shape == (4 * CONFIG.head_dim, 64)


def test_causal_future_tokens_do_not_leak():
    attn, x, pos = make()
    base, _ = attn(x, pos)
    changed = x.clone()
    changed[:, -1] += 5.0
    out, _ = attn(changed, pos)
    torch.testing.assert_close(out[:, :-1], base[:, :-1])
    assert not torch.allclose(out[:, -1], base[:, -1])


def test_padding_mask_removes_padded_keys():
    attn, x, pos = make()
    mask = torch.ones(2, 8)
    mask[:, 0] = 0
    masked, _ = attn(x, pos, attention_mask=mask)
    x_alt = x.clone()
    x_alt[:, 0] += 3.0
    masked_alt, _ = attn(x_alt, pos, attention_mask=mask)
    torch.testing.assert_close(masked[:, 1:], masked_alt[:, 1:], atol=1e-5, rtol=1e-5)


def test_matches_naive_reference():
    attn, x, (cos, sin) = make(seq_len=5, batch=1)
    q = attn.q_norm(attn.q_proj(x).view(1, 5, 4, -1))
    k = attn.k_norm(attn.k_proj(x).view(1, 5, 2, -1))
    v = attn.v_proj(x).view(1, 5, 2, -1)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    outputs = []
    for h in range(4):
        kh, vh = k[0, :, h // 2], v[0, :, h // 2]
        scores = q[0, :, h] @ kh.T / (CONFIG.head_dim**0.5)
        scores = scores.masked_fill(torch.ones(5, 5).triu(1).bool(), float("-inf"))
        outputs.append(torch.softmax(scores, -1) @ vh)
    expected = attn.o_proj(torch.cat(outputs, dim=-1)).unsqueeze(0)
    out, _ = attn(x, (cos, sin))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_fused_kernel_matches_manual_path():
    attn, x, pos = make(seq_len=8, batch=2)
    assert attn.flash
    fused, _ = attn(x, pos)
    attn.flash = False
    manual, _ = attn(x, pos)
    torch.testing.assert_close(fused, manual, atol=1e-5, rtol=1e-5)
    attn.flash = True
    mask = torch.ones(2, 8)
    mask[:, 0] = 0
    attn.flash = False
    manual_masked, _ = attn(x, pos, attention_mask=mask)
    attn.flash = True
    fused_masked, _ = attn(x, pos, attention_mask=mask)
    torch.testing.assert_close(fused_masked, manual_masked)


def test_kv_cache_matches_full_forward():
    attn, x, (cos, sin) = make(seq_len=8, batch=1)
    full, _ = attn(x, (cos, sin))
    prefix, cache = attn(x[:, :5], (cos[:5], sin[:5]), use_cache=True)
    assert cache[0].shape == (1, 5, 2, CONFIG.head_dim)
    step, cache = attn(x[:, 5:6], (cos[5:6], sin[5:6]), past_key_value=cache, use_cache=True)
    assert cache[0].shape[1] == 6
    tail, _ = attn(x[:, 6:], (cos[6:], sin[6:]), past_key_value=cache)
    torch.testing.assert_close(torch.cat([prefix, step, tail], dim=1), full, atol=1e-5, rtol=1e-5)
