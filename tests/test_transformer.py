import torch

from manas.config import ManasConfig
from manas.model.rope import precompute_freqs_cis
from manas.model.transformer import ManasBlock

CONFIG = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)


def make(seq_len=6):
    torch.manual_seed(0)
    block = ManasBlock(CONFIG).eval()
    x = torch.randn(2, seq_len, 64)
    cos, sin = precompute_freqs_cis(CONFIG.head_dim, end=seq_len)
    return block, x, (cos, sin)


def test_block_keeps_shape_and_returns_cache():
    block, x, pos = make()
    out, cache = block(x, pos, use_cache=True)
    assert out.shape == x.shape
    assert cache[0].shape == (2, 6, 2, CONFIG.head_dim)


def test_residual_paths_are_pre_norm():
    block, x, pos = make()
    attn_out, _ = block.self_attn(block.input_layernorm(x), pos)
    after_attn = x + attn_out
    expected = after_attn + block.mlp(block.post_attention_layernorm(after_attn))
    out, _ = block(x, pos)
    torch.testing.assert_close(out, expected)


def test_zeroed_sublayers_make_the_block_an_identity():
    block, x, pos = make()
    with torch.no_grad():
        block.self_attn.o_proj.weight.zero_()
        block.mlp.down_proj.weight.zero_()
    out, _ = block(x, pos)
    torch.testing.assert_close(out, x)
