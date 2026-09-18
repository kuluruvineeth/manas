import torch

from manas.config import ManasConfig
from manas.model.rope import precompute_freqs_cis
from manas.model.transformer import ManasBlock, ManasModel

CONFIG = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)


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


def test_model_shapes_and_rope_tables():
    torch.manual_seed(0)
    model = ManasModel(CONFIG).eval()
    ids = torch.randint(0, CONFIG.vocab_size, (2, 7))
    hidden, presents, aux_loss = model(ids)
    assert hidden.shape == (2, 7, 64)
    assert presents == [None, None]
    assert aux_loss.item() == 0.0
    cos, sin = model.rope_tables(hidden.device)
    assert cos.shape == sin.shape == (CONFIG.max_position_embeddings, CONFIG.head_dim)
    assert cos[0, 0] == 1.0
    assert not any("freqs" in key for key in model.state_dict())


def test_model_incremental_decode_matches_full():
    torch.manual_seed(0)
    model = ManasModel(CONFIG).eval()
    ids = torch.randint(0, CONFIG.vocab_size, (1, 9))
    full, _, _ = model(ids)
    prefix, cache, _ = model(ids[:, :6], use_cache=True)
    step, cache, _ = model(ids[:, 6:7], past_key_values=cache, use_cache=True)
    tail, _, _ = model(ids[:, 7:], past_key_values=cache)
    torch.testing.assert_close(torch.cat([prefix, step, tail], dim=1), full, atol=1e-5, rtol=1e-5)


def test_default_model_parameter_count():
    model = ManasModel(ManasConfig())
    params = sum(p.numel() for p in model.parameters())
    assert 63_000_000 < params < 65_000_000
