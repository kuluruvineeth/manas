import torch
import torch.nn.functional as F

from manas.config import ManasConfig
from manas.model.feed_forward import FeedForward

CONFIG = ManasConfig(hidden_size=64)


def test_shapes_follow_config():
    ffn = FeedForward(CONFIG)
    assert ffn.gate_proj.weight.shape == (CONFIG.intermediate_size, 64)
    assert ffn.down_proj.weight.shape == (64, CONFIG.intermediate_size)
    x = torch.randn(2, 5, 64)
    assert ffn(x).shape == x.shape


def test_swiglu_formula():
    torch.manual_seed(0)
    ffn = FeedForward(CONFIG)
    x = torch.randn(3, 64)
    expected = ffn.down_proj(F.silu(ffn.gate_proj(x)) * ffn.up_proj(x))
    torch.testing.assert_close(ffn(x), expected)


def test_intermediate_override_and_no_bias():
    ffn = FeedForward(CONFIG, intermediate_size=128)
    assert ffn.up_proj.weight.shape == (128, 64)
    assert all(m.bias is None for m in (ffn.gate_proj, ffn.up_proj, ffn.down_proj))
