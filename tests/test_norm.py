import torch
import torch.nn.functional as F

from manas.model.norm import RMSNorm


def test_matches_torch_reference():
    torch.manual_seed(0)
    norm = RMSNorm(96, eps=1e-6)
    x = torch.randn(2, 5, 96) * 3
    expected = F.rms_norm(x, (96,), weight=norm.weight, eps=1e-6)
    torch.testing.assert_close(norm(x), expected)


def test_output_has_unit_rms_before_scaling():
    torch.manual_seed(1)
    norm = RMSNorm(64)
    y = norm(torch.randn(4, 64) * 10)
    rms = y.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(4), atol=1e-4, rtol=0)


def test_keeps_input_dtype_and_scale_is_learnable():
    norm = RMSNorm(32)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16
    assert norm.weight.requires_grad
    assert y.float().pow(2).mean(-1).sqrt().mean().item() > 1.5
