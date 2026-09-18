from contextlib import nullcontext

import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.utils import (
    autocast_context,
    count_params,
    describe_params,
    get_lr,
    init_model,
    save_weights,
    setup_seed,
    unwrap,
    weight_path,
)

CONFIG = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)


def test_cosine_schedule_runs_from_lr_to_a_tenth():
    assert abs(get_lr(0, 100, 1.0) - 1.0) < 1e-9
    assert abs(get_lr(100, 100, 1.0) - 0.1) < 1e-9
    assert abs(get_lr(50, 100, 1.0) - 0.55) < 1e-9
    values = [get_lr(step, 100, 1.0) for step in range(101)]
    assert values == sorted(values, reverse=True)


def test_seed_makes_runs_reproducible():
    setup_seed(7)
    first = torch.randn(3)
    setup_seed(7)
    torch.testing.assert_close(torch.randn(3), first)


def test_save_and_init_from_weight(tmp_path):
    torch.manual_seed(0)
    model = ManasForCausalLM(CONFIG)
    path = weight_path(str(tmp_path), "pretrain", CONFIG)
    assert path.endswith("pretrain_64.pth")
    save_weights(model, path)
    saved = torch.load(path)
    assert all(v.dtype == torch.float16 for v in saved.values())
    loaded, tokenizer = init_model(CONFIG, "pretrain", save_dir=str(tmp_path))
    torch.testing.assert_close(loaded.lm_head.weight, model.lm_head.weight.half().float(), atol=1e-3, rtol=1e-3)
    assert tokenizer.eos_token_id == 2
    fresh, _ = init_model(CONFIG, "none")
    assert not torch.allclose(fresh.lm_head.weight, loaded.lm_head.weight)


def test_unwrap_and_param_count():
    model = ManasForCausalLM(CONFIG)
    assert unwrap(model) is model
    assert 0.5 < count_params(model) < 2.0
    assert describe_params(model, CONFIG).endswith("M") and "-A" not in describe_params(model, CONFIG)


def test_moe_naming_and_active_params():
    moe_config = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
                             use_moe=True)
    assert weight_path("out", "pretrain", moe_config).endswith("pretrain_64_moe.pth")
    description = describe_params(ManasForCausalLM(moe_config), moe_config)
    total, active = description.replace("M", "").split("-A")
    assert float(active) < float(total)


def test_autocast_only_on_cuda():
    assert isinstance(autocast_context("cpu", "bfloat16"), nullcontext)
    assert isinstance(autocast_context("mps", "bfloat16"), nullcontext)
