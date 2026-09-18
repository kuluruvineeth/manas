import torch

from manas.config import ManasConfig
from manas.lora import apply_lora, load_lora, lora_parameters, merge_lora, save_lora
from manas.model.causal_lm import ManasForCausalLM

CONFIG = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)


def make():
    torch.manual_seed(0)
    return ManasForCausalLM(CONFIG).eval()


def test_lora_wraps_only_square_linears_and_starts_as_identity():
    model = make()
    ids = torch.randint(3, CONFIG.vocab_size, (1, 6))
    before = model(ids).logits
    apply_lora(model, rank=4)
    wrapped = [name for name, module in model.named_modules() if hasattr(module, "lora")]
    assert all(name.endswith(("q_proj", "o_proj")) for name in wrapped) and len(wrapped) == 4
    assert not any(name.endswith(("k_proj", "v_proj", "gate_proj", "lm_head")) for name in wrapped)
    torch.testing.assert_close(model(ids).logits, before)
    assert len(lora_parameters(model)) == 8


def test_only_adapter_weights_are_saved_and_reload_round_trips(tmp_path):
    model = apply_lora(make(), rank=4)
    with torch.no_grad():
        for p in lora_parameters(model):
            p.add_(0.5)
    ids = torch.randint(3, CONFIG.vocab_size, (1, 6))
    tuned = model(ids).logits
    save_lora(model, tmp_path / "lora.pth")
    saved = torch.load(tmp_path / "lora.pth")
    assert all(".lora." in k for k in saved) and len(saved) == 8
    assert all(v.dtype == torch.float16 for v in saved.values())
    fresh = apply_lora(make(), rank=4)
    load_lora(fresh, tmp_path / "lora.pth")
    torch.testing.assert_close(fresh(ids).logits, tuned, atol=1e-2, rtol=1e-2)


def test_merge_folds_adapter_into_base_weights(tmp_path):
    model = apply_lora(make(), rank=4)
    with torch.no_grad():
        for p in lora_parameters(model):
            p.add_(0.3)
    ids = torch.randint(3, CONFIG.vocab_size, (1, 6))
    tuned = model(ids).logits
    save_lora(model, tmp_path / "lora.pth")
    merge_lora(model, tmp_path / "lora.pth", tmp_path / "merged.pth")
    merged_state = torch.load(tmp_path / "merged.pth")
    assert not any(".lora." in k for k in merged_state)
    plain = make()
    plain.load_state_dict({k: v.float() for k, v in merged_state.items()}, strict=False)
    torch.testing.assert_close(plain(ids).logits, tuned, atol=5e-2, rtol=5e-2)
