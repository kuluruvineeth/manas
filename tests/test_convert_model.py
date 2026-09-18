import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.utils import save_weights, weight_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from convert_model import export, qwen_config  # noqa: E402

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"
SMALL = ManasConfig(hidden_size=64, num_hidden_layers=2)


def test_qwen_config_mirrors_our_shape():
    dense = qwen_config(SMALL)
    assert dense.model_type == "qwen3"
    assert (dense.hidden_size, dense.num_hidden_layers) == (64, 2)
    assert (dense.num_attention_heads, dense.num_key_value_heads) == (8, 4)
    assert dense.rope_parameters["rope_theta"] == 1e6 and dense.vocab_size == 6400
    moe = qwen_config(ManasConfig(hidden_size=64, num_hidden_layers=2, use_moe=True))
    assert moe.model_type == "qwen3_moe"
    assert (moe.num_experts, moe.num_experts_per_tok) == (4, 1)


def test_exported_model_gives_identical_logits(tmp_path):
    torch.manual_seed(0)
    model = ManasForCausalLM(SMALL).eval()
    save_weights(model, weight_path(str(tmp_path), "full_sft", SMALL))
    out_dir = export("full_sft", SMALL, str(tmp_path), str(TOKENIZER_DIR), str(tmp_path / "hf"))
    exported = AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.float32).eval()
    ids = torch.randint(3, SMALL.vocab_size, (1, 7))
    ours = ManasForCausalLM(SMALL)
    ours.load_state_dict({k: v.float() for k, v in torch.load(weight_path(str(tmp_path), "full_sft", SMALL)).items()})
    torch.testing.assert_close(exported(ids).logits, ours.eval()(ids).logits, atol=2e-3, rtol=2e-3)
    config = json.loads((Path(out_dir) / "config.json").read_text())
    assert config["architectures"] == ["Qwen3ForCausalLM"]
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    assert tokenizer.eos_token_id == 2 and "<think>" in tokenizer.chat_template


def test_moe_export_round_trips(tmp_path):
    moe_config = ManasConfig(hidden_size=64, num_hidden_layers=2, use_moe=True)
    torch.manual_seed(0)
    model = ManasForCausalLM(moe_config).eval()
    save_weights(model, weight_path(str(tmp_path), "full_sft", moe_config))
    out_dir = export("full_sft", moe_config, str(tmp_path), str(TOKENIZER_DIR), str(tmp_path / "hf-moe"))
    config = json.loads((Path(out_dir) / "config.json").read_text())
    assert config["architectures"] == ["Qwen3MoeForCausalLM"]
    exported = AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.float32).eval()
    assert exported.config.num_experts == 4 and exported.config.num_experts_per_tok == 1
    ours = ManasForCausalLM(moe_config)
    saved = torch.load(weight_path(str(tmp_path), "full_sft", moe_config))
    ours.load_state_dict({k: v.float() for k, v in saved.items()})
    ids = torch.randint(3, moe_config.vocab_size, (1, 6))
    torch.testing.assert_close(exported(ids).logits, ours.eval()(ids).logits, atol=2e-3, rtol=2e-3)
