import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chat import build_input, parse_args, reply  # noqa: E402

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"


def test_pretrain_weights_get_raw_continuation_and_chat_weights_get_template():
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    history = [{"role": "user", "content": "Tell me a story"}]
    assert build_input(tokenizer, "pretrain", history, 0) == "<|im_start|>Tell me a story"
    chat = build_input(tokenizer, "full_sft", history, 0)
    assert chat.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert build_input(tokenizer, "full_sft", history, 1).endswith("<|im_start|>assistant\n<think>\n")


def test_reply_returns_only_new_text():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    config = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)
    model = ManasForCausalLM(config).eval()
    args = parse_args(["--weight", "full_sft", "--device", "cpu", "--max_new_tokens", "8", "--temperature", "1.0"])
    answer = reply(model, tokenizer, args, [{"role": "user", "content": "hi"}], stream=False)
    assert isinstance(answer, str) and "<|im_start|>" not in answer
