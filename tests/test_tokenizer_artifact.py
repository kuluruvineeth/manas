import json
from pathlib import Path

import pytest
from manas.tokenizer import ADDED_TOKENS, VOCAB_SIZE, render_chat
from transformers import AutoTokenizer

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"

pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(), reason="trained tokenizer not present")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def test_vocab_and_reserved_ids(tokenizer):
    assert len(tokenizer) == VOCAB_SIZE
    for index, token in enumerate(ADDED_TOKENS):
        assert tokenizer.convert_tokens_to_ids(token) == index
    assert (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (0, 1, 2)


def test_round_trip_english(tokenizer):
    text = "Manas is a small language model, trained from scratch on English text."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_control_tokens_survive_skip_special(tokenizer):
    text = "<|im_start|>assistant\n<think>\nplan\n</think>\n\nanswer<|im_end|>"
    ids = tokenizer.encode(text)
    kept = tokenizer.decode(ids, skip_special_tokens=True)
    assert "<think>" in kept and "</think>" in kept and "<|im_start|>" not in kept


def test_apply_chat_template_matches_reference(tokenizer):
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "why is the sky blue?"},
        {"role": "assistant", "content": "scattering", "reasoning_content": "physics"},
        {"role": "user", "content": "thanks"},
    ]
    hf = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=True)
    assert hf == render_chat(messages, add_generation_prompt=True, open_thinking=True)


def test_english_compression_beats_four_chars_per_token(tokenizer):
    sample = (
        "The committee reviewed the proposal carefully before publishing its recommendations, "
        "noting that smaller models trained on cleaner data often generalize surprisingly well."
    )
    ratio = len(sample) / len(tokenizer.encode(sample))
    assert ratio > 4.0


def test_saved_config_is_ours(tokenizer):
    config = json.loads((TOKENIZER_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert config["chat_template"] == tokenizer.chat_template
