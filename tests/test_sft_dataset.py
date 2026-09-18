import json
import random
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from manas.data.sft import (
    EMPTY_THINK,
    SFTDataset,
    assistant_labels,
    maybe_add_system_prompt,
    maybe_drop_empty_think,
    render_conversation,
)

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


CHAT = {"conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And Spain?"},
    {"role": "assistant", "content": "Madrid."},
]}
GET_TIME = json.dumps([{"type": "function", "function": {"name": "get_time", "parameters": {}}}])
TOOL_CHAT = {"conversations": [
    {"role": "system", "content": "help", "tools": GET_TIME},
    {"role": "user", "content": "time?"},
    {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "get_time", "arguments": {"tz": "UTC"}}])},
    {"role": "tool", "content": "14:30"},
    {"role": "assistant", "content": "It is 14:30."},
]}


def test_only_assistant_turns_are_supervised(tokenizer, tmp_path):
    path = tmp_path / "sft.jsonl"
    write_rows(path, [CHAT])
    random.seed(3)
    dataset = SFTDataset(str(path), tokenizer, max_length=96)
    input_ids, labels = dataset[0]
    supervised = tokenizer.decode(input_ids[labels != -100])
    assert "Paris.<|im_end|>" in supervised and "Madrid.<|im_end|>" in supervised
    assert "capital of France" not in supervised and "And Spain" not in supervised
    assert (labels[input_ids == tokenizer.pad_token_id] == -100).all()


def test_tool_conversations_render_and_mask(tokenizer, tmp_path):
    path = tmp_path / "tools.jsonl"
    write_rows(path, [TOOL_CHAT])
    dataset = SFTDataset(str(path), tokenizer, max_length=320)
    input_ids, labels = dataset[0]
    text = tokenizer.decode(input_ids[input_ids != tokenizer.pad_token_id])
    assert "<tools>" in text and "<tool_response>\n14:30" in text
    supervised = tokenizer.decode(input_ids[labels != -100])
    assert '"name": "get_time"' in supervised and "It is 14:30." in supervised
    assert "14:30\n</tool_response>" not in supervised


def test_tool_call_missing_arguments_still_renders(tokenizer):
    conversations = [
        {"role": "system", "content": "help", "tools": GET_TIME},
        {"role": "user", "content": "time?"},
        {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "get_time", "parameters": {}}])},
        {"role": "tool", "content": "14:30"},
        {"role": "assistant", "content": "It is 14:30."},
    ]
    text = render_conversation(tokenizer, conversations)
    assert '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>' in text


def test_system_prompt_injection_respects_tools_and_ratio():
    rng = random.Random(0)
    plain = CHAT["conversations"]
    results = [maybe_add_system_prompt(plain, add_system_ratio=0.5, rng=rng)[0]["role"] for _ in range(200)]
    assert 60 < results.count("system") < 140
    tool_turns = TOOL_CHAT["conversations"]
    assert maybe_add_system_prompt(tool_turns, add_system_ratio=1.0, rng=rng) is tool_turns
    assert maybe_add_system_prompt(plain, add_system_ratio=0.0, rng=rng) is plain


def test_empty_think_block_is_mostly_dropped(tokenizer):
    prompt = render_conversation(tokenizer, CHAT["conversations"])
    assert prompt.count(EMPTY_THINK) == 2
    rng = random.Random(1)
    kept = sum(EMPTY_THINK in maybe_drop_empty_think(prompt, 0.2, rng) for _ in range(500))
    assert 50 < kept < 150


def test_assistant_labels_algorithm_on_tiny_ids():
    start, end = [1, 9], [2, 7]
    ids = [5, 1, 9, 11, 12, 2, 7, 5, 1, 9, 13, 2, 7, 0, 0]
    labels = assistant_labels(ids, start, end, max_length=15)
    assert labels == [-100, -100, -100, 11, 12, 2, 7, -100, -100, -100, 13, 2, 7, -100, -100]
