import json

from manas.tokenizer import (
    ADDED_TOKENS,
    BOS_TOKEN,
    CONTROL_TOKENS,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    render_chat,
    token_id,
    tokenizer_config,
)


def test_added_token_layout():
    assert len(ADDED_TOKENS) == 36
    assert len(set(ADDED_TOKENS)) == 36
    assert (token_id(PAD_TOKEN), token_id(BOS_TOKEN), token_id(EOS_TOKEN)) == (0, 1, 2)
    assert token_id("<think>") == 25 and token_id("</think>") == 26
    assert ADDED_TOKENS[-1] == "<|buffer9|>"


def test_control_tokens_are_not_special():
    decoder = tokenizer_config()["added_tokens_decoder"]
    for token in CONTROL_TOKENS:
        assert decoder[str(token_id(token))]["special"] is False
    for token in SPECIAL_TOKENS:
        assert decoder[str(token_id(token))]["special"] is True


def test_config_shape():
    config = tokenizer_config()
    assert config["bos_token"] == "<|im_start|>" and config["eos_token"] == "<|im_end|>"
    assert config["pad_token"] == "<|endoftext|>"
    assert PAD_TOKEN not in config["additional_special_tokens"]
    assert config["chat_template"].startswith("{%- if tools %}")
    json.dumps(config)


def test_plain_chat_renders_with_empty_think_block():
    text = render_chat([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert text == (
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nhello<|im_end|>\n"
    )


def test_system_and_reasoning():
    text = render_chat([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "why?"},
        {"role": "assistant", "content": "because", "reasoning_content": "think first"},
    ])
    assert text.startswith("<|im_start|>system\nbe brief<|im_end|>\n")
    assert "<|im_start|>assistant\n<think>\nthink first\n</think>\n\nbecause<|im_end|>\n" in text


def test_generation_prompt_thinking_modes():
    messages = [{"role": "user", "content": "q"}]
    closed = render_chat(messages, add_generation_prompt=True)
    opened = render_chat(messages, add_generation_prompt=True, open_thinking=True)
    assert closed.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert opened.endswith("<|im_start|>assistant\n<think>\n")


def test_tools_and_tool_response():
    tool = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
    text = render_chat([
        {"role": "system", "content": "you help"},
        {"role": "user", "content": "time?"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "get_time", "arguments": {"tz": "UTC"}}]},
        {"role": "tool", "content": "14:30"},
        {"role": "assistant", "content": "It is 14:30."},
    ], tools=[tool])
    assert text.startswith("<|im_start|>system\nyou help\n\n# Tools")
    assert '<tools>\n{"function": {"name": "get_time"' in text
    assert '<tool_call>\n{"name": "get_time", "arguments": {"tz": "UTC"}}\n</tool_call><|im_end|>\n' in text
    assert "<|im_start|>user\n<tool_response>\n14:30\n</tool_response><|im_end|>\n" in text
