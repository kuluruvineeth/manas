import json

from manas.serve.protocol import (
    DONE,
    chat_completion,
    chunk,
    extract_tool_calls,
    split_reasoning,
    sse,
    strip_tool_calls,
)


def test_reasoning_is_split_from_the_answer():
    assert split_reasoning("plain answer") == (None, "plain answer")
    reasoning, answer = split_reasoning("<think>\nweigh the options\n</think>\n\nThe answer is four.")
    assert reasoning == "weigh the options" and answer == "The answer is four."
    empty, answer = split_reasoning("<think>\n\n</think>\n\nStraight to it.")
    assert empty is None and answer == "Straight to it."


def test_tool_calls_become_openai_shaped():
    text = '<tool_call>\n{"name": "get_time", "arguments": {"tz": "UTC"}}\n</tool_call>'
    calls = extract_tool_calls(text)
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function" and call["function"]["name"] == "get_time"
    assert json.loads(call["function"]["arguments"]) == {"tz": "UTC"}
    assert call["id"].startswith("call_")
    assert extract_tool_calls("<tool_call>\nbroken json\n</tool_call>") == []
    assert strip_tool_calls(f"before {text} after") == "before  after"


def test_completion_carries_every_required_field():
    payload = chat_completion("manas-64m", "Hello there.", prompt_tokens=5, completion_tokens=3)
    assert payload["object"] == "chat.completion" and payload["id"].startswith("chatcmpl-")
    assert isinstance(payload["created"], int) and payload["model"] == "manas-64m"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": "Hello there."}
    assert payload["usage"]["total_tokens"] == 8


def test_completion_reports_tool_calls_and_reasoning():
    text = '<think>\nneed the clock\n</think>\n\n<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'
    payload = chat_completion("manas-64m", text)
    message = payload["choices"][0]["message"]
    assert message["reasoning_content"] == "need the clock"
    assert message["tool_calls"][0]["function"]["name"] == "get_time"
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_chunks_are_well_formed_and_terminated():
    identifier = "chatcmpl-abc"
    payload = chunk("manas-64m", identifier, {"content": "Hi"})
    assert payload["object"] == "chat.completion.chunk" and payload["id"] == identifier
    assert payload["choices"][0]["delta"] == {"content": "Hi"}
    line = sse(payload)
    assert line.startswith("data: ") and line.endswith("\n\n")
    assert json.loads(line[6:].strip())["model"] == "manas-64m"
    assert DONE == "data: [DONE]\n\n"
