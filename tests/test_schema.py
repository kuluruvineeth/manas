import json

from datapipe.schema import validate_line

VALID = {
    "pretrain": {"text": "The mind is its own place."},
    "sft": {"conversations": [
        {"role": "system", "content": "sys", "tools": json.dumps([{"type": "function"}])},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_content": "greet back"},
    ]},
    "dpo": {
        "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
        "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}],
    },
    "agent": {"conversations": [
        {"role": "system", "content": "sys", "tools": json.dumps([{"type": "function"}])},
        {"role": "user", "content": "what is 2+2?"},
        {"role": "assistant", "content": "none"},
    ], "gt": ["4"]},
    "rlaif": {"conversations": [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "none"},
    ]},
}

INVALID = [
    ("pretrain", {"text": "   "}, "non-empty"),
    ("pretrain", {"body": "x"}, "non-empty"),
    ("sft", {"conversations": [{"role": "user", "content": "only one"}]}, "at least 2"),
    ("sft", {"conversations": [
        {"role": "user", "content": "q"}, {"role": "robot", "content": "a"},
    ]}, "invalid role"),
    ("sft", {"conversations": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": None},
    ]}, "content"),
    ("sft", {"conversations": [
        {"role": "system", "content": "s", "tools": [{"not": "a string"}]},
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
    ]}, "JSON string"),
    ("sft", {"conversations": [
        {"role": "user", "content": "q", "tool_calls": "[]"},
        {"role": "assistant", "content": "a"},
    ]}, "only allowed on assistant"),
    ("sft", {"conversations": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "result"},
    ]}, "end with an assistant"),
    ("sft", {"conversations": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "f", "parameters": {}}])},
    ]}, "needs arguments"),
    ("sft", {"conversations": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": json.dumps([{"arguments": {}}])},
    ]}, "needs a name"),
    ("dpo", {"chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
             "rejected": [{"role": "user", "content": "q"}, {"role": "tool", "content": "a"}]}, "invalid role"),
    ("agent", {"conversations": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "none"},
    ], "gt": []}, "non-empty list"),
    ("agent", {"conversations": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "none"},
    ], "gt": [{"nested": True}]}, "strings or numbers"),
]


def test_valid_rows():
    for kind, row in VALID.items():
        assert validate_line(kind, json.dumps(row)) is None, kind


def test_invalid_rows():
    for kind, row, fragment in INVALID:
        error = validate_line(kind, json.dumps(row))
        assert error is not None, (kind, row)
        assert fragment in error, (error, fragment)


def test_broken_json():
    assert validate_line("pretrain", "{not json") == "invalid JSON"
    assert validate_line("sft", '"just a string"') == "row must be an object"
