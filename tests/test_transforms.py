import json

from datapipe.schema import validate_line
from datapipe.sources import SOURCES
from datapipe.transforms import TRANSFORMS, parse_tools_block, transform_glaive, transform_hermes_tools

HERMES_SYSTEM = (
    "You are a function calling AI model. You are provided with function signatures within "
    "<tools> </tools> XML tags. You may call one or more functions.\n"
    '<tools>\n[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]\n</tools>'
)


def test_registry_transforms_all_exist():
    for source in SOURCES:
        assert source.transform in TRANSFORMS, source.transform


def test_tools_block_ignores_empty_decoy_pair():
    tools = parse_tools_block(HERMES_SYSTEM)
    assert tools and tools[0]["function"]["name"] == "get_weather"
    assert parse_tools_block("mentions <tools> </tools> but defines none") is None


def test_hermes_round_trip():
    row = {"conversations": [
        {"from": "system", "value": HERMES_SYSTEM},
        {"from": "human", "value": "weather in Paris?"},
        {"from": "gpt", "value": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
        {"from": "tool", "value": "<tool_response>\n18C\n</tool_response>"},
        {"from": "gpt", "value": "It is 18C in Paris."},
    ]}
    result = transform_hermes_tools(row)
    assert result is not None
    conversation = result["conversations"]
    assert json.loads(conversation[0]["tools"])[0]["function"]["name"] == "get_weather"
    assert json.loads(conversation[2]["tool_calls"]) == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    assert conversation[3] == {"role": "tool", "content": "18C"}
    assert validate_line("sft", json.dumps(result)) is None


def test_glaive_single_quoted_arguments():
    row = {
        "system": 'SYSTEM: You have access to: {"name": "get_news", "parameters": {"type": "object"}}',
        "chat": "USER: any news? ASSISTANT: <functioncall> "
                '{"name": "get_news", "arguments": \'{"country": "US"}\'} <|endoftext|> '
                "FUNCTION RESPONSE: {\"headlines\": []} ASSISTANT: Nothing today. <|endoftext|>",
    }
    result = transform_glaive(row)
    assert result is not None
    calls = json.loads(result["conversations"][2]["tool_calls"])
    assert calls[0]["arguments"] == {"country": "US"}
    assert result["conversations"][-1]["content"] == "Nothing today."
    assert validate_line("sft", json.dumps(result)) is None


def test_glaive_without_functions_keeps_plain_chat():
    row = {"system": "SYSTEM: You are a helpful assistant, with no access to external functions.",
           "chat": "USER: hi ASSISTANT: hello there <|endoftext|>"}
    result = transform_glaive(row)
    assert result == {"conversations": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]}


def test_thought_split_and_caps():
    good = {"conversations": [
        {"from": "user", "value": "why is the sky blue?"},
        {"from": "assistant", "value": "<|begin_of_thought|>scattering<|end_of_thought|>\n\n"
                                       "<|begin_of_solution|>Rayleigh scattering.<|end_of_solution|>"},
    ]}
    result = TRANSFORMS["thought_split"](good)
    assert result["conversations"][1]["reasoning_content"] == "scattering"
    long_row = {"conversations": [
        {"from": "user", "value": "q"},
        {"from": "assistant", "value": "<|begin_of_thought|>" + "x" * 3000 + "<|end_of_thought|>"
                                       "<|begin_of_solution|>a<|end_of_solution|>"},
    ]}
    assert TRANSFORMS["thought_split"](long_row) is None


def test_gsm8k_extracts_numeric_gt():
    row = {"question": "Tom has 3 apples and buys 4 more. How many?",
           "answer": "3+4=<<3+4=7>>7\n#### 7"}
    result = TRANSFORMS["gsm8k"](row)
    assert result["gt"] == ["7"]
    assert validate_line("agent", json.dumps(result)) is None


def test_multiple_choice_formats():
    row = {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}
    result = TRANSFORMS["multiple_choice"](row)
    content = result["conversations"][0]["content"]
    assert "A. 3" in content and "B. 4" in content
    assert result["conversations"][1]["content"].endswith("B. 4")
    assert TRANSFORMS["multiple_choice"]({"question": "q", "choices": ["a"], "answer": 3}) is None
