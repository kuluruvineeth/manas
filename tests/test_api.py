import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.serve.api import (
    ChatRequest,
    create_app,
    effective_repetition_penalty,
    render,
    wants_thinking,
)
from manas.tokenizer import EMPTY_THINK

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"
CONFIG = ManasConfig(hidden_size=64, num_hidden_layers=2)


@pytest.fixture(scope="module")
def client():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    model = ManasForCausalLM(CONFIG).eval()
    return TestClient(create_app(model, tokenizer))


def test_health_and_model_listing(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model"] == "manas-64m"
    models = client.get("/v1/models").json()
    assert models["object"] == "list" and models["data"][0]["id"] == "manas-64m"


def test_chat_completion_has_the_openai_shape(client):
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}], "max_tokens": 6,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion" and payload["id"].startswith("chatcmpl-")
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["finish_reason"] in {"stop", "tool_calls"}
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["usage"]["total_tokens"] >= payload["usage"]["prompt_tokens"]


def test_streaming_ends_with_the_done_sentinel(client):
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}], "max_tokens": 6, "stream": True,
    }) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line.strip()]
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0][6:])
    assert first["object"] == "chat.completion.chunk" and first["choices"][0]["delta"]["role"] == "assistant"
    identifiers = {json.loads(line[6:])["id"] for line in lines[:-1]}
    assert len(identifiers) == 1, "every chunk in a stream shares one id"
    assert json.loads(lines[-2][6:])["choices"][0]["finish_reason"] in {"stop", "tool_calls"}


def test_thinking_can_be_requested_three_ways():
    base = {"messages": [{"role": "user", "content": "hi"}]}
    assert not wants_thinking(ChatRequest(**base))
    assert wants_thinking(ChatRequest(**base, open_thinking=True))
    assert wants_thinking(ChatRequest(**base, chat_template_kwargs={"open_thinking": True}))
    assert wants_thinking(ChatRequest(**base, chat_template_kwargs={"enable_thinking": True}))


def test_empty_think_block_is_stripped_when_not_thinking():
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    base = {"messages": [{"role": "user", "content": "hi"}]}
    assert EMPTY_THINK not in render(tokenizer, ChatRequest(**base))
    assert EMPTY_THINK in render(tokenizer, ChatRequest(**base, strip_empty_think=False))
    thinking = render(tokenizer, ChatRequest(**base, open_thinking=True))
    assert thinking.endswith("<think>\n") and EMPTY_THINK not in thinking


def test_tools_are_accepted_in_the_request(client):
    tool = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "time?"}], "tools": [tool], "max_tokens": 6,
    })
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["role"] == "assistant"


def test_repetition_penalty_is_dropped_when_tools_are_offered():
    # A tool call repeats braces, quotes and key names by design; penalising that truncates
    # the call mid-expression and the calculator gets "847" instead of "847 * 23".
    tools = [{"type": "function", "function": {"name": "calculator", "parameters": {}}}]
    assert effective_repetition_penalty(ChatRequest(messages=[], tools=tools)) == 1.0
    assert effective_repetition_penalty(ChatRequest(messages=[])) == 1.1
