import json
import re
import time
import uuid

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_END = "</think>"


def completion_id():
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def split_reasoning(text):
    if THINK_END not in text:
        return None, text
    reasoning, _, answer = text.partition(THINK_END)
    return reasoning.replace("<think>", "").strip() or None, answer.lstrip("\n")


def extract_tool_calls(text):
    calls = []
    for index, block in enumerate(TOOL_CALL_BLOCK.findall(text)):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        arguments = parsed.get("arguments", {})
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:16]}_{index}",
            "type": "function",
            "function": {
                "name": parsed.get("name", ""),
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            },
        })
    return calls


def strip_tool_calls(text):
    return TOOL_CALL_BLOCK.sub("", text).strip()


def chat_completion(model, text, prompt_tokens=0, completion_tokens=0):
    reasoning, answer = split_reasoning(text)
    tool_calls = extract_tool_calls(answer)
    message = {"role": "assistant", "content": strip_tool_calls(answer) if tool_calls else answer}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chunk(model, identifier, delta, finish_reason=None):
    return {
        "id": identifier,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


DONE = "data: [DONE]\n\n"
