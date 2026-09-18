import json
from functools import partial

ROLES = {"system", "user", "assistant", "tool"}
DPO_ROLES = {"system", "user", "assistant"}


def check_text(row):
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return "text must be a non-empty string"
    return None


def check_message(message, index):
    role = message.get("role")
    if role not in ROLES:
        return f"message {index}: invalid role {role!r}"
    if not isinstance(message.get("content"), str):
        return f"message {index}: content must be a string"
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        return f"message {index}: reasoning_content must be a string"
    for field, expect_role in (("tools", "system"), ("tool_calls", "assistant")):
        value = message.get(field)
        if value is None:
            continue
        if role != expect_role:
            return f"message {index}: {field} only allowed on {expect_role} messages"
        if not isinstance(value, str):
            return f"message {index}: {field} must be a JSON string"
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return f"message {index}: {field} is not valid JSON"
        if not isinstance(parsed, list):
            return f"message {index}: {field} must encode a list"
        if field == "tool_calls":
            for call in parsed:
                error = check_tool_call(call, index)
                if error:
                    return error
    return None


def check_tool_call(call, index):
    if not isinstance(call, dict):
        return f"message {index}: each tool call must be an object"
    call = call.get("function", call)
    if not isinstance(call.get("name"), str) or not call["name"]:
        return f"message {index}: tool call needs a name"
    if not isinstance(call.get("arguments"), dict | str):
        return f"message {index}: tool call needs arguments"
    return None


def check_conversations(row, require_gt=False):
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return "conversations must be a list with at least 2 messages"
    for index, message in enumerate(conversations):
        error = check_message(message, index)
        if error:
            return error
    roles = [m["role"] for m in conversations]
    if "user" not in roles:
        return "conversations must contain a user message"
    if conversations[-1]["role"] != "assistant":
        return "conversations must end with an assistant message"
    if require_gt:
        gt = row.get("gt")
        if not isinstance(gt, list) or not gt:
            return "gt must be a non-empty list"
        if not all(isinstance(item, str | int | float) for item in gt):
            return "gt items must be strings or numbers"
    return None


def check_dpo(row):
    for side in ("chosen", "rejected"):
        messages = row.get(side)
        if not isinstance(messages, list) or len(messages) < 2:
            return f"{side} must be a list with at least 2 messages"
        for index, message in enumerate(messages):
            if message.get("role") not in DPO_ROLES:
                return f"{side} message {index}: invalid role"
            if not isinstance(message.get("content"), str):
                return f"{side} message {index}: content must be a string"
        if messages[-1]["role"] != "assistant":
            return f"{side} must end with an assistant message"
    return None


CHECKERS = {
    "pretrain": check_text,
    "sft": check_conversations,
    "rlaif": check_conversations,
    "agent": partial(check_conversations, require_gt=True),
    "dpo": check_dpo,
}


def validate_line(kind, line):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return "invalid JSON"
    if not isinstance(row, dict):
        return "row must be an object"
    return CHECKERS[kind](row)


def validate_file(path, kind, limit=None, max_errors=10):
    rows = 0
    errors = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if limit is not None and rows >= limit:
                break
            rows += 1
            error = validate_line(kind, line)
            if error and len(errors) < max_errors:
                errors.append(f"line {line_number}: {error}")
    return rows, errors
