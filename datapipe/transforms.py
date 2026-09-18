import json
import random
import re

from datapipe.tools_en import tools_json

AGENT_SYSTEM_PROMPT = "You are Manas, a helpful assistant. Use the provided tools when they help you answer."

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}

TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
HERMES_SIGNATURE_RE = re.compile(r"You are provided with function signatures.*?$", re.DOTALL)
GLAIVE_TURN_RE = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s*", re.DOTALL)
GLAIVE_FCALL_RE = re.compile(r"<functioncall>\s*(.*?)(?:<\|endoftext\|>|$)", re.DOTALL)
GLAIVE_SQ_ARGS_RE = re.compile(r"'(\{.*?\})'", re.DOTALL)

MAX_REASONING_CHARS = 2400
MAX_SOLUTION_CHARS = 2400

_rng = random.Random(42)

MC_LETTERS = ["A", "B", "C", "D", "E"]
MC_PREFIXES = [
    "Answer the following multiple-choice question.",
    "Choose the correct option.",
    "Pick the right answer.",
    "",
]


def _clean(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text or None


def transform_text(row):
    text = _clean(row.get("text"))
    if text is None:
        return None
    return {"text": text}


def _conversation_from(messages, role_key, content_key):
    conversation = []
    for message in messages:
        role = ROLE_MAP.get(message.get(role_key))
        content = _clean(message.get(content_key))
        if role is None or content is None:
            return None
        conversation.append({"role": role, "content": content})
    return {"conversations": conversation}


def transform_messages(row):
    return _conversation_from(row.get("messages") or [], "role", "content")


def transform_from_value(row):
    return _conversation_from(row.get("conversations") or [], "from", "value")


def parse_tools_block(text):
    candidates = [c for c in TOOLS_RE.findall(text) if c.strip()]
    if not candidates:
        return None
    raw = max(candidates, key=len).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass
    items = []
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            return None
    return items or None


def transform_hermes_tools(row):
    tools = None
    conversation = []
    for message in row.get("conversations") or []:
        role = ROLE_MAP.get(message.get("from"))
        value = message.get("value") or ""
        if role == "system":
            tools = parse_tools_block(value)
            content = HERMES_SIGNATURE_RE.sub("", TOOLS_RE.sub("", value)).strip()
            entry = {"role": "system", "content": content or "You are a function calling AI model."}
            if tools:
                entry["tools"] = json.dumps(tools, ensure_ascii=False)
            conversation.append(entry)
        elif role == "assistant":
            calls = []
            for block in TOOL_CALL_RE.findall(value):
                try:
                    calls.append(json.loads(block))
                except json.JSONDecodeError:
                    return None
            entry = {"role": "assistant", "content": TOOL_CALL_RE.sub("", value).strip()}
            if calls:
                entry["tool_calls"] = json.dumps(calls, ensure_ascii=False)
            conversation.append(entry)
        elif role == "tool":
            responses = TOOL_RESP_RE.findall(value)
            content = "\n".join(r.strip() for r in responses) if responses else value.strip()
            conversation.append({"role": "tool", "content": content})
        elif role == "user":
            content = _clean(value)
            if content is None:
                return None
            conversation.append({"role": "user", "content": content})
        else:
            return None
    if not tools or not conversation:
        return None
    return {"conversations": conversation}


def parse_glaive_functions(system_text):
    start_index = system_text.find("{")
    if start_index == -1:
        return None
    chunk = system_text[start_index:]
    functions = []
    depth = 0
    start = None
    for index, char in enumerate(chunk):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    functions.append(json.loads(chunk[start:index + 1]))
                except json.JSONDecodeError:
                    return None
    if not functions:
        return None
    return [{"type": "function", "function": f} for f in functions]


def transform_glaive(row):
    tools = parse_glaive_functions(row.get("system") or "")
    parts = GLAIVE_TURN_RE.split(row.get("chat") or "")
    if tools:
        conversation = [{
            "role": "system",
            "content": "You are a helpful assistant with access to functions. Use them when needed.",
            "tools": json.dumps(tools, ensure_ascii=False),
        }]
    else:
        conversation = []
    for label, text in zip(parts[1::2], parts[2::2], strict=False):
        text = text.replace("<|endoftext|>", "").strip()
        if label == "USER":
            if not text:
                return None
            conversation.append({"role": "user", "content": text})
        elif label == "FUNCTION RESPONSE":
            conversation.append({"role": "tool", "content": text})
        else:
            match = GLAIVE_FCALL_RE.search(text)
            if match:
                raw = GLAIVE_SQ_ARGS_RE.sub(r"\1", match.group(1).strip())
                try:
                    call = json.loads(raw)
                    if isinstance(call.get("arguments"), str):
                        call["arguments"] = json.loads(call["arguments"])
                except json.JSONDecodeError:
                    return None
                conversation.append({
                    "role": "assistant",
                    "content": GLAIVE_FCALL_RE.sub("", text).strip(),
                    "tool_calls": json.dumps([call], ensure_ascii=False),
                })
            else:
                conversation.append({"role": "assistant", "content": text})
    if not conversation:
        return None
    return {"conversations": conversation}


THOUGHT_RE = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
SOLUTION_RE = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)


def transform_thought_split(row):
    messages = row.get("conversations") or []
    if len(messages) < 2:
        return None
    user = _clean(messages[0].get("value"))
    assistant = messages[1].get("value") or ""
    thought = THOUGHT_RE.search(assistant)
    solution = SOLUTION_RE.search(assistant)
    if not (user and thought and solution):
        return None
    reasoning = thought.group(1).strip()
    answer = solution.group(1).strip()
    if len(reasoning) > MAX_REASONING_CHARS or len(answer) > MAX_SOLUTION_CHARS:
        return None
    return {"conversations": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer, "reasoning_content": reasoning},
    ]}


def transform_chosen_rejected(row):
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not chosen or not rejected:
        return None
    sides = {}
    for name, messages in (("chosen", chosen), ("rejected", rejected)):
        side = []
        for message in messages:
            content = _clean(message.get("content"))
            if content is None or message.get("role") not in ("user", "assistant"):
                return None
            side.append({"role": message["role"], "content": content})
        if side[-1]["role"] != "assistant":
            return None
        sides[name] = side
    if sides["chosen"][-1]["content"] == sides["rejected"][-1]["content"]:
        return None
    return sides


def transform_input_output(row):
    question = _clean(row.get("input"))
    answer = _clean(row.get("output"))
    if not question or not answer:
        return None
    return {"conversations": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def transform_gsm8k(row):
    question = _clean(row.get("question"))
    answer = (row.get("answer") or "").split("####")[-1].strip().replace(",", "")
    if not question or not answer:
        return None
    return {
        "conversations": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT, "tools": tools_json(["calculate_math"])},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "none"},
        ],
        "gt": [answer],
    }


def _multiple_choice(question, choices, answer_index):
    if question is None or answer_index is None or not 0 <= answer_index < len(choices):
        return None
    lines = [f"{MC_LETTERS[i]}. {choice}" for i, choice in enumerate(choices)]
    prefix = _rng.choice(MC_PREFIXES)
    user = (prefix + "\n\n" if prefix else "") + question.strip() + "\n" + "\n".join(lines)
    answer = f"{MC_LETTERS[answer_index]}. {choices[answer_index]}"
    return {"conversations": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": f"The correct answer is {answer}"},
    ]}


def transform_multiple_choice(row):
    return _multiple_choice(row.get("question"), row.get("choices") or [], row.get("answer"))


def _labeled_choice(question, choices, answer_key):
    texts = choices.get("text") or []
    labels = choices.get("label") or []
    index = labels.index(answer_key) if answer_key in labels else None
    return _multiple_choice(question, texts, index)


def transform_arc(row):
    return _labeled_choice(row.get("question"), row.get("choices") or {}, row.get("answerKey"))


def transform_openbookqa(row):
    return _labeled_choice(row.get("question_stem"), row.get("choices") or {}, row.get("answerKey"))


TRANSFORMS = {
    "text": transform_text,
    "messages": transform_messages,
    "from_value": transform_from_value,
    "hermes_tools": transform_hermes_tools,
    "glaive": transform_glaive,
    "thought_split": transform_thought_split,
    "chosen_rejected": transform_chosen_rejected,
    "input_output": transform_input_output,
    "gsm8k": transform_gsm8k,
    "multiple_choice": transform_multiple_choice,
    "arc": transform_arc,
    "openbookqa": transform_openbookqa,
}
