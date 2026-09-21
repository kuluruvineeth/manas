import json
import os
import sqlite3
from pathlib import Path

import chainlit as cl
import httpx
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from openai import AsyncOpenAI

BASE_URL = os.environ.get("MANAS_BASE_URL", "http://localhost:8000/v1")
SYSTEM_PROMPT = os.environ.get("MANAS_SYSTEM", "You are Manas, a small language model trained from scratch.")
COMPARE = "Compare all"
TOOL_PROFILES = {"agent"}
# A tool call is JSON, and at 0.3 this model leaks the schema into its own arguments.
# The turn that decides the call is sampled cold; the turn that writes prose is not.
TOOL_TEMPERATURE = 0.1
TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. 847 * 23"}},
            "required": ["expression"],
        },
    },
}]
DB_PATH = Path(os.environ.get("MANAS_DB", Path(__file__).parent / "manas_chats.db"))


def ensure_schema():
    schema = (Path(__file__).parent / "schema.sql").read_text()
    with sqlite3.connect(DB_PATH) as db:
        db.executescript(schema)


class SqliteDataLayer(SQLAlchemyDataLayer):
    async def execute_sql(self, query: str, parameters: dict):
        safe = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in parameters.items()}
        return await super().execute_sql(query, safe)


@cl.data_layer
def data_layer():
    ensure_schema()
    return SqliteDataLayer(conninfo=f"sqlite+aiosqlite:///{DB_PATH}")


@cl.password_auth_callback
def auth(username: str, password: str):
    if username == os.environ.get("MANAS_USER", "manas") and password == os.environ.get("MANAS_PASS", "manas"):
        return cl.User(identifier=username, metadata={"role": "admin"})
    return None


@cl.on_chat_resume
async def resume(thread):
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for step in thread.get("steps", []):
        if step.get("type") == "user_message":
            history.append({"role": "user", "content": step.get("output", "")})
        elif step.get("type") == "assistant_message":
            history.append({"role": "assistant", "content": step.get("output", "")})
    cl.user_session.set("history", history[-5:])

STAGES = {
    "pretrain-mini": "raw text prediction, 280M tokens",
    "pretrain": "raw text prediction, 2.2B tokens",
    "sft-mini": "taught to answer, 280M tokens",
    "sft": "taught to answer, 2.2B tokens",
    "dpo": "nudged toward preferred answers",
    "ppo": "trained against a reward model",
    "grpo": "trained against its own batch",
    "agent": "trained to call tools",
    "moe": "four experts, one runs per token",
    "distilled": "taught by a larger model",
    "identity": "a LoRA adapter over sft — 0.62% of the weights",
}

client = AsyncOpenAI(base_url=BASE_URL, api_key=os.environ.get("MANAS_API_KEY", "not-needed"))

print(f"[manas] gallery at {BASE_URL}")


async def available():
    try:
        models = await client.models.list()
        return [m.id for m in models.data]
    except Exception as exc:
        print(f"[manas] could not list models: {exc}")
        return []


@cl.set_chat_profiles
async def profiles():
    names = await available()
    ordered = [n for n in STAGES if n in names] + [n for n in names if n not in STAGES]
    rows = [
        cl.ChatProfile(name=COMPARE, markdown_description="Ask every stage the same question at once.")
    ]
    rows += [
        cl.ChatProfile(name=n, markdown_description=STAGES.get(n, "a Manas checkpoint"))
        for n in ordered
    ]
    return rows


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [{"role": "system", "content": SYSTEM_PROMPT}])
    settings = await cl.ChatSettings([
        cl.input_widget.Slider(id="temperature", label="Temperature", initial=0.3, min=0.1, max=1.5, step=0.05),
        cl.input_widget.Slider(id="top_p", label="Top-p", initial=0.95, min=0.1, max=1.0, step=0.05),
        cl.input_widget.Slider(id="max_tokens", label="Max new tokens", initial=90, min=16, max=1024, step=16),
        cl.input_widget.Switch(id="thinking", label="Show thinking", initial=False),
    ]).send()
    cl.user_session.set("settings", settings)


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("settings", settings)


@cl.set_starters
async def starters():
    return [
        cl.Starter(label="A fact it knows", message="What is the capital of France?"),
        cl.Starter(label="Write something", message="Write a short poem about the ocean."),
        cl.Starter(label="Follow a format", message="Give me three tips for learning to code."),
        cl.Starter(label="Watch it break", message="What is 12 plus 7?"),
    ]


def run_tool(name, arguments):
    if name != "calculator":
        return f"no tool named {name}"
    try:
        expression = json.loads(arguments).get("expression", "")
    except json.JSONDecodeError:
        return f"could not read arguments: {arguments}"
    try:
        from manas.tools.safe_math import safe_math_eval

        return f"{expression} = {safe_math_eval(str(expression))}"
    except Exception as exc:
        return f"tool error: {exc}"


async def compare_all(message, settings):
    payload = {
        "messages": [{"role": "user", "content": message.content}],
        "temperature": settings.get("temperature", 0.3),
        "top_p": settings.get("top_p", 0.95),
        "max_tokens": int(settings.get("max_tokens", 90)),
        "open_thinking": bool(settings.get("thinking", False)),
    }
    holder = cl.Message(content="Asking every stage...")
    await holder.send()
    async with httpx.AsyncClient(timeout=600) as http:
        response = await http.post(BASE_URL.rstrip("/").removesuffix("/v1") + "/v1/compare", json=payload)
    response.raise_for_status()
    rows = response.json()["data"]
    order = [n for n in STAGES if any(r["model"] == n for r in rows)]
    order += [r["model"] for r in rows if r["model"] not in STAGES]
    by_name = {r["model"]: r for r in rows}
    lines = [f"**{message.content}**", ""]
    for name in order:
        answer = by_name[name]["answer"].replace("\n", " ").strip() or "_(nothing)_"
        lines += [f"**{name}** — _{STAGES.get(name, '')}_", "", f"> {answer}", ""]
    holder.content = "\n".join(lines)
    await holder.update()


@cl.on_message
async def on_message(message: cl.Message):
    settings = cl.user_session.get("settings") or {}
    profile = cl.user_session.get("chat_profile")

    if profile == COMPARE:
        await compare_all(message, settings)
        return

    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    answer = cl.Message(content="")
    thinking_step = None
    thinking_text = ""
    pending = []

    extra = {"chat_template_kwargs": {"open_thinking": bool(settings.get("thinking", False))}}
    if profile in TOOL_PROFILES:
        extra["tools"] = TOOLS

    temperature = settings.get("temperature", 0.3)
    stream = await client.chat.completions.create(
        model=profile,
        messages=history,
        temperature=TOOL_TEMPERATURE if profile in TOOL_PROFILES else temperature,
        top_p=settings.get("top_p", 0.95),
        max_tokens=int(settings.get("max_tokens", 90)),
        stream=True,
        extra_body=extra,
    )

    async for part in stream:
        delta = part.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if thinking_step is None:
                thinking_step = cl.Step(name="thinking", type="tool")
                await thinking_step.__aenter__()
            thinking_text += reasoning
            thinking_step.output = thinking_text
            await thinking_step.update()
        for call in getattr(delta, "tool_calls", None) or []:
            fn = getattr(call, "function", None) or {}
            name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "tool")
            args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else "{}")
            pending.append({"name": name, "arguments": args})
        if delta.content:
            await answer.stream_token(delta.content)

    if thinking_step is not None:
        await thinking_step.__aexit__(None, None, None)

    if pending:
        # The model asked for a tool. Run it, hand the result back, and let it finish the
        # sentence — a call with no second turn leaves the answer bubble empty.
        history.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": c} for c in pending
        ]})
        for call in pending:
            async with cl.Step(name=call["name"], type="tool") as step:
                step.input = call["arguments"]
                step.output = run_tool(call["name"], call["arguments"])
                history.append({"role": "tool", "content": step.output})
        follow_up = await client.chat.completions.create(
            model=profile,
            messages=history,
            temperature=temperature,
            top_p=settings.get("top_p", 0.95),
            max_tokens=int(settings.get("max_tokens", 90)),
            stream=True,
            extra_body={"chat_template_kwargs": extra["chat_template_kwargs"]},
        )
        async for part in follow_up:
            if part.choices[0].delta.content:
                await answer.stream_token(part.choices[0].delta.content)

    await answer.send()
    history.append({"role": "assistant", "content": answer.content})
    cl.user_session.set("history", history[-6:])
