import os

import chainlit as cl
from openai import AsyncOpenAI

BASE_URL = os.environ.get("MANAS_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MANAS_MODEL", "manas-64m")
SYSTEM_PROMPT = os.environ.get("MANAS_SYSTEM", "You are Manas, a small language model trained from scratch.")

client = AsyncOpenAI(base_url=BASE_URL, api_key=os.environ.get("MANAS_API_KEY", "not-needed"))

print(f"[manas] serving {MODEL} from {BASE_URL}")


@cl.set_starters
async def starters():
    return [
        cl.Starter(label="A fact it should know", message="What is the capital of France?"),
        cl.Starter(label="Follow an instruction", message="Write one sentence about the sea."),
        cl.Starter(label="Explain something", message="Why is the sky blue?"),
        cl.Starter(label="Watch it break", message="Who won the 2019 Cricket World Cup final?"),
    ]


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [{"role": "system", "content": SYSTEM_PROMPT}])
    settings = await cl.ChatSettings([
        cl.input_widget.Slider(id="temperature", label="Temperature", initial=0.85, min=0.1, max=1.5, step=0.05),
        cl.input_widget.Slider(id="top_p", label="Top-p", initial=0.95, min=0.1, max=1.0, step=0.05),
        cl.input_widget.Slider(id="max_tokens", label="Max new tokens", initial=256, min=16, max=2048, step=16),
        cl.input_widget.Switch(id="thinking", label="Show thinking", initial=False),
    ]).send()
    cl.user_session.set("settings", settings)


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("settings", settings)


@cl.on_message
async def on_message(message: cl.Message):
    settings = cl.user_session.get("settings") or {}
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    answer = cl.Message(content="")
    thinking_step = None
    thinking_text = ""

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=history,
        temperature=settings.get("temperature", 0.85),
        top_p=settings.get("top_p", 0.95),
        max_tokens=int(settings.get("max_tokens", 256)),
        stream=True,
        extra_body={"chat_template_kwargs": {"open_thinking": bool(settings.get("thinking", False))}},
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
        if delta.content:
            await answer.stream_token(delta.content)

    if thinking_step is not None:
        await thinking_step.__aexit__(None, None, None)
    await answer.send()
    history.append({"role": "assistant", "content": answer.content})
    cl.user_session.set("history", history[-9:])
