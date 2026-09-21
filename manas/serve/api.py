import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from manas.serve.protocol import (
    DONE,
    chat_completion,
    chunk,
    completion_id,
    extract_tool_calls,
    split_reasoning,
    sse,
    strip_tool_calls,
)
from manas.tokenizer import EMPTY_THINK

MODEL_NAME = "manas-64m"


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list
    temperature: float = 0.85
    top_p: float = 0.95
    max_tokens: int = 512
    repetition_penalty: float = 1.1
    stream: bool = False
    tools: list | None = None
    open_thinking: bool = False
    chat_template_kwargs: dict | None = None
    strip_empty_think: bool = True


def wants_thinking(request):
    extra = request.chat_template_kwargs or {}
    return bool(request.open_thinking or extra.get("open_thinking") or extra.get("enable_thinking"))


def render(tokenizer, request):
    thinking = wants_thinking(request)
    prompt = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=request.tools or None,
        open_thinking=thinking,
    )
    if not thinking and request.strip_empty_think:
        prompt = prompt.replace(EMPTY_THINK, "")
    return prompt


def effective_repetition_penalty(request):
    """A tool call is JSON: it repeats braces, quotes and key names by design. Penalising
    repetition truncates it mid-expression, so the penalty is off whenever tools are offered."""
    return 1.0 if request.tools else request.repetition_penalty


def generate_text(model, tokenizer, request, device):
    prompt = render(tokenizer, request)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=effective_repetition_penalty(request),
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)
    return text, input_ids.shape[1], output.shape[1] - input_ids.shape[1]


def stream_text(model, tokenizer, request, device, model_name):
    identifier = completion_id()
    text, _, _ = generate_text(model, tokenizer, request, device)
    reasoning, answer = split_reasoning(text)
    yield sse(chunk(model_name, identifier, {"role": "assistant"}))
    if reasoning:
        yield sse(chunk(model_name, identifier, {"reasoning_content": reasoning}))
    tool_calls = extract_tool_calls(answer)
    body = strip_tool_calls(answer) if tool_calls else answer
    for piece in body.split(" "):
        if piece:
            yield sse(chunk(model_name, identifier, {"content": piece + " "}))
    if tool_calls:
        yield sse(chunk(model_name, identifier, {"tool_calls": tool_calls}))
    yield sse(chunk(model_name, identifier, {}, finish_reason="tool_calls" if tool_calls else "stop"))
    yield DONE


def create_app(model, tokenizer, device="cpu", model_name=MODEL_NAME):
    app = FastAPI(title="Manas")
    started = time.time()

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "manas"}]}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "uptime_seconds": round(time.time() - started, 1)}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest):
        if request.stream:
            return StreamingResponse(
                stream_text(model, tokenizer, request, device, model_name), media_type="text/event-stream"
            )
        text, prompt_tokens, completion_tokens = generate_text(model, tokenizer, request, device)
        return chat_completion(model_name, text, prompt_tokens, completion_tokens)

    return app


class CompareRequest(BaseModel):
    messages: list
    models: list | None = None
    temperature: float = 0.85
    top_p: float = 0.95
    max_tokens: int = 128
    repetition_penalty: float = 1.1
    open_thinking: bool = False


def create_gallery_app(models, tokenizer, device="cpu"):
    app = FastAPI(title="Manas gallery")
    started = time.time()

    def pick(name):
        if name not in models:
            raise HTTPException(status_code=404, detail=f"unknown model {name}; have {sorted(models)}")
        return models[name]

    @app.get("/v1/models")
    def list_models():
        data = [{"id": name, "object": "model", "owned_by": "manas"} for name in sorted(models)]
        return {"object": "list", "data": data}

    @app.get("/health")
    def health():
        return {"status": "ok", "models": sorted(models), "uptime_seconds": round(time.time() - started, 1)}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest):
        model = pick(request.model)
        if request.stream:
            return StreamingResponse(
                stream_text(model, tokenizer, request, device, request.model), media_type="text/event-stream"
            )
        text, prompt_tokens, completion_tokens = generate_text(model, tokenizer, request, device)
        return chat_completion(request.model, text, prompt_tokens, completion_tokens)

    @app.post("/v1/compare")
    def compare(request: CompareRequest):
        names = request.models or sorted(models)
        answers = []
        for name in names:
            one = ChatRequest(
                model=name,
                messages=request.messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                open_thinking=request.open_thinking,
            )
            text, _, _ = generate_text(pick(name), tokenizer, one, device)
            reasoning, answer = split_reasoning(text)
            answers.append({"model": name, "answer": answer.strip(), "reasoning": reasoning})
        return {"object": "list", "prompt": request.messages, "data": answers}

    return app
