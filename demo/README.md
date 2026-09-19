# Chat demo

A Chainlit chat UI that talks to any OpenAI-compatible endpoint, so the same interface works
against our own FastAPI server, vLLM, or SGLang without changing a line.

```bash
uv pip install chainlit openai

# point it at whichever engine is running
export MANAS_BASE_URL=http://localhost:8000/v1   # vLLM
export MANAS_MODEL=manas-64m

chainlit run demo/app.py -w
```

The sidebar controls temperature, top-p, max tokens, and whether the model is asked to think
before answering. When it does think, the reasoning appears in a collapsible step above the
answer rather than mixed into it.

Why Chainlit rather than Streamlit: Streamlit re-runs the whole script on every interaction,
which makes token streaming and chat history awkward to keep straight. Chainlit is built around
async message streams, so `stream_token` is the natural unit and thinking blocks get their own
step out of the box.
