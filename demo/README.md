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

## Saved conversations

Chainlit keeps a sidebar of past chats when a data layer is configured. This app uses SQLite,
which needs no server.

```bash
uv pip install chainlit openai httpx sqlalchemy aiosqlite greenlet

export CHAINLIT_AUTH_SECRET=$(python -c "import secrets;print(secrets.token_hex(32))")
export MANAS_BASE_URL=https://vineethreddy3268--manas-gallery-gallery.modal.run/v1
CHAINLIT_APP_ROOT=demo chainlit run demo/app.py --port 8501
```

Log in with `manas` / `manas` (override with `MANAS_USER` and `MANAS_PASS`). Conversations land
in `demo/manas_chats.db` and appear in the left sidebar; clicking one resumes it with its
history restored.

Persistence requires authentication because threads belong to a user — that is a Chainlit
constraint, not a choice. `schema.sql` is the table definition the SQLAlchemy data layer
expects; it is created automatically on first run.
