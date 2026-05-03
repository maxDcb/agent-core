# agent-core examples

## Quickstart with OpenAI

From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`, then run:

```bash
.venv/bin/python examples/quickstart.py
```

You can pass a prompt directly:

```bash
.venv/bin/python examples/quickstart.py "Echo hello through the echo tool."
```

Or start a small REPL:

```bash
.venv/bin/python examples/quickstart.py --interactive
```

The example persists session memory under `.agent-core-demo/`.

## Pending tool result / resume

`pending_tool_resume.py` shows the async integration pattern without calling an
LLM provider. It uses a fake provider, returns a pending tool result, then
resumes the same turn with host-provided output:

```bash
.venv/bin/python examples/pending_tool_resume.py
```
