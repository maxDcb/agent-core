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

You can also run a provider compatibility check against OpenAI-compatible local
endpoints. This exercises plain chat, tool calls, `response_format` JSON object,
`response_format` JSON schema with fallback, and `StructuredTaskRunner` final
output:

```bash
.venv/bin/python examples/quickstart.py --compat-check
```

The example persists session memory under `.agent-core-demo/`.

## Pending tool result / resume

`pending_tool_resume.py` shows the async integration pattern without calling an
LLM provider. It uses a fake provider, returns a pending tool result, then
resumes the same turn with host-provided output:

```bash
.venv/bin/python examples/pending_tool_resume.py
```
