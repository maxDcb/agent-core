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

You can also run a provider compatibility check against the provider selected
by `LLM_PROVIDER`. This exercises plain chat, tool calls, `response_format`
JSON Schema enforcement, and `StructuredTaskRunner` final output:

```bash
.venv/bin/python examples/quickstart.py --compat-check
```

To run the same checks through LangChain's Azure OpenAI model integration:

```bash
LLM_PROVIDER=azure_openai
AGENT_CORE_MODEL_BACKEND=langchain
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AGENT_CORE_MODEL=<deployment-name>
AGENT_CORE_MEMORY_MODEL=<deployment-name>

.venv/bin/python examples/quickstart.py --compat-check
```

Only model invocation changes. The agent loop, tool execution, structured
tasks, persistence, and memory lifecycle remain implemented by agent-core.

For Azure Anthropic / Claude on Azure Foundry, use the `/anthropic` endpoint:

```bash
LLM_PROVIDER=azure_anthropic
AZURE_ANTHROPIC_ENDPOINT=https://<resource>.services.ai.azure.com/anthropic
AZURE_ANTHROPIC_API_KEY=...
AZURE_ANTHROPIC_VERSION=2023-06-01
AGENT_CORE_MODEL=claude-opus-4-6
AGENT_CORE_MEMORY_MODEL=claude-opus-4-6
```

The Azure model version shown in the portal, for example `1`, is the model
deployment version and is not the runtime `api-version` query parameter.

The example persists session memory under `.agent-core-demo/`.

## Pending tool result / resume

`pending_tool_resume.py` shows the async integration pattern without calling an
LLM provider. It uses a fake provider, returns a pending tool result, then
resumes the same turn with host-provided output:

```bash
.venv/bin/python examples/pending_tool_resume.py
```
