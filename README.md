# agent-core

Reusable, domain-agnostic runtime for tool-using LLM agents.

`agent-core` provides the orchestration layer that applications can compose with
their own prompts, tools, policy rules, storage and domain memory. It is meant
to be the generic runtime under an agent application, not a domain-specific
agent package.

Version `0.3.0` is an alpha release. The runtime is usable and tested, but the
public API may still evolve before `1.0.0`.

## What It Provides

- Tool-calling assistant loop with pending tool result and resume support
- Three run modes: `direct`, `investigate`, and `deep_investigate`
- Bounded investigation state with auditable facts, gaps, actions and confidence
- Run trace persistence with prompt snapshots, events and compact summaries
- Generic structured task runner for bounded JSON-producing subtasks
- Tool registry and small tool protocol
- Session persistence and memory lifecycle helpers
- Policy guardrail entry points for tool execution
- Domain hooks for application-specific prompt and memory extensions
- Provider adapters for OpenAI, Azure OpenAI and Azure Anthropic backends
- Provider-enforced JSON Schema contracts for structured task final outputs
- OpenAI/Azure request normalization and adaptive retry handling
- Typed Python package marker (`py.typed`)

## What Stays Outside Core

`agent-core` intentionally does not ship domain-specific prompts, checklists,
tools, reports or integrations. Domain packages should depend on this runtime
and provide their own behavior through:

- application prompts
- `DomainHooks`
- registered tools
- policy rules
- external storage or domain memory

This keeps the repository usable for many application domains instead of
specializing it for one workflow.

## Install

From a tagged Git repository:

```bash
python -m pip install "agent-core @ git+https://github.com/maxDcb/agent-core.git@v0.3.0"
```

For local development:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

## Quickstart

The repository includes a minimal working example with two small tools:
`get_current_time` and `echo`.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then run:

```bash
.venv/bin/python examples/quickstart.py
```

Use a custom one-shot prompt:

```bash
.venv/bin/python examples/quickstart.py "Echo hello through the echo tool."
```

Or run a small REPL:

```bash
.venv/bin/python examples/quickstart.py --interactive
```

See [examples/README.md](examples/README.md) for the pending tool result and
resume example.

## Core Concepts

- `AgentOrchestrator` coordinates one user turn from prompt build to provider
  call, tool execution, memory persistence and optional trace persistence.
- `ToolRegistry` stores application tools and exposes model-facing tool
  definitions.
- `PolicyEngine` authorizes tool calls before execution.
- `SessionManager` and `SessionRepository` persist conversation blocks, memory
  state, metadata and run traces.
- `RunOptions` selects the execution mode and investigation budgets.
- `DomainHooks` let applications add domain prompt blocks and memory payloads
  without adding domain logic to the core package.

## Run Modes

The default mode is `direct`, which preserves the ordinary assistant/tool loop.
For multi-step work, `investigate` and `deep_investigate` add bounded planning,
reflection, decision and optional final critique phases while storing only
auditable artifacts, not raw chain-of-thought.

```python
from agent_core import RunOptions

result = orchestrator.run_turn_result(
    "Investigate this issue using the available tools.",
    options=RunOptions.investigate(),
)
```

See [docs/investigation_modes.md](docs/investigation_modes.md) for mode details
and metadata returned by completed investigation runs.

## Structured Tasks

`StructuredTaskRunner` runs a bounded tool-using subtask and asks the model for
a JSON object. It is useful when an application needs a generic sub-agent-like
step without introducing domain-specific specialist profiles into core.

```python
from agent_core import StructuredTaskSpec

spec = StructuredTaskSpec(
    task_id="workspace_summary",
    system_prompt="Return JSON only.",
    objective="Summarize the provided workspace context.",
    allowed_tools=["search_code"],
    output_schema={
        "type": "object",
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    },
)

result = structured_task_runner.run(spec=spec, session_id="default")
```

## Pending Tool Result Flow

Tools that need an external asynchronous result can return:

```python
from agent_core.types import ToolResult

return ToolResult.pending_result(
    "Waiting for command output.",
    metadata={"command_id": "cmd-123"},
)
```

The application stores the returned `pending_id`, then resumes the turn when the
external result arrives:

```python
completed = orchestrator.resume_turn(
    pending_id=pending.pending_id,
    tool_content="command output",
)
```

## Minimal Integration Pattern

1. Build `CoreSettings` from your application config.
2. Create a `BaseLLMProvider` implementation or use a provider from
   `agent_core.llm`.
3. Register tools in `ToolRegistry`.
4. Instantiate `SessionRepository` and `SessionManager`.
5. Instantiate `PolicyEngine`.
6. Optionally implement `DomainHooks`.
7. Build `AgentOrchestrator` and call `run_turn_result()`.

## Public API

```python
from agent_core import (
    AgentOrchestrator,
    AgentRunMode,
    AgentTurnResult,
    BaseTool,
    ContextBudget,
    CoreSettings,
    DomainHooks,
    EvidenceItem,
    ExecutionContext,
    FinalCritique,
    Hypothesis,
    InvestigationDecision,
    InvestigationPromptSet,
    InvestigationState,
    JsonFileSessionStore,
    PolicyEngine,
    PromptBlock,
    PromptSnapshot,
    RunOptions,
    RunTrace,
    SessionManager,
    SessionRepository,
    SessionStore,
    StepReflection,
    StructuredTaskResult,
    StructuredTaskRunner,
    StructuredTaskSpec,
    ToolRegistry,
    ToolResult,
    TraceEvent,
    build_tool_definition,
)
```

## Development Checks

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy agent_core
.venv/bin/python -m build
```

## Repository Scope

This repository should stay focused on the generic runtime. Domain packages
should depend on it instead of adding their prompts, tools or reporting logic
here.
