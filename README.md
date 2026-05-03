# agent-core

Reusable, domain-agnostic runtime for tool-using LLM agents.

`agent-core` provides the orchestration layer that applications can compose with
their own prompts, tools, policy rules and domain memory. It intentionally does
not ship pentest-specific prompts, checklists or C2 integrations.

## Features

- Tool-calling orchestration loop with pending tool result / resume support
- Provider abstraction for OpenAI-compatible and Azure-backed LLMs
- Tool registry and small tool protocol
- Session persistence and memory lifecycle helpers
- Policy guardrail entry points
- Domain hooks for application-specific context and memory extensions
- Typed Python package marker (`py.typed`)

## Install

From a tagged Git repository:

```bash
python -m pip install "agent-core @ git+https://github.com/maxDcb/agent-core.git@v0.1.0"
```

For local development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Public API

```python
from agent_core import (
    AgentOrchestrator,
    BaseTool,
    CoreSettings,
    DomainHooks,
    ExecutionContext,
    PolicyEngine,
    SessionManager,
    SessionRepository,
    ToolRegistry,
    build_tool_definition,
)
```

## Minimal Integration Pattern

1. Build `CoreSettings` from your application config.
2. Create a `BaseLLMProvider` implementation.
3. Register tools in `ToolRegistry`.
4. Instantiate `SessionRepository` and `SessionManager`.
5. Instantiate `PolicyEngine`.
6. Optionally implement `DomainHooks`.
7. Build `AgentOrchestrator` and call `run_turn_result()`.

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

## Repository Scope

This repository should stay focused on the generic runtime. Domain packages
should depend on it instead of adding their prompts or tools here.
