# agent-core

Reusable, domain-agnostic runtime for tool-using LLM agents.

This package provides:
- orchestration loop (`AgentOrchestrator`)
- tool registry and tool protocol (`ToolRegistry`, `BaseTool`)
- provider abstraction (`BaseLLMProvider`)
- session persistence and memory lifecycle (`SessionManager`, `SessionRepository`)
- policy guardrails (`PolicyEngine`)
- domain extension hooks (`DomainHooks`)

## Design Scope

`agent-core` intentionally does not ship domain-specific prompts, checklists, or reporting logic.
Those concerns should live in an application/domain layer that composes the core runtime.

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

1. Build `CoreSettings` from your app config.
2. Create one `BaseLLMProvider` implementation.
3. Register tools in `ToolRegistry`.
4. Instantiate `SessionRepository` + `SessionManager`.
5. Instantiate `PolicyEngine`.
6. Optionally implement `DomainHooks` for app-specific prompt blocks and memory payload extension.
7. Build `AgentOrchestrator` and call `run_turn()`.

## Provider Notes

Built-in provider adapters are available under `agent_core.llm`.
The runtime depends on:
- `openai`
- `requests`

## Packaging Notes

- Typed package marker: `py.typed`
- Python: `>=3.11`
- Distribution metadata lives in the repository `pyproject.toml`
