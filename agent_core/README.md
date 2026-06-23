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
- `anthropic`
- `openai`
- `requests`

### OpenAI Chat Completions compatibility

`OpenAIProvider` and `AzureOpenAIProvider` use Chat Completions.
Before dispatch, they normalize model-sensitive parameters through a shared
request policy:

- known non-reasoning chat models omit unsupported `reasoning_effort`
- known reasoning model families omit custom `temperature`
- reasoning model families use `max_completion_tokens` instead of deprecated
  `max_tokens`
- opaque model names, common with Azure deployment names, keep the requested
  payload first and learn unsupported parameters from controlled `BadRequest`
  retries

The adaptive retry path is a safety net for unknown deployments and provider
drift. It should not be the normal path for known OpenAI model families.

## Packaging Notes

- Typed package marker: `py.typed`
- Python: `>=3.11`
- Distribution metadata lives in the repository `pyproject.toml`
