# agent-core

Reusable, domain-agnostic runtime for tool-using LLM agents.

This package provides:

- a persisted autonomous run service and structured task runner;
- an optional conversation adapter with incremental memory;
- a tool registry, tool protocol and policy guardrails;
- provider abstractions and built-in provider adapters;
- always-on lossless tool-result artifacts;
- domain extension hooks.

## Design Scope

`agent-core` intentionally does not ship domain-specific prompts, checklists, or reporting logic.
Those concerns should live in an application/domain layer that composes the core runtime.

## Public API surfaces

Autonomous run applications import the run engine from `agent_core`:

```python
from agent_core import (
    AgentRunService,
    ArtifactResultEnvelope,
    CoreSettings,
    JsonFileArtifactStore,
    JsonFileRunStore,
    RunContext,
    RunOptions,
    StructuredTaskSpec,
    ToolArtifactPolicy,
)
```

Host extension contracts live in `agent_core.spi`:

```python
from agent_core.spi import (
    BaseLLMProvider,
    BaseTool,
    DomainHooks,
    PolicyEngine,
    ToolRegistry,
    build_tool_definition,
)
```

Conversation-only components live in `agent_core.conversation`:

```python
from agent_core.conversation import (
    AgentOrchestrator,
    ConversationAgent,
    JsonFileSessionStore,
    SessionManager,
    SessionRepository,
)
```

Only names listed in the `__all__` of these supported facades are public.
See [Public API boundary](../docs/public_api.md) for the complete rule.

## Integration patterns

For a headless run:

1. Build `CoreSettings`, a provider, `ToolRegistry`, `PolicyEngine` and
   `JsonFileRunStore` or another `RunStore`.
2. Build `AgentRunService`.
3. Execute a `StructuredTaskSpec` with a bound `RunContext`.

For a conversation with incremental memory:

1. Build `CoreSettings` from your app config.
2. Create the main provider and, optionally, a separate memory provider.
3. Register tools in `ToolRegistry`.
4. Instantiate `SessionRepository` + `SessionManager`.
5. Instantiate `PolicyEngine` and optional `DomainHooks`.
6. Build `AgentOrchestrator`, then wrap it in `ConversationAgent` with a
   `RunStore`.
7. Call `ConversationAgent.execute_turn()`.

Conversation memory uses append-only exchange and turn journals plus a bounded
operational handoff. Tool results are always externalized to an artifact store.
See [Conversation memory and tool-result artifacts](../docs/memory_and_artifacts.md)
for lifecycle, recovery, projection and persistence details.

## Provider Notes

Built-in provider adapters are available under `agent_core.llm`.
The runtime depends on:

- `anthropic`
- `jsonschema`
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
