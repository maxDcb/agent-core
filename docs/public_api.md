# Public API boundary

agent-core exposes four supported import surfaces. This boundary keeps the run
engine usable without forcing consumers to depend on conversation memory,
provider implementations, or internal prompt assembly.

## Run engine: `agent_core`

The package root contains the autonomous run lifecycle, execution context,
structured task contracts, run stores, and their result models. Pipeline
applications should start here.

`LLMBudget`, `LLMBudgetUsage`, and `LLMBudgetExceededError` are part of this
surface. They configure run-wide model limits and expose their enforcement
state without making the internal controller a public extension point.

`LLMContextPolicy`, `LLMContextPlan`, `LLMContextUsage`, and
`LLMContextOverflowError` are also public. They configure the provider-window
planner and expose content-free planning telemetry; the stateful planner itself
remains an implementation detail.

`ToolArtifactPolicy`, `ToolArtifactDescriptor`, `ArtifactResultEnvelope`,
`ToolArtifactUsage`, `ArtifactChunk`, and `JsonFileArtifactStore` configure and
describe lossless tool-result externalization. `ArtifactStore` is exported from
both the package root and the SPI for hosts that provide remote or
transactional artifact persistence.

## Extension SPI: `agent_core.spi`

The SPI contains contracts intended to be implemented or assembled by a host
application: LLM providers, domain hooks, policies, tools, provider factories,
artifact stores, and prompt loading. Changes to these contracts are part of the
public semantic versioning policy.

Text-only and tool-enabled provider calls both return `LLMCompletionResult`.
`LLMTokenUsage`, `LLMCallRecord`, and `LLMUsageSummary` carry exact provider
usage through checkpoints and run results. Custom providers leave `usage`
unset when the upstream service does not report it; local estimates must not be
presented as provider usage.

`LLMProviderConfig.model_backend` and `CoreSettings.llm_model_backend` select
the implementation behind the stable provider contract. `native` is the
default. `langchain` currently selects the Azure OpenAI LangChain adapter;
provider classes and LangChain message types remain internal implementation
details. A dedicated memory provider may override this with
`CoreSettings.memory_llm_model_backend`.

Every built-in completion sets `LLMCompletionResult.model_backend` to `native`
or `langchain`. `LLMCallRecord` persists the same additive field through run
results and structured-task checkpoints; older payloads without the field
continue to load with `model_backend=None`.

LangSmith tracing is disabled for LangChain model calls unless
`CoreSettings.langchain_tracing_enabled` is explicitly enabled. This scoped
override takes precedence over an inherited `LANGSMITH_TRACING` environment
setting so host applications do not export prompts or tool results by accident.

`CoreSettings.agent_kernel_backend` independently selects `native` or
`langgraph` control flow for direct conversation turns. LangGraph state, nodes,
and compiled graphs are internal implementation details and are not exported
from a supported facade. Investigation modes and headless structured runs are
unchanged by this setting.

## Optional conversation API: `agent_core.conversation`

Conversation support is an adapter over runs. It contains the orchestrator,
conversation agent, session stores and manager, and a detached read-only-style
`ConversationStateView` passed to domain hooks. The mutable internal thread
state is deliberately not public. The detached view exposes the materialized
`session_view` and detached exchange/turn journal snapshots; turn-memory
synthesis hooks receive the public `TurnMemoryContextView`, a smaller bounded
context containing the previous handoff and current-turn exchange memories.
Pipeline-only consumers do not need this module.

## Observability: `agent_core.observability`

Logging configuration and run-trace models are exposed separately so a host can
integrate diagnostics without importing logging or trace implementation modules.

## Compatibility rule

Only names listed in `__all__` by these four modules are public. Other
`agent_core.*` modules remain importable for the runtime's own implementation
and tests, but external projects must not depend on them. They may be reorganized
without a compatibility layer.

The official
[maxDcb/agent-core-exemple](https://github.com/maxDcb/agent-core-exemple)
repository demonstrates an application built against these supported facades.
agent-core also tests the exact root export set and verifies that external
domain, policy, provider, and tool integrations can be declared through the
public SPI.
