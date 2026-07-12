# Public API boundary

agent-core exposes four supported import surfaces. This boundary keeps the run
engine usable without forcing consumers to depend on conversation memory,
provider implementations, or internal prompt assembly.

## Run engine: `agent_core`

The package root contains the autonomous run lifecycle, execution context,
structured task contracts, run stores, and their result models. Pipeline
applications should start here.

## Extension SPI: `agent_core.spi`

The SPI contains contracts intended to be implemented or assembled by a host
application: LLM providers, domain hooks, policies, tools, provider factories,
and prompt loading. Changes to these contracts are part of the public semantic
versioning policy.

Text-only and tool-enabled provider calls both return `LLMCompletionResult`.
`LLMTokenUsage`, `LLMCallRecord`, and `LLMUsageSummary` carry exact provider
usage through checkpoints and run results. Custom providers leave `usage`
unset when the upstream service does not report it; local estimates must not be
presented as provider usage.

## Optional conversation API: `agent_core.conversation`

Conversation support is an adapter over runs. It contains the orchestrator,
conversation agent, session stores and manager, and a detached read-only-style
`ConversationStateView` passed to domain hooks. The mutable internal thread
state is deliberately not public. Pipeline-only consumers do not need this
module.

## Observability: `agent_core.observability`

Logging configuration and run-trace models are exposed separately so a host can
integrate diagnostics without importing logging or trace implementation modules.

## Compatibility rule

Only names listed in `__all__` by these four modules are public. Other
`agent_core.*` modules remain importable for the runtime's own implementation
and tests, but external projects must not depend on them. They may be reorganized
without a compatibility layer.

PentestAssistant is the reference integration. Its architecture test scans all
production Python modules and fails if they import any agent-core module outside
these four facades. agent-core also tests the exact root export set and verifies
that external domain, policy, provider, and tool integrations can be declared
through the public SPI.
