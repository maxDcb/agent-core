# Run-first architecture

`agent-core` is a run engine. A run is the durable unit of execution; a
conversation is an optional adapter that groups turns in a thread.

## Identifiers

- `namespace_id` partitions application data and agent runs. Its meaning is
  caller-owned (workspace, assessment, tenant, etc.).
- `run_id` identifies exactly one technical execution. It is mandatory before
  a tool can execute.
- `parent_id` links a run to an application aggregate such as a job.
- `thread_id` is present only for conversational runs.

The core never interprets `parent_id` or values in `correlation`.

## Context contract

`RunContext` separates four concerns that were previously easy to conflate:

- `scope`: enforceable filesystem and HTTP boundaries;
- `correlation`: operational identifiers such as phase and attempt;
- `application_context`: domain configuration visible to tools;
- `thread_id`: optional conversation memory selection.

`ExecutionContext` is derived from a bound `RunContext`. Tools receive
`namespace_id`, `run_id`, the effective scope, correlation and application
context. They do not receive conversation storage or mutable session state.

## Headless runs

`AgentRunService` executes a `StructuredTaskSpec` without a session manager. It
persists `AgentRunState` before execution and its terminal `AgentRunResult`
after execution. Repeating a completed or failed `run_id` with the identical
request returns the stored result and never replays tools. Rebinding a run id
to another specification or context is rejected.

During execution, the structured runner persists a lossless, versioned
checkpoint before each provider request, before and after every tool call, and
after receiving the final provider response. The checkpoint contains the exact
transcript, counters, tool cursor, tool history and a fingerprint of the task
specification. `AgentRunService.resume()` continues a non-terminal run from
that checkpoint. A changed specification is rejected.

Every successful provider response also contributes a typed `LLMCallRecord`.
OpenAI, Azure OpenAI and Azure Anthropic usage fields are normalized without
estimating missing values. Calls and token usage are persisted at the same
checkpoint boundary as the response, then exposed on `AgentRunResult` with an
aggregate usage summary. Exact totals are `null` whenever any call lacks
provider usage; partial reported totals remain available separately. Cached
input and reasoning-token details are retained when the provider exposes them.

A tool call persisted as completed is never replayed. A tool call left in
`running` is considered ambiguous because its external effect may have happened
before the process stopped. Recovery becomes `blocked` until the host reconciles
the effect and calls `resolve_ambiguous_tool()` with the observed result.

Each execution or recovery is recorded as an `AgentRunAttempt`. Interrupted,
blocked and completed attempts remain in the run state for audit.

`RunStore` is a protocol and includes an execution-ownership context manager.
`JsonFileRunStore` uses an in-process lock plus an OS file lock, so two local
threads or processes cannot execute or resume the same run concurrently.
Applications can supply a transactional distributed store implementing the
same ownership contract.

## Lifecycle test matrix

The lifecycle suite treats persistence boundaries as public behavior:

- declared transitions are accepted and illegal terminal transitions are rejected;
- checkpoints, attempts and terminal results survive serialization round trips;
- missing, foreign or unsupported checkpoints block recovery without calling the provider;
- completed and blocked runs remain idempotent across repeated resume requests;
- completed tools are not replayed, while ambiguous effects require explicit reconciliation;
- invalid reconciliation and context rebinding leave the persisted run unchanged;
- multi-tool recovery executes only the remaining calls;
- locks reject duplicate ownership while allowing unrelated runs and namespaces;
- the conversation adapter preserves pending, resumed and terminal attempts.

Deterministic crash injection scenarios use the `chaos` marker and can be run
independently with `python -m pytest -m chaos`.

## Optional conversation

`ConversationAgent` adds thread memory and pending-tool resume semantics over
the same persisted lifecycle. `AgentOrchestrator`, `SessionManager` and the
session repository are implementation components of this adapter, not a
requirement for headless pipeline runs.

Conversation execution captures all built-in provider calls made in the turn,
including internal synthesis calls. Pending resumes append stable call records
instead of replacing or duplicating previous usage.

## Application pipelines

Applications own jobs, phases, retries, artifacts and domain state. A pipeline
creates one core run per LLM/tool execution and links its `run_id` to the
application attempt. Pipeline state is passed through explicit correlation;
it is never injected into conversation memory.
