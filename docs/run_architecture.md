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

When configured, one `LLMBudget` controller spans every LLM call in the run,
including internal synthesis and post-turn memory calls. It accounts before a
request, reconciles estimates with provider usage after the response, and
persists its usage in pending conversation payloads or structured checkpoints.
`observe` mode reports violations without altering execution; `enforce` mode
rejects the next request before provider invocation. A resumed run restores the
previous counters instead of receiving a fresh budget. The duration counter is
the cumulative time spent in provider calls and is enforced between calls; it
does not interrupt a request already in flight.

An optional `LLMContextPolicy` is evaluated before the run-level budget for
every provider call. The planner measures the complete request envelope:
messages, tool definitions and structured response schema. It subtracts the
configured output reserve and safety margin from the provider context window,
then removes only complete historical message groups until the request fits.
System messages and the entire current user/tool turn are mandatory, so tool
calls are never separated from their responses. An irreducible overflow fails
before provider invocation. Planner aggregates are persisted in pending turns
and structured checkpoints and exposed in final metadata.

When `ToolArtifactPolicy` is active, completed application-tool outputs are
published to an `ArtifactStore` before the corresponding tool execution is
checkpointed as complete. Persisted transcripts contain an opaque descriptor;
the provider projection rehydrates only the newest results within the hot-byte
budget. Large results stay cold from their first projection. The model can use
the bounded `agent_core_read_artifact` runtime tool to retrieve more data.
Runtime artifact reads bypass application policy and tool-call accounting, but
are namespace-scoped and enforce separate per-run call and byte limits.
The default file store is plaintext and local; deployments with sensitive tool
outputs must provide storage encryption, permissions, and retention through a
custom `ArtifactStore`.

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

Conversation memory has three layers:

1. `ExchangeMemory` records each atomic assistant/tool interaction. Runtime
   records are written atomically with their context block; investigation
   reflections add grounded facts, gaps and next actions without another model
   call.
2. `TurnMemory` is one bounded delta synthesized only from the completed
   current turn. It never reads the full thread or a historical overflow.
3. `SessionView` is a bounded materialized projection merged
   deterministically from committed turn deltas. Its generation and terminal
   turn id make it rebuildable from the append-only journal.

The raw transcript remains the recovery source until a turn memory is
committed. Invalid model JSON, provider failure or budget exhaustion produces a
deterministic `TurnMemory` fallback. An interruption between transcript and
memory persistence is reconciled on the next prompt build without calling the
model. Memory persistence failure is isolated from the user-visible result;
the persisted raw turn remains available for the same reconciliation path.

Recent raw history is still selected in whole turn groups for provider
continuity. Historical blocks outside that window are retained for audit, but
long-term memory no longer depends on a block crossing an overflow threshold.

## Application pipelines

Applications own jobs, phases, retries, domain artifacts and domain state.
Agent-core owns the optional storage lifecycle for raw tool-result artifacts
needed to virtualize its internal transcript. A pipeline
creates one core run per LLM/tool execution and links its `run_id` to the
application attempt. Pipeline state is passed through explicit correlation;
it is never injected into conversation memory.
