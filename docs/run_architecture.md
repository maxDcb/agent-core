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

`RunStore` is a protocol. `JsonFileRunStore` is the reference single-process
implementation; applications can supply a transactional multi-worker store.

## Optional conversation

`ConversationAgent` adds thread memory and pending-tool resume semantics over
the same persisted lifecycle. `AgentOrchestrator`, `SessionManager` and the
session repository are implementation components of this adapter, not a
requirement for headless pipeline runs.

## Application pipelines

Applications own jobs, phases, retries, artifacts and domain state. A pipeline
creates one core run per LLM/tool execution and links its `run_id` to the
application attempt. Pipeline state is passed through explicit correlation;
it is never injected into conversation memory.
