# Conversation memory and tool-result artifacts

This document defines the current runtime contracts for incremental conversation
memory and lossless application-tool results. Conversation memory is optional
because headless structured tasks do not use sessions. Tool-result artifact
storage is always enabled for both conversation and structured runs.

## Incremental conversation memory

### Stored layers

Conversation memory has three persisted layers:

1. `ExchangeMemory` is an append-only event for an atomic tool exchange,
   investigation reflection, final response or provider failure. A runtime
   exchange and its raw `ContextBlock` are saved in one session transaction.
2. `TurnMemory` is committed after a completed turn. It contains an immutable
   `turn_summary` plus `handoff_after_turn`, the complete bounded working note
   for the next turn.
3. `SessionView` is derived from the latest valid `TurnMemory`. It is not an
   independent semantic merge and can be rebuilt deterministically from the
   turn journal.

The raw context blocks remain the audit and recovery source. Only the latest
`SessionView` and a recent suffix of raw history are projected into a later
prompt; the full exchange and turn journals remain persisted.

### One bounded synthesis call

At most one memory-model call is made after a completed turn. Its input contains:

- the previous bounded handoff;
- the current user request and assistant outcome;
- ordered `ExchangeMemory` events for the current turn;
- bounded runtime and investigation-controller state;
- optional domain payload and guidance.

It never receives the complete conversation or the raw historical overflow.
`CoreSettings` applies the following character bounds:

- `memory_max_turn_input_chars` to the complete synthesis payload;
- `memory_max_turn_summary_chars` to `turn_summary`;
- `memory_max_handoff_chars` to both the previous and replacement handoff.

`turn_memory_synthesis_prompt` can replace the built-in instructions. An empty
value uses the built-in prompt.

The built-in prompt treats the current turn as a delta over the previous
handoff. A prior fact is not dropped merely because the current turn does not
repeat it. Facts, failed approaches, constraints and unresolved contradictions
are retained while they affect the current objective. Conflicting evidence is
kept with its provenance until it is resolved. An item is removed only when the
objective makes it obsolete or current evidence explicitly resolves,
supersedes or invalidates it.

### Domain extensions

`DomainHooks.extend_turn_memory_payload()` may add serializable bounded context,
and `DomainHooks.turn_memory_guidance()` may add prose instructions. Both hooks
receive a detached `TurnMemoryContextView` containing only the thread id, turn
index, metadata, previous handoff and current-turn exchange memories.

Extensions do not create a second memory schema. If an extension hook or the
model synthesis fails, the runtime uses the same deterministic fallback as any
other synthesis failure.

### Failure and interruption recovery

Invalid model output, a provider error, a budget failure, an oversized result or
any other synthesis exception produces a deterministic degraded `TurnMemory`.
Memory failure does not replace or block the user-visible answer.

If persistence fails after the raw context block is saved, the next prompt build
calls `SessionManager.reconcile_memory()`. It reconstructs missing exchange
events and a deterministic turn fallback from the raw blocks without an LLM
call. Appends and turn commits are idempotent by stable memory id.

Memory journal schema version `"2"` is intentionally incompatible with schema
1. A schema-1 or otherwise unsupported memory payload loads as an empty journal;
there is no compatibility parser or automatic migration.

### Raw-history compaction

`max_active_context_tokens` is a soft budget for replayed raw history after the
fixed prompts, current user message and `SessionView` have reserved their
estimated space. `HistoryCompactor`:

- groups conversation and tool blocks by complete turn;
- scans backward from the newest turn;
- keeps one contiguous chronological suffix;
- stops at the first older group that would exceed the remaining budget;
- always keeps the newest group, even if that group alone exceeds the budget;
- does not retain isolated older groups because they are marked `pinned`.

The overflow prefix remains stored. An enforced `LLMContextPolicy` is the
separate hard guard that measures the complete provider request and rejects an
irreducible overflow before provider invocation.

## Lossless tool-result artifacts

### Always-on externalization

Every application-tool result is written to an `ArtifactStore`, including
pending, failed, denied and synthetic budget-exhausted results. The default
`CoreSettings.tool_artifact_policy` is `ToolArtifactPolicy()` and cannot be
disabled with `None`. `RunOptions.tool_artifact_policy` and
`StructuredTaskSpec.tool_artifact_policy` optionally replace that default for
one run.

The built-in `JsonFileArtifactStore` writes UTF-8 content plus JSON metadata
under a directory derived from the namespace. Artifact ids are opaque
`art_<32 lowercase hex characters>` values; model-facing reads never accept a
filesystem path.

### Result envelope

An application-tool message always uses an `artifact_result` envelope:

```json
{
  "schema_version": "1",
  "kind": "artifact_result",
  "status": "ok",
  "artifact": {
    "artifact_id": "art_0123456789abcdef0123456789abcdef",
    "tool_name": "example_tool",
    "size_bytes": 12000,
    "sha256": "<content digest>",
    "content_type": "text/plain; charset=utf-8"
  },
  "materialization": "preview",
  "content": "<bounded UTF-8 prefix>",
  "returned_bytes": 4096,
  "complete": false,
  "next_offset": 4096,
  "read_tool": "agent_core_read_artifact"
}
```

Materialization has three states:

- `complete`: the full result fits `max_complete_result_bytes`; `content`
  contains it, `complete` is true and `next_offset` is null.
- `preview`: the result is larger; `content` contains up to `preview_bytes`,
  `complete` is false and `next_offset` continues after the returned UTF-8
  bytes.
- `reference`: no content is projected; `returned_bytes` is zero and reading
  starts at offset 0.

Before each model request, projection scans stored artifact messages from newest
to oldest. It rehydrates a full small result or a large-result preview while
`hot_context_bytes` remains. Older results become references. If stored content
cannot be read or verified, projection safely leaves a reference.

Persisted messages always use the reference form, even when the live provider
projection was complete or a preview. Raw content therefore exists once in the
artifact store instead of being copied into session state or structured
checkpoints.

### Bounded reads

The reserved `agent_core_read_artifact` tool is exposed only after the run has a
readable artifact. Applications cannot register a tool with that name. Its
arguments are:

- `artifact_id`: required opaque id;
- `offset`: non-negative byte offset, default 0;
- `limit`: requested positive byte limit.

The effective limit is the minimum of the requested value,
`max_read_bytes`, and the run's remaining `max_total_read_bytes`. The response
is an `artifact_chunk`:

```json
{
  "schema_version": "1",
  "kind": "artifact_chunk",
  "artifact_id": "art_0123456789abcdef0123456789abcdef",
  "offset": 4096,
  "next_offset": 8192,
  "returned_bytes": 4096,
  "size_bytes": 12000,
  "eof": false,
  "content": "<next UTF-8 chunk>"
}
```

Continue with each `next_offset` until `eof` is true. Reads have independent
`max_reads_per_run` and `max_total_read_bytes` accounting. They do not consume
the application tool-call budget and do not pass through application
`PolicyEngine` rules.

Reads are namespace-scoped: the runtime supplies the current namespace and the
model supplies only the artifact id. The default store does not add a second
run-level authorization check, so applications that require finer isolation
must enforce it in a custom store.

### Store extension and security

`ArtifactStore` is a public protocol with three operations:

- `put_text(namespace_id, run_id, tool_name, content, metadata)`;
- `read_text(namespace_id, artifact_id, offset, limit)`;
- `read_all_text(namespace_id, artifact_id)`.

Inject a custom implementation through `AgentRunService`,
`StructuredTaskRunner` or `AgentOrchestrator`. The returned descriptors must
remain compatible with `ToolArtifactDescriptor`.

`JsonFileArtifactStore` uses atomic replacement and verifies size and SHA-256
when reading full content. It stores plaintext and provides no retention,
encryption or cross-machine transaction policy. Production deployments handling
credentials, cookies or browser-session data should provide those controls in a
custom store.

### Persistence and memory interaction

Pending conversation state schema version `"2"` and structured checkpoint
schema version `6` persist reference envelopes, the effective
`ToolArtifactPolicy`, and `ToolArtifactUsage`. Resume restores the same read
budgets instead of starting fresh.

Conversation `ExchangeMemory` and `TurnMemory` retain relevant artifact ids, not
raw artifact content. This preserves provenance without injecting large tool
results into the long-term handoff. A later model must use the bounded read tool
when details are no longer materialized in its prompt.
