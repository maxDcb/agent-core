# Reading artifacts reliably

Application tools must return a complete result or an explicitly selected view.
They must not discard content merely to fit a model preview. Agent-core stores
the tool result before projecting the bounded `artifact_result` envelope.

The existing `agent_core_read_artifact` tool now supports:

| Arguments | Behavior |
| --- | --- |
| `artifact_id`, optional `offset` and `limit` | Existing UTF-8 byte reads |
| `artifact_id`, optional `offset`, `max_bytes` | Raw reads bounded by serialized UTF-8 page bytes, including metadata |
| `artifact_id`, `operation: inspect`, optional `json_pointer` | Bounded structural inventory, without leaf values |
| `artifact_id`, `json_pointer`, optional `fields` | JSON selection; complete array elements and explicit field projection |
| `artifact_id`, `start_line`, optional `line_count` and `max_bytes` | One-based text range, split into byte-bounded pages |
| `artifact_id`, `operation: search`, `query` | Case-sensitive literal search with localized excerpts |
| `artifact_id`, `continuation` | Resume the exact immutable view returned by `next_read` |

Use an empty JSON pointer to select the JSON root. Pointers follow RFC 6901.
Search can also select JSON with `json_pointer`; its character offsets refer to
the compact serialization of that selection. Search reports non-overlapping
matches, at most 100 per page, and explicitly indicates whether the selected
source has been fully searched. An empty partial page is not proof of absence.

Choose the requested view before following a cursor. The original result
envelope's `next_read` continues raw bytes after its preview; it does not select
a requested line range or JSON subtree. A selected page's `next_read` continues
that selected view.

`next_read` contains the tool name and ready-to-use arguments. A continuation
preserves operation, selection, projection, and position. Repeating matching
arguments, including an implicit `operation="read"`, is accepted. A page budget
may be reduced, never increased beyond the cursor's budget or runtime cap.
Conflicting selections, positions, projections, or budgets return
`continuation_conflict` and the exact valid continuation call. Invalid or stale
cursors still require a fresh selection. Cursors carry an artifact ID and SHA-256 version. They are navigation
data, not authorization tokens: the store checks namespace ownership before
every continuation and cache hit. Custom stores should populate the additive
`ArtifactChunk.sha256` field from their immutable descriptor.

The v1 envelopes, offsets, and stored references remain readable. New selected
views return `artifact_page` schema version 2. `selection_complete` describes
only the selected view; it does not claim the entire source was read. Projected
fields are explicitly reported. JSON arrays are not split inside elements;
oversized objects return an actionable error pointing to structural inspection.
Selected strings are chunked without losing Unicode characters. Huge text lines
can be read through the raw byte interface; their correction starts at the
oversized line's byte offset. This fallback changes the view to raw text and
does not claim completion of the original line range.

For example, `{"start_line":40,"line_count":120,"max_bytes":8192}` selects
lines 40 through 159 inclusive. Every continuation retains this range and the
page budget. `selection_complete=true` means the range, or its available prefix
at end-of-file, is complete. The line immediately after EOF returns an empty
complete selection; positions beyond that are rejected. `line_count` requires
`start_line`. Line ranges cannot be mixed with byte offsets or JSON selections.

## Budgets and caching

Structured pages and search results fit both the configured byte limit and the
runtime context predicate. Their serialized envelopes count toward the total
read budget. Existing raw reads retain their content-byte accounting. Navigation
does not increase application-tool, internal-read, LLM, or context budgets.

Prefer `max_bytes` for new callers: it always counts the entire serialized
successful page, including metadata. Legacy `limit` remains a byte budget,
never a line count; raw offset calls retain content-byte accounting. Supplying
both budgets with different values is an error. Continuations preserve the
selected accounting mode and requested page size. Diagnostic errors are not
delivered artifact content and do not increment `artifact_bytes_read`.

Capacity errors distinguish `page_too_small` (caller page size),
`budget_exhausted` (cumulative read bytes), and `context_exhausted` (model context).
A too-small caller page gets a same-position corrected read only after a dry
render proves the next page fits the remaining hard limits. Dry rendering does
not deliver content, increment read counts, or change limits. A runtime page
cap that cannot hold metadata remains non-recoverable; oversized structured
items may instead offer a smaller selection. Genuine exhaustion is terminal.

`max_navigation_source_bytes` defaults to 8 MiB. Larger documents remain available
through raw reads; structured navigation returns an explicit limit error.
`max_navigation_cache_bytes` defaults to 16 MiB of source data, with at most 16
cached documents per runtime. Parsed JSON is reused across selections. This is
a source-byte bound, not an exact Python heap bound. Cache entries are local to
the namespace-bound runtime and are not serialized in checkpoints.

## Recovery and continuity

Navigation errors use `kind: artifact_read_error`, a stable `code`, a
`recoverable` flag, and a `suggested_action` when applicable. An application may
use the same contract for an invalid selection, provided the fallback action
actually retrieves the original data.

In conversation investigation mode, a decision to finalize or stop as blocked
immediately after a recoverable artifact-read error can be reconsidered at most
twice per run. Direct conversation and structured-task finalization use the same
contract, with native and LangGraph kernels. The runtime asks for a read correction; it never executes a
suggested action itself. Authorization, context exhaustion, and iteration/tool
budgets still take precedence. There is no forced full-document traversal.
Internal artifact reads retain their separate read budget even when the
application-tool budget has been used. Structured reconsideration consumes an
iteration and the next call uses the normal LLM and context accounting.

The common runtime helper emits a bounded correction message and an
`artifact_read_recovery` log event with run, namespace, attempt, error code, and
a hash of the correction. Repeated identical corrections are not offered again.
Only the latest tool batch is considered. A small application read-error
diagnostic projected to an artifact reference can be recovered from its original
authorized artifact (at most 12,000 bytes); larger diagnostics are not loaded by
this mechanism. No target document is inferred from a truncated error.

The existing persisted `ToolArtifactUsage` stores the last delivered view for
up to 16 artifacts, the recovery counter, and at most two correction hashes.
Structured checkpoints save the claimed correction before the next model call,
so resume cannot reset its allowance. After a checkpoint resume,
`inspect` can report the previous continuation if it fits the page. Tool schemas
stay static: progress is not inserted into mandatory tool descriptions, which
could otherwise overflow an already-full context. Previously delivered does
not mean still present in context, and these counters do not carry automatically
into an unrelated new turn.

Logs contain artifact IDs, completion flags, and error codes, without contents.
Tests cover pagination, Unicode, selected large strings, cache bounds, context
limits, checkpoint continuity, namespace isolation, and bounded conversation
recovery with scripted providers. They do not assert live-model reliability.

`tests/test_live_artifact_navigation.py` adds opt-in GPT-5.4-mini tests using
synthetic documents: exact line-range retrieval with redundant continuation
arguments, and recovery from a deliberately tiny first page. It covers direct
conversation, investigation, and structured tasks on both kernels. Use the
existing Azure live-test environment variables (`AGENT_CORE_RUN_LIVE_LLM_TESTS=1`,
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, optional API version, and
`AGENT_CORE_LIVE_LLM_MODEL=gpt-5.4-mini`) and run:

```powershell
python -m pytest tests/test_live_artifact_navigation.py -q
```

`AGENT_CORE_ARTIFACT_EVAL_OUTPUT` optionally selects a directory for per-case
synthetic answers, read arguments, errors, and token/latency measurements. These
controlled examples validate integration, not a population-wide success rate.

The [2026-09-12 validation report](artifact-reading-validation-2026-09-12.md)
records the implementation checks, final live matrix, observed limitations,
and local-versus-published delivery status.
