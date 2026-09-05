# Reading artifacts reliably

Application tools must return a complete result or an explicitly selected view.
They must not discard content merely to fit a model preview. Agent-core stores
the tool result before projecting the bounded `artifact_result` envelope.

The existing `agent_core_read_artifact` tool now supports:

| Arguments | Behavior |
| --- | --- |
| `artifact_id`, optional `offset` and `limit` | Existing UTF-8 byte reads |
| `artifact_id`, `operation: inspect`, optional `json_pointer` | Bounded structural inventory, without leaf values |
| `artifact_id`, `json_pointer`, optional `fields` | JSON selection; complete array elements and explicit field projection |
| `artifact_id`, `start_line` | Text reading from a one-based line number |
| `artifact_id`, `operation: search`, `query` | Case-sensitive literal search with localized excerpts |
| `artifact_id`, `continuation` | Resume the exact immutable view returned by `next_read` |

Use an empty JSON pointer to select the JSON root. Pointers follow RFC 6901.
Search can also select JSON with `json_pointer`; its character offsets refer to
the compact serialization of that selection. Search reports non-overlapping
matches, at most 100 per page, and explicitly indicates whether the selected
source has been fully searched. An empty partial page is not proof of absence.

`next_read` contains the tool name and ready-to-use arguments. Pass a continuation
with its artifact ID only; it preserves operation, selection, projection, and
position. Cursors carry an artifact ID and SHA-256 version. They are navigation
data, not authorization tokens: the store checks namespace ownership before
every continuation and cache hit. Custom stores should populate the additive
`ArtifactChunk.sha256` field from their immutable descriptor.

The v1 envelopes, offsets, and stored references remain readable. New selected
views return `artifact_page` schema version 2. `selection_complete` describes
only the selected view; it does not claim the entire source was read. Projected
fields are explicitly reported. JSON arrays are not split inside elements;
oversized objects return an actionable error pointing to structural inspection.
Selected strings are chunked without losing Unicode characters. Huge text lines
can always be read through the raw byte interface.

## Budgets and caching

Structured pages and search results fit both the configured byte limit and the
runtime context predicate. Their serialized envelopes count toward the total
read budget. Existing raw reads retain their content-byte accounting. Navigation
does not increase application-tool, internal-read, LLM, or context budgets.

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
twice per run. The runtime asks for a read correction; it never executes a
suggested action itself. Authorization, context exhaustion, and iteration/tool
budgets still take precedence. There is no forced full-document traversal.

The existing persisted `ToolArtifactUsage` stores the last delivered view for
up to 16 artifacts and the recovery counter. After a checkpoint resume,
`inspect` can report the previous continuation if it fits the page. Tool schemas
stay static: progress is not inserted into mandatory tool descriptions, which
could otherwise overflow an already-full context. Previously delivered does
not mean still present in context, and these counters do not carry automatically
into an unrelated new turn.

Logs contain artifact IDs, completion flags, and error codes, without contents.
Tests cover pagination, Unicode, selected large strings, cache bounds, context
limits, checkpoint continuity, namespace isolation, and bounded conversation
recovery with scripted providers. They do not assert live-model reliability.
