# Artifact reader validation — 2026-09-12

ART-001, ART-002, and ART-003 are implemented on top of agent-core revision
`e96804ec9e2f919f35b0cbbe6bbe0201fba08445`. This report describes the local
working-tree changes; it is not evidence of a published dependency revision.

## Result

- 344 deterministic agent-core tests passed; 28 opt-in live tests were excluded
  from that command.
- 10 PentestAssistant integration tests passed against the editable sibling
  checkout (`test_pipeline_artifact_tools.py` and `test_app_factory.py`).
- All 12 artifact-specific GPT-5.4-mini cases passed in the final controlled
  validation, with exact equality of all 12 requested code values, in order.
- Ruff checks, the five changed runtime modules' mypy check, and diff whitespace
  checks passed.

## Changes exercised

Continuation normalization accepts matching repeated arguments and reduced page
budgets while preserving artifact/version/namespace binding and the selected
view. Conflicting arguments return the exact usable continuation.

`start_line` plus `line_count` defines the selected text range; `max_bytes`
defines the serialized page budget. Legacy `limit` retains byte semantics.
Page-size errors, cumulative-read exhaustion, and model-context exhaustion are
distinct. Recoverable page-size errors retain the selected range and position.

Direct conversation, conversation investigation, and structured tasks share the
bounded read-correction mechanism. Tests cover native and LangGraph kernels,
successful correction, duplicate failures, authorization/unavailable artifacts,
application errors projected to previews/references, hard budgets, and persisted
recovery state. Structured checkpoint resume and conversation pending-tool
resume preserve the counter and correction hashes.

## Real-model method and results

Tests use the Azure OpenAI provider with `gpt-5.4-mini` and the native provider
adapter. The kernel backend is independently varied between native and
LangGraph. Conversation investigation uses real model calls for reflection and
decision as well as tool use, with `recover_internal_synthesis_errors=true`,
matching PentestAssistant's existing conversation default.

The only application tool returns a synthetic 150-line document. Each line
contains a deterministic opaque code. The objective requests lines 40–51 and
all twelve codes in their original order. Expected codes are checked by Python;
they are not supplied in the user prompt.

The range scenario starts with `max_bytes=1100` and repeats `operation=read`
with continuations. The small-page scenario deliberately starts with
`max_bytes=120`, then requires a corrected read. No target, repository source,
user document, or account data is sent to the model.

| Mode | Kernel | Range: read calls | Small page: read calls | Exact answers |
| --- | --- | ---: | ---: | --- |
| Direct conversation | Native | 6 | 2 | Both passed |
| Direct conversation | LangGraph | 6 | 2 | Both passed |
| Investigation | Native | 6 | 2 | Both passed |
| Investigation | LangGraph | 6 | 2 | Both passed |
| Structured task | Native | 7 | 2 | Both passed |
| Structured task | LangGraph | 6 | 2 | Both passed |

Range cases have no read errors. Every small-page case has exactly the expected
`page_too_small` error followed by successful retrieval. Runtime completion of a
task alone is not the oracle: assertions also check the requested range/budget,
continuation behavior, and exact final content.

The final twelve case records report 392,901 tokens in aggregate, including
internal synthesis and memory calls. This is not total development-run usage:
earlier failed or interrupted experiments are excluded. Per-case JSON records
are available locally under `artifacts/artifact-reading-validation/`.

## What the real tests changed

Early experiments exposed a misleading default instruction to continue the
raw preview. That sometimes displaced the requested line selection and, in one
case, produced codes from the wrong part of the document. The reader schema
now explicitly distinguishes choosing a view from continuing raw bytes. The
same distinction is provided to investigation reflection so recommended next
actions preserve the user's range and budget.

Some early investigation experiments also encountered invalid internal decision
JSON. The final matrix uses the existing bounded synthesis-error recovery
option already enabled by PentestAssistant. This is distinct from artifact
read recovery and does not relax artifact assertions or enlarge runtime limits.

These are controlled integration checks after iteration, not a statistical
claim of universal model reliability. Scripted-provider tests, rather than
model cooperation, verify the runtime's behavior when the model attempts to
finalize prematurely and when counters/budgets must prevent another recovery.

## Reproduce and deliver

See [artifact_navigation.md](artifact_navigation.md) for the API contract and
live-test environment variables. Deterministic checks use:

```powershell
python -m pytest -m "not live_llm" -q
python -m pytest tests/test_live_artifact_navigation.py -q
```

The second command is opt-in and requires the documented Azure credentials and
`AGENT_CORE_RUN_LIVE_LLM_TESTS=1`. It incurs model usage.

PentestAssistant's local Python environment now imports the editable sibling
checkout. Standard installations still use the previous Git pin. Publication
of agent-core and an accompanying application pin/lockfile update remain a
separate delivery step; no unresolvable remote revision was written into the
dependency files.
