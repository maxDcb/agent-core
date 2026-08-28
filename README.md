# agent-core

Reusable, domain-agnostic runtime for tool-using LLM agents.

`agent-core` provides the orchestration layer that applications can compose with
their own prompts, tools, policy rules, storage and domain memory. It is meant
to be the generic runtime under an agent application, not a domain-specific
agent package.

Version `0.4.0` is an alpha release. The runtime is usable and tested, but the
public API may still evolve before `1.0.0`.

## What It Provides

- Tool-calling assistant loop with pending tool result and resume support
- Three run modes: `direct`, `investigate`, and `deep_investigate`
- Bounded investigation state with auditable facts, gaps, actions and confidence
- Run trace persistence with prompt snapshots, events and compact summaries
- Generic structured task runner for bounded JSON-producing subtasks
- Tool registry and small tool protocol
- Persisted, idempotent autonomous run lifecycle
- Optional conversation threads and memory lifecycle helpers
- Policy guardrail entry points for tool execution
- Domain hooks for application-specific prompt and memory extensions
- Provider adapters for OpenAI, Azure OpenAI and Azure Anthropic backends
- Exact provider token usage, per-call telemetry and run-level usage summaries
- Optional run-level LLM budgets covering main, reflection, finalization and memory calls
- Optional provider-window context planning with atomic history compaction
- Always-on lossless tool-result artifacts with bounded on-demand rereads
- Provider-enforced JSON Schema contracts for structured task final outputs
- OpenAI/Azure request normalization and adaptive retry handling
- Typed Python package marker (`py.typed`)

## What Stays Outside Core

`agent-core` intentionally does not ship domain-specific prompts, checklists,
tools, reports or integrations. Domain packages should depend on this runtime
and provide their own behavior through:

- application prompts
- `DomainHooks`
- registered tools
- policy rules
- external storage or domain memory

This keeps the repository usable for many application domains instead of
specializing it for one workflow.

## Install

From a tagged Git repository:

```bash
python -m pip install "agent-core @ git+https://github.com/maxDcb/agent-core.git@v0.4.0"
```

For local development:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

## Quickstart

The repository includes a minimal working example with two small tools:
`get_current_time` and `echo`.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then run:

```bash
.venv/bin/python examples/quickstart.py
```

Use a custom one-shot prompt:

```bash
.venv/bin/python examples/quickstart.py "Echo hello through the echo tool."
```

Or run a small REPL:

```bash
.venv/bin/python examples/quickstart.py --interactive
```

### Optional LangChain model backend for Azure OpenAI

Azure OpenAI can use LangChain for model invocation while the agent loop,
tools, policies, checkpoints, memory lifecycle, and public provider contract
remain owned by agent-core. The native SDK backend remains the default.

```bash
LLM_PROVIDER=azure_openai
AGENT_CORE_MODEL_BACKEND=langchain
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AGENT_CORE_MODEL=<deployment-name>
AGENT_CORE_MEMORY_MODEL=<deployment-name>

.venv/bin/python examples/quickstart.py --compat-check
```

`AGENT_CORE_MEMORY_MODEL_BACKEND` can override the backend for a dedicated
memory provider. If it is omitted, memory synthesis uses the primary backend;
`native` and `langchain` can therefore be compared without changing the agent
or memory contracts. The LangChain backend currently supports Azure OpenAI
only.

See [examples/README.md](examples/README.md) for the pending tool result and
resume example.

For a complete application built on the supported public API, see the official
[maxDcb/agent-core-exemple](https://github.com/maxDcb/agent-core-exemple)
repository.

## Core Concepts

- `RunContext` carries the namespace, run identity, execution scope,
  correlation and caller-owned application context.
- `AgentRunService` executes and persists autonomous headless runs without a
  conversation or session manager.
- `ConversationAgent` optionally adds thread memory and pending-tool resumes.
- `ToolRegistry` stores application tools and exposes model-facing tool
  definitions.
- `PolicyEngine` authorizes tool calls before execution.
- `RunStore` persists run state; `JsonFileRunStore` is the reference
  local filesystem implementation with thread and process execution locks.
- Structured runs persist lossless checkpoints and can resume after process
  interruption without replaying completed tools.
- Ambiguous tool effects fail closed until the host supplies a reconciled
  result.
- `RunOptions` selects the execution mode and investigation budgets.
- `DomainHooks` let applications add domain prompt blocks and memory payloads
  without adding domain logic to the core package.

## Incremental Conversation Memory

Conversation memory is committed as an append-only journal instead of being
re-summarized from the full thread:

- every completed assistant/tool exchange produces an `ExchangeMemory`;
- investigation reflections are reused as structured exchange-memory events;
- one bounded `TurnMemory` is synthesized from the current turn's ordered
  events after the final answer;
- each turn stores an immutable turn summary and a complete replacement
  operational handoff;
- `SessionView` is the latest valid handoff and is rebuilt deterministically
  from the append-only turn journal.

The turn-memory synthesizer never receives the complete conversation or an
overflow backlog. A synthesis failure commits a deterministic fallback, so it
cannot block the user-visible answer or leave the same historical backlog to be
retried indefinitely. If execution stops after the raw conversation block is
saved but before its memory commit, the next prompt build reconstructs the
missing exchange and turn records without an LLM call.

The replacement handoff is updated conservatively from the previous handoff and
the current turn. Still-relevant facts, constraints, failed approaches and
unresolved contradictions are retained even when the current turn does not
repeat them. They are removed only when the objective changes or current
evidence explicitly resolves, supersedes or invalidates them.

`CoreSettings.turn_memory_synthesis_prompt` customizes the single post-turn
memory call. `memory_max_turn_input_chars`, `memory_max_handoff_chars`, and
`memory_max_turn_summary_chars` bound its input and prose outputs. The complete
turn journal remains persisted even though only the latest compact handoff is
injected into later prompts.

For long turns, cumulative controller state and runtime/domain JSON are
projected first. The serialized fixed payload is then measured and its exact
remaining character budget is used for ordered exchange memories, prioritizing
facts, evidence and artifact references, tested actions, hypotheses, risks and
next actions.

Raw history compaction keeps a contiguous suffix of complete turn groups. It
stops at the first older group that does not fit instead of creating holes or
retaining isolated pinned groups. The newest group is always kept, even when it
alone exceeds the soft `max_active_context_tokens` history budget. Configure an
enforced `LLMContextPolicy` when the complete provider request must stay inside
a hard context limit.

Memory journal schema 2 is intentionally breaking while the package is
pre-1.0. Schema 1 memory payloads are ignored; no compatibility parser or
migration path is provided.

Domain packages can implement `DomainHooks.extend_turn_memory_payload()` and
`DomainHooks.turn_memory_guidance()` to add bounded context and prose guidance
without introducing a second memory schema. Headless structured tasks and
application pipelines do not use this conversation journal.

See [docs/memory_and_artifacts.md](docs/memory_and_artifacts.md) for the memory
commit, compaction, recovery and persistence contracts.

## Run Modes

The default mode is `direct`, which preserves the ordinary assistant/tool loop.
For multi-step work, `investigate` and `deep_investigate` add bounded planning,
reflection, decision and optional final critique phases while storing only
auditable artifacts, not raw chain-of-thought.

Investigation modes are conversation modes by default: after the investigation
stops, a dedicated no-tool model call renders the user-visible answer as natural
text from the original request, candidate draft, and auditable state. Empty,
tool-calling, raw-JSON, or failed final responses fall back to the bounded
deterministic state renderer so the conversation still completes. The result
metadata reports `final_response_origin` as `model` or `fallback`.

Planning, reflection, decision and critique phases use internal JSON synthesis.
If a caller needs a structured final answer, pass
`final_output_mode="json_schema"` with a `StructuredOutputContract`; this stays
on its separate provider-enforced and locally validated JSON Schema path.

```python
from agent_core import RunContext, RunOptions

result = conversation_agent.execute_turn(
    thread_id="incident-1",
    context=RunContext(namespace_id="workspace-1", thread_id="incident-1"),
    user_input="Investigate this issue using the available tools.",
    options=RunOptions.investigate(),
)
```

See [docs/investigation_modes.md](docs/investigation_modes.md) for mode details
and metadata returned by completed investigation runs.

## Structured Tasks

`StructuredTaskRunner` runs a bounded tool-using subtask. Without an
`output_contract`, it returns raw text and the caller owns any parsing. With an
`output_contract`, the provider JSON Schema contract is enforced only on the
final no-tool output, after any tool investigation is complete. Providers that
cannot enforce the contract fail the task instead of silently downgrading to
loose JSON mode. Provider enforcement is not trusted as the application
boundary: agent-core also validates the parsed object locally with JSON Schema
Draft 2020-12 and format checking before returning success. The contract schema
itself is checked when `StructuredOutputContract` is constructed.

An invalid structured result fails explicitly. agent-core does not silently
coerce values, remove extra fields, invent missing fields, or retry. Validation
metadata contains the contract name, validation phase, validator, instance
path, schema path, and a safe message; payload values are excluded so secrets
are not copied into diagnostics. The original raw completion remains available
on the failed task result for controlled recovery or audit by the caller.

```python
from agent_core import StructuredOutputContract, StructuredTaskSpec

spec = StructuredTaskSpec(
    task_id="workspace_summary",
    system_prompt="Return JSON only.",
    objective="Summarize the provided workspace context.",
    allowed_tools=["search_code"],
    output_contract=StructuredOutputContract(
        name="workspace_summary",
        strict=True,
        schema={
            "type": "object",
            "required": ["summary"],
            "additionalProperties": False,
            "properties": {"summary": {"type": "string"}},
        },
    ),
)

from agent_core import AgentRunService, JsonFileRunStore, RunContext

service = AgentRunService(
    settings=settings,
    provider=provider,
    tool_registry=tool_registry,
    policy_engine=policy_engine,
    run_store=JsonFileRunStore(".agent-core/runs"),
)
result = service.execute(
    spec=spec,
    context=RunContext(namespace_id="workspace-1", parent_id="job-1"),
)
```

## Run-Level LLM Budgets

`LLMBudget` applies one shared budget to every model call made by a run,
including assistant steps, internal investigation synthesis, finalization and
post-turn memory refresh. No budget is applied unless one is configured, so
existing integrations preserve their current behavior.

Set a default on `CoreSettings`, or override it for one conversation or
structured task through `RunOptions.llm_budget` or
`StructuredTaskSpec.llm_budget`:

```python
from agent_core import LLMBudget, RunOptions

budget = LLMBudget(
    max_calls=12,
    max_input_tokens=80_000,
    max_output_tokens=12_000,
    max_total_tokens=90_000,
    max_duration_seconds=180,
    mode="enforce",
)

options = RunOptions.investigate(llm_budget=budget)
```

`mode="observe"` records limit violations without blocking calls or changing
provider output limits. In enforcement mode, a call that would exceed the call,
input-token, total-token or cumulative provider-call-duration limit is rejected
before provider invocation. The duration limit is checked between calls; it
does not cancel a provider request already in flight. The controller also caps
the provider's per-call maximum output to the remaining output/total budget
when the provider accepts call options.

Completed and pending results expose `llm_budget` and `llm_budget_usage` in
their metadata. Pending conversation state and structured-task checkpoints
persist the same usage so resume cannot reset a budget. Exact provider token
usage replaces pre-call estimates when available; otherwise the controller
retains its conservative estimate.

## Provider-Window Context Planning

`LLMContextPolicy` plans the complete provider input immediately before every
LLM call. Unlike history-only compaction, it counts system prompts, conversation
messages, tool definitions and structured response schemas, then reserves room
for the model output and a safety margin.

Set a default on `CoreSettings`, or override it through
`RunOptions.llm_context_policy` or `StructuredTaskSpec.llm_context_policy`:

```python
from agent_core import LLMContextPolicy, RunOptions

context_policy = LLMContextPolicy(
    max_context_tokens=128_000,
    reserved_output_tokens=8_192,
    safety_margin_tokens=512,
    mode="enforce",
)

options = RunOptions.investigate(llm_context_policy=context_policy)
```

Enforcement preserves all system messages and the complete current turn,
including each assistant tool call with all matching tool responses. If the
request is too large, it removes only complete older history groups, newest
first. If the mandatory context still cannot fit, the call is rejected before
provider invocation with `LLMContextOverflowError`. `mode="observe"` records
the overflow without changing the request.

Results expose `llm_context_policy` and `llm_context_usage`; pending turns and
structured checkpoints persist the same planner usage across resume. Token
planning uses a deterministic character heuristic, so the safety margin should
cover tokenizer variance. Output reservation is provider-enforced only when
the provider supports `LLMCallOptions.max_output_tokens`.

## Lossless Tool-Result Artifacts

Every application-tool result is stored outside the transcript, including
pending, failed, denied and budget-exhausted results.
The model always receives an `artifact_result` envelope: a small result is
`complete`, a large result is a bounded `preview`, and an older result that has
left the hot context is a `reference`. The runtime exposes
`agent_core_read_artifact` after at least one readable artifact exists so the
model can continue reading bounded chunks on demand.

```python
from pathlib import Path

from agent_core import CoreSettings, ToolArtifactPolicy

settings = CoreSettings(
    artifacts_directory=Path("./artifacts"),
    tool_artifact_policy=ToolArtifactPolicy(
        hot_context_bytes=64 * 1024,
        max_complete_result_bytes=32 * 1024,
        preview_bytes=4 * 1024,
        max_read_bytes=16 * 1024,
        max_reads_per_run=20,
        max_total_read_bytes=256 * 1024,
    ),
)
```

The policy can also be overridden with `RunOptions.tool_artifact_policy` or
`StructuredTaskSpec.tool_artifact_policy`. `JsonFileArtifactStore` is the
default; hosts may inject an `ArtifactStore` into `AgentRunService` or
`AgentOrchestrator` for remote or transactional storage. Artifact reads are
namespace-scoped, never accept a filesystem path, have their own call/byte
limits, and do not consume the application tool-call budget. Continue a
`preview` from its `next_offset`; subsequent `artifact_chunk` responses provide
another `next_offset` and an `eof` flag. Pending conversation state schema
version 2 and structured checkpoint schema version 6 persist `artifact_result`
references and artifact usage rather than copying the full result.

Artifact storage is always enabled. `JsonFileArtifactStore` stores plaintext
UTF-8 files and is intended for trusted local deployments; production hosts
handling credentials or browser session data should inject an encrypted store
with explicit access control and retention.

See [docs/memory_and_artifacts.md](docs/memory_and_artifacts.md) for the envelope
schema, projection rules, read limits and custom-store contract.

## Pending Tool Result Flow

Tools that need an external asynchronous result can return:

```python
from agent_core.spi import ToolResult

return ToolResult.pending_result(
    "Waiting for command output.",
    metadata={"command_id": "cmd-123"},
)
```

The application stores the returned `pending_id`, then resumes the turn when the
external result arrives:

```python
completed = conversation_agent.resume(
    namespace_id="workspace-1",
    run_id=pending.run_id,
    pending_id=pending.metadata["pending_id"],
    tool_content="command output",
)
```

## Minimal Integration Pattern

1. Build `CoreSettings` from your application config.
2. Create a `BaseLLMProvider` implementation or configure a bundled provider
   through the factories exported by `agent_core.spi`.
3. Register tools in `ToolRegistry`.
4. Instantiate a `RunStore` and `PolicyEngine`.
5. Build `AgentRunService` and execute a `StructuredTaskSpec` with an explicit
   `RunContext`.
6. Only if conversation is needed, add `SessionManager`, `AgentOrchestrator`
   and `ConversationAgent`.

See [docs/run_architecture.md](docs/run_architecture.md) for identifier,
context, idempotence and pipeline ownership rules.

## Public API

The supported API is intentionally split by responsibility. The package root
contains the autonomous run engine:

```python
from agent_core import (
    AgentRunResult,
    AgentRunService,
    AgentRunState,
    AgentRunMode,
    ArtifactChunk,
    ArtifactResultEnvelope,
    ArtifactStore,
    CoreSettings,
    ExecutionContext,
    ExecutionScope,
    JsonFileArtifactStore,
    JsonFileRunStore,
    LLMBudget,
    LLMBudgetExceededError,
    LLMBudgetUsage,
    LLMContextOverflowError,
    LLMContextPlan,
    LLMContextPolicy,
    LLMContextUsage,
    RunContext,
    RunOptions,
    RunStore,
    StructuredOutputContract,
    StructuredTaskResult,
    StructuredTaskRunner,
    StructuredTaskSpec,
    ToolArtifactDescriptor,
    ToolArtifactPolicy,
    ToolArtifactUsage,
)
```

Provider, policy, domain and tool extension contracts live in `agent_core.spi`:

```python
from agent_core.spi import (
    BaseLLMProvider,
    DomainHooks,
    LLMProviderError,
    PolicyEngine,
    ToolRegistry,
    ToolResult,
    build_tool_definition,
)
```

Conversation and session support is optional and imported separately:

```python
from agent_core.conversation import (
    AgentOrchestrator,
    ConversationAgent,
    SessionManager,
    SessionRepository,
)
from agent_core.observability import RunTrace, configure_logging, get_logger
```

Only names exported by these four facades are covered by the public compatibility
contract. Direct imports from other `agent_core.*` modules are implementation
dependencies and may change without notice.

See [docs/public_api.md](docs/public_api.md) for the ownership and compatibility
rules of each facade.

## Development Checks

```bash
.venv/bin/python -m ruff check agent_core tests examples
.venv/bin/python -m mypy agent_core
.venv/bin/python -m pytest
.venv/bin/python -m pytest --cov=agent_core --cov-report=term
.venv/bin/python -m build
```

The coverage configuration measures branches and enforces a ratchetable global
minimum. Deterministic fault-injection tests use the `chaos` marker and can be
run separately with `python -m pytest -m chaos`. Property-based resilience
tests are part of the normal suite; longer diagnostic runs are scheduled in CI.

## Repository Scope

This repository should stay focused on the generic runtime. Domain packages
should depend on it instead of adding their prompts, tools or reporting logic
here.
