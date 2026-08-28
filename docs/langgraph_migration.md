# LangGraph kernel migration

This document records the first four steps of the incremental migration of the
conversation agent kernel. The public conversation API and every durable
storage contract remain unchanged.

## Implemented scope

1. The existing direct and investigation control flows are mapped below.
2. Direct execution has a typed internal `AgentGraphState`.
3. `CoreSettings.agent_kernel_backend` selects `native` or `langgraph`; native
   remains the default.
4. The direct model/tool loop is a real multi-node LangGraph graph. The native
   fallback executes the same node implementations with an ordinary loop.

`investigate` and `deep_investigate` remain on `InvestigationController` during
this phase. `AgentRunService` and `StructuredTaskRunner` are also outside this
conversation-kernel migration.

## Existing orchestration map

The public entry point binds a `RunContext`, creates budget, context-planning,
and artifact scopes, acquires the session scope, builds the prompt, and starts
the run trace. It then routes by `RunOptions.mode`:

- `direct` enters the model/tool loop described below;
- `investigate` and `deep_investigate` enter `InvestigationController`;
- all paths finalize the existing `RunTrace` and return `AgentTurnResult`.

Pending tools stop the current invocation after agent-core has persisted the
exact provider transcript, counters, artifact usage, budget usage, trace id,
and remaining tool cursor. `resume_turn()` restores those values, finishes any
remaining calls in the same tool exchange, clears the pending marker, then
continues through the selected direct kernel or investigation controller.

## Direct graph

```mermaid
flowchart TD
    S([START]) --> M[call_model]
    M -->|provider failure| E([END])
    M -->|no tool calls| C[complete_response]
    M -->|tool calls| T[execute_tools]
    T -->|pending result| E
    T -->|tool budget exhausted| B[complete_budget]
    T -->|exchange completed| M
    C --> E
    B --> E
```

The graph nodes are intentionally coarse enough that a node completes one
agent-core atomic effect boundary:

- `call_model` accounts the request, invokes the configured provider, appends
  the assistant message, and records response telemetry;
- `execute_tools` delegates authorization, execution, artifact publication,
  tool history, atomic exchange persistence, and pending persistence to the
  existing implementation;
- `complete_response` persists the conversation turn and commits memory;
- `complete_budget` writes the deterministic budget terminal response and
  commits memory.

Provider failures are handled in `call_model` by the existing deterministic
failure path and route directly to `END`.

## Typed ephemeral state

`agent_core.agent_graph.state.AgentGraphState` contains only the values needed
to route one in-process direct invocation:

- identity and input: `user_input`, `session_id`, `context`, `turn_index`;
- provider transcript: `messages`, `assistant_message`;
- loop counters: `model_call_index`, `tool_calls_used`, `exchange_index`;
- context-reserve accounting and its one-shot warning flag;
- current `ToolExecutionStepResult`, terminal `AgentTurnResult`, and `RunTrace`.

This type is internal and is not a serialized schema. LangGraph message types
are not introduced: the graph keeps using agent-core's `LLMMessage` and
`AgentTurnResult` contracts.

## State ownership

| Concern | Owner in this phase | Reason |
| --- | --- | --- |
| In-process direct transitions | LangGraph when enabled | Makes the loop and terminal routes explicit |
| Provider and model translation | Existing `BaseLLMProvider` adapters | Preserves the Chantier 1 boundary |
| Tool authorization and execution | Agent-core | Preserves policy and SPI contracts |
| Tool artifacts and transcript projection | Agent-core | Preserves lossless artifact semantics |
| Session and conversation memory | `SessionManager` and memory journal | Avoids storage/schema migration |
| Pending tool resume | Agent-core pending payload plus `RunStore` checkpoint | Preserves restart and idempotence behavior |
| LLM budget and context planning | Existing scoped controllers | One controller still spans model and memory calls |
| Run traces | Existing `RunTrace` repository | Preserves current audit format |
| LangGraph checkpointing | Disabled | Prevents dual writes and ambiguous recovery authority |
| LangSmith export | Explicit agent-core opt-in | Graph state can contain the complete transcript |

The graph is therefore compiled without a LangGraph checkpointer. A pending
resume creates a fresh in-process graph invocation from agent-core's durable
pending transcript. This is deliberate, not an accidental limitation.
Graph execution is wrapped in a disabled LangSmith tracing context unless
`CoreSettings.langchain_tracing_enabled` is explicitly set, even when tracing
is enabled in the surrounding process environment.

## Backend selection and rollback

The default remains:

```python
CoreSettings(agent_kernel_backend="native")
```

The migrated direct path is enabled with:

```python
CoreSettings(agent_kernel_backend="langgraph")
```

The quickstart equivalent is
`AGENT_CORE_AGENT_KERNEL_BACKEND=langgraph`. Unknown values fail when the
orchestrator is constructed. Switching back to native does not require a data
migration because both backends use the same persistence contracts and node
behavior.

## Compatibility evidence

The deterministic parity suite runs the native and LangGraph direct kernels
against the same scripted provider and compares the observable result, number
of provider calls, context-block sequence, tool history, and trace-event
sequence. Dedicated cases cover pending/resume, provider failure, and tool-call
budget exhaustion. The existing direct, pending, trace, memory, budget,
context-planning, and investigation suites remain the broader regression net.
The opt-in paid Azure suite additionally runs the native and LangGraph kernels
against the same real LangChain-backed model, checking the complete tool loop,
persisted trace telemetry, and a LangGraph pending/resume cycle.

## Remaining full-graph work

The following work is intentionally not part of these first four steps:

- migrate investigation planning, reflection, decision, critique, and final
  synthesis into subgraphs;
- decide whether LangGraph or agent-core becomes the single durable checkpoint
  authority, then design and test a one-way storage migration;
- model external pending tools with LangGraph interrupts only after tool-node
  side effects are made explicitly idempotent at the interrupt boundary;
- migrate headless structured runs only if a graph adds value without weakening
  their stricter before/after-tool checkpoint protocol;
- expose graph streaming or inspection through a deliberate public contract,
  rather than leaking internal LangGraph objects.
