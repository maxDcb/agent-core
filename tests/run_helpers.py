from __future__ import annotations

from uuid import uuid4
from weakref import WeakKeyDictionary

from agent_core.execution_context import ExecutionContext
from agent_core.orchestrator import AgentOrchestrator
from agent_core.run_context import RunContext
from agent_core.run_options import RunOptions
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskResult, StructuredTaskRunner, StructuredTaskSpec
from agent_core.types import AgentTurnResult

_ACTIVE_CONTEXTS: WeakKeyDictionary[AgentOrchestrator, RunContext] = WeakKeyDictionary()


def turn_memory_payload(
    *,
    objective: str = "Test objective",
    user_intent: str = "Test request",
    assistant_outcome: str = "Test outcome",
    confirmed_facts: list[str] | None = None,
    next_actions: list[str] | None = None,
) -> dict:
    return {
        "memory_id": "turn-0000-memory",
        "thread_id": "default",
        "turn_index": 0,
        "user_intent": user_intent,
        "assistant_outcome": assistant_outcome,
        "active_task": {
            "objective": objective,
            "status": "active",
            "next_action": next_actions[0] if next_actions else None,
            "open_questions": [],
            "constraints": [],
        },
        "exchange_memory_ids": [],
        "source_block_ids": [],
        "confirmed_facts": confirmed_facts or [],
        "superseded_facts": [],
        "open_hypotheses": [],
        "rejected_hypotheses": [],
        "open_questions": [],
        "resolved_questions": [],
        "decisions": [],
        "completed_actions": [],
        "next_actions": next_actions or [],
        "relevant_artifacts": [],
        "risk_notes": [],
        "domain_extensions": {},
        "origin": "model",
        "schema_version": "1",
    }


def execution_context(
    settings: CoreSettings,
    *,
    namespace_id: str = "default",
    application_context: dict | None = None,
) -> ExecutionContext:
    return ExecutionContext.from_run_context(
        context=RunContext(
            namespace_id=namespace_id,
            run_id=f"test-run-{uuid4().hex}",
            application_context=application_context or {},
        ),
        settings=settings,
    )


def run_structured(
    runner: StructuredTaskRunner,
    *,
    spec: StructuredTaskSpec,
    namespace_id: str = "default",
    application_context: dict | None = None,
) -> StructuredTaskResult:
    return runner.run(
        spec=spec,
        context=execution_context(
            runner.settings,
            namespace_id=namespace_id,
            application_context=application_context,
        ),
    )


def run_turn(
    orchestrator: AgentOrchestrator,
    user_input: str,
    *,
    options: RunOptions | None = None,
    thread_id: str = "default",
) -> AgentTurnResult:
    context = RunContext(
        namespace_id=thread_id,
        run_id=f"test-run-{uuid4().hex}",
        thread_id=thread_id,
    )
    _ACTIVE_CONTEXTS[orchestrator] = context
    return orchestrator.run_turn_result(
        user_input=user_input,
        thread_id=thread_id,
        context=context,
        options=options,
    )


def resume_turn(
    orchestrator: AgentOrchestrator,
    *,
    pending_id: str,
    tool_content: str,
    ok: bool = True,
) -> AgentTurnResult:
    context = _ACTIVE_CONTEXTS.get(orchestrator)
    if context is None:
        context = RunContext(
            namespace_id="default",
            run_id=f"test-resume-{uuid4().hex}",
            thread_id="default",
        )
    return orchestrator.resume_turn(
        pending_id=pending_id,
        tool_content=tool_content,
        thread_id=context.thread_id or "default",
        context=context,
        ok=ok,
    )
