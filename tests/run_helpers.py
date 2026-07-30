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
    fact_lines = "\n".join(f"- {fact}" for fact in (confirmed_facts or []))
    action_lines = "\n".join(f"- {action}" for action in (next_actions or []))
    return {
        "turn_summary": f"Request: {user_intent}\nOutcome: {assistant_outcome}",
        "next_handoff": "\n\n".join(
            section
            for section in (
                f"Current objective:\n{objective}",
                f"Grounded observations:\n{fact_lines}" if fact_lines else "",
                f"Next useful action:\n{action_lines}" if action_lines else "",
            )
            if section
        ),
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
