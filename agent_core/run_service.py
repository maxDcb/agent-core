from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import BaseLLMProvider
from agent_core.policy_engine import PolicyEngine
from agent_core.run_context import RunContext
from agent_core.run_models import AgentRunError, AgentRunResult, AgentRunState
from agent_core.run_store import RunStore
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskRunner, StructuredTaskSpec
from agent_core.tool_registry import ToolRegistry


class AgentRunService:
    """Autonomous, persisted run service used by headless pipelines.

    Pipeline ownership stays outside the core. ``parent_id`` and ``correlation``
    link the technical run to an application job, phase, or attempt.
    """

    def __init__(
        self,
        *,
        settings: CoreSettings,
        provider: BaseLLMProvider,
        tool_registry: ToolRegistry,
        policy_engine: PolicyEngine,
        run_store: RunStore,
    ) -> None:
        self.settings = settings
        self.run_store = run_store
        self.executor = StructuredTaskRunner(
            settings=settings,
            provider=provider,
            tool_registry=tool_registry,
            policy_engine=policy_engine,
        )

    def execute(
        self,
        *,
        spec: StructuredTaskSpec,
        context: RunContext,
        run_id: str | None = None,
        on_started: Callable[[str], None] | None = None,
    ) -> AgentRunResult:
        resolved_run_id = (run_id or context.run_id or f"run-{uuid4().hex}").strip()
        bound_context = context.with_run_id(resolved_run_id)
        existing = self.run_store.load(namespace_id=bound_context.namespace_id, run_id=resolved_run_id)
        if existing is not None:
            if existing.strategy != "structured" or existing.spec_id != spec.task_id or existing.context != bound_context:
                raise ValueError(f"Run id is already bound to a different request: {resolved_run_id}")
            if existing.result is not None and existing.status in {"completed", "failed", "cancelled", "blocked"}:
                return existing.result
            raise RuntimeError(f"Run already exists in non-terminal state: {resolved_run_id} ({existing.status})")
        state = AgentRunState(
            run_id=resolved_run_id,
            strategy="structured",
            spec_id=spec.task_id,
            context=bound_context,
        )
        self.run_store.create(state)
        if on_started is not None:
            on_started(resolved_run_id)
        state.transition("running")
        self.run_store.save(state)

        execution_context = ExecutionContext.from_run_context(context=bound_context, settings=self.settings)
        try:
            task_result = self.executor.run(spec=spec, context=execution_context)
        except Exception as exc:
            error = AgentRunError(
                kind="run_execution_error",
                message="Agent run execution failed.",
                detail=str(exc),
            )
            result = AgentRunResult(run_id=resolved_run_id, status="failed", error=error)
            state.error = error
            state.result = result
            state.transition("failed")
            self.run_store.save(state)
            return result

        if task_result.ok:
            result = AgentRunResult(
                run_id=resolved_run_id,
                status="completed",
                output=task_result.output,
                raw_content=task_result.raw_content,
                tool_history=list(task_result.tool_history),
                iterations=task_result.iterations,
                tool_calls_used=task_result.tool_calls_used,
                metadata={**task_result.metadata, "spec_id": spec.task_id},
            )
            state.result = result
            state.transition("completed")
        else:
            error = AgentRunError(
                kind="structured_run_failed",
                message=task_result.failure_reason or "Structured run failed.",
                detail=task_result.raw_content,
            )
            result = AgentRunResult(
                run_id=resolved_run_id,
                status="failed",
                output=task_result.output,
                raw_content=task_result.raw_content,
                error=error,
                tool_history=list(task_result.tool_history),
                iterations=task_result.iterations,
                tool_calls_used=task_result.tool_calls_used,
                metadata={**task_result.metadata, "spec_id": spec.task_id},
            )
            state.error = error
            state.result = result
            state.transition("failed")
        self.run_store.save(state)
        return result

    def get(self, *, namespace_id: str, run_id: str) -> AgentRunState | None:
        return self.run_store.load(namespace_id=namespace_id, run_id=run_id)

    def list(self, *, namespace_id: str, parent_id: str | None = None) -> list[AgentRunState]:
        return self.run_store.list(namespace_id=namespace_id, parent_id=parent_id)
