from __future__ import annotations

import json
from collections.abc import Callable
from uuid import uuid4

from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import BaseLLMProvider, LLMMessage
from agent_core.logging_utils import safe_preview
from agent_core.policy_engine import PolicyEngine
from agent_core.run_context import RunContext
from agent_core.run_models import (
    AgentRunAttempt,
    AgentRunError,
    AgentRunResult,
    AgentRunState,
    RunCheckpoint,
)
from agent_core.run_store import RunStore
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import (
    StructuredTaskCheckpoint,
    StructuredTaskRecoveryError,
    StructuredTaskResult,
    StructuredTaskRunner,
    StructuredTaskSpec,
)
from agent_core.tool_registry import ToolRegistry
from agent_core.types import ToolResult


class AgentRunService:
    """Persisted autonomous run service with lossless checkpoint recovery."""

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
        with self.run_store.acquire_execution(
            namespace_id=bound_context.namespace_id,
            run_id=resolved_run_id,
        ):
            existing = self.run_store.load(namespace_id=bound_context.namespace_id, run_id=resolved_run_id)
            if existing is not None:
                self._validate_binding(state=existing, spec=spec, context=bound_context)
                if not self._checkpoint_matches_spec(state=existing, spec=spec):
                    raise ValueError(f"Run id is already bound to a different specification: {resolved_run_id}")
                if existing.result is not None and existing.status in {
                    "completed",
                    "failed",
                    "cancelled",
                    "blocked",
                }:
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
            return self._start_or_resume_locked(state=state, spec=spec, resume_checkpoint=None)

    def resume(
        self,
        *,
        spec: StructuredTaskSpec,
        context: RunContext,
        run_id: str,
    ) -> AgentRunResult:
        resolved_run_id = run_id.strip()
        bound_context = context.with_run_id(resolved_run_id)
        with self.run_store.acquire_execution(
            namespace_id=bound_context.namespace_id,
            run_id=resolved_run_id,
        ):
            state = self.run_store.load(namespace_id=bound_context.namespace_id, run_id=resolved_run_id)
            if state is None:
                raise KeyError(f"Unknown run: {resolved_run_id}")
            self._validate_binding(state=state, spec=spec, context=bound_context)
            if not self._checkpoint_matches_spec(state=state, spec=spec):
                return self._block_recovery(
                    state=state,
                    kind="spec_mismatch",
                    message="The structured task specification does not match the persisted run checkpoint.",
                )
            if state.result is not None and state.status in {"completed", "failed", "cancelled", "blocked"}:
                return state.result

            if state.status in {"running", "pending"}:
                self._mark_active_attempt_interrupted(state)
                state.transition("interrupted")
                self.run_store.save(state)

            resume_checkpoint: StructuredTaskCheckpoint | None = None
            if state.checkpoint is not None:
                if state.checkpoint.kind != "structured_task":
                    return self._block_recovery(
                        state=state,
                        kind="invalid_checkpoint_kind",
                        message="Run checkpoint is not a structured-task checkpoint.",
                    )
                resume_checkpoint = StructuredTaskCheckpoint.from_dict(state.checkpoint.payload)
                if resume_checkpoint is None:
                    return self._block_recovery(
                        state=state,
                        kind="invalid_checkpoint",
                        message="Run checkpoint is invalid or uses an unsupported schema version.",
                    )
            elif state.status != "created":
                return self._block_recovery(
                    state=state,
                    kind="missing_checkpoint",
                    message="Interrupted run has no checkpoint from which it can safely resume.",
                )

            return self._start_or_resume_locked(
                state=state,
                spec=spec,
                resume_checkpoint=resume_checkpoint,
            )

    def get(self, *, namespace_id: str, run_id: str) -> AgentRunState | None:
        return self.run_store.load(namespace_id=namespace_id, run_id=run_id)

    def list(self, *, namespace_id: str, parent_id: str | None = None) -> list[AgentRunState]:
        return self.run_store.list(namespace_id=namespace_id, parent_id=parent_id)

    def resolve_ambiguous_tool(
        self,
        *,
        spec: StructuredTaskSpec,
        context: RunContext,
        run_id: str,
        tool_call_id: str,
        result: ToolResult,
    ) -> AgentRunResult:
        """Resume a blocked run after the host reconciles an ambiguous tool effect."""

        resolved_run_id = run_id.strip()
        bound_context = context.with_run_id(resolved_run_id)
        with self.run_store.acquire_execution(
            namespace_id=bound_context.namespace_id,
            run_id=resolved_run_id,
        ):
            state = self.run_store.load(namespace_id=bound_context.namespace_id, run_id=resolved_run_id)
            if state is None:
                raise KeyError(f"Unknown run: {resolved_run_id}")
            self._validate_binding(state=state, spec=spec, context=bound_context)
            if state.status != "blocked" or state.error is None or state.error.kind != "ambiguous_tool_execution":
                raise ValueError(f"Run is not blocked on an ambiguous tool execution: {resolved_run_id}")
            if state.checkpoint is None or state.checkpoint.kind != "structured_task":
                raise ValueError("Blocked run has no structured-task checkpoint")
            checkpoint = StructuredTaskCheckpoint.from_dict(state.checkpoint.payload)
            if checkpoint is None or checkpoint.spec_fingerprint != spec.fingerprint():
                raise ValueError("Blocked run checkpoint does not match the supplied specification")
            pending_index = checkpoint.next_tool_call_index
            if pending_index >= len(checkpoint.pending_tool_calls):
                raise ValueError("Blocked run has no pending tool call to reconcile")
            pending = checkpoint.pending_tool_calls[pending_index]
            if pending.tool_call_id != tool_call_id or pending.status != "running":
                raise ValueError(f"Tool call is not the ambiguous execution recorded by the run: {tool_call_id}")

            try:
                loaded_arguments = json.loads(pending.arguments_json or "{}")
                arguments = loaded_arguments if isinstance(loaded_arguments, dict) else {}
            except json.JSONDecodeError:
                arguments = {}
            checkpoint.messages.append(
                self._reconciled_tool_message(tool_call_id=tool_call_id, content=result.content)
            )
            checkpoint.tool_history.append(
                {
                    "tool_name": pending.tool_name,
                    "arguments": arguments,
                    "status": "ok" if result.ok else "tool_error",
                    "content_preview": safe_preview(result.content, limit=500),
                    "reconciled_after_interruption": True,
                }
            )
            pending.status = "completed"
            checkpoint.next_tool_call_index += 1
            checkpoint.sequence += 1
            state.checkpoint = RunCheckpoint(
                kind="structured_task",
                sequence=checkpoint.sequence,
                payload=checkpoint.to_dict(),
            )
            state.result = None
            state.error = None
            return self._start_or_resume_locked(
                state=state,
                spec=spec,
                resume_checkpoint=checkpoint,
            )

    def _start_or_resume_locked(
        self,
        *,
        state: AgentRunState,
        spec: StructuredTaskSpec,
        resume_checkpoint: StructuredTaskCheckpoint | None,
    ) -> AgentRunResult:
        if state.status in {"created", "interrupted", "blocked"}:
            state.transition("running")
        attempt = AgentRunAttempt(
            attempt_id=f"attempt-{uuid4().hex}",
            resumed_from_sequence=(resume_checkpoint.sequence if resume_checkpoint is not None else None),
        )
        state.attempts.append(attempt)
        self.run_store.save(state)

        def persist_checkpoint(checkpoint: StructuredTaskCheckpoint) -> None:
            state.checkpoint = RunCheckpoint(
                kind="structured_task",
                sequence=checkpoint.sequence,
                payload=checkpoint.to_dict(),
            )
            self.run_store.save(state)

        execution_context = ExecutionContext.from_run_context(context=state.context, settings=self.settings)
        try:
            if resume_checkpoint is None:
                task_result = self.executor.run(
                    spec=spec,
                    context=execution_context,
                    on_checkpoint=persist_checkpoint,
                )
            else:
                task_result = self.executor.resume(
                    spec=spec,
                    context=execution_context,
                    checkpoint=resume_checkpoint,
                    on_checkpoint=persist_checkpoint,
                )
        except StructuredTaskRecoveryError as exc:
            attempt.finish("blocked", failure_reason=str(exc))
            return self._block_recovery(
                state=state,
                kind=exc.kind,
                message=str(exc),
                metadata={"tool_call_id": exc.tool_call_id} if exc.tool_call_id else {},
            )
        except Exception as exc:
            error = AgentRunError(
                kind="run_execution_error",
                message="Agent run execution failed.",
                detail=str(exc),
            )
            result = AgentRunResult(run_id=state.run_id, status="failed", error=error)
            state.error = error
            state.result = result
            attempt.finish("failed", failure_reason=error.message)
            state.transition("failed")
            self.run_store.save(state)
            return result

        return self._commit_task_result(state=state, attempt=attempt, task_result=task_result, spec=spec)

    def _commit_task_result(
        self,
        *,
        state: AgentRunState,
        attempt: AgentRunAttempt,
        task_result: StructuredTaskResult,
        spec: StructuredTaskSpec,
    ) -> AgentRunResult:
        if task_result.ok:
            result = AgentRunResult(
                run_id=state.run_id,
                status="completed",
                output=task_result.output,
                raw_content=task_result.raw_content,
                tool_history=list(task_result.tool_history),
                iterations=task_result.iterations,
                tool_calls_used=task_result.tool_calls_used,
                metadata={**task_result.metadata, "spec_id": spec.task_id},
            )
            state.result = result
            attempt.finish("completed")
            state.transition("completed")
        else:
            error = AgentRunError(
                kind="structured_run_failed",
                message=task_result.failure_reason or "Structured run failed.",
                detail=task_result.raw_content,
            )
            result = AgentRunResult(
                run_id=state.run_id,
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
            attempt.finish("failed", failure_reason=error.message)
            state.transition("failed")
        self.run_store.save(state)
        return result

    def _block_recovery(
        self,
        *,
        state: AgentRunState,
        kind: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> AgentRunResult:
        error = AgentRunError(
            kind=kind,
            message=message,
            retryable=False,
        )
        result = AgentRunResult(
            run_id=state.run_id,
            status="blocked",
            error=error,
            metadata={"recovery_blocked": True, **(metadata or {})},
        )
        state.error = error
        state.result = result
        active = next((attempt for attempt in reversed(state.attempts) if attempt.status == "running"), None)
        if active is not None:
            active.finish("blocked", failure_reason=message)
        if state.status != "blocked":
            state.transition("blocked")
        self.run_store.save(state)
        return result

    @staticmethod
    def _mark_active_attempt_interrupted(state: AgentRunState) -> None:
        active = next((attempt for attempt in reversed(state.attempts) if attempt.status == "running"), None)
        if active is not None:
            active.finish(
                "interrupted",
                failure_reason="Execution ownership was lost before the run reached a terminal state.",
            )

    @staticmethod
    def _validate_binding(*, state: AgentRunState, spec: StructuredTaskSpec, context: RunContext) -> None:
        if state.strategy != "structured" or state.spec_id != spec.task_id or state.context != context:
            raise ValueError(f"Run id is already bound to a different request: {state.run_id}")

    @staticmethod
    def _checkpoint_matches_spec(*, state: AgentRunState, spec: StructuredTaskSpec) -> bool:
        checkpoint = state.checkpoint
        if checkpoint is not None and checkpoint.kind == "structured_task":
            persisted = StructuredTaskCheckpoint.from_dict(checkpoint.payload)
            return persisted is None or persisted.spec_fingerprint == spec.fingerprint()
        return True

    @staticmethod
    def _reconciled_tool_message(*, tool_call_id: str, content: str) -> LLMMessage:
        return LLMMessage(role="tool", tool_call_id=tool_call_id, content=content)
