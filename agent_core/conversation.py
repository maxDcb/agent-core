from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from agent_core.conversation_state import ConversationStateView
from agent_core.llm.base import LLMCallRecord, capture_llm_calls
from agent_core.llm_budget import LLMBudgetExceededError
from agent_core.orchestrator import AgentOrchestrator
from agent_core.run_context import RunContext
from agent_core.run_models import (
    AgentRunAttempt,
    AgentRunError,
    AgentRunResult,
    AgentRunState,
    RunCheckpoint,
    RunStatus,
)
from agent_core.run_options import RunOptions
from agent_core.run_store import RunStore
from agent_core.session_manager import SessionManager
from agent_core.session_repo import JsonFileSessionStore, SessionRepository, SessionStore
from agent_core.types import AgentTurnResult, SessionState

__all__ = [
    "AgentOrchestrator",
    "AgentTurnResult",
    "ConversationAgent",
    "ConversationStateView",
    "JsonFileSessionStore",
    "SessionManager",
    "SessionRepository",
    "SessionState",
    "SessionStore",
]


class ConversationAgent:
    """Optional thread adapter over the autonomous run lifecycle."""

    def __init__(self, *, orchestrator: AgentOrchestrator, run_store: RunStore) -> None:
        self.orchestrator = orchestrator
        self.run_store = run_store

    def execute_turn(
        self,
        *,
        thread_id: str,
        context: RunContext,
        user_input: str,
        options: RunOptions | None = None,
        run_id: str | None = None,
    ) -> AgentRunResult:
        resolved_run_id = run_id or context.run_id or f"run-{uuid4().hex}"
        bound_context = RunContext(
            namespace_id=context.namespace_id,
            run_id=resolved_run_id,
            parent_id=context.parent_id,
            thread_id=thread_id,
            scope=context.scope,
            correlation=context.correlation,
            application_context=context.application_context,
        )
        with self.run_store.acquire_execution(
            namespace_id=bound_context.namespace_id,
            run_id=resolved_run_id,
        ):
            return self._execute_turn_locked(
                thread_id=thread_id,
                context=bound_context,
                user_input=user_input,
                options=options or RunOptions.direct(),
            )

    def resume(
        self,
        *,
        namespace_id: str,
        run_id: str,
        pending_id: str,
        tool_content: str,
        ok: bool = True,
    ) -> AgentRunResult:
        with self.run_store.acquire_execution(namespace_id=namespace_id, run_id=run_id):
            state = self.run_store.load(namespace_id=namespace_id, run_id=run_id)
            if state is None:
                raise KeyError(f"Unknown run: {run_id}")
            if state.status != "pending":
                if state.result is not None:
                    return state.result
                raise ValueError(f"Run is not pending: {run_id}")
            if state.context.thread_id is None:
                raise ValueError("Pending conversation run has no thread_id")
            expected_pending_id = (
                state.checkpoint.payload.get("pending_id")
                if state.checkpoint is not None and state.checkpoint.kind == "conversation"
                else None
            )
            if expected_pending_id != pending_id:
                raise ValueError(f"Pending id does not match the persisted conversation checkpoint: {pending_id}")

            state.transition("running")
            attempt = AgentRunAttempt(
                attempt_id=f"attempt-{uuid4().hex}",
                resumed_from_sequence=state.checkpoint.sequence if state.checkpoint is not None else None,
            )
            state.attempts.append(attempt)
            self.run_store.save(state)
            previous_calls = list(state.result.llm_calls) if state.result is not None else []
            captured_calls: list[LLMCallRecord] = []
            try:
                with capture_llm_calls() as captured_calls:
                    turn = self.orchestrator.resume_turn(
                        pending_id=pending_id,
                        tool_content=tool_content,
                        thread_id=state.context.thread_id,
                        context=state.context,
                        ok=ok,
                    )
            except Exception as exc:
                return self._commit_conversation_failure(
                    state=state,
                    attempt=attempt,
                    exc=exc,
                    llm_calls=self._merge_llm_calls(previous_calls, captured_calls),
                )
            return self._commit_turn(
                state=state,
                attempt=attempt,
                turn=turn,
                llm_calls=self._merge_llm_calls(previous_calls, captured_calls),
            )

    def _execute_turn_locked(
        self,
        *,
        thread_id: str,
        context: RunContext,
        user_input: str,
        options: RunOptions,
    ) -> AgentRunResult:
        existing = self.run_store.load(namespace_id=context.namespace_id, run_id=context.run_id or "")
        if existing is not None:
            if existing.spec_id != "conversation_turn" or existing.context != context:
                raise ValueError(f"Run id is already bound to a different request: {context.run_id}")
            if existing.result is not None:
                return existing.result
            raise RuntimeError(f"Conversation run already exists without a result: {context.run_id}")

        state = AgentRunState(
            run_id=context.run_id or "",
            strategy=options.mode,
            spec_id="conversation_turn",
            context=context,
        )
        self.run_store.create(state)
        state.transition("running")
        attempt = AgentRunAttempt(attempt_id=f"attempt-{uuid4().hex}")
        state.attempts.append(attempt)
        self.run_store.save(state)
        captured_calls: list[LLMCallRecord] = []
        try:
            with capture_llm_calls() as captured_calls:
                turn = self.orchestrator.run_turn_result(
                    user_input=user_input,
                    thread_id=thread_id,
                    context=context,
                    options=options,
                )
        except Exception as exc:
            return self._commit_conversation_failure(
                state=state,
                attempt=attempt,
                exc=exc,
                llm_calls=captured_calls,
            )
        return self._commit_turn(state=state, attempt=attempt, turn=turn, llm_calls=captured_calls)

    def _commit_turn(
        self,
        *,
        state: AgentRunState,
        attempt: AgentRunAttempt,
        turn: AgentTurnResult,
        llm_calls: list[LLMCallRecord],
    ) -> AgentRunResult:
        status: RunStatus = "pending" if turn.is_pending else "completed"
        result = AgentRunResult(
            run_id=state.run_id,
            status=status,
            raw_content=turn.content,
            tool_calls_used=int(turn.metadata.get("tool_calls_used", 0)),
            iterations=int(turn.metadata.get("iterations_used", 0)),
            llm_calls=list(llm_calls),
            metadata={
                **turn.metadata,
                "pending_id": turn.pending_id,
                "tool_name": turn.tool_name,
                "tool_arguments": dict(turn.tool_arguments),
            },
        )
        previous_sequence = state.checkpoint.sequence if state.checkpoint is not None else 0
        state.result = result
        state.checkpoint = (
            RunCheckpoint(
                kind="conversation",
                sequence=previous_sequence + 1,
                payload={"pending_id": turn.pending_id},
            )
            if turn.pending_id
            else None
        )
        attempt.finish("pending" if turn.is_pending else "completed")
        state.transition(status)
        self.run_store.save(state)
        return result

    def _commit_conversation_failure(
        self,
        *,
        state: AgentRunState,
        attempt: AgentRunAttempt,
        exc: Exception,
        llm_calls: list[LLMCallRecord],
    ) -> AgentRunResult:
        error = AgentRunError(
            kind="conversation_run_error",
            message="Conversation run failed.",
            detail=str(exc),
        )
        budget_metadata = exc.budget_metadata if isinstance(exc, LLMBudgetExceededError) else {}
        result = AgentRunResult(
            run_id=state.run_id,
            status="failed",
            error=error,
            llm_calls=list(llm_calls),
            metadata=budget_metadata,
        )
        state.error = error
        state.result = result
        attempt.finish("failed", failure_reason=error.message)
        state.transition("failed")
        self.run_store.save(state)
        return result

    @staticmethod
    def _merge_llm_calls(
        previous_calls: list[LLMCallRecord],
        new_calls: list[LLMCallRecord],
    ) -> list[LLMCallRecord]:
        merged = list(previous_calls)
        for call in new_calls:
            call_index = len(merged) + 1
            merged.append(replace(call, call_id=f"llm-{call_index:04d}", call_index=call_index))
        return merged
