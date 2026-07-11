from __future__ import annotations

from uuid import uuid4

from agent_core.orchestrator import AgentOrchestrator
from agent_core.run_context import RunContext
from agent_core.run_models import AgentRunError, AgentRunResult, AgentRunState, RunStatus
from agent_core.run_options import RunOptions
from agent_core.run_store import RunStore


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
        existing = self.run_store.load(namespace_id=bound_context.namespace_id, run_id=resolved_run_id)
        if existing is not None:
            if existing.spec_id != "conversation_turn" or existing.context != bound_context:
                raise ValueError(f"Run id is already bound to a different request: {resolved_run_id}")
            if existing.result is not None:
                return existing.result
            raise RuntimeError(f"Conversation run already exists without a result: {resolved_run_id}")
        selected_options = options or RunOptions.direct()
        state = AgentRunState(
            run_id=resolved_run_id,
            strategy=selected_options.mode,
            spec_id="conversation_turn",
            context=bound_context,
        )
        self.run_store.create(state)
        state.transition("running")
        self.run_store.save(state)
        try:
            turn = self.orchestrator.run_turn_result(
                user_input=user_input,
                thread_id=thread_id,
                context=bound_context,
                options=selected_options,
            )
        except Exception as exc:
            error = AgentRunError(kind="conversation_run_error", message="Conversation run failed.", detail=str(exc))
            result = AgentRunResult(run_id=resolved_run_id, status="failed", error=error)
            state.error = error
            state.result = result
            state.transition("failed")
            self.run_store.save(state)
            return result

        status: RunStatus = "pending" if turn.is_pending else "completed"
        result = AgentRunResult(
            run_id=resolved_run_id,
            status=status,
            raw_content=turn.content,
            tool_calls_used=int(turn.metadata.get("tool_calls_used", 0)),
            iterations=int(turn.metadata.get("iterations_used", 0)),
            metadata={
                **turn.metadata,
                "pending_id": turn.pending_id,
                "tool_name": turn.tool_name,
                "tool_arguments": dict(turn.tool_arguments),
            },
        )
        state.result = result
        state.checkpoint = {"pending_id": turn.pending_id} if turn.pending_id else {}
        state.transition(status)
        self.run_store.save(state)
        return result

    def resume(
        self,
        *,
        namespace_id: str,
        run_id: str,
        pending_id: str,
        tool_content: str,
        ok: bool = True,
    ) -> AgentRunResult:
        state = self.run_store.load(namespace_id=namespace_id, run_id=run_id)
        if state is None:
            raise KeyError(f"Unknown run: {run_id}")
        if state.status != "pending":
            if state.result is not None:
                return state.result
            raise ValueError(f"Run is not pending: {run_id}")
        if state.context.thread_id is None:
            raise ValueError("Pending conversation run has no thread_id")

        state.transition("running")
        self.run_store.save(state)
        turn = self.orchestrator.resume_turn(
            pending_id=pending_id,
            tool_content=tool_content,
            thread_id=state.context.thread_id,
            context=state.context,
            ok=ok,
        )
        status: RunStatus = "pending" if turn.is_pending else "completed"
        result = AgentRunResult(
            run_id=run_id,
            status=status,
            raw_content=turn.content,
            tool_calls_used=int(turn.metadata.get("tool_calls_used", 0)),
            iterations=int(turn.metadata.get("iterations_used", 0)),
            metadata={
                **turn.metadata,
                "pending_id": turn.pending_id,
                "tool_name": turn.tool_name,
                "tool_arguments": dict(turn.tool_arguments),
            },
        )
        state.result = result
        state.checkpoint = {"pending_id": turn.pending_id} if turn.pending_id else {}
        state.transition(status)
        self.run_store.save(state)
        return result
