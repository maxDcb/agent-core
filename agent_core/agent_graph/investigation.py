from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal, Protocol, cast

from agent_core.agent_graph.state import (
    InvestigationGraphState,
    InvestigationGraphUpdate,
    build_graph_checkpoint,
    normalize_agent_kernel_backend,
)
from agent_core.execution_context import ExecutionContext
from agent_core.investigation_state import InvestigationState
from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage
from agent_core.llm.errors import LLMProviderError
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.run_options import RunOptions
from agent_core.settings import CoreSettings
from agent_core.tool_artifacts import active_tool_artifact_runtime
from agent_core.turn_steps import PendingResumeState, ToolExecutionStepResult
from agent_core.types import AgentTurnResult

logger = get_logger("core.agent_graph.investigation")

EntryRoute = Literal["initialize_plan", "reflect_decide"]
InitializeRoute = Literal["assistant_step", "end"]
AssistantRoute = Literal["execute_tools", "handle_final_draft", "complete_max_tools", "end"]
ToolRoute = Literal["reflect_decide", "complete_max_tools", "end"]
ContinueRoute = Literal["assistant_step", "complete_max_iterations", "end"]


def _llm_failure_stop_reason(error: LLMProviderError) -> str:
    if error.kind == "budget_exhausted":
        return "llm_budget_exhausted"
    if error.kind == "context_overflow":
        return "llm_context_overflow"
    return "provider_failure"


class InvestigationOperations(Protocol):
    settings: CoreSettings

    def _record_event(
        self,
        *,
        event_type: str,
        summary: str,
        iteration: int | None = None,
        payload: dict[str, Any] | None = None,
        related_tool_call_id: str | None = None,
    ) -> None: ...

    def _synthesize_initial_plan(
        self,
        *,
        user_input: str,
        state: InvestigationState,
        options: RunOptions,
    ) -> InvestigationState: ...

    def _messages_with_iteration_state(
        self,
        *,
        messages: list[LLMMessage],
        state: InvestigationState,
        iteration: int,
    ) -> list[LLMMessage]: ...

    def _call_options(self, *, options: RunOptions, target: str) -> LLMCallOptions: ...

    def call_model_once(
        self,
        *,
        messages: list[LLMMessage],
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult: ...

    def handle_provider_failure(
        self,
        *,
        error: LLMProviderError,
        user_input: str,
        turn_index: int,
    ) -> AgentTurnResult: ...

    def _attach_metadata(
        self,
        result: AgentTurnResult,
        *,
        options: RunOptions,
        iterations_used: int,
        tool_calls_used: int,
        stop_reason: str,
        state: InvestigationState,
    ) -> AgentTurnResult: ...

    def execute_tool_calls_once(
        self,
        *,
        user_input: str,
        session_id: str,
        context: ExecutionContext,
        messages: list[LLMMessage],
        turn_index: int,
        exchange_index: int,
        tool_calls_used: int,
        assistant_message: LLMMessage,
        max_tool_calls: int,
        pending_metadata_extra: dict[str, Any] | None = None,
    ) -> ToolExecutionStepResult: ...

    def _reflect_and_decide_after_tools(
        self,
        *,
        user_input: str,
        turn_index: int,
        options: RunOptions,
        state: InvestigationState,
        messages: list[LLMMessage],
        tool_step: ToolExecutionStepResult,
        iterations_used: int,
        no_progress_iterations: int,
    ) -> tuple[AgentTurnResult | None, int]: ...

    def _evaluate_final_draft(
        self,
        *,
        user_input: str,
        messages: list[LLMMessage],
        turn_index: int,
        options: RunOptions,
        state: InvestigationState,
        final_draft: str,
        iterations_used: int,
        tool_calls_used: int,
    ) -> AgentTurnResult | None: ...

    def _complete_with_budget_answer(
        self,
        *,
        user_input: str,
        turn_index: int,
        options: RunOptions,
        state: InvestigationState,
        messages: list[LLMMessage],
        iterations_used: int,
        tool_calls_used: int,
        stop_reason: str,
    ) -> AgentTurnResult: ...


class InvestigationTurnNodes:
    """Multi-node investigate/deep-investigate control flow."""

    def __init__(self, operations: InvestigationOperations) -> None:
        self.operations = operations

    def initial_state(
        self,
        *,
        user_input: str,
        session_id: str,
        context: ExecutionContext,
        messages: list[LLMMessage],
        turn_index: int,
        options: RunOptions,
        investigation_state: InvestigationState | None = None,
        iterations_used: int = 0,
        tool_calls_used: int = 0,
        exchange_index: int = 0,
        no_progress_iterations: int = 0,
        resume_tool_step: ToolExecutionStepResult | None = None,
    ) -> InvestigationGraphState:
        return InvestigationGraphState(
            user_input=user_input,
            session_id=session_id,
            context=context,
            messages=list(messages),
            turn_index=turn_index,
            options=options,
            investigation_state=investigation_state or InvestigationState.create_template(objective=user_input),
            iterations_used=iterations_used,
            tool_calls_used=tool_calls_used,
            exchange_index=exchange_index,
            no_progress_iterations=no_progress_iterations,
            assistant_message=None,
            tool_step=resume_tool_step,
            final_draft=None,
            result=resume_tool_step.pending_result if resume_tool_step is not None else None,
            entrypoint="after_tools" if resume_tool_step is not None else "initialize",
        )

    @staticmethod
    def route_entry(state: InvestigationGraphState) -> EntryRoute:
        return "reflect_decide" if state["entrypoint"] == "after_tools" else "initialize_plan"

    def initialize_plan(self, state: InvestigationGraphState) -> InvestigationGraphUpdate:
        options = state["options"]
        investigation_state = state["investigation_state"]
        if not options.require_initial_plan:
            return {"result": None}

        self.operations._record_event(
            event_type="initial_plan_started",
            summary="Initial investigation plan synthesis started",
            payload={"mode": options.mode},
        )
        try:
            investigation_state = self.operations._synthesize_initial_plan(
                user_input=state["user_input"],
                state=investigation_state,
                options=options,
            )
        except LLMProviderError as exc:
            self.operations._record_event(
                event_type="llm_provider_failure",
                summary="Initial plan provider failure handled",
                payload={"kind": exc.kind},
            )
            failure_result = self.operations.handle_provider_failure(
                error=exc,
                user_input=state["user_input"],
                turn_index=state["turn_index"],
            )
            return {
                "result": self.operations._attach_metadata(
                    failure_result,
                    options=options,
                    iterations_used=0,
                    tool_calls_used=0,
                    stop_reason=_llm_failure_stop_reason(exc),
                    state=investigation_state,
                )
            }
        except ValueError as exc:
            if not options.recover_internal_synthesis_errors:
                raise
            logger.warning(
                "Initial investigation plan synthesis failed; continuing with template state",
                extra={"error_preview": safe_preview(str(exc), limit=200)},
            )
            investigation_state.metadata["initial_plan_synthesis_error"] = safe_preview(str(exc), limit=200)
            self.operations._record_event(
                event_type="structured_synthesis_recovered",
                summary="Initial plan synthesis failed; continuing with template investigation state",
                payload={
                    "target": "investigation_initial_plan",
                    "error_preview": investigation_state.metadata["initial_plan_synthesis_error"],
                },
            )
        else:
            self.operations._record_event(
                event_type="initial_plan_created",
                summary="Initial investigation plan created",
                payload={"investigation_state": investigation_state.compact_summary()},
            )
        return {"investigation_state": investigation_state, "result": None}

    @staticmethod
    def route_after_initialize(state: InvestigationGraphState) -> InitializeRoute:
        return "end" if state["result"] is not None else "assistant_step"

    def assistant_step(self, state: InvestigationGraphState) -> InvestigationGraphUpdate:
        options = state["options"]
        iterations_used = state["iterations_used"] + 1
        messages = list(state["messages"])
        investigation_state = state["investigation_state"]
        self.operations._record_event(
            event_type="investigation_iteration_started",
            summary="Investigation iteration started",
            iteration=iterations_used,
            payload={
                "tool_calls_used": state["tool_calls_used"],
                "max_iterations": options.max_iterations,
                "max_tool_calls": options.max_tool_calls,
            },
        )
        try:
            assistant_messages = self.operations._messages_with_iteration_state(
                messages=messages,
                state=investigation_state,
                iteration=iterations_used,
            )
            llm_response = self.operations.call_model_once(
                messages=assistant_messages,
                options=self.operations._call_options(options=options, target="assistant_step"),
            )
        except LLMProviderError as exc:
            self.operations._record_event(
                event_type="llm_provider_failure",
                summary="Assistant step provider failure handled",
                iteration=iterations_used,
                payload={"kind": exc.kind},
            )
            failure_result = self.operations.handle_provider_failure(
                error=exc,
                user_input=state["user_input"],
                turn_index=state["turn_index"],
            )
            return {
                "iterations_used": iterations_used,
                "result": self.operations._attach_metadata(
                    failure_result,
                    options=options,
                    iterations_used=iterations_used,
                    tool_calls_used=state["tool_calls_used"],
                    stop_reason=_llm_failure_stop_reason(exc),
                    state=investigation_state,
                ),
            }

        assistant_message = LLMMessage(
            role="assistant",
            content=llm_response.content,
            tool_calls=list(llm_response.tool_calls),
        )
        messages.append(assistant_message)
        self.operations._record_event(
            event_type="assistant_step_completed",
            summary="Assistant investigation step completed",
            iteration=iterations_used,
            payload={
                "content_length": len(llm_response.content),
                "tool_call_count": len(llm_response.tool_calls),
                "tool_calls": [
                    {"id": tool_call.id, "name": tool_call.name} for tool_call in llm_response.tool_calls
                ],
            },
        )
        final_draft = None
        if not llm_response.tool_calls:
            final_draft = llm_response.content
            self.operations._record_event(
                event_type="final_draft_received",
                summary="Assistant produced a final draft",
                iteration=iterations_used,
                payload={"content_length": len(llm_response.content)},
            )
        return {
            "messages": messages,
            "iterations_used": iterations_used,
            "assistant_message": assistant_message,
            "tool_step": None,
            "final_draft": final_draft,
            "result": None,
        }

    @staticmethod
    def route_after_assistant(state: InvestigationGraphState) -> AssistantRoute:
        if state["result"] is not None:
            return "end"
        assistant_message = state["assistant_message"]
        if assistant_message is None:
            raise RuntimeError("Investigation graph has no assistant message")
        if not assistant_message.tool_calls:
            return "handle_final_draft"
        artifact_runtime = active_tool_artifact_runtime()
        only_internal_calls = (
            artifact_runtime is not None
            and all(artifact_runtime.is_internal_tool(tool_call.name) for tool_call in assistant_message.tool_calls)
        )
        if state["tool_calls_used"] >= state["options"].max_tool_calls and not only_internal_calls:
            return "complete_max_tools"
        return "execute_tools"

    def execute_tools(self, state: InvestigationGraphState) -> InvestigationGraphUpdate:
        assistant_message = state["assistant_message"]
        if assistant_message is None:
            raise RuntimeError("Investigation graph cannot execute tools without an assistant message")
        options = state["options"]
        investigation_state = state["investigation_state"]
        tool_step = self.operations.execute_tool_calls_once(
            user_input=state["user_input"],
            session_id=state["session_id"],
            context=state["context"],
            messages=list(state["messages"]),
            turn_index=state["turn_index"],
            exchange_index=state["exchange_index"],
            tool_calls_used=state["tool_calls_used"],
            assistant_message=assistant_message,
            max_tool_calls=options.max_tool_calls,
            pending_metadata_extra={
                "mode": options.mode,
                "run_options": asdict(options),
                "investigation_state": investigation_state.to_dict(),
                "iterations_used": state["iterations_used"],
                "no_progress_iterations": state["no_progress_iterations"],
                "agent_graph_checkpoint": build_graph_checkpoint(
                    graph="investigation",
                    backend=normalize_agent_kernel_backend(self.operations.settings.agent_kernel_backend),
                    resume_node="resume_tool_exchange",
                ),
            },
        )
        self.operations._record_event(
            event_type="tool_step_completed",
            summary="Investigation tool step completed",
            iteration=state["iterations_used"],
            payload={
                "tool_names": list(tool_step.tool_names),
                "tool_statuses": list(tool_step.tool_statuses),
                "tool_calls_used": tool_step.tool_calls_used,
                "budget_exhausted": tool_step.budget_exhausted,
                "pending": tool_step.pending_result is not None,
            },
        )
        result = tool_step.pending_result
        if result is not None:
            result = self.operations._attach_metadata(
                result,
                options=options,
                iterations_used=state["iterations_used"],
                tool_calls_used=tool_step.tool_calls_used,
                stop_reason="pending_tool_result",
                state=investigation_state,
            )
        return {
            "messages": tool_step.messages,
            "exchange_index": tool_step.exchange_index,
            "tool_calls_used": tool_step.tool_calls_used,
            "tool_step": tool_step,
            "result": result,
        }

    @staticmethod
    def route_after_tools(state: InvestigationGraphState) -> ToolRoute:
        tool_step = state["tool_step"]
        if tool_step is None:
            raise RuntimeError("Investigation graph has no completed tool step")
        if state["result"] is not None:
            return "end"
        if tool_step.budget_exhausted:
            return "complete_max_tools"
        return "reflect_decide"

    def reflect_decide(self, state: InvestigationGraphState) -> InvestigationGraphUpdate:
        tool_step = state["tool_step"]
        if tool_step is None:
            raise RuntimeError("Investigation graph cannot reflect without a tool step")
        result, no_progress_iterations = self.operations._reflect_and_decide_after_tools(
            user_input=state["user_input"],
            turn_index=state["turn_index"],
            options=state["options"],
            state=state["investigation_state"],
            messages=state["messages"],
            tool_step=tool_step,
            iterations_used=state["iterations_used"],
            no_progress_iterations=state["no_progress_iterations"],
        )
        return {
            "no_progress_iterations": no_progress_iterations,
            "tool_calls_used": tool_step.tool_calls_used,
            "exchange_index": tool_step.exchange_index,
            "result": result,
        }

    @staticmethod
    def route_after_continue(state: InvestigationGraphState) -> ContinueRoute:
        if state["result"] is not None:
            return "end"
        if state["iterations_used"] >= state["options"].max_iterations:
            return "complete_max_iterations"
        return "assistant_step"

    def handle_final_draft(self, state: InvestigationGraphState) -> InvestigationGraphUpdate:
        final_draft = state["final_draft"]
        if final_draft is None:
            raise RuntimeError("Investigation graph cannot finalize without a draft")
        result = self.operations._evaluate_final_draft(
            user_input=state["user_input"],
            messages=state["messages"],
            turn_index=state["turn_index"],
            options=state["options"],
            state=state["investigation_state"],
            final_draft=final_draft,
            iterations_used=state["iterations_used"],
            tool_calls_used=state["tool_calls_used"],
        )
        return {"result": result, "final_draft": None}

    def complete_max_tools(self, state: InvestigationGraphState) -> InvestigationGraphUpdate:
        return {
            "result": self.operations._complete_with_budget_answer(
                user_input=state["user_input"],
                turn_index=state["turn_index"],
                options=state["options"],
                state=state["investigation_state"],
                messages=state["messages"],
                iterations_used=state["iterations_used"],
                tool_calls_used=state["tool_calls_used"],
                stop_reason="max_tool_calls",
            )
        }

    def complete_max_iterations(self, state: InvestigationGraphState) -> InvestigationGraphUpdate:
        return {
            "result": self.operations._complete_with_budget_answer(
                user_input=state["user_input"],
                turn_index=state["turn_index"],
                options=state["options"],
                state=state["investigation_state"],
                messages=state["messages"],
                iterations_used=state["iterations_used"],
                tool_calls_used=state["tool_calls_used"],
                stop_reason="max_iterations",
            )
        }


class LangGraphInvestigationKernel:
    backend = "langgraph"

    def __init__(self, operations: InvestigationOperations) -> None:
        from langgraph.graph import END, START, StateGraph

        self.nodes = InvestigationTurnNodes(operations)
        builder = StateGraph(InvestigationGraphState)
        builder.add_node("initialize_plan", self.nodes.initialize_plan)
        builder.add_node("assistant_step", self.nodes.assistant_step)
        builder.add_node("execute_tools", self.nodes.execute_tools)
        builder.add_node("reflect_decide", self.nodes.reflect_decide)
        builder.add_node("handle_final_draft", self.nodes.handle_final_draft)
        builder.add_node("complete_max_tools", self.nodes.complete_max_tools)
        builder.add_node("complete_max_iterations", self.nodes.complete_max_iterations)
        builder.add_conditional_edges(
            START,
            self.nodes.route_entry,
            {"initialize_plan": "initialize_plan", "reflect_decide": "reflect_decide"},
        )
        builder.add_conditional_edges(
            "initialize_plan",
            self.nodes.route_after_initialize,
            {"assistant_step": "assistant_step", "end": END},
        )
        builder.add_conditional_edges(
            "assistant_step",
            self.nodes.route_after_assistant,
            {
                "execute_tools": "execute_tools",
                "handle_final_draft": "handle_final_draft",
                "complete_max_tools": "complete_max_tools",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "execute_tools",
            self.nodes.route_after_tools,
            {
                "reflect_decide": "reflect_decide",
                "complete_max_tools": "complete_max_tools",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "reflect_decide",
            self.nodes.route_after_continue,
            {
                "assistant_step": "assistant_step",
                "complete_max_iterations": "complete_max_iterations",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "handle_final_draft",
            self.nodes.route_after_continue,
            {
                "assistant_step": "assistant_step",
                "complete_max_iterations": "complete_max_iterations",
                "end": END,
            },
        )
        builder.add_edge("complete_max_tools", END)
        builder.add_edge("complete_max_iterations", END)
        self.graph = builder.compile()

    def run(
        self,
        *,
        user_input: str,
        session_id: str,
        context: ExecutionContext,
        messages: list[LLMMessage],
        turn_index: int,
        options: RunOptions,
    ) -> AgentTurnResult:
        return self._invoke(
            self.nodes.initial_state(
                user_input=user_input,
                session_id=session_id,
                context=context,
                messages=messages,
                turn_index=turn_index,
                options=options,
            )
        )

    def resume_after_pending(
        self,
        *,
        pending: PendingResumeState,
        session_id: str,
        context: ExecutionContext,
        options: RunOptions,
        state: InvestigationState,
        iterations_used: int,
        no_progress_iterations: int,
        tool_step: ToolExecutionStepResult | None = None,
    ) -> AgentTurnResult:
        completed_tool_step = tool_step or ToolExecutionStepResult(
            messages=pending.messages,
            tool_messages=pending.tool_messages,
            exchange_index=pending.exchange_index,
            tool_calls_used=pending.tool_calls_used,
            tool_statuses=pending.tool_statuses or [pending.tool_status],
            tool_names=pending.tool_names or [str(pending.pending_payload.get("tool_name") or "unknown")],
        )
        return self._invoke(
            self.nodes.initial_state(
                user_input=pending.user_input,
                session_id=session_id,
                context=context,
                messages=completed_tool_step.messages,
                turn_index=pending.turn_index,
                options=options,
                investigation_state=state,
                iterations_used=iterations_used,
                tool_calls_used=completed_tool_step.tool_calls_used,
                exchange_index=completed_tool_step.exchange_index,
                no_progress_iterations=no_progress_iterations,
                resume_tool_step=completed_tool_step,
            )
        )

    def _invoke(self, initial_state: InvestigationGraphState) -> AgentTurnResult:
        import langsmith as ls

        options = initial_state["options"]
        recursion_limit = max(50, (options.max_iterations * 8) + 16)
        with ls.tracing_context(enabled=self.nodes.operations.settings.langchain_tracing_enabled):
            final_state = cast(
                InvestigationGraphState,
                self.graph.invoke(initial_state, {"recursion_limit": recursion_limit}),
            )
        result = final_state["result"]
        if result is None:
            raise RuntimeError("LangGraph investigation kernel completed without a result")
        return result
