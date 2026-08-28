from __future__ import annotations

from typing import Any, Literal, Protocol, cast

from agent_core.agent_graph.state import (
    AgentGraphState,
    AgentGraphUpdate,
    build_graph_checkpoint,
    normalize_agent_kernel_backend,
)
from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import LLMCompletionResult, LLMMessage
from agent_core.llm.errors import LLMProviderError
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.run_trace import RunTrace
from agent_core.settings import CoreSettings
from agent_core.tool_artifacts import active_tool_artifact_runtime
from agent_core.turn_steps import ToolExecutionStepResult
from agent_core.types import AgentTurnResult, ToolExecutionStatus

logger = get_logger("core.agent_graph.direct")

AgentKernelBackend = Literal["native", "langgraph"]
AfterModelRoute = Literal["execute_tools", "complete_response", "end"]
AfterToolsRoute = Literal["call_model", "complete_budget", "end"]


class DirectTurnOperations(Protocol):
    """Existing agent-core effects used by either control-flow backend."""

    settings: CoreSettings

    def _estimate_prompt_tokens(self, *, messages: list[LLMMessage]) -> int: ...

    def _record_trace_event(
        self,
        trace: RunTrace | None,
        *,
        event_type: str,
        summary: str,
        iteration: int | None = None,
        payload: dict[str, Any] | None = None,
        related_tool_call_id: str | None = None,
    ) -> None: ...

    def _call_model_once(self, *, messages: list[LLMMessage]) -> LLMCompletionResult: ...

    def _handle_provider_failure(
        self,
        *,
        error: LLMProviderError,
        user_input: str,
        turn_index: int,
    ) -> AgentTurnResult: ...

    def _persist_conversation_turn(
        self,
        *,
        turn_index: int,
        user_input: str,
        assistant_content: str,
        provider_failure: bool = False,
    ) -> None: ...

    def _refresh_memory_after_turn(self, *, turn_index: int) -> None: ...

    def _execute_tool_calls_once(
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
        start_tool_call_index: int = 0,
        existing_tool_messages: list[LLMMessage] | None = None,
        existing_tool_statuses: list[ToolExecutionStatus] | None = None,
        existing_tool_names: list[str] | None = None,
        reuse_exchange_index: bool = False,
        pending_metadata_extra: dict[str, Any] | None = None,
        trace: RunTrace | None = None,
    ) -> ToolExecutionStepResult: ...


class DirectTurnNodes:
    """Behavior shared by the native loop and LangGraph transitions."""

    def __init__(self, operations: DirectTurnOperations) -> None:
        self.operations = operations

    def initial_state(
        self,
        *,
        user_input: str,
        session_id: str,
        context: ExecutionContext,
        messages: list[LLMMessage],
        turn_index: int,
        tool_calls_used: int,
        exchange_index: int,
        trace: RunTrace | None,
        resume_tool_step: ToolExecutionStepResult | None = None,
    ) -> AgentGraphState:
        return AgentGraphState(
            user_input=user_input,
            session_id=session_id,
            context=context,
            messages=list(messages),
            turn_index=turn_index,
            tool_calls_used=tool_calls_used,
            exchange_index=exchange_index,
            trace=trace,
            start_prompt_tokens=self.operations._estimate_prompt_tokens(messages=messages),
            tool_loop_reserve_tokens=max(1, self.operations.settings.max_active_context_tokens),
            prompt_reserve_warning_emitted=False,
            model_call_index=0,
            assistant_message=None,
            tool_step=resume_tool_step,
            result=resume_tool_step.pending_result if resume_tool_step is not None else None,
            entrypoint="after_tools" if resume_tool_step is not None else "model",
        )

    @staticmethod
    def route_entry(state: AgentGraphState) -> AfterToolsRoute:
        if state["entrypoint"] == "model":
            return "call_model"
        return DirectTurnNodes.route_after_tools(state)

    def call_model(self, state: AgentGraphState) -> AgentGraphUpdate:
        messages = list(state["messages"])
        model_call_index = state["model_call_index"] + 1
        prompt_tokens = self.operations._estimate_prompt_tokens(messages=messages)
        prompt_growth_tokens = max(0, prompt_tokens - state["start_prompt_tokens"])
        warning_emitted = state["prompt_reserve_warning_emitted"]
        logger.debug(
            "Calling LLM",
            extra={
                "model": self.operations.settings.model,
                "message_count": len(messages),
                "estimated_prompt_tokens": prompt_tokens,
                "start_turn_prompt_tokens": state["start_prompt_tokens"],
                "tool_loop_reserve_tokens": state["tool_loop_reserve_tokens"],
                "prompt_growth_tokens": prompt_growth_tokens,
            },
        )
        if (
            state["tool_calls_used"] > 0
            and prompt_growth_tokens >= state["tool_loop_reserve_tokens"]
            and not warning_emitted
        ):
            logger.warning(
                "Tool loop consumed the start-turn prompt reserve",
                extra={
                    "estimated_prompt_tokens": prompt_tokens,
                    "start_turn_prompt_tokens": state["start_prompt_tokens"],
                    "prompt_growth_tokens": prompt_growth_tokens,
                    "tool_loop_reserve_tokens": state["tool_loop_reserve_tokens"],
                    "tool_calls_used": state["tool_calls_used"],
                },
            )
            warning_emitted = True

        self.operations._record_trace_event(
            state["trace"],
            event_type="llm_call_started",
            summary="LLM call started",
            iteration=model_call_index,
            payload={
                "message_count": len(messages),
                "estimated_prompt_tokens": prompt_tokens,
                "tool_calls_used": state["tool_calls_used"],
                "exchange_index": state["exchange_index"],
            },
        )
        try:
            llm_response = self.operations._call_model_once(messages=messages)
        except LLMProviderError as exc:
            self.operations._record_trace_event(
                state["trace"],
                event_type="llm_provider_failure",
                summary="LLM provider failure handled",
                iteration=model_call_index,
                payload={
                    "kind": exc.kind,
                    "detail_preview": safe_preview(exc.detail or exc.user_message, limit=200),
                },
            )
            return {
                "model_call_index": model_call_index,
                "prompt_reserve_warning_emitted": warning_emitted,
                "assistant_message": None,
                "tool_step": None,
                "result": self.operations._handle_provider_failure(
                    error=exc,
                    user_input=state["user_input"],
                    turn_index=state["turn_index"],
                ),
            }

        logger.debug(
            "Received LLM response",
            extra={
                "content_length": len(llm_response.content),
                "tool_call_count": len(llm_response.tool_calls),
                "provider": llm_response.provider,
                "model_backend": llm_response.model_backend,
                "model": llm_response.model,
                "provider_attempts": llm_response.provider_attempts,
            },
        )
        assistant_message = LLMMessage(
            role="assistant",
            content=llm_response.content,
            tool_calls=list(llm_response.tool_calls),
        )
        messages.append(assistant_message)
        self.operations._record_trace_event(
            state["trace"],
            event_type="assistant_response_received",
            summary="Assistant response received",
            iteration=model_call_index,
            payload={
                "content_length": len(llm_response.content),
                "tool_call_count": len(llm_response.tool_calls),
                "tool_calls": [
                    {"id": tool_call.id, "name": tool_call.name} for tool_call in llm_response.tool_calls
                ],
                "provider": llm_response.provider,
                "model_backend": llm_response.model_backend,
                "model": llm_response.model,
                "provider_request_id": llm_response.provider_request_id,
                "provider_attempts": llm_response.provider_attempts,
                "usage": llm_response.usage.to_dict() if llm_response.usage is not None else None,
            },
        )
        return {
            "messages": messages,
            "model_call_index": model_call_index,
            "prompt_reserve_warning_emitted": warning_emitted,
            "assistant_message": assistant_message,
            "tool_step": None,
            "result": None,
        }

    @staticmethod
    def route_after_model(state: AgentGraphState) -> AfterModelRoute:
        if state["result"] is not None:
            return "end"
        assistant_message = state["assistant_message"]
        if assistant_message is None:
            raise RuntimeError("Direct agent graph has no assistant message after a successful model call")
        return "execute_tools" if assistant_message.tool_calls else "complete_response"

    def complete_response(self, state: AgentGraphState) -> AgentGraphUpdate:
        assistant_message = state["assistant_message"]
        if assistant_message is None:
            raise RuntimeError("Direct agent graph cannot complete without an assistant message")
        self.operations._persist_conversation_turn(
            turn_index=state["turn_index"],
            user_input=state["user_input"],
            assistant_content=assistant_message.content,
        )
        self.operations._refresh_memory_after_turn(turn_index=state["turn_index"])
        logger.info("Completing run_turn without additional tool calls")
        return {"result": AgentTurnResult(status="completed", content=assistant_message.content)}

    def execute_tools(self, state: AgentGraphState) -> AgentGraphUpdate:
        assistant_message = state["assistant_message"]
        if assistant_message is None:
            raise RuntimeError("Direct agent graph cannot execute tools without an assistant message")
        tool_step = self.operations._execute_tool_calls_once(
            user_input=state["user_input"],
            session_id=state["session_id"],
            context=state["context"],
            messages=list(state["messages"]),
            turn_index=state["turn_index"],
            exchange_index=state["exchange_index"],
            tool_calls_used=state["tool_calls_used"],
            assistant_message=assistant_message,
            max_tool_calls=self.operations.settings.max_tool_calls_per_turn,
            pending_metadata_extra={
                "agent_graph_checkpoint": build_graph_checkpoint(
                    graph="direct",
                    backend=normalize_agent_kernel_backend(self.operations.settings.agent_kernel_backend),
                    resume_node="resume_tool_exchange",
                )
            },
            trace=state["trace"],
        )
        return {
            "messages": tool_step.messages,
            "exchange_index": tool_step.exchange_index,
            "tool_calls_used": tool_step.tool_calls_used,
            "tool_step": tool_step,
            "result": tool_step.pending_result,
        }

    @staticmethod
    def route_after_tools(state: AgentGraphState) -> AfterToolsRoute:
        tool_step = state["tool_step"]
        if tool_step is None:
            raise RuntimeError("Direct agent graph has no tool step after tool execution")
        if tool_step.pending_result is not None:
            return "end"
        if tool_step.budget_exhausted:
            return "complete_budget"
        return "call_model"

    def complete_budget(self, state: AgentGraphState) -> AgentGraphUpdate:
        message = "Maximum number of tool calls reached for this turn."
        logger.error(message)
        self.operations._record_trace_event(
            state["trace"],
            event_type="tool_budget_exhausted",
            summary=message,
            iteration=state["model_call_index"],
            payload={"tool_calls_used": state["tool_calls_used"]},
        )
        self.operations._persist_conversation_turn(
            turn_index=state["turn_index"],
            user_input=state["user_input"],
            assistant_content=message,
        )
        self.operations._refresh_memory_after_turn(turn_index=state["turn_index"])
        return {"result": AgentTurnResult(status="completed", content=message)}


class DirectTurnKernel(Protocol):
    backend: AgentKernelBackend

    def run(self, initial_state: AgentGraphState) -> AgentTurnResult: ...


class NativeDirectTurnKernel:
    backend: AgentKernelBackend = "native"

    def __init__(self, nodes: DirectTurnNodes) -> None:
        self.nodes = nodes

    def run(self, initial_state: AgentGraphState) -> AgentTurnResult:
        state = initial_state
        entry_route = self.nodes.route_entry(state)
        if entry_route == "end":
            result = state["result"]
            if result is None:
                raise RuntimeError("Native direct agent kernel resumed without a result")
            return result
        if entry_route == "complete_budget":
            state.update(self.nodes.complete_budget(state))
            result = state["result"]
            if result is None:
                raise RuntimeError("Native direct agent kernel budget completion produced no result")
            return result

        while True:
            state.update(self.nodes.call_model(state))
            model_route = self.nodes.route_after_model(state)
            if model_route == "end":
                break
            if model_route == "complete_response":
                state.update(self.nodes.complete_response(state))
                break

            state.update(self.nodes.execute_tools(state))
            tools_route = self.nodes.route_after_tools(state)
            if tools_route == "end":
                break
            if tools_route == "complete_budget":
                state.update(self.nodes.complete_budget(state))
                break

        result = state["result"]
        if result is None:
            raise RuntimeError("Native direct agent kernel completed without a result")
        return result


class LangGraphDirectTurnKernel:
    backend: AgentKernelBackend = "langgraph"

    def __init__(self, nodes: DirectTurnNodes) -> None:
        from langgraph.graph import END, START, StateGraph

        self.nodes = nodes
        builder = StateGraph(AgentGraphState)
        builder.add_node("call_model", nodes.call_model)
        builder.add_node("complete_response", nodes.complete_response)
        builder.add_node("execute_tools", nodes.execute_tools)
        builder.add_node("complete_budget", nodes.complete_budget)
        builder.add_conditional_edges(
            START,
            nodes.route_entry,
            {
                "call_model": "call_model",
                "complete_budget": "complete_budget",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "call_model",
            nodes.route_after_model,
            {
                "execute_tools": "execute_tools",
                "complete_response": "complete_response",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "execute_tools",
            nodes.route_after_tools,
            {
                "call_model": "call_model",
                "complete_budget": "complete_budget",
                "end": END,
            },
        )
        builder.add_edge("complete_response", END)
        builder.add_edge("complete_budget", END)
        self.graph = builder.compile()

    def run(self, initial_state: AgentGraphState) -> AgentTurnResult:
        import langsmith as ls

        # A model/tool exchange consumes two graph supersteps. Keep the limit
        # proportional to the configured tool budget instead of LangGraph's
        # low generic default.
        artifact_runtime = active_tool_artifact_runtime()
        max_internal_tool_calls = artifact_runtime.policy.max_reads_per_run if artifact_runtime is not None else 0
        recursion_limit = max(
            25,
            ((self.nodes.operations.settings.max_tool_calls_per_turn + max_internal_tool_calls) * 2) + 8,
        )
        # Graph execution can contain the complete transcript in its state.
        # Reuse the explicit LangSmith opt-in from the LangChain model backend
        # instead of inheriting process-wide LANGSMITH_TRACING implicitly.
        with ls.tracing_context(enabled=self.nodes.operations.settings.langchain_tracing_enabled):
            final_state = cast(
                AgentGraphState,
                self.graph.invoke(initial_state, {"recursion_limit": recursion_limit}),
            )
        result = final_state["result"]
        if result is None:
            raise RuntimeError("LangGraph direct agent kernel completed without a result")
        return result


def build_direct_turn_kernel(
    *,
    backend: str,
    operations: DirectTurnOperations,
) -> tuple[DirectTurnNodes, DirectTurnKernel]:
    normalized = normalize_agent_kernel_backend(backend)
    nodes = DirectTurnNodes(operations)
    if normalized == "native":
        return nodes, NativeDirectTurnKernel(nodes)
    if normalized == "langgraph":
        return nodes, LangGraphDirectTurnKernel(nodes)
    raise ValueError(
        f"Unsupported agent kernel backend: {backend!r}. Expected 'native' or 'langgraph'."
    )
