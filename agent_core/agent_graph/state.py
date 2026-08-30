from __future__ import annotations

from typing import Literal, TypedDict

from agent_core.execution_context import ExecutionContext
from agent_core.investigation_state import InvestigationState
from agent_core.llm.base import LLMMessage
from agent_core.run_options import RunOptions
from agent_core.run_trace import RunTrace
from agent_core.turn_steps import ToolExecutionStepResult
from agent_core.types import AgentTurnResult


class AgentGraphState(TypedDict):
    """Typed, ephemeral state for the direct conversation graph.

    Durable conversation state remains owned by ``SessionManager`` and
    ``RunStore``. This state exists only for one in-process graph invocation.
    """

    user_input: str
    session_id: str
    context: ExecutionContext
    messages: list[LLMMessage]
    turn_index: int
    tool_calls_used: int
    exchange_index: int
    trace: RunTrace | None
    start_prompt_tokens: int
    tool_loop_reserve_tokens: int
    prompt_reserve_warning_emitted: bool
    model_call_index: int
    assistant_message: LLMMessage | None
    tool_step: ToolExecutionStepResult | None
    result: AgentTurnResult | None
    entrypoint: Literal["model", "after_tools"]


class AgentGraphUpdate(TypedDict, total=False):
    """Partial update emitted by one direct-graph node."""

    messages: list[LLMMessage]
    tool_calls_used: int
    exchange_index: int
    prompt_reserve_warning_emitted: bool
    model_call_index: int
    assistant_message: LLMMessage | None
    tool_step: ToolExecutionStepResult | None
    result: AgentTurnResult | None


class InvestigationGraphState(TypedDict):
    """Ephemeral state for investigate and deep-investigate graphs."""

    user_input: str
    session_id: str
    context: ExecutionContext
    messages: list[LLMMessage]
    turn_index: int
    options: RunOptions
    investigation_state: InvestigationState
    iterations_used: int
    tool_calls_used: int
    exchange_index: int
    no_progress_iterations: int
    assistant_message: LLMMessage | None
    tool_step: ToolExecutionStepResult | None
    final_draft: str | None
    result: AgentTurnResult | None
    entrypoint: Literal["initialize", "after_tools"]


class InvestigationGraphUpdate(TypedDict, total=False):
    messages: list[LLMMessage]
    investigation_state: InvestigationState
    iterations_used: int
    tool_calls_used: int
    exchange_index: int
    no_progress_iterations: int
    assistant_message: LLMMessage | None
    tool_step: ToolExecutionStepResult | None
    final_draft: str | None
    result: AgentTurnResult | None


def normalize_agent_kernel_backend(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def build_graph_checkpoint(
    *,
    graph: Literal["direct", "investigation"],
    backend: str,
    resume_node: Literal["resume_tool_exchange"],
) -> dict[str, str]:
    """Return the durable graph cursor embedded in agent-core pending state."""

    return {
        "schema_version": "1",
        "graph": graph,
        "backend": backend,
        "resume_node": resume_node,
    }
