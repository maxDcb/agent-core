from __future__ import annotations

from typing import TypedDict

from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import LLMMessage
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
