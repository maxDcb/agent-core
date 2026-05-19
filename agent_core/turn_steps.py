from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.llm.base import LLMMessage
from agent_core.types import AgentTurnResult, ToolExecutionStatus


@dataclass(slots=True)
class ToolExecutionStepResult:
    messages: list[LLMMessage]
    tool_messages: list[LLMMessage]
    exchange_index: int
    tool_calls_used: int
    pending_result: AgentTurnResult | None = None
    budget_exhausted: bool = False
    tool_statuses: list[ToolExecutionStatus] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)

    @property
    def executed_count(self) -> int:
        return len(self.tool_messages)


@dataclass(slots=True)
class PendingResumeState:
    user_input: str
    messages: list[LLMMessage]
    turn_index: int
    exchange_index: int
    tool_calls_used: int
    tool_messages: list[LLMMessage]
    tool_status: ToolExecutionStatus = "ok"
    pending_payload: dict[str, Any] = field(default_factory=dict)
