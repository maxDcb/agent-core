from __future__ import annotations

from agent_core.investigation_models import StepReflection
from agent_core.llm.base import LLMMessage
from agent_core.memory.journal import ExchangeMemory
from agent_core.tool_artifacts import artifact_descriptor_from_message
from agent_core.types import ToolExecutionStatus


def derive_tool_exchange_memory(
    *,
    thread_id: str,
    turn_index: int,
    exchange_index: int,
    assistant_message: LLMMessage,
    tool_messages: list[LLMMessage],
    tool_names: list[str] | None = None,
    tool_statuses: list[ToolExecutionStatus] | None = None,
    source_block_id: str,
    origin: str = "runtime",
) -> ExchangeMemory:
    names = list(tool_names or [tool_call.name for tool_call in assistant_message.tool_calls])
    statuses = list(tool_statuses or [])
    completed_actions: list[str] = []
    risk_notes: list[str] = []
    for index, name in enumerate(names):
        status = statuses[index] if index < len(statuses) else "unknown"
        completed_actions.append(f"{name}: {status}")
        if status not in {"ok", "unknown"}:
            risk_notes.append(f"Tool {name} completed with status {status}.")

    artifacts = [
        descriptor.artifact_id
        for message in tool_messages
        if (descriptor := artifact_descriptor_from_message(message)) is not None
    ]
    tool_summary = ", ".join(completed_actions) or "tool exchange completed"
    observations = [
        f"{names[index] if index < len(names) else 'tool'} returned: {message.content}"
        for index, message in enumerate(tool_messages)
        if message.content.strip()
    ]
    assistant_summary = assistant_message.content.strip()
    summary_parts = [assistant_summary, f"Tools: {tool_summary}.", *observations]
    summary = " ".join(part for part in summary_parts if part)
    memory = ExchangeMemory.from_any(
        {
            "memory_id": f"turn-{turn_index:04d}-exchange-{exchange_index:02d}-runtime",
            "thread_id": thread_id,
            "turn_index": turn_index,
            "exchange_index": exchange_index,
            "kind": "tool_exchange",
            "summary": summary,
            "origin": origin,
            "completed_actions": completed_actions,
            "relevant_artifacts": artifacts,
            "risk_notes": risk_notes,
            "confidence": 1.0,
            "source_block_ids": [source_block_id],
        }
    )
    if memory is None:
        raise ValueError("Could not derive tool exchange memory")
    return memory


def derive_reflection_memory(
    *,
    thread_id: str,
    turn_index: int,
    exchange_index: int,
    reflection: StepReflection,
    relevant_artifacts: list[str],
    source_block_id: str,
) -> ExchangeMemory:
    memory = ExchangeMemory.from_any(
        {
            "memory_id": f"turn-{turn_index:04d}-exchange-{exchange_index:02d}-reflection",
            "thread_id": thread_id,
            "turn_index": turn_index,
            "exchange_index": exchange_index,
            "kind": "reflection",
            "summary": reflection.observation_summary,
            "origin": "reflection",
            "confirmed_facts": reflection.new_facts,
            "open_hypotheses": reflection.updated_hypotheses,
            "rejected_hypotheses": reflection.rejected_hypotheses,
            "open_questions": reflection.remaining_gaps,
            "resolved_questions": reflection.resolved_gaps,
            "completed_actions": [reflection.observation_summary] if reflection.observation_summary else [],
            "next_actions": reflection.recommended_next_actions,
            "relevant_artifacts": relevant_artifacts,
            "risk_notes": reflection.risk_notes,
            "confidence": reflection.confidence,
            "source_block_ids": [source_block_id],
        }
    )
    if memory is None:
        raise ValueError("Could not derive reflection memory")
    return memory


def derive_final_response_memory(
    *,
    thread_id: str,
    turn_index: int,
    exchange_index: int,
    assistant_content: str,
    source_block_id: str,
    provider_failure: bool = False,
    origin: str = "runtime",
) -> ExchangeMemory:
    memory = ExchangeMemory.from_any(
        {
            "memory_id": f"turn-{turn_index:04d}-final-response",
            "thread_id": thread_id,
            "turn_index": turn_index,
            "exchange_index": exchange_index,
            "kind": "provider_failure" if provider_failure else "final_response",
            "summary": assistant_content,
            "origin": origin,
            "risk_notes": ["The provider failed while completing this turn."] if provider_failure else [],
            "confidence": 1.0,
            "source_block_ids": [source_block_id],
        }
    )
    if memory is None:
        raise ValueError("Could not derive final response memory")
    return memory
