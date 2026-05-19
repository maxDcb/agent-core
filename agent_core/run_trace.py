from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

from agent_core.llm.base import LLMMessage
from agent_core.memory.context_block import estimate_token_count
from agent_core.types import utc_now_iso

TRACE_SCHEMA_VERSION = 1

TraceStatus = Literal["running", "completed", "pending_tool_result", "failed"]
ContextBudgetStatus = Literal["unknown", "ok", "warning", "critical", "exceeded"]


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable copy without dropping audit context."""

    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


@dataclass(slots=True)
class ContextBudget:
    estimated_prompt_tokens: int
    context_window_tokens: int | None
    usage_percent: float | None
    status: ContextBudgetStatus

    @classmethod
    def from_estimate(cls, *, estimated_prompt_tokens: int, context_window_tokens: int | None) -> "ContextBudget":
        if context_window_tokens is None or context_window_tokens <= 0:
            return cls(
                estimated_prompt_tokens=max(0, estimated_prompt_tokens),
                context_window_tokens=None,
                usage_percent=None,
                status="unknown",
            )

        usage_percent = round((max(0, estimated_prompt_tokens) / context_window_tokens) * 100, 2)
        if usage_percent >= 100:
            status: ContextBudgetStatus = "exceeded"
        elif usage_percent >= 95:
            status = "critical"
        elif usage_percent >= 80:
            status = "warning"
        else:
            status = "ok"
        return cls(
            estimated_prompt_tokens=max(0, estimated_prompt_tokens),
            context_window_tokens=context_window_tokens,
            usage_percent=usage_percent,
            status=status,
        )

    @classmethod
    def from_any(cls, payload: Any) -> "ContextBudget | None":
        data = _coerce_dict(payload)
        if not data:
            return None
        estimated = data.get("estimated_prompt_tokens", 0)
        window = data.get("context_window_tokens")
        usage_percent = data.get("usage_percent")
        status = data.get("status")
        return cls(
            estimated_prompt_tokens=estimated if isinstance(estimated, int) else 0,
            context_window_tokens=window if isinstance(window, int) else None,
            usage_percent=float(usage_percent) if isinstance(usage_percent, (int, float)) else None,
            status=status if status in {"unknown", "ok", "warning", "critical", "exceeded"} else "unknown",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "context_window_tokens": self.context_window_tokens,
            "usage_percent": self.usage_percent,
            "status": self.status,
        }


@dataclass(slots=True)
class PromptBlock:
    block_id: str
    type: str
    title: str
    source: str
    content: Any
    estimated_tokens: int = 0
    redacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_message(cls, *, message: LLMMessage, index: int, is_last: bool) -> "PromptBlock":
        history_payload = message.to_history_dict()
        block_type = _classify_message(message=message, index=index, is_last=is_last)
        return cls(
            block_id=f"prompt-message-{index:04d}",
            type=block_type,
            title=_title_for_message(message=message, block_type=block_type, index=index),
            source="llm_messages",
            content=history_payload,
            estimated_tokens=estimate_token_count([history_payload]),
            metadata={
                "message_index": index,
                "role": message.role,
                "tool_call_count": len(message.tool_calls),
                "has_tool_call_id": bool(message.tool_call_id),
            },
        )

    @classmethod
    def from_any(cls, payload: Any) -> "PromptBlock | None":
        data = _coerce_dict(payload)
        if not data:
            return None
        block_id = data.get("block_id")
        block_type = data.get("type")
        title = data.get("title")
        source = data.get("source")
        estimated_tokens = data.get("estimated_tokens")
        if not isinstance(block_id, str) or not isinstance(block_type, str):
            return None
        return cls(
            block_id=block_id,
            type=block_type,
            title=title if isinstance(title, str) else block_type,
            source=source if isinstance(source, str) else "unknown",
            content=json_safe(data.get("content")),
            estimated_tokens=estimated_tokens if isinstance(estimated_tokens, int) else 0,
            redacted=bool(data.get("redacted", False)),
            metadata=_coerce_dict(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "type": self.type,
            "title": self.title,
            "source": self.source,
            "content": json_safe(self.content),
            "estimated_tokens": self.estimated_tokens,
            "redacted": self.redacted,
            "metadata": json_safe(self.metadata),
        }


@dataclass(slots=True)
class PromptSnapshot:
    created_at: str
    message_count: int
    blocks: list[PromptBlock] = field(default_factory=list)
    estimated_tokens: int = 0
    context_budget: ContextBudget = field(
        default_factory=lambda: ContextBudget.from_estimate(
            estimated_prompt_tokens=0,
            context_window_tokens=None,
        )
    )

    @classmethod
    def from_messages(
        cls,
        *,
        messages: list[LLMMessage],
        context_window_tokens: int | None,
        extra_blocks: list[PromptBlock] | None = None,
    ) -> "PromptSnapshot":
        blocks = [
            PromptBlock.from_message(
                message=message,
                index=index,
                is_last=index == len(messages) - 1,
            )
            for index, message in enumerate(messages)
        ]
        if extra_blocks:
            blocks.extend(extra_blocks)

        history_payload = [message.to_history_dict() for message in messages]
        estimated_tokens = estimate_token_count(history_payload)
        return cls(
            created_at=utc_now_iso(),
            message_count=len(messages),
            blocks=blocks,
            estimated_tokens=estimated_tokens,
            context_budget=ContextBudget.from_estimate(
                estimated_prompt_tokens=estimated_tokens,
                context_window_tokens=context_window_tokens,
            ),
        )

    @classmethod
    def from_any(cls, payload: Any) -> "PromptSnapshot | None":
        data = _coerce_dict(payload)
        if not data:
            return None
        budget = ContextBudget.from_any(data.get("context_budget")) or ContextBudget.from_estimate(
            estimated_prompt_tokens=0,
            context_window_tokens=None,
        )
        message_count = data.get("message_count")
        estimated_tokens = data.get("estimated_tokens")
        return cls(
            created_at=str(data.get("created_at") or ""),
            message_count=message_count if isinstance(message_count, int) else 0,
            blocks=[
                block
                for item in _coerce_list(data.get("blocks"))
                if (block := PromptBlock.from_any(item)) is not None
            ],
            estimated_tokens=estimated_tokens if isinstance(estimated_tokens, int) else 0,
            context_budget=budget,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "message_count": self.message_count,
            "estimated_tokens": self.estimated_tokens,
            "context_budget": self.context_budget.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(slots=True)
class TraceEvent:
    event_id: str
    timestamp: str
    type: str
    summary: str
    iteration: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    related_tool_call_id: str | None = None
    related_prompt_block_ids: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        event_type: str,
        summary: str,
        iteration: int | None = None,
        payload: dict[str, Any] | None = None,
        related_tool_call_id: str | None = None,
        related_prompt_block_ids: list[str] | None = None,
    ) -> "TraceEvent":
        return cls(
            event_id=event_id,
            timestamp=utc_now_iso(),
            type=event_type,
            summary=summary,
            iteration=iteration,
            payload=json_safe(payload or {}),
            related_tool_call_id=related_tool_call_id,
            related_prompt_block_ids=list(related_prompt_block_ids or []),
        )

    @classmethod
    def from_any(cls, payload: Any) -> "TraceEvent | None":
        data = _coerce_dict(payload)
        if not data:
            return None
        event_id = data.get("event_id")
        event_type = data.get("type")
        summary = data.get("summary")
        timestamp = data.get("timestamp")
        if not isinstance(event_id, str) or not isinstance(event_type, str):
            return None
        return cls(
            event_id=event_id,
            timestamp=timestamp if isinstance(timestamp, str) else "",
            type=event_type,
            summary=summary if isinstance(summary, str) else event_type,
            iteration=data.get("iteration") if isinstance(data.get("iteration"), int) else None,
            payload=_coerce_dict(data.get("payload")),
            related_tool_call_id=(
                data.get("related_tool_call_id") if isinstance(data.get("related_tool_call_id"), str) else None
            ),
            related_prompt_block_ids=[
                str(item)
                for item in _coerce_list(data.get("related_prompt_block_ids"))
                if isinstance(item, str)
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "summary": self.summary,
            "iteration": self.iteration,
            "payload": json_safe(self.payload),
            "related_tool_call_id": self.related_tool_call_id,
            "related_prompt_block_ids": list(self.related_prompt_block_ids),
        }


@dataclass(slots=True)
class RunTrace:
    run_id: str
    session_id: str
    mode: str
    turn_index: int
    started_at: str
    schema_version: int = TRACE_SCHEMA_VERSION
    status: TraceStatus = "running"
    completed_at: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    prompt_snapshot: PromptSnapshot | None = None
    events: list[TraceEvent] = field(default_factory=list)
    final_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        session_id: str,
        mode: str,
        turn_index: int,
        options: dict[str, Any] | None = None,
        prompt_snapshot: PromptSnapshot | None = None,
    ) -> "RunTrace":
        trace = cls(
            run_id=run_id,
            session_id=session_id,
            mode=mode,
            turn_index=turn_index,
            started_at=utc_now_iso(),
            options=json_safe(options or {}),
            prompt_snapshot=prompt_snapshot,
        )
        trace.add_event(
            event_type="run_started",
            summary="Run started",
            payload={"mode": mode, "turn_index": turn_index},
        )
        if prompt_snapshot is not None:
            trace.add_event(
                event_type="prompt_snapshot_created",
                summary="Initial prompt snapshot captured",
                payload={
                    "message_count": prompt_snapshot.message_count,
                    "estimated_tokens": prompt_snapshot.estimated_tokens,
                    "context_budget": prompt_snapshot.context_budget.to_dict(),
                },
            )
        return trace

    @classmethod
    def from_any(cls, payload: Any) -> "RunTrace | None":
        data = _coerce_dict(payload)
        if not data:
            return None
        run_id = data.get("run_id")
        session_id = data.get("session_id")
        mode = data.get("mode")
        turn_index = data.get("turn_index")
        started_at = data.get("started_at")
        if not isinstance(run_id, str) or not isinstance(session_id, str):
            return None
        schema_version = data.get("schema_version")
        status = data.get("status")
        normalized_status: TraceStatus = (
            status if status in {"running", "completed", "pending_tool_result", "failed"} else "failed"
        )
        return cls(
            run_id=run_id,
            session_id=session_id,
            mode=mode if isinstance(mode, str) else "direct",
            turn_index=turn_index if isinstance(turn_index, int) else 0,
            started_at=started_at if isinstance(started_at, str) else "",
            schema_version=schema_version if isinstance(schema_version, int) else 0,
            status=normalized_status,
            completed_at=data.get("completed_at") if isinstance(data.get("completed_at"), str) else None,
            options=_coerce_dict(data.get("options")),
            prompt_snapshot=PromptSnapshot.from_any(data.get("prompt_snapshot")),
            events=[
                event
                for item in _coerce_list(data.get("events"))
                if (event := TraceEvent.from_any(item)) is not None
            ],
            final_metadata=_coerce_dict(data.get("final_metadata")),
        )

    def add_event(
        self,
        *,
        event_type: str,
        summary: str,
        iteration: int | None = None,
        payload: dict[str, Any] | None = None,
        related_tool_call_id: str | None = None,
        related_prompt_block_ids: list[str] | None = None,
    ) -> TraceEvent:
        event = TraceEvent.create(
            event_id=f"event-{len(self.events) + 1:04d}",
            event_type=event_type,
            summary=summary,
            iteration=iteration,
            payload=payload,
            related_tool_call_id=related_tool_call_id,
            related_prompt_block_ids=related_prompt_block_ids,
        )
        self.events.append(event)
        return event

    def complete(self, *, status: TraceStatus, final_metadata: dict[str, Any] | None = None) -> None:
        self.status = status
        self.completed_at = utc_now_iso()
        self.final_metadata = json_safe(final_metadata or {})

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "mode": self.mode,
            "turn_index": self.turn_index,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "event_count": len(self.events),
            "estimated_prompt_tokens": self.prompt_snapshot.estimated_tokens if self.prompt_snapshot else None,
            "context_budget": self.prompt_snapshot.context_budget.to_dict() if self.prompt_snapshot else None,
            "final_metadata": json_safe(self.final_metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "mode": self.mode,
            "turn_index": self.turn_index,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "options": json_safe(self.options),
            "prompt_snapshot": self.prompt_snapshot.to_dict() if self.prompt_snapshot is not None else None,
            "events": [event.to_dict() for event in self.events],
            "final_metadata": json_safe(self.final_metadata),
        }


def _classify_message(*, message: LLMMessage, index: int, is_last: bool) -> str:
    if message.role == "system":
        if index == 0:
            return "system"
        if message.content.startswith("Execution scope:"):
            return "execution_scope"
        if message.content.startswith("Run mode:"):
            return "run_mode_guidance"
        return "system_context"
    if message.role == "user" and is_last:
        return "user_query"
    if message.role in {"assistant", "tool"}:
        return "session_history"
    return message.role


def _title_for_message(*, message: LLMMessage, block_type: str, index: int) -> str:
    if block_type == "system":
        return "System prompt"
    if block_type == "execution_scope":
        return "Execution scope"
    if block_type == "run_mode_guidance":
        return "Run mode guidance"
    if block_type == "user_query":
        return "User query"
    if block_type == "session_history":
        return f"Session history message {index}"
    return f"{message.role.title()} message {index}"
