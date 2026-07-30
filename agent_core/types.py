from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

ToolExecutionStatus: TypeAlias = Literal[
    "ok",
    "pending",
    "tool_error",
    "policy_denied",
    "invalid_arguments",
    "execution_failed",
    "budget_exhausted",
]

AgentTurnStatus: TypeAlias = Literal["completed", "pending_tool_result"]


class SessionState(TypedDict):
    context_blocks: list[dict[str, Any]]
    active_block_ids: list[str]
    overflow_block_ids: list[str]
    memory: dict[str, Any]
    tool_history: list[dict[str, Any]]
    domain_state: dict[str, Any]
    meta: dict[str, Any]
    execution_scope: NotRequired[dict[str, Any]]


SESSION_SCHEMA_VERSION = 5


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_empty_session_state(*, session_id: str = "default", storage_backend: str = "json") -> SessionState:
    return {
        "context_blocks": [],
        "active_block_ids": [],
        "overflow_block_ids": [],
        "memory": {
            "schema_version": "2",
            "thread_id": session_id,
            "exchanges": [],
            "turns": [],
            "view_policy": {
                "max_handoff_chars": 6000,
            },
            "session_view": {
                "thread_id": session_id,
                "generation": 0,
                "through_turn_index": -1,
                "through_turn_memory_id": None,
                "updated_at": utc_now_iso(),
                "content": "",
                "origin": "runtime",
                "degraded": False,
                "schema_version": "2",
            },
        },
        "tool_history": [],
        "domain_state": {},
        "meta": {
            "session_id": session_id,
            "schema_version": SESSION_SCHEMA_VERSION,
            "storage_backend": storage_backend,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        },
    }


@dataclass(slots=True)
class ToolResult:
    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pending: bool = False
    pending_id: str | None = None

    @classmethod
    def pending_result(
        cls,
        content: str,
        *,
        pending_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        return cls(
            ok=True,
            content=content,
            metadata=metadata or {},
            pending=True,
            pending_id=pending_id,
        )


@dataclass(slots=True)
class AgentTurnResult:
    status: AgentTurnStatus
    content: str
    pending_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        return self.status == "pending_tool_result"


@dataclass(slots=True)
class AuthorizationResult:
    allowed: bool
    reason: str = ""
