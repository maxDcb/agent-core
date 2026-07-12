from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from agent_core.llm.base import LLMCallRecord, LLMUsageSummary
from agent_core.run_context import ExecutionScope, RunContext
from agent_core.types import utc_now_iso

RunStatus: TypeAlias = Literal[
    "created",
    "running",
    "pending",
    "interrupted",
    "completed",
    "failed",
    "cancelled",
    "blocked",
]
RunStrategy: TypeAlias = Literal["structured", "direct", "investigate", "deep_investigate"]
RunAttemptStatus: TypeAlias = Literal[
    "running",
    "pending",
    "interrupted",
    "completed",
    "failed",
    "cancelled",
    "blocked",
]
RUN_SCHEMA_VERSION = 3


@dataclass(slots=True)
class RunCheckpoint:
    kind: str
    sequence: int
    payload: dict[str, Any]
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sequence": self.sequence,
            "payload": dict(self.payload),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: object) -> RunCheckpoint | None:
        if not isinstance(payload, dict):
            return None
        kind = payload.get("kind")
        sequence = payload.get("sequence")
        checkpoint_payload = payload.get("payload")
        if not isinstance(kind, str) or not isinstance(sequence, int) or not isinstance(checkpoint_payload, dict):
            return None
        updated_at = payload.get("updated_at")
        return cls(
            kind=kind,
            sequence=max(0, sequence),
            payload=dict(checkpoint_payload),
            updated_at=updated_at if isinstance(updated_at, str) else utc_now_iso(),
        )


@dataclass(slots=True)
class AgentRunAttempt:
    attempt_id: str
    status: RunAttemptStatus = "running"
    started_at: str = field(default_factory=utc_now_iso)
    completed_at: str | None = None
    resumed_from_sequence: int | None = None
    failure_reason: str = ""

    def finish(self, status: RunAttemptStatus, *, failure_reason: str = "") -> None:
        self.status = status
        self.completed_at = utc_now_iso()
        self.failure_reason = failure_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "resumed_from_sequence": self.resumed_from_sequence,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, payload: object) -> AgentRunAttempt | None:
        if not isinstance(payload, dict):
            return None
        attempt_id = payload.get("attempt_id")
        status = payload.get("status")
        if not isinstance(attempt_id, str) or not attempt_id:
            return None
        normalized_status: RunAttemptStatus = (
            status
            if status in {"running", "pending", "interrupted", "completed", "failed", "cancelled", "blocked"}
            else "interrupted"
        )
        started_at = payload.get("started_at")
        completed_at = payload.get("completed_at")
        resumed_from_sequence = payload.get("resumed_from_sequence")
        failure_reason = payload.get("failure_reason")
        return cls(
            attempt_id=attempt_id,
            status=normalized_status,
            started_at=started_at if isinstance(started_at, str) else utc_now_iso(),
            completed_at=completed_at if isinstance(completed_at, str) else None,
            resumed_from_sequence=resumed_from_sequence if isinstance(resumed_from_sequence, int) else None,
            failure_reason=failure_reason if isinstance(failure_reason, str) else "",
        )


@dataclass(slots=True)
class AgentRunError:
    kind: str
    message: str
    retryable: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "retryable": self.retryable,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: object) -> AgentRunError | None:
        if not isinstance(payload, dict):
            return None
        kind = payload.get("kind")
        message = payload.get("message")
        if not isinstance(kind, str) or not isinstance(message, str):
            return None
        detail = payload.get("detail")
        return cls(
            kind=kind,
            message=message,
            retryable=bool(payload.get("retryable", False)),
            detail=detail if isinstance(detail, str) else "",
        )


@dataclass(slots=True)
class AgentRunResult:
    run_id: str
    status: RunStatus
    output: dict[str, Any] | None = None
    raw_content: str = ""
    error: AgentRunError | None = None
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tool_calls_used: int = 0
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.error is None

    @property
    def failure_reason(self) -> str:
        return self.error.message if self.error is not None else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "output": self.output,
            "raw_content": self.raw_content,
            "error": self.error.to_dict() if self.error is not None else None,
            "tool_history": list(self.tool_history),
            "iterations": self.iterations,
            "tool_calls_used": self.tool_calls_used,
            "llm_calls": [call.to_dict() for call in self.llm_calls],
            "usage": LLMUsageSummary.from_calls(self.llm_calls).to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: object) -> AgentRunResult | None:
        if not isinstance(payload, dict):
            return None
        run_id = payload.get("run_id")
        status = payload.get("status")
        if not isinstance(run_id, str) or status not in {
            "created", "running", "pending", "interrupted", "completed", "failed", "cancelled", "blocked"
        }:
            return None
        output = payload.get("output")
        history = payload.get("tool_history")
        metadata = payload.get("metadata")
        raw_content = payload.get("raw_content")
        iterations = payload.get("iterations")
        tool_calls_used = payload.get("tool_calls_used")
        raw_llm_calls = payload.get("llm_calls")
        return cls(
            run_id=run_id,
            status=status,
            output=dict(output) if isinstance(output, dict) else None,
            raw_content=raw_content if isinstance(raw_content, str) else "",
            error=AgentRunError.from_dict(payload.get("error")),
            tool_history=[dict(item) for item in history if isinstance(item, dict)] if isinstance(history, list) else [],
            iterations=iterations if isinstance(iterations, int) else 0,
            tool_calls_used=tool_calls_used if isinstance(tool_calls_used, int) else 0,
            llm_calls=(
                [call for item in raw_llm_calls if (call := LLMCallRecord.from_dict(item)) is not None]
                if isinstance(raw_llm_calls, list)
                else []
            ),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )


@dataclass(slots=True)
class AgentRunState:
    run_id: str
    strategy: RunStrategy
    spec_id: str
    context: RunContext
    status: RunStatus = "created"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    result: AgentRunResult | None = None
    error: AgentRunError | None = None
    checkpoint: RunCheckpoint | None = None
    attempts: list[AgentRunAttempt] = field(default_factory=list)
    schema_version: int = RUN_SCHEMA_VERSION

    def transition(self, status: RunStatus) -> None:
        allowed: dict[RunStatus, set[RunStatus]] = {
            "created": {"running", "cancelled"},
            "running": {"pending", "interrupted", "completed", "failed", "cancelled", "blocked"},
            "pending": {"running", "interrupted", "completed", "failed", "cancelled", "blocked"},
            "interrupted": {"running", "cancelled", "blocked"},
            "completed": set(),
            "failed": set(),
            "cancelled": set(),
            "blocked": {"running", "cancelled"},
        }
        if status != self.status and status not in allowed[self.status]:
            raise ValueError(f"Invalid run transition: {self.status} -> {status}")
        self.status = status
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "strategy": self.strategy,
            "spec_id": self.spec_id,
            "context": self.context.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result.to_dict() if self.result is not None else None,
            "error": self.error.to_dict() if self.error is not None else None,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint is not None else None,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @classmethod
    def from_dict(cls, payload: object) -> AgentRunState | None:
        if not isinstance(payload, dict):
            return None
        run_id = payload.get("run_id")
        strategy = payload.get("strategy")
        spec_id = payload.get("spec_id")
        context_payload = payload.get("context")
        status = payload.get("status")
        if not isinstance(run_id, str) or strategy not in {"structured", "direct", "investigate", "deep_investigate"}:
            return None
        if not isinstance(spec_id, str) or not isinstance(context_payload, dict):
            return None
        namespace_id = context_payload.get("namespace_id")
        if not isinstance(namespace_id, str):
            return None
        correlation = context_payload.get("correlation")
        application_context = context_payload.get("application_context")
        context = RunContext(
            namespace_id=namespace_id,
            run_id=run_id,
            parent_id=context_payload.get("parent_id") if isinstance(context_payload.get("parent_id"), str) else None,
            thread_id=context_payload.get("thread_id") if isinstance(context_payload.get("thread_id"), str) else None,
            scope=ExecutionScope.from_dict(context_payload.get("scope")),
            correlation=dict(correlation) if isinstance(correlation, dict) else {},
            application_context=(
                dict(application_context) if isinstance(application_context, dict) else {}
            ),
        )
        normalized_status: RunStatus = (
            status
            if status in {
                "created",
                "running",
                "pending",
                "interrupted",
                "completed",
                "failed",
                "cancelled",
                "blocked",
            }
            else "failed"
        )
        checkpoint = RunCheckpoint.from_dict(payload.get("checkpoint"))
        raw_attempts = payload.get("attempts")
        attempts = (
            [attempt for item in raw_attempts if (attempt := AgentRunAttempt.from_dict(item)) is not None]
            if isinstance(raw_attempts, list)
            else []
        )
        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")
        schema_version = payload.get("schema_version")
        return cls(
            run_id=run_id,
            strategy=strategy,
            spec_id=spec_id,
            context=context,
            status=normalized_status,
            created_at=created_at if isinstance(created_at, str) else utc_now_iso(),
            updated_at=updated_at if isinstance(updated_at, str) else utc_now_iso(),
            result=AgentRunResult.from_dict(payload.get("result")),
            error=AgentRunError.from_dict(payload.get("error")),
            checkpoint=checkpoint,
            attempts=attempts,
            schema_version=schema_version if isinstance(schema_version, int) else 0,
        )
