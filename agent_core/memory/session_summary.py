from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.memory.context_block import ContextBlock, estimate_token_count
from agent_core.types import utc_now_iso


def _normalize_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_optional_str(value: object) -> str:
    return value if isinstance(value, str) else ""


@dataclass(slots=True)
class SessionSummary:
    summary_id: str
    thread_id: str
    covers_blocks_until: str
    generated_at: str
    source_block_count: int
    facts_confirmed: list[str] = field(default_factory=list)
    hypotheses_open: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    completed_actions: list[str] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)
    relevant_artifacts: list[str] = field(default_factory=list)
    token_estimate: int = 0
    schema_version: str = "1"
    domain_extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_estimate <= 0:
            self.token_estimate = estimate_token_count(self.render_text())

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "thread_id": self.thread_id,
            "covers_blocks_until": self.covers_blocks_until,
            "generated_at": self.generated_at,
            "source_block_count": self.source_block_count,
            "facts_confirmed": self.facts_confirmed,
            "hypotheses_open": self.hypotheses_open,
            "decisions": self.decisions,
            "completed_actions": self.completed_actions,
            "pending_actions": self.pending_actions,
            "relevant_artifacts": self.relevant_artifacts,
            "token_estimate": self.token_estimate,
            "schema_version": self.schema_version,
            "domain_extensions": self.domain_extensions,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SessionSummary | None":
        if not isinstance(payload, dict):
            return None

        domain_ext = payload.get("domain_extensions")
        return cls(
            summary_id=_coerce_optional_str(payload.get("summary_id")) or "summary",
            thread_id=_coerce_optional_str(payload.get("thread_id")),
            covers_blocks_until=_coerce_optional_str(payload.get("covers_blocks_until")),
            generated_at=_coerce_optional_str(payload.get("generated_at")) or utc_now_iso(),
            source_block_count=payload.get("source_block_count", 0) if isinstance(payload.get("source_block_count"), int) else 0,
            facts_confirmed=_normalize_str_list(payload.get("facts_confirmed")),
            hypotheses_open=_normalize_str_list(payload.get("hypotheses_open")),
            decisions=_normalize_str_list(payload.get("decisions")),
            completed_actions=_normalize_str_list(payload.get("completed_actions")),
            pending_actions=_normalize_str_list(payload.get("pending_actions")),
            relevant_artifacts=_normalize_str_list(payload.get("relevant_artifacts")),
            token_estimate=payload.get("token_estimate", 0) if isinstance(payload.get("token_estimate"), int) else 0,
            schema_version=_coerce_optional_str(payload.get("schema_version")) or "1",
            domain_extensions=dict(domain_ext) if isinstance(domain_ext, dict) else {},
        )

    @classmethod
    def from_any(
        cls,
        payload: object,
        *,
        thread_id: str = "",
        covers_blocks_until: str = "",
    ) -> "SessionSummary | None":
        if payload is None:
            return None
        if isinstance(payload, SessionSummary):
            return payload

        parsed = cls.from_dict(payload)
        if parsed is not None:
            if not parsed.thread_id:
                parsed.thread_id = thread_id
            if not parsed.covers_blocks_until:
                parsed.covers_blocks_until = covers_blocks_until
            return parsed

        if isinstance(payload, str) and payload.strip():
            return cls(
                summary_id="legacy-summary",
                thread_id=thread_id,
                covers_blocks_until=covers_blocks_until,
                generated_at=utc_now_iso(),
                source_block_count=0,
                facts_confirmed=[payload.strip()],
            )

        if isinstance(payload, dict) and payload:
            legacy_lines = [f"{key}: {value}" for key, value in payload.items()]
            return cls(
                summary_id="legacy-summary",
                thread_id=thread_id,
                covers_blocks_until=covers_blocks_until,
                generated_at=utc_now_iso(),
                source_block_count=0,
                facts_confirmed=legacy_lines,
            )

        return None

    @classmethod
    def create_template(
        cls,
        *,
        thread_id: str,
        covers_blocks_until: str,
        source_block_count: int,
        previous_summary: "SessionSummary | None" = None,
    ) -> "SessionSummary":
        return cls(
            summary_id=previous_summary.summary_id if previous_summary is not None else f"summary-{thread_id}",
            thread_id=thread_id,
            covers_blocks_until=covers_blocks_until,
            generated_at=utc_now_iso(),
            source_block_count=source_block_count,
            facts_confirmed=list(previous_summary.facts_confirmed) if previous_summary is not None else [],
            hypotheses_open=list(previous_summary.hypotheses_open) if previous_summary is not None else [],
            decisions=list(previous_summary.decisions) if previous_summary is not None else [],
            completed_actions=list(previous_summary.completed_actions) if previous_summary is not None else [],
            pending_actions=list(previous_summary.pending_actions) if previous_summary is not None else [],
            relevant_artifacts=list(previous_summary.relevant_artifacts) if previous_summary is not None else [],
            domain_extensions=dict(previous_summary.domain_extensions) if previous_summary is not None else {},
        )

    def render_text(self) -> str:
        lines = ["Structured session summary:"]
        lines.append(f"- Summary ID: {self.summary_id}")
        lines.append(f"- Thread ID: {self.thread_id or '-'}")
        lines.append(f"- Covers blocks until: {self.covers_blocks_until or '-'}")
        lines.append(f"- Generated at: {self.generated_at}")
        lines.append(f"- Source block count: {self.source_block_count}")
        lines.extend(self._render_section("Confirmed facts", self.facts_confirmed))
        lines.extend(self._render_section("Open hypotheses", self.hypotheses_open))
        lines.extend(self._render_section("Decisions", self.decisions))
        lines.extend(self._render_section("Completed actions", self.completed_actions))
        lines.extend(self._render_section("Pending actions", self.pending_actions))
        lines.extend(self._render_section("Relevant artifacts", self.relevant_artifacts))
        for key, value in self.domain_extensions.items():
            if isinstance(value, list):
                lines.extend(self._render_section(key, [str(v) for v in value]))
            else:
                lines.append(f"- {key}: {value or '-'}")
        return "\n".join(lines)

    def as_context_block(self, *, pinned: bool = True, priority: int = 100, source: str = "runtime") -> ContextBlock:
        return ContextBlock(
            block_id=f"summary:{self.summary_id}",
            kind="summary",
            content={"summary": self.to_dict()},
            token_estimate=self.token_estimate,
            pinned=pinned,
            priority=priority,
            source=source,
            metadata={"thread_id": self.thread_id, "covers_blocks_until": self.covers_blocks_until},
        )

    def _render_section(self, label: str, values: list[str]) -> list[str]:
        if not values:
            return [f"{label}: -"]
        lines = [f"{label}:"]
        lines.extend(f"- {value}" for value in values)
        return lines
