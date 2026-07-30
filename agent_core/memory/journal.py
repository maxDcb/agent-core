from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_core.memory.context_block import ContextBlock, estimate_token_count
from agent_core.types import utc_now_iso

MemoryOrigin = Literal["runtime", "reflection", "model", "fallback", "recovery"]
ExchangeMemoryKind = Literal["tool_exchange", "reflection", "final_response", "provider_failure"]

DEFAULT_MAX_SESSION_ITEMS = 100
DEFAULT_MAX_RECENT_OUTCOMES = 12
DEFAULT_MAX_TEXT_CHARS = 2_000


def _normalize_text(value: object, *, limit: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 1)].rstrip()}…"


def _normalize_optional_text(value: object, *, limit: int = DEFAULT_MAX_TEXT_CHARS) -> str | None:
    normalized = _normalize_text(value, limit=limit)
    return normalized or None


def _normalize_text_list(value: object, *, item_limit: int = DEFAULT_MAX_TEXT_CHARS) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _normalize_text(item, limit=item_limit)
        key = text.casefold()
        if not text or key in seen:
            continue
        normalized.append(text)
        seen.add(key)
    return normalized


def _normalize_string_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item))


def _normalize_extensions(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _clamp_confidence(value: object, *, default: float = 0.5) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def _bounded_merge(existing: list[str], additions: list[str], *, limit: int) -> list[str]:
    merged: list[str] = []
    positions: dict[str, int] = {}
    for item in [*existing, *additions]:
        normalized = _normalize_text(item)
        if not normalized:
            continue
        key = normalized.casefold()
        previous_position = positions.get(key)
        if previous_position is not None:
            merged.pop(previous_position)
            positions = {value.casefold(): index for index, value in enumerate(merged)}
        merged.append(normalized)
        positions[key] = len(merged) - 1
    return merged[-max(1, limit) :]


def _remove_items(existing: list[str], removals: list[str]) -> list[str]:
    removal_keys = {item.casefold() for item in removals if item}
    return [item for item in existing if item.casefold() not in removal_keys]


def _coerce_int(value: object, *, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _positive_int(value: object, *, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


@dataclass(slots=True)
class ActiveTask:
    """Small operational projection used to steer the next conversation turn."""

    objective: str
    status: str = "active"
    next_action: str | None = None
    open_questions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "status": self.status,
            "next_action": self.next_action,
            "open_questions": list(self.open_questions),
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_any(cls, payload: object) -> ActiveTask | None:
        if isinstance(payload, ActiveTask):
            return payload
        if not isinstance(payload, dict):
            return None
        objective = _normalize_text(payload.get("objective"))
        if not objective:
            return None
        status = _normalize_text(payload.get("status"), limit=100) or "active"
        return cls(
            objective=objective,
            status=status,
            next_action=_normalize_optional_text(payload.get("next_action")),
            open_questions=_normalize_text_list(payload.get("open_questions")),
            constraints=_normalize_text_list(payload.get("constraints")),
        )

    @classmethod
    def create_template(cls, *, objective: str = "") -> ActiveTask:
        return cls(objective=objective)


@dataclass(slots=True)
class ExchangeMemory:
    """Append-only memory event for one atomic model/tool interaction."""

    memory_id: str
    thread_id: str
    turn_index: int
    exchange_index: int
    kind: ExchangeMemoryKind
    summary: str
    created_at: str = field(default_factory=utc_now_iso)
    origin: MemoryOrigin = "runtime"
    confirmed_facts: list[str] = field(default_factory=list)
    open_hypotheses: list[str] = field(default_factory=list)
    rejected_hypotheses: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    resolved_questions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    completed_actions: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    relevant_artifacts: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    confidence: float = 0.5
    source_block_ids: list[str] = field(default_factory=list)
    domain_extensions: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "thread_id": self.thread_id,
            "turn_index": self.turn_index,
            "exchange_index": self.exchange_index,
            "kind": self.kind,
            "summary": self.summary,
            "created_at": self.created_at,
            "origin": self.origin,
            "confirmed_facts": list(self.confirmed_facts),
            "open_hypotheses": list(self.open_hypotheses),
            "rejected_hypotheses": list(self.rejected_hypotheses),
            "open_questions": list(self.open_questions),
            "resolved_questions": list(self.resolved_questions),
            "decisions": list(self.decisions),
            "completed_actions": list(self.completed_actions),
            "next_actions": list(self.next_actions),
            "relevant_artifacts": list(self.relevant_artifacts),
            "risk_notes": list(self.risk_notes),
            "confidence": self.confidence,
            "source_block_ids": list(self.source_block_ids),
            "domain_extensions": dict(self.domain_extensions),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_any(cls, payload: object) -> ExchangeMemory | None:
        if isinstance(payload, ExchangeMemory):
            return payload
        if not isinstance(payload, dict):
            return None
        memory_id = payload.get("memory_id")
        thread_id = payload.get("thread_id")
        kind = payload.get("kind")
        if (
            not isinstance(memory_id, str)
            or not memory_id
            or not isinstance(thread_id, str)
            or kind not in {"tool_exchange", "reflection", "final_response", "provider_failure"}
        ):
            return None
        origin = payload.get("origin")
        return cls(
            memory_id=memory_id,
            thread_id=thread_id,
            turn_index=_coerce_int(payload.get("turn_index")),
            exchange_index=_coerce_int(payload.get("exchange_index")),
            kind=kind,
            summary=_normalize_text(payload.get("summary")),
            created_at=_normalize_text(payload.get("created_at"), limit=100) or utc_now_iso(),
            origin=origin if origin in {"runtime", "reflection", "model", "fallback", "recovery"} else "runtime",
            confirmed_facts=_normalize_text_list(payload.get("confirmed_facts")),
            open_hypotheses=_normalize_text_list(payload.get("open_hypotheses")),
            rejected_hypotheses=_normalize_text_list(payload.get("rejected_hypotheses")),
            open_questions=_normalize_text_list(payload.get("open_questions")),
            resolved_questions=_normalize_text_list(payload.get("resolved_questions")),
            decisions=_normalize_text_list(payload.get("decisions")),
            completed_actions=_normalize_text_list(payload.get("completed_actions")),
            next_actions=_normalize_text_list(payload.get("next_actions")),
            relevant_artifacts=_normalize_string_ids(payload.get("relevant_artifacts")),
            risk_notes=_normalize_text_list(payload.get("risk_notes")),
            confidence=_clamp_confidence(payload.get("confidence")),
            source_block_ids=_normalize_string_ids(payload.get("source_block_ids")),
            domain_extensions=_normalize_extensions(payload.get("domain_extensions")),
            schema_version=_normalize_text(payload.get("schema_version"), limit=20) or "1",
        )


@dataclass(slots=True)
class TurnMemory:
    """Bounded delta synthesized only from memory events belonging to one turn."""

    memory_id: str
    thread_id: str
    turn_index: int
    user_intent: str
    assistant_outcome: str
    active_task: ActiveTask
    created_at: str = field(default_factory=utc_now_iso)
    origin: MemoryOrigin = "model"
    exchange_memory_ids: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    confirmed_facts: list[str] = field(default_factory=list)
    superseded_facts: list[str] = field(default_factory=list)
    open_hypotheses: list[str] = field(default_factory=list)
    rejected_hypotheses: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    resolved_questions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    completed_actions: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    relevant_artifacts: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    domain_extensions: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "thread_id": self.thread_id,
            "turn_index": self.turn_index,
            "user_intent": self.user_intent,
            "assistant_outcome": self.assistant_outcome,
            "active_task": self.active_task.to_dict(),
            "created_at": self.created_at,
            "origin": self.origin,
            "exchange_memory_ids": list(self.exchange_memory_ids),
            "source_block_ids": list(self.source_block_ids),
            "confirmed_facts": list(self.confirmed_facts),
            "superseded_facts": list(self.superseded_facts),
            "open_hypotheses": list(self.open_hypotheses),
            "rejected_hypotheses": list(self.rejected_hypotheses),
            "open_questions": list(self.open_questions),
            "resolved_questions": list(self.resolved_questions),
            "decisions": list(self.decisions),
            "completed_actions": list(self.completed_actions),
            "next_actions": list(self.next_actions),
            "relevant_artifacts": list(self.relevant_artifacts),
            "risk_notes": list(self.risk_notes),
            "domain_extensions": dict(self.domain_extensions),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_any(cls, payload: object) -> TurnMemory | None:
        if isinstance(payload, TurnMemory):
            return payload
        if not isinstance(payload, dict):
            return None
        memory_id = payload.get("memory_id")
        thread_id = payload.get("thread_id")
        active_task = ActiveTask.from_any(payload.get("active_task"))
        if not isinstance(memory_id, str) or not memory_id or not isinstance(thread_id, str) or active_task is None:
            return None
        origin = payload.get("origin")
        return cls(
            memory_id=memory_id,
            thread_id=thread_id,
            turn_index=_coerce_int(payload.get("turn_index")),
            user_intent=_normalize_text(payload.get("user_intent")),
            assistant_outcome=_normalize_text(payload.get("assistant_outcome")),
            active_task=active_task,
            created_at=_normalize_text(payload.get("created_at"), limit=100) or utc_now_iso(),
            origin=origin if origin in {"runtime", "reflection", "model", "fallback", "recovery"} else "model",
            exchange_memory_ids=_normalize_string_ids(payload.get("exchange_memory_ids")),
            source_block_ids=_normalize_string_ids(payload.get("source_block_ids")),
            confirmed_facts=_normalize_text_list(payload.get("confirmed_facts")),
            superseded_facts=_normalize_text_list(payload.get("superseded_facts")),
            open_hypotheses=_normalize_text_list(payload.get("open_hypotheses")),
            rejected_hypotheses=_normalize_text_list(payload.get("rejected_hypotheses")),
            open_questions=_normalize_text_list(payload.get("open_questions")),
            resolved_questions=_normalize_text_list(payload.get("resolved_questions")),
            decisions=_normalize_text_list(payload.get("decisions")),
            completed_actions=_normalize_text_list(payload.get("completed_actions")),
            next_actions=_normalize_text_list(payload.get("next_actions")),
            relevant_artifacts=_normalize_string_ids(payload.get("relevant_artifacts")),
            risk_notes=_normalize_text_list(payload.get("risk_notes")),
            domain_extensions=_normalize_extensions(payload.get("domain_extensions")),
            schema_version=_normalize_text(payload.get("schema_version"), limit=20) or "1",
        )

    @classmethod
    def create_template(
        cls,
        *,
        memory_id: str,
        thread_id: str,
        turn_index: int,
        objective: str,
    ) -> TurnMemory:
        return cls(
            memory_id=memory_id,
            thread_id=thread_id,
            turn_index=turn_index,
            user_intent="",
            assistant_outcome="",
            active_task=ActiveTask.create_template(objective=objective),
        )

    def with_runtime_identity(
        self,
        *,
        memory_id: str,
        thread_id: str,
        turn_index: int,
        exchange_memory_ids: list[str],
        source_block_ids: list[str],
        origin: MemoryOrigin,
    ) -> TurnMemory:
        return TurnMemory(
            memory_id=memory_id,
            thread_id=thread_id,
            turn_index=turn_index,
            user_intent=self.user_intent,
            assistant_outcome=self.assistant_outcome,
            active_task=self.active_task,
            created_at=utc_now_iso(),
            origin=origin,
            exchange_memory_ids=list(exchange_memory_ids),
            source_block_ids=list(source_block_ids),
            confirmed_facts=list(self.confirmed_facts),
            superseded_facts=list(self.superseded_facts),
            open_hypotheses=list(self.open_hypotheses),
            rejected_hypotheses=list(self.rejected_hypotheses),
            open_questions=list(self.open_questions),
            resolved_questions=list(self.resolved_questions),
            decisions=list(self.decisions),
            completed_actions=list(self.completed_actions),
            next_actions=list(self.next_actions),
            relevant_artifacts=list(self.relevant_artifacts),
            risk_notes=list(self.risk_notes),
            domain_extensions=dict(self.domain_extensions),
            schema_version=self.schema_version,
        )


@dataclass(slots=True)
class SessionView:
    """Materialized, bounded projection of the append-only turn journal."""

    thread_id: str
    generation: int = 0
    through_turn_index: int = -1
    through_turn_memory_id: str | None = None
    updated_at: str = field(default_factory=utc_now_iso)
    active_task: ActiveTask | None = None
    confirmed_facts: list[str] = field(default_factory=list)
    open_hypotheses: list[str] = field(default_factory=list)
    rejected_hypotheses: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    completed_actions: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    relevant_artifacts: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    recent_outcomes: list[str] = field(default_factory=list)
    domain_extensions: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "generation": self.generation,
            "through_turn_index": self.through_turn_index,
            "through_turn_memory_id": self.through_turn_memory_id,
            "updated_at": self.updated_at,
            "active_task": self.active_task.to_dict() if self.active_task is not None else None,
            "confirmed_facts": list(self.confirmed_facts),
            "open_hypotheses": list(self.open_hypotheses),
            "rejected_hypotheses": list(self.rejected_hypotheses),
            "open_questions": list(self.open_questions),
            "decisions": list(self.decisions),
            "completed_actions": list(self.completed_actions),
            "next_actions": list(self.next_actions),
            "relevant_artifacts": list(self.relevant_artifacts),
            "risk_notes": list(self.risk_notes),
            "recent_outcomes": list(self.recent_outcomes),
            "domain_extensions": dict(self.domain_extensions),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_any(cls, payload: object, *, thread_id: str) -> SessionView | None:
        if isinstance(payload, SessionView):
            return payload
        if not isinstance(payload, dict):
            return None
        stored_thread_id = payload.get("thread_id")
        if stored_thread_id not in {None, "", thread_id}:
            return None
        through_memory_id = payload.get("through_turn_memory_id")
        return cls(
            thread_id=thread_id,
            generation=max(0, _coerce_int(payload.get("generation"))),
            through_turn_index=_coerce_int(payload.get("through_turn_index"), default=-1),
            through_turn_memory_id=through_memory_id if isinstance(through_memory_id, str) else None,
            updated_at=_normalize_text(payload.get("updated_at"), limit=100) or utc_now_iso(),
            active_task=ActiveTask.from_any(payload.get("active_task")),
            confirmed_facts=_normalize_text_list(payload.get("confirmed_facts")),
            open_hypotheses=_normalize_text_list(payload.get("open_hypotheses")),
            rejected_hypotheses=_normalize_text_list(payload.get("rejected_hypotheses")),
            open_questions=_normalize_text_list(payload.get("open_questions")),
            decisions=_normalize_text_list(payload.get("decisions")),
            completed_actions=_normalize_text_list(payload.get("completed_actions")),
            next_actions=_normalize_text_list(payload.get("next_actions")),
            relevant_artifacts=_normalize_string_ids(payload.get("relevant_artifacts")),
            risk_notes=_normalize_text_list(payload.get("risk_notes")),
            recent_outcomes=_normalize_text_list(payload.get("recent_outcomes")),
            domain_extensions=_normalize_extensions(payload.get("domain_extensions")),
            schema_version=_normalize_text(payload.get("schema_version"), limit=20) or "1",
        )

    @classmethod
    def empty(cls, *, thread_id: str) -> SessionView:
        return cls(thread_id=thread_id)

    def merge(
        self,
        turn_memory: TurnMemory,
        *,
        max_items: int = DEFAULT_MAX_SESSION_ITEMS,
        max_recent_outcomes: int = DEFAULT_MAX_RECENT_OUTCOMES,
    ) -> SessionView:
        if turn_memory.thread_id != self.thread_id:
            raise ValueError("Cannot merge memory from a different thread")
        confirmed_facts = _remove_items(self.confirmed_facts, turn_memory.superseded_facts)
        open_hypotheses = _remove_items(self.open_hypotheses, turn_memory.rejected_hypotheses)
        open_questions = _remove_items(self.open_questions, turn_memory.resolved_questions)
        pending_actions = _remove_items(self.next_actions, turn_memory.completed_actions)
        recent_outcome = f"Turn {turn_memory.turn_index}: {turn_memory.assistant_outcome}".strip()
        return SessionView(
            thread_id=self.thread_id,
            generation=self.generation + 1,
            through_turn_index=turn_memory.turn_index,
            through_turn_memory_id=turn_memory.memory_id,
            updated_at=turn_memory.created_at,
            active_task=turn_memory.active_task,
            confirmed_facts=_bounded_merge(confirmed_facts, turn_memory.confirmed_facts, limit=max_items),
            open_hypotheses=_bounded_merge(open_hypotheses, turn_memory.open_hypotheses, limit=max_items),
            rejected_hypotheses=_bounded_merge(
                self.rejected_hypotheses,
                turn_memory.rejected_hypotheses,
                limit=max_items,
            ),
            open_questions=_bounded_merge(open_questions, turn_memory.open_questions, limit=max_items),
            decisions=_bounded_merge(self.decisions, turn_memory.decisions, limit=max_items),
            completed_actions=_bounded_merge(
                self.completed_actions,
                turn_memory.completed_actions,
                limit=max_items,
            ),
            next_actions=_bounded_merge(pending_actions, turn_memory.next_actions, limit=max_items),
            relevant_artifacts=_bounded_merge(
                self.relevant_artifacts,
                turn_memory.relevant_artifacts,
                limit=max_items,
            ),
            risk_notes=_bounded_merge(self.risk_notes, turn_memory.risk_notes, limit=max_items),
            recent_outcomes=_bounded_merge(
                self.recent_outcomes,
                [recent_outcome] if recent_outcome else [],
                limit=max_recent_outcomes,
            ),
            domain_extensions=_merge_domain_extensions(
                self.domain_extensions,
                turn_memory.domain_extensions,
                max_items=max_items,
            ),
        )

    def render_text(self) -> str:
        lines = [
            "Conversation memory view:",
            f"- Generation: {self.generation}",
            f"- Covers through turn: {self.through_turn_index}",
        ]
        if self.active_task is not None:
            lines.extend(
                [
                    "Active task:",
                    f"- Objective: {self.active_task.objective}",
                    f"- Status: {self.active_task.status}",
                    f"- Next action: {self.active_task.next_action or '-'}",
                ]
            )
            lines.extend(_render_section("Task open questions", self.active_task.open_questions))
            lines.extend(_render_section("Task constraints", self.active_task.constraints))
        lines.extend(_render_section("Confirmed facts", self.confirmed_facts))
        lines.extend(_render_section("Open hypotheses", self.open_hypotheses))
        lines.extend(_render_section("Rejected hypotheses", self.rejected_hypotheses))
        lines.extend(_render_section("Open questions", self.open_questions))
        lines.extend(_render_section("Decisions", self.decisions))
        lines.extend(_render_section("Completed actions", self.completed_actions))
        lines.extend(_render_section("Next actions", self.next_actions))
        lines.extend(_render_section("Relevant artifacts", self.relevant_artifacts))
        lines.extend(_render_section("Risk notes", self.risk_notes))
        lines.extend(_render_section("Recent outcomes", self.recent_outcomes))
        for key, value in self.domain_extensions.items():
            values = value if isinstance(value, list) else [value]
            lines.extend(_render_section(str(key), [str(item) for item in values]))
        return "\n".join(lines)

    def as_context_block(self) -> ContextBlock:
        payload = self.to_dict()
        return ContextBlock(
            block_id=f"memory-view:{self.thread_id}:{self.generation}",
            kind="memory_view",
            content={"session_view": payload},
            token_estimate=estimate_token_count(payload),
            pinned=True,
            priority=100,
            source="memory_journal",
            metadata={
                "thread_id": self.thread_id,
                "generation": self.generation,
                "through_turn_index": self.through_turn_index,
            },
        )


@dataclass(slots=True)
class IncrementalMemoryJournal:
    """Append-only exchanges and turns plus a rebuildable SessionView."""

    thread_id: str
    exchanges: list[ExchangeMemory] = field(default_factory=list)
    turns: list[TurnMemory] = field(default_factory=list)
    session_view: SessionView | None = None
    max_session_items: int = DEFAULT_MAX_SESSION_ITEMS
    max_recent_outcomes: int = DEFAULT_MAX_RECENT_OUTCOMES
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.session_view is None:
            self.session_view = SessionView.empty(thread_id=self.thread_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "thread_id": self.thread_id,
            "exchanges": [item.to_dict() for item in self.exchanges],
            "turns": [item.to_dict() for item in self.turns],
            "session_view": self.session_view.to_dict() if self.session_view is not None else None,
            "view_policy": {
                "max_session_items": self.max_session_items,
                "max_recent_outcomes": self.max_recent_outcomes,
            },
        }

    @classmethod
    def from_any(
        cls,
        payload: object,
        *,
        thread_id: str,
        max_session_items: int | None = None,
        max_recent_outcomes: int | None = None,
    ) -> IncrementalMemoryJournal:
        if isinstance(payload, IncrementalMemoryJournal):
            return payload
        if not isinstance(payload, dict) or payload.get("schema_version") != "1":
            return cls(
                thread_id=thread_id,
                max_session_items=max_session_items or DEFAULT_MAX_SESSION_ITEMS,
                max_recent_outcomes=max_recent_outcomes or DEFAULT_MAX_RECENT_OUTCOMES,
            )

        raw_exchanges = payload.get("exchanges")
        exchanges = (
            [
                exchange_memory
                for item in raw_exchanges
                if (exchange_memory := ExchangeMemory.from_any(item)) is not None
                and exchange_memory.thread_id == thread_id
            ]
            if isinstance(raw_exchanges, list)
            else []
        )
        exchanges = _deduplicate_by_memory_id(exchanges)

        raw_turns = payload.get("turns")
        turns = (
            [
                turn_memory
                for item in raw_turns
                if (turn_memory := TurnMemory.from_any(item)) is not None and turn_memory.thread_id == thread_id
            ]
            if isinstance(raw_turns, list)
            else []
        )
        turns = sorted(_deduplicate_by_memory_id(turns), key=lambda item: item.turn_index)

        raw_policy = payload.get("view_policy")
        stored_max_items = (
            _positive_int(raw_policy.get("max_session_items"), default=DEFAULT_MAX_SESSION_ITEMS)
            if isinstance(raw_policy, dict)
            else DEFAULT_MAX_SESSION_ITEMS
        )
        stored_max_outcomes = (
            _positive_int(raw_policy.get("max_recent_outcomes"), default=DEFAULT_MAX_RECENT_OUTCOMES)
            if isinstance(raw_policy, dict)
            else DEFAULT_MAX_RECENT_OUTCOMES
        )
        effective_max_items = max_session_items or stored_max_items
        effective_max_outcomes = max_recent_outcomes or stored_max_outcomes
        journal = cls(
            thread_id=thread_id,
            exchanges=exchanges,
            turns=turns,
            session_view=SessionView.empty(thread_id=thread_id),
            max_session_items=effective_max_items,
            max_recent_outcomes=effective_max_outcomes,
        )
        journal.rebuild_session_view(
            max_session_items=effective_max_items,
            max_recent_outcomes=effective_max_outcomes,
        )
        return journal

    def append_exchange(self, memory: ExchangeMemory) -> bool:
        if memory.thread_id != self.thread_id:
            raise ValueError("Cannot append exchange memory from a different thread")
        existing = self._exchange_by_id(memory.memory_id)
        if existing is not None:
            if existing.to_dict() != memory.to_dict():
                raise ValueError(f"Exchange memory id already exists with different content: {memory.memory_id}")
            return False
        self.exchanges.append(memory)
        return True

    def commit_turn(
        self,
        memory: TurnMemory,
        *,
        max_session_items: int = DEFAULT_MAX_SESSION_ITEMS,
        max_recent_outcomes: int = DEFAULT_MAX_RECENT_OUTCOMES,
    ) -> bool:
        if memory.thread_id != self.thread_id:
            raise ValueError("Cannot append turn memory from a different thread")
        self.max_session_items = max_session_items
        self.max_recent_outcomes = max_recent_outcomes
        existing = self.turn_for_index(memory.turn_index)
        if existing is not None:
            if existing.to_dict() != memory.to_dict():
                raise ValueError(f"Turn memory already committed with different content: {memory.turn_index}")
            return False
        out_of_order = bool(self.turns and memory.turn_index < self.turns[-1].turn_index)
        self.turns.append(memory)
        self.turns.sort(key=lambda item: item.turn_index)
        if out_of_order:
            self.rebuild_session_view(
                max_session_items=max_session_items,
                max_recent_outcomes=max_recent_outcomes,
            )
            return True
        current_view = self.session_view or SessionView.empty(thread_id=self.thread_id)
        self.session_view = current_view.merge(
            memory,
            max_items=max_session_items,
            max_recent_outcomes=max_recent_outcomes,
        )
        return True

    def exchanges_for_turn(self, turn_index: int) -> list[ExchangeMemory]:
        return sorted(
            [item for item in self.exchanges if item.turn_index == turn_index],
            key=lambda item: (item.exchange_index, item.created_at, item.memory_id),
        )

    def turn_for_index(self, turn_index: int) -> TurnMemory | None:
        return next((item for item in self.turns if item.turn_index == turn_index), None)

    def rebuild_session_view(
        self,
        *,
        max_session_items: int = DEFAULT_MAX_SESSION_ITEMS,
        max_recent_outcomes: int = DEFAULT_MAX_RECENT_OUTCOMES,
    ) -> SessionView:
        view = SessionView.empty(thread_id=self.thread_id)
        self.max_session_items = max_session_items
        self.max_recent_outcomes = max_recent_outcomes
        for turn in sorted(self.turns, key=lambda item: item.turn_index):
            view = view.merge(
                turn,
                max_items=max_session_items,
                max_recent_outcomes=max_recent_outcomes,
            )
        self.session_view = view
        return view

    def _exchange_by_id(self, memory_id: str) -> ExchangeMemory | None:
        return next((item for item in self.exchanges if item.memory_id == memory_id), None)


def build_fallback_turn_memory(
    *,
    thread_id: str,
    turn_index: int,
    user_intent: str,
    assistant_outcome: str,
    exchanges: list[ExchangeMemory],
    source_block_ids: list[str],
    previous_active_task: ActiveTask | None,
    origin: MemoryOrigin = "fallback",
) -> TurnMemory:
    active_objective = previous_active_task.objective if previous_active_task is not None else user_intent
    active_task = ActiveTask(
        objective=_normalize_text(active_objective) or "Continue the current conversation",
        status=previous_active_task.status if previous_active_task is not None else "active",
        next_action=_first_item(exchanges, "next_actions")
        or (previous_active_task.next_action if previous_active_task is not None else None),
        open_questions=_aggregate_exchange_field(exchanges, "open_questions"),
        constraints=list(previous_active_task.constraints) if previous_active_task is not None else [],
    )
    return TurnMemory(
        memory_id=f"turn-{turn_index:04d}-memory",
        thread_id=thread_id,
        turn_index=turn_index,
        user_intent=_normalize_text(user_intent),
        assistant_outcome=_normalize_text(assistant_outcome),
        active_task=active_task,
        origin=origin,
        exchange_memory_ids=[item.memory_id for item in exchanges],
        source_block_ids=list(dict.fromkeys(source_block_ids)),
        confirmed_facts=_aggregate_exchange_field(exchanges, "confirmed_facts"),
        open_hypotheses=_aggregate_exchange_field(exchanges, "open_hypotheses"),
        rejected_hypotheses=_aggregate_exchange_field(exchanges, "rejected_hypotheses"),
        open_questions=_aggregate_exchange_field(exchanges, "open_questions"),
        resolved_questions=_aggregate_exchange_field(exchanges, "resolved_questions"),
        decisions=_aggregate_exchange_field(exchanges, "decisions"),
        completed_actions=_aggregate_exchange_field(exchanges, "completed_actions"),
        next_actions=_aggregate_exchange_field(exchanges, "next_actions"),
        relevant_artifacts=_aggregate_exchange_field(exchanges, "relevant_artifacts"),
        risk_notes=_aggregate_exchange_field(exchanges, "risk_notes"),
        domain_extensions=_aggregate_domain_extensions(exchanges),
    )


def _aggregate_exchange_field(exchanges: list[ExchangeMemory], field_name: str) -> list[str]:
    aggregate: list[str] = []
    for exchange in exchanges:
        values = getattr(exchange, field_name)
        if isinstance(values, list):
            aggregate = _bounded_merge(aggregate, values, limit=DEFAULT_MAX_SESSION_ITEMS)
    return aggregate


def _first_item(exchanges: list[ExchangeMemory], field_name: str) -> str | None:
    for exchange in reversed(exchanges):
        values = getattr(exchange, field_name)
        if isinstance(values, list) and values:
            first = values[0]
            if isinstance(first, str):
                return first
    return None


def _aggregate_domain_extensions(exchanges: list[ExchangeMemory]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for exchange in exchanges:
        aggregate = _merge_domain_extensions(
            aggregate,
            exchange.domain_extensions,
            max_items=DEFAULT_MAX_SESSION_ITEMS,
        )
    return aggregate


def _merge_domain_extensions(
    existing: dict[str, Any],
    additions: dict[str, Any],
    *,
    max_items: int,
) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in additions.items():
        if isinstance(value, list):
            current = merged.get(key)
            merged[key] = _bounded_merge_json(
                current if isinstance(current, list) else [],
                value,
                limit=max_items,
            )
        elif value is not None:
            merged[key] = deepcopy(value)
    return merged


def _bounded_merge_json(existing: list[Any], additions: list[Any], *, limit: int) -> list[Any]:
    merged: list[Any] = []
    positions: dict[str, int] = {}
    for item in [*existing, *additions]:
        key = _json_identity(item)
        previous_position = positions.get(key)
        if previous_position is not None:
            merged.pop(previous_position)
            positions = {_json_identity(value): index for index, value in enumerate(merged)}
        merged.append(deepcopy(item))
        positions[key] = len(merged) - 1
    return merged[-max(1, limit) :]


def _json_identity(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _render_section(label: str, values: list[str]) -> list[str]:
    if not values:
        return [f"{label}: -"]
    return [f"{label}:", *(f"- {item}" for item in values)]


def _deduplicate_by_memory_id(items: list[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for item in items:
        memory_id = item.memory_id
        if memory_id in seen:
            continue
        unique.append(item)
        seen.add(memory_id)
    return unique
