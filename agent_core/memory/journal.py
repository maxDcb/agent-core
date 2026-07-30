from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent_core.memory.context_block import ContextBlock, estimate_token_count
from agent_core.types import utc_now_iso

MemoryOrigin = Literal["runtime", "reflection", "model", "fallback", "recovery"]
ExchangeMemoryKind = Literal["tool_exchange", "reflection", "final_response", "provider_failure"]

MEMORY_SCHEMA_VERSION = "2"
DEFAULT_MAX_HANDOFF_CHARS = 6_000
DEFAULT_MAX_TURN_SUMMARY_CHARS = 4_000
DEFAULT_MAX_TEXT_CHARS = 2_000
MAX_PERSISTED_DOCUMENT_CHARS = 64_000


def _normalize_text(value: object, *, limit: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 1)].rstrip()}…"


def _normalize_document(value: object, *, limit: int = MAX_PERSISTED_DOCUMENT_CHARS) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in normalized.splitlines()]
    normalized = "\n".join(lines).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 1)].rstrip()}…"


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


def _coerce_int(value: object, *, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _valid_origin(value: object, *, default: MemoryOrigin) -> MemoryOrigin:
    if value in {"runtime", "reflection", "model", "fallback", "recovery"}:
        return value
    return default


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
        return cls(
            memory_id=memory_id,
            thread_id=thread_id,
            turn_index=_coerce_int(payload.get("turn_index")),
            exchange_index=_coerce_int(payload.get("exchange_index")),
            kind=kind,
            summary=_normalize_text(payload.get("summary")),
            created_at=_normalize_text(payload.get("created_at"), limit=100) or utc_now_iso(),
            origin=_valid_origin(payload.get("origin"), default="runtime"),
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
    """Immutable record of one turn and its resulting operational handoff."""

    memory_id: str
    thread_id: str
    turn_index: int
    user_intent: str
    assistant_outcome: str
    turn_summary: str
    handoff_after_turn: str
    created_at: str = field(default_factory=utc_now_iso)
    origin: MemoryOrigin = "model"
    degraded: bool = False
    exchange_memory_ids: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    relevant_artifacts: list[str] = field(default_factory=list)
    schema_version: str = MEMORY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "thread_id": self.thread_id,
            "turn_index": self.turn_index,
            "user_intent": self.user_intent,
            "assistant_outcome": self.assistant_outcome,
            "turn_summary": self.turn_summary,
            "handoff_after_turn": self.handoff_after_turn,
            "created_at": self.created_at,
            "origin": self.origin,
            "degraded": self.degraded,
            "exchange_memory_ids": list(self.exchange_memory_ids),
            "source_block_ids": list(self.source_block_ids),
            "relevant_artifacts": list(self.relevant_artifacts),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_any(cls, payload: object) -> TurnMemory | None:
        if isinstance(payload, TurnMemory):
            return payload
        if not isinstance(payload, dict) or payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
            return None
        memory_id = payload.get("memory_id")
        thread_id = payload.get("thread_id")
        turn_index = payload.get("turn_index")
        turn_summary = _normalize_document(payload.get("turn_summary"))
        handoff = _normalize_document(payload.get("handoff_after_turn"))
        if (
            not isinstance(memory_id, str)
            or not memory_id
            or not isinstance(thread_id, str)
            or not isinstance(turn_index, int)
            or isinstance(turn_index, bool)
            or turn_index < 0
            or not turn_summary
            or not handoff
        ):
            return None
        return cls(
            memory_id=memory_id,
            thread_id=thread_id,
            turn_index=turn_index,
            user_intent=_normalize_text(payload.get("user_intent"), limit=4_000),
            assistant_outcome=_normalize_text(payload.get("assistant_outcome"), limit=8_000),
            turn_summary=turn_summary,
            handoff_after_turn=handoff,
            created_at=_normalize_text(payload.get("created_at"), limit=100) or utc_now_iso(),
            origin=_valid_origin(payload.get("origin"), default="model"),
            degraded=payload.get("degraded") is True,
            exchange_memory_ids=_normalize_string_ids(payload.get("exchange_memory_ids")),
            source_block_ids=_normalize_string_ids(payload.get("source_block_ids")),
            relevant_artifacts=_normalize_string_ids(payload.get("relevant_artifacts")),
        )


@dataclass(slots=True)
class SessionView:
    """Latest bounded operational handoff derived from the turn journal."""

    thread_id: str
    generation: int = 0
    through_turn_index: int = -1
    through_turn_memory_id: str | None = None
    content: str = ""
    origin: MemoryOrigin = "runtime"
    degraded: bool = False
    updated_at: str = field(default_factory=utc_now_iso)
    schema_version: str = MEMORY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "generation": self.generation,
            "through_turn_index": self.through_turn_index,
            "through_turn_memory_id": self.through_turn_memory_id,
            "content": self.content,
            "origin": self.origin,
            "degraded": self.degraded,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_any(cls, payload: object, *, thread_id: str) -> SessionView | None:
        if isinstance(payload, SessionView):
            return payload
        if not isinstance(payload, dict) or payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
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
            content=_normalize_document(payload.get("content")),
            origin=_valid_origin(payload.get("origin"), default="runtime"),
            degraded=payload.get("degraded") is True,
            updated_at=_normalize_text(payload.get("updated_at"), limit=100) or utc_now_iso(),
        )

    @classmethod
    def empty(cls, *, thread_id: str) -> SessionView:
        return cls(thread_id=thread_id)

    @classmethod
    def from_turn_memory(cls, turn_memory: TurnMemory, *, generation: int) -> SessionView:
        return cls(
            thread_id=turn_memory.thread_id,
            generation=max(1, generation),
            through_turn_index=turn_memory.turn_index,
            through_turn_memory_id=turn_memory.memory_id,
            content=turn_memory.handoff_after_turn,
            origin=turn_memory.origin,
            degraded=turn_memory.degraded,
            updated_at=turn_memory.created_at,
        )

    def render_text(self) -> str:
        if not self.content:
            return ""
        return f"Operational handoff from previous turns:\n\n{self.content}"

    def as_context_block(self) -> ContextBlock:
        payload = self.to_dict()
        rendered = self.render_text()
        return ContextBlock(
            block_id=f"memory-view:{self.thread_id}:{self.generation}",
            kind="memory_view",
            content={"session_view": payload},
            token_estimate=estimate_token_count(rendered),
            pinned=True,
            priority=100,
            source="memory_journal",
            metadata={
                "thread_id": self.thread_id,
                "generation": self.generation,
                "through_turn_index": self.through_turn_index,
                "origin": self.origin,
                "degraded": self.degraded,
            },
        )


@dataclass(slots=True)
class IncrementalMemoryJournal:
    """Append-only exchanges and turns plus the latest operational handoff."""

    thread_id: str
    exchanges: list[ExchangeMemory] = field(default_factory=list)
    turns: list[TurnMemory] = field(default_factory=list)
    session_view: SessionView | None = None
    max_handoff_chars: int = DEFAULT_MAX_HANDOFF_CHARS
    max_turn_summary_chars: int = DEFAULT_MAX_TURN_SUMMARY_CHARS
    schema_version: str = MEMORY_SCHEMA_VERSION

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
                "max_handoff_chars": self.max_handoff_chars,
                "max_turn_summary_chars": self.max_turn_summary_chars,
            },
        }

    @classmethod
    def from_any(
        cls,
        payload: object,
        *,
        thread_id: str,
        max_handoff_chars: int | None = None,
        max_turn_summary_chars: int | None = None,
    ) -> IncrementalMemoryJournal:
        effective_handoff_limit = _positive_int(
            max_handoff_chars,
            default=DEFAULT_MAX_HANDOFF_CHARS,
        )
        effective_summary_limit = _positive_int(
            max_turn_summary_chars,
            default=DEFAULT_MAX_TURN_SUMMARY_CHARS,
        )
        if isinstance(payload, IncrementalMemoryJournal):
            return payload
        if not isinstance(payload, dict) or payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
            return cls(
                thread_id=thread_id,
                max_handoff_chars=effective_handoff_limit,
                max_turn_summary_chars=effective_summary_limit,
            )

        raw_policy = payload.get("view_policy")
        if max_handoff_chars is None and isinstance(raw_policy, dict):
            effective_handoff_limit = _positive_int(
                raw_policy.get("max_handoff_chars"),
                default=DEFAULT_MAX_HANDOFF_CHARS,
            )
        if max_turn_summary_chars is None and isinstance(raw_policy, dict):
            effective_summary_limit = _positive_int(
                raw_policy.get("max_turn_summary_chars"),
                default=DEFAULT_MAX_TURN_SUMMARY_CHARS,
            )

        raw_exchanges = payload.get("exchanges")
        exchanges = (
            [
                memory
                for item in raw_exchanges
                if (memory := ExchangeMemory.from_any(item)) is not None and memory.thread_id == thread_id
            ]
            if isinstance(raw_exchanges, list)
            else []
        )

        raw_turns = payload.get("turns")
        turns = (
            [
                memory
                for item in raw_turns
                if (memory := TurnMemory.from_any(item)) is not None
                and memory.thread_id == thread_id
            ]
            if isinstance(raw_turns, list)
            else []
        )

        journal = cls(
            thread_id=thread_id,
            exchanges=_deduplicate_by_memory_id(exchanges),
            turns=sorted(_deduplicate_by_memory_id(turns), key=lambda item: item.turn_index),
            session_view=SessionView.empty(thread_id=thread_id),
            max_handoff_chars=effective_handoff_limit,
            max_turn_summary_chars=effective_summary_limit,
        )
        journal.rebuild_session_view(max_handoff_chars=effective_handoff_limit)
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
        max_handoff_chars: int = DEFAULT_MAX_HANDOFF_CHARS,
        max_turn_summary_chars: int = DEFAULT_MAX_TURN_SUMMARY_CHARS,
    ) -> bool:
        if memory.thread_id != self.thread_id:
            raise ValueError("Cannot append turn memory from a different thread")
        if len(memory.turn_summary) > max_turn_summary_chars:
            raise ValueError(
                "Turn summary exceeds the configured bound "
                f"({len(memory.turn_summary)} > {max_turn_summary_chars} characters)"
            )
        if len(memory.handoff_after_turn) > max_handoff_chars:
            raise ValueError(
                "Turn handoff exceeds the configured bound "
                f"({len(memory.handoff_after_turn)} > {max_handoff_chars} characters)"
            )
        existing = self.turn_for_index(memory.turn_index)
        if existing is not None:
            if existing.to_dict() != memory.to_dict():
                raise ValueError(f"Turn memory already committed with different content: {memory.turn_index}")
            return False

        self.max_handoff_chars = max_handoff_chars
        self.max_turn_summary_chars = max_turn_summary_chars
        self.turns.append(memory)
        self.turns.sort(key=lambda item: item.turn_index)
        self.rebuild_session_view(max_handoff_chars=max_handoff_chars)
        return True

    def exchanges_for_turn(self, turn_index: int) -> list[ExchangeMemory]:
        return sorted(
            [item for item in self.exchanges if item.turn_index == turn_index],
            key=lambda item: (item.exchange_index, item.created_at, item.memory_id),
        )

    def turn_for_index(self, turn_index: int) -> TurnMemory | None:
        return next((item for item in self.turns if item.turn_index == turn_index), None)

    def handoff_before_turn(self, turn_index: int) -> str:
        previous_turns = [
            item
            for item in self.turns
            if item.turn_index < turn_index and len(item.handoff_after_turn) <= self.max_handoff_chars
        ]
        if not previous_turns:
            return ""
        return max(previous_turns, key=lambda item: item.turn_index).handoff_after_turn

    def rebuild_session_view(
        self,
        *,
        max_handoff_chars: int = DEFAULT_MAX_HANDOFF_CHARS,
    ) -> SessionView:
        self.max_handoff_chars = max_handoff_chars
        valid_turns = [
            turn
            for turn in sorted(self.turns, key=lambda item: item.turn_index)
            if len(turn.handoff_after_turn) <= max_handoff_chars
        ]
        self.session_view = (
            SessionView.from_turn_memory(valid_turns[-1], generation=len(valid_turns))
            if valid_turns
            else SessionView.empty(thread_id=self.thread_id)
        )
        return self.session_view

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
    previous_handoff: str = "",
    controller_state: dict[str, Any] | None = None,
    max_handoff_chars: int = DEFAULT_MAX_HANDOFF_CHARS,
    max_turn_summary_chars: int = DEFAULT_MAX_TURN_SUMMARY_CHARS,
    origin: MemoryOrigin = "fallback",
) -> TurnMemory:
    turn_summary = _build_fallback_turn_summary(
        user_intent=user_intent,
        assistant_outcome=assistant_outcome,
        exchanges=exchanges,
        max_chars=max_turn_summary_chars,
    )
    handoff = _build_fallback_handoff(
        previous_handoff=previous_handoff,
        user_intent=user_intent,
        assistant_outcome=assistant_outcome,
        exchanges=exchanges,
        controller_state=controller_state,
        max_chars=max_handoff_chars,
    )
    return TurnMemory(
        memory_id=f"turn-{turn_index:04d}-memory",
        thread_id=thread_id,
        turn_index=turn_index,
        user_intent=_normalize_text(user_intent, limit=4_000),
        assistant_outcome=_normalize_text(assistant_outcome, limit=8_000),
        turn_summary=turn_summary,
        handoff_after_turn=handoff,
        origin=origin,
        degraded=True,
        exchange_memory_ids=[item.memory_id for item in exchanges],
        source_block_ids=list(dict.fromkeys(source_block_ids)),
        relevant_artifacts=_aggregate_artifact_refs(exchanges),
    )


def _build_fallback_turn_summary(
    *,
    user_intent: str,
    assistant_outcome: str,
    exchanges: list[ExchangeMemory],
    max_chars: int,
) -> str:
    action_lines = _exchange_summary_lines(exchanges, max_items=6)
    sections = [
        f"User request:\n{_normalize_text(user_intent, limit=max(200, max_chars // 4))}",
        f"Assistant outcome:\n{_normalize_text(assistant_outcome, limit=max(300, max_chars // 3))}",
    ]
    if action_lines:
        sections.append("Actions and results:\n" + "\n".join(f"- {line}" for line in action_lines))
    return _bounded_document("\n\n".join(sections), max_chars=max_chars)


def _build_fallback_handoff(
    *,
    previous_handoff: str,
    user_intent: str,
    assistant_outcome: str,
    exchanges: list[ExchangeMemory],
    controller_state: dict[str, Any] | None,
    max_chars: int,
) -> str:
    sections = [
        "Current objective:\n" + _normalize_text(user_intent, limit=max(300, max_chars // 5)),
    ]

    next_actions = _controller_text_list(controller_state, "next_actions") or _aggregate_exchange_field(
        exchanges,
        "next_actions",
    )
    if next_actions:
        sections.append(
            "Next useful action:\n"
            + "\n".join(f"- {item}" for item in next_actions[:3])
        )

    action_lines = _exchange_summary_lines(exchanges, max_items=5)
    if action_lines:
        sections.append("Actions and results from this turn:\n" + "\n".join(f"- {line}" for line in action_lines))

    outcome = _normalize_text(assistant_outcome, limit=max(300, max_chars // 5))
    if outcome:
        sections.append(f"Latest outcome:\n{outcome}")

    risks = _controller_text_list(controller_state, "risk_notes") or _aggregate_exchange_field(
        exchanges,
        "risk_notes",
    )
    if risks:
        sections.append("Important constraints or risks:\n" + "\n".join(f"- {item}" for item in risks[:3]))

    previous = _normalize_document(previous_handoff, limit=max(500, max_chars // 3))
    if previous:
        sections.append(f"Prior operational context retained for recovery:\n{previous}")

    return _bounded_document("\n\n".join(sections), max_chars=max_chars)


def _bounded_document(value: str, *, max_chars: int) -> str:
    normalized = _normalize_document(value, limit=max_chars)
    return normalized or "No operational handoff was available."


def _exchange_summary_lines(exchanges: list[ExchangeMemory], *, max_items: int) -> list[str]:
    summaries: list[str] = []
    for exchange in exchanges:
        summary = _normalize_text(exchange.summary, limit=600)
        if summary and summary.casefold() not in {item.casefold() for item in summaries}:
            summaries.append(summary)
    return summaries[-max_items:]


def _aggregate_exchange_field(exchanges: list[ExchangeMemory], field_name: str) -> list[str]:
    aggregate: list[str] = []
    seen: set[str] = set()
    for exchange in exchanges:
        values = getattr(exchange, field_name, None)
        if not isinstance(values, list):
            continue
        for item in values:
            normalized = _normalize_text(item)
            key = normalized.casefold()
            if normalized and key not in seen:
                aggregate.append(normalized)
                seen.add(key)
    return aggregate


def _aggregate_artifact_refs(exchanges: list[ExchangeMemory]) -> list[str]:
    refs: list[str] = []
    for exchange in exchanges:
        refs.extend(exchange.relevant_artifacts)
    return list(dict.fromkeys(ref for ref in refs if ref))


def _controller_text_list(controller_state: dict[str, Any] | None, key: str) -> list[str]:
    if not isinstance(controller_state, dict):
        return []
    return _normalize_text_list(controller_state.get(key))


def _positive_int(value: object, *, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


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
