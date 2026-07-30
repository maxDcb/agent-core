from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent_core.memory.journal import (
    DEFAULT_MAX_HANDOFF_CHARS,
    DEFAULT_MAX_TURN_SUMMARY_CHARS,
    ExchangeMemory,
    TurnMemory,
)
from agent_core.structured_synthesizer import StructuredSynthesisRequest, StructuredSynthesizer
from agent_core.types import utc_now_iso

DEFAULT_TURN_MEMORY_SYNTHESIS_PROMPT = """\
Create memory for exactly one completed conversation turn and rewrite the
operational handoff for the next turn.

Use only the previous handoff, current user request, assistant outcome,
controller state, runtime context, and ordered exchange-memory events supplied
in the payload. Never reconstruct or summarize the full conversation.

The turn summary is an immutable, concise record of what happened in this
turn. The next handoff is a compact working note, not an exhaustive history.
Rewrite it completely and keep only:
- the current objective and active scope constraints;
- actions already performed when repeating them would be wasteful;
- observations that affect the next decision;
- current blockers, important risks, and the next useful action.

Remove completed details that no longer affect future work. When the objective
changes, discard obsolete operational context. Reconcile apparently
contradictory observations chronologically. Do not invent facts, actions,
evidence, artifact identifiers, or decisions.

Return exactly one JSON object matching the requested format."""


@dataclass(frozen=True, slots=True)
class MemorySynthesisResult:
    """Provider output containing only prose documents."""

    turn_summary: str
    next_handoff: str

    def to_dict(self) -> dict[str, str]:
        return {
            "turn_summary": self.turn_summary,
            "next_handoff": self.next_handoff,
        }

    @classmethod
    def from_any(cls, payload: object) -> MemorySynthesisResult | None:
        if isinstance(payload, MemorySynthesisResult):
            payload = payload.to_dict()
        if not isinstance(payload, dict):
            return None
        turn_summary = _normalize_document(payload.get("turn_summary"))
        next_handoff = _normalize_document(payload.get("next_handoff"))
        if (
            not turn_summary
            or not next_handoff
            or _is_serialized_container(turn_summary)
            or _is_serialized_container(next_handoff)
        ):
            return None
        return cls(turn_summary=turn_summary, next_handoff=next_handoff)

    @classmethod
    def create_template(cls) -> MemorySynthesisResult:
        return cls(turn_summary="", next_handoff="")


@dataclass(slots=True)
class TurnMemorySynthesisInput:
    thread_id: str
    turn_index: int
    user_intent: str
    assistant_outcome: str
    exchange_memories: list[ExchangeMemory]
    source_block_ids: list[str]
    previous_handoff: str
    runtime_context: dict[str, Any]
    controller_state: dict[str, Any] | None = None
    domain_payload: dict[str, Any] | None = None
    domain_guidance: str = ""


class TurnMemoryCommitter:
    """Create one bounded turn record and its replacement operational handoff."""

    def __init__(
        self,
        *,
        synthesizer: StructuredSynthesizer,
        instructions: str = "",
        max_input_chars: int = 64_000,
        max_handoff_chars: int = DEFAULT_MAX_HANDOFF_CHARS,
        max_turn_summary_chars: int = DEFAULT_MAX_TURN_SUMMARY_CHARS,
    ) -> None:
        self.synthesizer = synthesizer
        self.instructions = instructions.strip() or DEFAULT_TURN_MEMORY_SYNTHESIS_PROMPT
        self.max_input_chars = max_input_chars
        self.max_handoff_chars = max_handoff_chars
        self.max_turn_summary_chars = max_turn_summary_chars

    def synthesize(self, synthesis_input: TurnMemorySynthesisInput) -> TurnMemory:
        payload = {
            "thread_id": synthesis_input.thread_id,
            "turn_index": synthesis_input.turn_index,
            "runtime_context": synthesis_input.runtime_context,
            "previous_handoff": _clip_document(
                synthesis_input.previous_handoff,
                self.max_handoff_chars,
            ),
            "user_request": _clip_text(synthesis_input.user_intent, 4_000),
            "assistant_outcome": _clip_text(synthesis_input.assistant_outcome, 8_000),
            "controller_state": synthesis_input.controller_state,
            "exchange_memories": self._project_exchanges(synthesis_input.exchange_memories),
            "domain_context": synthesis_input.domain_payload or {},
            "domain_guidance": _clip_document(synthesis_input.domain_guidance, 4_000),
            "output_limits": {
                "turn_summary_max_chars": self.max_turn_summary_chars,
                "next_handoff_max_chars": self.max_handoff_chars,
            },
        }
        payload_chars = _json_chars(payload)
        if payload_chars > self.max_input_chars:
            raise ValueError(
                "TurnMemory synthesis payload exceeds the configured bound "
                f"({payload_chars} > {self.max_input_chars} characters)"
            )

        synthesized = self.synthesizer.synthesize(
            request=StructuredSynthesisRequest(
                target_name="turn_memory",
                instructions=self.instructions,
                output_format=MemorySynthesisResult.create_template().to_dict(),
                payload=payload,
                parser=MemorySynthesisResult.from_any,
            )
        )
        if len(synthesized.turn_summary) > self.max_turn_summary_chars:
            raise ValueError(
                "Turn summary exceeds the configured bound "
                f"({len(synthesized.turn_summary)} > {self.max_turn_summary_chars} characters)"
            )
        if len(synthesized.next_handoff) > self.max_handoff_chars:
            raise ValueError(
                "Turn handoff exceeds the configured bound "
                f"({len(synthesized.next_handoff)} > {self.max_handoff_chars} characters)"
            )

        return TurnMemory(
            memory_id=f"turn-{synthesis_input.turn_index:04d}-memory",
            thread_id=synthesis_input.thread_id,
            turn_index=synthesis_input.turn_index,
            user_intent=_clip_text(synthesis_input.user_intent, 4_000),
            assistant_outcome=_clip_text(synthesis_input.assistant_outcome, 8_000),
            turn_summary=synthesized.turn_summary,
            handoff_after_turn=synthesized.next_handoff,
            created_at=utc_now_iso(),
            origin="model",
            degraded=False,
            exchange_memory_ids=[item.memory_id for item in synthesis_input.exchange_memories],
            source_block_ids=list(dict.fromkeys(synthesis_input.source_block_ids)),
            relevant_artifacts=_artifact_refs(synthesis_input.exchange_memories),
        )

    def _project_exchanges(self, exchanges: list[ExchangeMemory]) -> list[dict[str, Any]]:
        if not exchanges:
            return []
        per_exchange_chars = max(300, int(self.max_input_chars * 0.65) // len(exchanges))
        return [_project_exchange(memory, max_chars=per_exchange_chars) for memory in exchanges]


def _project_exchange(memory: ExchangeMemory, *, max_chars: int) -> dict[str, Any]:
    payload = {
        "memory_id": memory.memory_id,
        "exchange_index": memory.exchange_index,
        "kind": memory.kind,
        "summary": memory.summary,
        "confirmed_facts": memory.confirmed_facts,
        "open_hypotheses": memory.open_hypotheses,
        "rejected_hypotheses": memory.rejected_hypotheses,
        "open_questions": memory.open_questions,
        "resolved_questions": memory.resolved_questions,
        "decisions": memory.decisions,
        "completed_actions": memory.completed_actions,
        "next_actions": memory.next_actions,
        "relevant_artifacts": memory.relevant_artifacts,
        "risk_notes": memory.risk_notes,
        "confidence": memory.confidence,
    }
    if _json_chars(payload) <= max_chars:
        return payload

    compact = {
        "memory_id": memory.memory_id,
        "exchange_index": memory.exchange_index,
        "kind": memory.kind,
        "summary": _clip_text(memory.summary, max(200, max_chars // 2)),
        "completed_actions": _clip_items(memory.completed_actions, max_chars=max_chars // 8),
        "next_actions": _clip_items(memory.next_actions, max_chars=max_chars // 8),
        "relevant_artifacts": list(memory.relevant_artifacts),
        "risk_notes": _clip_items(memory.risk_notes, max_chars=max_chars // 8),
    }
    if _json_chars(compact) <= max_chars:
        return compact

    return {
        "memory_id": memory.memory_id,
        "exchange_index": memory.exchange_index,
        "kind": memory.kind,
        "summary": _clip_text(memory.summary, max(100, max_chars - 200)),
    }


def _artifact_refs(exchanges: list[ExchangeMemory]) -> list[str]:
    refs: list[str] = []
    for exchange in exchanges:
        refs.extend(exchange.relevant_artifacts)
    return list(dict.fromkeys(ref for ref in refs if ref))


def _clip_items(items: list[str], *, max_chars: int) -> list[str]:
    projected: list[str] = []
    used = 0
    for item in items:
        remaining = max_chars - used
        if remaining <= 0:
            break
        clipped = _clip_text(item, min(1_000, remaining))
        if not clipped:
            continue
        projected.append(clipped)
        used += len(clipped)
    return projected


def _clip_text(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 1)].rstrip()}…"


def _clip_document(value: str, limit: int) -> str:
    normalized = _normalize_document(value)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 1)].rstrip()}…"


def _normalize_document(value: object) -> str:
    if not isinstance(value, str):
        return ""
    lines = [
        line.rstrip()
        for line in value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").splitlines()
    ]
    return "\n".join(lines).strip()


def _is_serialized_container(value: str) -> bool:
    """Reject raw JSON documents accidentally wrapped in a prose field."""

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, (dict, list))


def _json_chars(payload: object) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
