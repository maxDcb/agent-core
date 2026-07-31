from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent_core.logging_utils import get_logger
from agent_core.memory.journal import (
    DEFAULT_MAX_HANDOFF_CHARS,
    DEFAULT_MAX_TURN_SUMMARY_CHARS,
    ExchangeMemory,
    TurnMemory,
)
from agent_core.structured_synthesizer import StructuredSynthesisRequest, StructuredSynthesizer
from agent_core.types import utc_now_iso

logger = get_logger(__name__)

DEFAULT_TURN_MEMORY_SYNTHESIS_PROMPT = """\
Create memory for exactly one completed conversation turn and rewrite the
operational handoff for the next turn.

Use only the previous handoff, current user request, assistant outcome,
controller state, runtime context, and ordered exchange-memory events supplied
in the payload. Never reconstruct or summarize the full conversation.

The turn summary is an immutable, concise record of what happened in this
turn. The next handoff is a compact working note, not an exhaustive history.

Treat the previous handoff as the operational memory to update and the current
turn as a delta. Rewrite the output document, but update its meaning
conservatively: do not drop a previous fact merely because the current turn
does not repeat it. Keep a previous item while the objective is still current
and the item:
- changes which next actions are valid or promising;
- rules an approach in or out, records an unresolved contradiction, or
  explains a prior failure;
- captures a scope, safety, parsing, validation, protocol, or runtime
  constraint that still applies;
- prevents repeating completed work; or
- is required to interpret the next action or retrieve supporting evidence.

Drop a previous item only when the objective has changed enough to make it
irrelevant, or when current evidence explicitly resolves, supersedes, or
invalidates it. Silence in the current turn is not evidence that a fact became
irrelevant. When evidence conflicts, preserve the conflict and its provenance
until it is resolved instead of silently replacing either side.

Keep only the operationally useful subset:
- the current objective and active scope constraints;
- actions already performed when repeating them would be wasteful;
- established facts and unresolved observations that affect the next
  decision, with concise provenance when source, simulation, and live
  observation differ;
- current blockers, important risks, and the next useful action.

Remove completed details that no longer affect future work. When the objective
changes, discard obsolete operational context. Before returning, ensure that a
fresh model given only the next handoff could avoid repeating known failures
and choose the next action without losing a still-relevant constraint. Do not
invent facts, actions, evidence, artifact identifiers, or decisions.

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
        controller_state = _project_json_mapping(
            synthesis_input.controller_state,
            max_chars=max(512, int(self.max_input_chars * 0.25)),
            priority_keys=(
                "objective",
                "scope",
                "constraints",
                "facts",
                "confirmed_facts",
                "evidence",
                "hypotheses",
                "open_hypotheses",
                "rejected_hypotheses",
                "completed_actions",
                "evidence_gaps",
                "open_questions",
                "next_actions",
                "risk_notes",
                "stop_reason",
                "confidence",
            ),
        )
        runtime_context = _project_json_mapping(
            synthesis_input.runtime_context,
            max_chars=max(256, int(self.max_input_chars * 0.05)),
        )
        domain_context = _project_json_mapping(
            synthesis_input.domain_payload,
            max_chars=max(256, int(self.max_input_chars * 0.05)),
        )
        payload: dict[str, Any] = {
            "thread_id": synthesis_input.thread_id,
            "turn_index": synthesis_input.turn_index,
            "runtime_context": runtime_context,
            "previous_handoff": _clip_document(
                synthesis_input.previous_handoff,
                self.max_handoff_chars,
            ),
            "user_request": _clip_text(synthesis_input.user_intent, 4_000),
            "assistant_outcome": _clip_text(synthesis_input.assistant_outcome, 8_000),
            "controller_state": controller_state,
            "exchange_memories": [],
            "domain_context": domain_context,
            "domain_guidance": _clip_document(synthesis_input.domain_guidance, 4_000),
            "output_limits": {
                "turn_summary_max_chars": self.max_turn_summary_chars,
                "next_handoff_max_chars": self.max_handoff_chars,
            },
        }
        fixed_payload_chars = _json_chars(payload)
        exchange_budget = self.max_input_chars - fixed_payload_chars
        if synthesis_input.exchange_memories and exchange_budget > 2:
            payload["exchange_memories"] = self._project_exchanges(
                synthesis_input.exchange_memories,
                max_chars=exchange_budget,
            )
        payload_chars = _json_chars(payload)
        if payload_chars > self.max_input_chars:
            raise ValueError(
                "TurnMemory synthesis payload exceeds the configured bound "
                f"({payload_chars} > {self.max_input_chars} characters)"
            )
        original_variable_chars = _json_chars(
            {
                "runtime_context": synthesis_input.runtime_context,
                "controller_state": synthesis_input.controller_state,
                "exchange_memories": [item.to_dict() for item in synthesis_input.exchange_memories],
                "domain_context": synthesis_input.domain_payload or {},
            }
        )
        projected_variable_chars = _json_chars(
            {
                "runtime_context": payload["runtime_context"],
                "controller_state": payload["controller_state"],
                "exchange_memories": payload["exchange_memories"],
                "domain_context": payload["domain_context"],
            }
        )
        if projected_variable_chars < original_variable_chars:
            logger.debug(
                "Projected TurnMemory synthesis input to the configured bound",
                extra={
                    "turn_index": synthesis_input.turn_index,
                    "original_variable_chars": original_variable_chars,
                    "projected_variable_chars": projected_variable_chars,
                    "payload_chars": payload_chars,
                    "max_input_chars": self.max_input_chars,
                    "exchange_count": len(synthesis_input.exchange_memories),
                    "projected_exchange_count": len(payload["exchange_memories"]),
                },
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

    def _project_exchanges(
        self,
        exchanges: list[ExchangeMemory],
        *,
        max_chars: int,
    ) -> list[dict[str, Any]]:
        if not exchanges or max_chars <= 2:
            return []
        available_chars = max_chars - 2
        minimum_exchange_chars = 220
        max_exchange_count = max(1, available_chars // minimum_exchange_chars)
        selected = _select_exchanges(exchanges, max_items=max_exchange_count)
        separator_chars = max(0, len(selected) - 1)
        per_exchange_chars = max(
            64,
            (available_chars - separator_chars) // len(selected),
        )
        projected = [
            _project_exchange(memory, max_chars=per_exchange_chars)
            for memory in selected
        ]
        while projected and _json_chars(projected) > max_chars:
            projected.pop(0)
        return projected


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

    compact: dict[str, Any] = {
        "memory_id": memory.memory_id,
        "exchange_index": memory.exchange_index,
        "kind": memory.kind,
    }
    if _json_chars(compact) > max_chars:
        return {
            "memory_id": _clip_json_string(memory.memory_id, max(8, max_chars - 70)),
            "exchange_index": memory.exchange_index,
            "kind": memory.kind,
        }

    artifact_budget = max(32, max_chars // 6)
    artifacts = _project_json_list(memory.relevant_artifacts, max_chars=artifact_budget)
    if artifacts:
        compact["relevant_artifacts"] = artifacts

    remaining_chars = max(0, max_chars - _json_chars(compact))
    summary = _clip_json_string(memory.summary, max(16, int(remaining_chars * 0.45)))
    if summary:
        compact["summary"] = summary

    remaining_chars = max(0, max_chars - _json_chars(compact))
    operational = _project_json_mapping(
        {
            "confirmed_facts": memory.confirmed_facts,
            "open_hypotheses": memory.open_hypotheses,
            "rejected_hypotheses": memory.rejected_hypotheses,
            "decisions": memory.decisions,
            "completed_actions": memory.completed_actions,
            "next_actions": memory.next_actions,
            "risk_notes": memory.risk_notes,
            "confidence": memory.confidence,
        },
        max_chars=remaining_chars,
        priority_keys=(
            "confirmed_facts",
            "decisions",
            "open_hypotheses",
            "rejected_hypotheses",
            "completed_actions",
            "next_actions",
            "risk_notes",
            "confidence",
        ),
    )
    for key, value in operational.items():
        candidate = {**compact, key: value}
        if _json_chars(candidate) <= max_chars:
            compact[key] = value
    if _json_chars(compact) <= max_chars:
        return compact

    minimal = {
        "memory_id": memory.memory_id,
        "exchange_index": memory.exchange_index,
        "kind": memory.kind,
    }
    summary_budget = max(0, max_chars - _json_chars(minimal) - 12)
    if summary_budget:
        minimal["summary"] = _clip_json_string(memory.summary, summary_budget)
    return minimal


def _select_exchanges(
    exchanges: list[ExchangeMemory],
    *,
    max_items: int,
) -> list[ExchangeMemory]:
    if len(exchanges) <= max_items:
        return list(exchanges)
    ranked = sorted(
        enumerate(exchanges),
        key=lambda item: (
            item[1].kind == "final_response",
            bool(item[1].relevant_artifacts),
            item[1].kind == "reflection",
            item[1].exchange_index,
        ),
        reverse=True,
    )
    selected_indexes = {index for index, _ in ranked[:max_items]}
    selected_indexes.add(len(exchanges) - 1)
    if len(selected_indexes) > max_items:
        selected_indexes.remove(min(selected_indexes))
    return [memory for index, memory in enumerate(exchanges) if index in selected_indexes]


def _project_json_mapping(
    value: dict[str, Any] | None,
    *,
    max_chars: int,
    priority_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not isinstance(value, dict) or max_chars < 2:
        return {}
    if _json_chars(value) <= max_chars:
        return dict(value)

    priority = {key: index for index, key in enumerate(priority_keys)}
    ordered_keys = sorted(
        value,
        key=lambda key: (priority.get(key, len(priority)), list(value).index(key)),
    )
    projected: dict[str, Any] = {}
    for index, key in enumerate(ordered_keys):
        remaining = max_chars - _json_chars(projected)
        if remaining <= len(json.dumps(key, ensure_ascii=False)) + 4:
            break
        remaining_keys = max(1, len(ordered_keys) - index)
        value_budget = max(16, remaining // remaining_keys)
        projected_value = _project_json_value(value[key], max_chars=value_budget)
        candidate = {**projected, key: projected_value}
        if _json_chars(candidate) <= max_chars:
            projected[key] = projected_value
    return projected


def _project_json_value(value: Any, *, max_chars: int) -> Any:
    if max_chars <= 0:
        return None
    try:
        if _json_chars(value) <= max_chars:
            return value
    except (TypeError, ValueError):
        value = str(value)
    if isinstance(value, str):
        return _clip_json_string(value, max_chars)
    if isinstance(value, list):
        return _project_json_list(value, max_chars=max_chars)
    if isinstance(value, dict):
        return _project_json_mapping(value, max_chars=max_chars)
    return value if _json_chars(value) <= max_chars else None


def _project_json_list(items: list[Any], *, max_chars: int) -> list[Any]:
    if max_chars < 2 or not items:
        return []
    if _json_chars(items) <= max_chars:
        return list(items)

    max_items = min(len(items), max(1, max_chars // 96))
    head_count = (max_items + 1) // 2
    tail_count = max_items - head_count
    indexes = list(range(head_count))
    if tail_count:
        indexes.extend(range(len(items) - tail_count, len(items)))
    selected = [items[index] for index in sorted(set(indexes))]
    separator_chars = max(0, len(selected) - 1)
    per_item_chars = max(4, (max_chars - 2 - separator_chars) // len(selected))
    projected = [
        _project_json_value(item, max_chars=per_item_chars)
        for item in selected
    ]
    while projected and _json_chars(projected) > max_chars:
        projected.pop(len(projected) // 2)
    return projected


def _clip_json_string(value: str, max_chars: int) -> str:
    if max_chars < 2:
        return ""
    if _json_chars(value) <= max_chars:
        return value
    low = 0
    high = len(value)
    clipped = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = _clip_text(value, midpoint)
        if _json_chars(candidate) <= max_chars:
            clipped = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return clipped


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
