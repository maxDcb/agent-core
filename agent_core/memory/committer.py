from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from agent_core.memory.journal import (
    ActiveTask,
    ExchangeMemory,
    TurnMemory,
    build_fallback_turn_memory,
)
from agent_core.structured_synthesizer import StructuredSynthesisRequest, StructuredSynthesizer

DEFAULT_TURN_MEMORY_SYNTHESIS_PROMPT = """\
Synthesize a memory delta for exactly one completed conversation turn.

Use only the supplied user request, assistant outcome, controller state, and
ordered exchange-memory events. Do not use or reconstruct the full
conversation. Do not invent facts, decisions, evidence, artifact identifiers,
or completed actions.

Keep durable confirmed facts separate from hypotheses. Put corrected prior
facts in superseded_facts, disproven hypotheses in rejected_hypotheses, and
answered questions in resolved_questions. Keep active_task concise and useful
for the next turn. Preserve exact artifact identifiers.

Return exactly one JSON object matching the requested format."""


@dataclass(slots=True)
class TurnMemorySynthesisInput:
    thread_id: str
    turn_index: int
    user_intent: str
    assistant_outcome: str
    exchange_memories: list[ExchangeMemory]
    source_block_ids: list[str]
    previous_active_task: ActiveTask | None
    runtime_context: dict[str, Any]
    controller_state: dict[str, Any] | None = None
    domain_payload: dict[str, Any] | None = None
    domain_extensions_template: dict[str, Any] | None = None


class TurnMemoryCommitter:
    """Create one bounded TurnMemory delta without revisiting old turns."""

    def __init__(
        self,
        *,
        synthesizer: StructuredSynthesizer,
        instructions: str = "",
        max_input_chars: int = 64_000,
    ) -> None:
        self.synthesizer = synthesizer
        self.instructions = instructions.strip() or DEFAULT_TURN_MEMORY_SYNTHESIS_PROMPT
        self.max_input_chars = max_input_chars

    def synthesize(self, synthesis_input: TurnMemorySynthesisInput) -> TurnMemory:
        memory_id = f"turn-{synthesis_input.turn_index:04d}-memory"
        objective = (
            synthesis_input.previous_active_task.objective
            if synthesis_input.previous_active_task is not None
            else synthesis_input.user_intent
        )
        template = TurnMemory.create_template(
            memory_id=memory_id,
            thread_id=synthesis_input.thread_id,
            turn_index=synthesis_input.turn_index,
            objective=objective,
        )
        template.domain_extensions = dict(synthesis_input.domain_extensions_template or {})

        payload = {
            "thread_id": synthesis_input.thread_id,
            "turn_index": synthesis_input.turn_index,
            "runtime_context": synthesis_input.runtime_context,
            "previous_active_task": (
                synthesis_input.previous_active_task.to_dict()
                if synthesis_input.previous_active_task is not None
                else None
            ),
            "user_request": _clip_text(synthesis_input.user_intent, 4_000),
            "assistant_outcome": _clip_text(synthesis_input.assistant_outcome, 8_000),
            "controller_state": synthesis_input.controller_state,
            "exchange_memories": self._project_exchanges(synthesis_input.exchange_memories),
            "domain_context": synthesis_input.domain_payload or {},
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
                output_format=template.to_dict(),
                payload=payload,
                parser=TurnMemory.from_any,
            )
        )
        synthesized = synthesized.with_runtime_identity(
            memory_id=memory_id,
            thread_id=synthesis_input.thread_id,
            turn_index=synthesis_input.turn_index,
            exchange_memory_ids=[item.memory_id for item in synthesis_input.exchange_memories],
            source_block_ids=synthesis_input.source_block_ids,
            origin="model",
        )
        baseline = build_fallback_turn_memory(
            thread_id=synthesis_input.thread_id,
            turn_index=synthesis_input.turn_index,
            user_intent=synthesis_input.user_intent,
            assistant_outcome=synthesis_input.assistant_outcome,
            exchanges=synthesis_input.exchange_memories,
            source_block_ids=synthesis_input.source_block_ids,
            previous_active_task=synthesis_input.previous_active_task,
            origin="model",
        )
        return replace(
            synthesized,
            user_intent=synthesized.user_intent or baseline.user_intent,
            assistant_outcome=synthesized.assistant_outcome or baseline.assistant_outcome,
            confirmed_facts=_merge_unique(baseline.confirmed_facts, synthesized.confirmed_facts),
            open_hypotheses=_merge_unique(baseline.open_hypotheses, synthesized.open_hypotheses),
            rejected_hypotheses=_merge_unique(
                baseline.rejected_hypotheses,
                synthesized.rejected_hypotheses,
            ),
            open_questions=_merge_unique(baseline.open_questions, synthesized.open_questions),
            resolved_questions=_merge_unique(
                baseline.resolved_questions,
                synthesized.resolved_questions,
            ),
            decisions=_merge_unique(baseline.decisions, synthesized.decisions),
            completed_actions=_merge_unique(
                baseline.completed_actions,
                synthesized.completed_actions,
            ),
            next_actions=_merge_unique(baseline.next_actions, synthesized.next_actions),
            relevant_artifacts=_merge_unique(
                baseline.relevant_artifacts,
                synthesized.relevant_artifacts,
            ),
            risk_notes=_merge_unique(baseline.risk_notes, synthesized.risk_notes),
            domain_extensions={
                **baseline.domain_extensions,
                **synthesized.domain_extensions,
            },
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
        "domain_extensions": memory.domain_extensions,
    }
    if _json_chars(payload) <= max_chars:
        return payload

    compact = {
        "memory_id": memory.memory_id,
        "exchange_index": memory.exchange_index,
        "kind": memory.kind,
        "summary": _clip_text(memory.summary, max(200, max_chars // 3)),
        "confirmed_facts": _clip_items(memory.confirmed_facts, max_chars=max_chars // 5),
        "open_questions": _clip_items(memory.open_questions, max_chars=max_chars // 8),
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


def _json_chars(payload: object) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _merge_unique(existing: list[str], additions: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*existing, *additions]:
        key = item.casefold()
        if not item or key in seen:
            continue
        merged.append(item)
        seen.add(key)
    return merged
