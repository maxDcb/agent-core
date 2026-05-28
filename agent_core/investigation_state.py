from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_core.investigation_models import FinalCritique, StepReflection


def _normalize_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = ""
            for key in ("summary", "fact", "statement", "gap", "action", "note", "reason", "value"):
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate.strip()
                    break
            if not text:
                text = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _clamp_confidence(value: object, *, default: float = 0.5) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def _merge_unique(existing: list[str], additions: list[str]) -> list[str]:
    seen = {item for item in existing if item}
    merged = list(existing)
    for item in additions:
        normalized = item.strip()
        if normalized and normalized not in seen:
            merged.append(normalized)
            seen.add(normalized)
    return merged


def _remove_resolved(existing: list[str], resolved: list[str]) -> list[str]:
    resolved_set = {item.strip() for item in resolved if item.strip()}
    if not resolved_set:
        return existing
    return [item for item in existing if item.strip() not in resolved_set]


@dataclass(slots=True)
class EvidenceItem:
    id: str
    source: str
    summary: str
    confidence: float = 0.5
    artifact_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "summary": self.summary,
            "confidence": self.confidence,
            "artifact_ref": self.artifact_ref,
            "metadata": self.metadata,
        }

    @classmethod
    def from_any(cls, payload: object) -> "EvidenceItem | None":
        if isinstance(payload, EvidenceItem):
            return payload
        if not isinstance(payload, dict):
            return None
        item_id = payload.get("id")
        source = payload.get("source")
        summary = payload.get("summary")
        if not isinstance(item_id, str) or not isinstance(source, str) or not isinstance(summary, str):
            return None
        metadata = payload.get("metadata")
        return cls(
            id=item_id,
            source=source,
            summary=summary,
            confidence=_clamp_confidence(payload.get("confidence")),
            artifact_ref=_normalize_optional_str(payload.get("artifact_ref")),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )


HypothesisStatus = Literal["open", "supported", "rejected"]


@dataclass(slots=True)
class Hypothesis:
    id: str
    statement: str
    status: HypothesisStatus = "open"
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "status": self.status,
            "evidence_refs": self.evidence_refs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_any(cls, payload: object) -> "Hypothesis | None":
        if isinstance(payload, Hypothesis):
            return payload
        if not isinstance(payload, dict):
            return None
        item_id = payload.get("id")
        statement = payload.get("statement")
        status = payload.get("status", "open")
        if not isinstance(item_id, str) or not isinstance(statement, str):
            return None
        if status not in {"open", "supported", "rejected"}:
            status = "open"
        metadata = payload.get("metadata")
        return cls(
            id=item_id,
            statement=statement,
            status=status,
            evidence_refs=_normalize_str_list(payload.get("evidence_refs")),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )


@dataclass(slots=True)
class InvestigationState:
    objective: str
    plan: list[str] = field(default_factory=list)
    facts: list[EvidenceItem] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    completed_actions: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "plan": self.plan,
            "facts": [fact.to_dict() for fact in self.facts],
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "evidence_gaps": self.evidence_gaps,
            "completed_actions": self.completed_actions,
            "next_actions": self.next_actions,
            "risk_notes": self.risk_notes,
            "confidence": self.confidence,
            "stop_reason": self.stop_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_any(cls, payload: object) -> "InvestigationState | None":
        if isinstance(payload, InvestigationState):
            return payload
        if not isinstance(payload, dict):
            return None
        objective = payload.get("objective")
        if not isinstance(objective, str):
            return None
        raw_facts = payload.get("facts", [])
        raw_hypotheses = payload.get("hypotheses", [])
        metadata = payload.get("metadata")
        return cls(
            objective=objective,
            plan=_normalize_str_list(payload.get("plan")),
            facts=[
                fact
                for item in raw_facts
                if (fact := EvidenceItem.from_any(item)) is not None
            ]
            if isinstance(raw_facts, list)
            else [],
            hypotheses=[
                hypothesis
                for item in raw_hypotheses
                if (hypothesis := Hypothesis.from_any(item)) is not None
            ]
            if isinstance(raw_hypotheses, list)
            else [],
            evidence_gaps=_normalize_str_list(payload.get("evidence_gaps")),
            completed_actions=_normalize_str_list(payload.get("completed_actions")),
            next_actions=_normalize_str_list(payload.get("next_actions")),
            risk_notes=_normalize_str_list(payload.get("risk_notes")),
            confidence=_clamp_confidence(payload.get("confidence"), default=0.0),
            stop_reason=_normalize_optional_str(payload.get("stop_reason")),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )

    @classmethod
    def create_template(cls, *, objective: str = "") -> "InvestigationState":
        return cls(objective=objective, confidence=0.0)

    def apply_reflection(self, reflection: StepReflection | dict[str, Any]) -> "InvestigationState":
        parsed = StepReflection.from_any(reflection)
        if parsed is None:
            raise ValueError("Invalid step reflection")

        if parsed.observation_summary:
            self.completed_actions = _merge_unique(self.completed_actions, [parsed.observation_summary])

        for fact_summary in parsed.new_facts:
            if any(fact.summary == fact_summary for fact in self.facts):
                continue
            fact_index = len(self.facts) + 1
            self.facts.append(
                EvidenceItem(
                    id=f"fact-{fact_index:03d}",
                    source="step_reflection",
                    summary=fact_summary,
                    confidence=parsed.confidence,
                )
            )

        self._apply_supported_hypotheses(parsed.updated_hypotheses)
        self._apply_rejected_hypotheses(parsed.rejected_hypotheses)
        self.evidence_gaps = _remove_resolved(
            _merge_unique(self.evidence_gaps, parsed.remaining_gaps),
            parsed.resolved_gaps,
        )
        self.next_actions = _merge_unique([], parsed.recommended_next_actions)
        self.risk_notes = _merge_unique(self.risk_notes, parsed.risk_notes)
        self.confidence = parsed.confidence
        if parsed.stop_reason:
            self.stop_reason = parsed.stop_reason
        return self

    def apply_critique(self, critique: FinalCritique | dict[str, Any]) -> "InvestigationState":
        parsed = FinalCritique.from_any(critique)
        if parsed is None:
            raise ValueError("Invalid final critique")
        self.evidence_gaps = _merge_unique(self.evidence_gaps, parsed.unsupported_claims + parsed.missing_evidence)
        self.risk_notes = _merge_unique(self.risk_notes, parsed.scope_or_safety_issues)
        self.next_actions = _merge_unique(self.next_actions, parsed.suggested_followup_actions)
        if not parsed.approved and not self.stop_reason:
            self.stop_reason = "final_critique_rejected"
        return self

    def progress_fingerprint(self) -> str:
        payload = {
            "facts": [(fact.summary, round(fact.confidence, 3)) for fact in self.facts],
            "hypotheses": [(hypothesis.statement, hypothesis.status) for hypothesis in self.hypotheses],
            "evidence_gaps": self.evidence_gaps,
            "completed_actions": self.completed_actions,
            "next_actions": self.next_actions,
            "risk_notes": self.risk_notes,
            "confidence": round(self.confidence, 3),
            "stop_reason": self.stop_reason,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def compact_summary(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "facts": [fact.summary for fact in self.facts],
            "hypotheses": [
                {"statement": hypothesis.statement, "status": hypothesis.status}
                for hypothesis in self.hypotheses
            ],
            "evidence_gaps": self.evidence_gaps,
            "completed_actions": self.completed_actions,
            "next_actions": self.next_actions,
            "risk_notes": self.risk_notes,
            "confidence": self.confidence,
            "stop_reason": self.stop_reason,
        }

    def _apply_supported_hypotheses(self, statements: list[str]) -> None:
        for statement in statements:
            existing = self._find_hypothesis(statement)
            if existing is None:
                self.hypotheses.append(
                    Hypothesis(
                        id=f"hypothesis-{len(self.hypotheses) + 1:03d}",
                        statement=statement,
                        status="supported",
                    )
                )
            else:
                existing.status = "supported"

    def _apply_rejected_hypotheses(self, statements: list[str]) -> None:
        for statement in statements:
            existing = self._find_hypothesis(statement)
            if existing is None:
                self.hypotheses.append(
                    Hypothesis(
                        id=f"hypothesis-{len(self.hypotheses) + 1:03d}",
                        statement=statement,
                        status="rejected",
                    )
                )
            else:
                existing.status = "rejected"

    def _find_hypothesis(self, statement: str) -> Hypothesis | None:
        normalized = statement.strip().lower()
        for hypothesis in self.hypotheses:
            if hypothesis.statement.strip().lower() == normalized:
                return hypothesis
        return None
