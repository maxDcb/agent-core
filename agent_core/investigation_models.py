from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal


def _coerce_text_item(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in (
            "summary",
            "fact",
            "statement",
            "gap",
            "action",
            "note",
            "reason",
            "value",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value).strip()
    return str(value).strip()


def _normalize_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        text = _coerce_text_item(item)
        if text:
            normalized.append(text)
    return normalized


def _normalize_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _clamp_confidence(value: object, *, default: float = 0.5) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


@dataclass(slots=True)
class StepReflection:
    observation_summary: str
    new_facts: list[str] = field(default_factory=list)
    updated_hypotheses: list[str] = field(default_factory=list)
    rejected_hypotheses: list[str] = field(default_factory=list)
    remaining_gaps: list[str] = field(default_factory=list)
    resolved_gaps: list[str] = field(default_factory=list)
    recommended_next_actions: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    confidence: float = 0.5
    should_continue: bool = True
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_summary": self.observation_summary,
            "new_facts": self.new_facts,
            "updated_hypotheses": self.updated_hypotheses,
            "rejected_hypotheses": self.rejected_hypotheses,
            "remaining_gaps": self.remaining_gaps,
            "resolved_gaps": self.resolved_gaps,
            "recommended_next_actions": self.recommended_next_actions,
            "risk_notes": self.risk_notes,
            "confidence": self.confidence,
            "should_continue": self.should_continue,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_any(cls, payload: object) -> "StepReflection | None":
        if isinstance(payload, StepReflection):
            return payload
        if not isinstance(payload, dict):
            return None
        summary = payload.get("observation_summary")
        if not isinstance(summary, str):
            return None
        return cls(
            observation_summary=summary,
            new_facts=_normalize_str_list(payload.get("new_facts")),
            updated_hypotheses=_normalize_str_list(payload.get("updated_hypotheses")),
            rejected_hypotheses=_normalize_str_list(payload.get("rejected_hypotheses")),
            remaining_gaps=_normalize_str_list(payload.get("remaining_gaps")),
            resolved_gaps=_normalize_str_list(payload.get("resolved_gaps")),
            recommended_next_actions=_normalize_str_list(payload.get("recommended_next_actions")),
            risk_notes=_normalize_str_list(payload.get("risk_notes")),
            confidence=_clamp_confidence(payload.get("confidence")),
            should_continue=bool(payload.get("should_continue", True)),
            stop_reason=_normalize_optional_str(payload.get("stop_reason")),
        )

    @classmethod
    def create_template(cls) -> "StepReflection":
        return cls(observation_summary="")


InvestigationDecisionKind = Literal["continue", "final", "ask_user", "blocked"]


@dataclass(slots=True)
class InvestigationDecision:
    kind: InvestigationDecisionKind
    reason_summary: str
    next_action: str | None = None
    question: str | None = None
    required_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason_summary": self.reason_summary,
            "next_action": self.next_action,
            "question": self.question,
            "required_approval": self.required_approval,
        }

    @classmethod
    def from_any(cls, payload: object) -> "InvestigationDecision | None":
        if isinstance(payload, InvestigationDecision):
            return payload
        if not isinstance(payload, dict):
            return None
        kind = payload.get("kind")
        reason_summary = payload.get("reason_summary")
        if kind not in {"continue", "final", "ask_user", "blocked"} or not isinstance(reason_summary, str):
            return None
        return cls(
            kind=kind,
            reason_summary=reason_summary,
            next_action=_normalize_optional_str(payload.get("next_action")),
            question=_normalize_optional_str(payload.get("question")),
            required_approval=bool(payload.get("required_approval", False)),
        )

    @classmethod
    def create_template(cls) -> "InvestigationDecision":
        return cls(kind="continue", reason_summary="", next_action=None, question=None)


@dataclass(slots=True)
class FinalCritique:
    approved: bool
    unsupported_claims: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    scope_or_safety_issues: list[str] = field(default_factory=list)
    suggested_followup_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "unsupported_claims": self.unsupported_claims,
            "missing_evidence": self.missing_evidence,
            "scope_or_safety_issues": self.scope_or_safety_issues,
            "suggested_followup_actions": self.suggested_followup_actions,
        }

    @classmethod
    def from_any(cls, payload: object) -> "FinalCritique | None":
        if isinstance(payload, FinalCritique):
            return payload
        if not isinstance(payload, dict):
            return None
        approved = payload.get("approved")
        if not isinstance(approved, bool):
            return None
        return cls(
            approved=approved,
            unsupported_claims=_normalize_str_list(payload.get("unsupported_claims")),
            missing_evidence=_normalize_str_list(payload.get("missing_evidence")),
            scope_or_safety_issues=_normalize_str_list(payload.get("scope_or_safety_issues")),
            suggested_followup_actions=_normalize_str_list(payload.get("suggested_followup_actions")),
        )

    @classmethod
    def create_template(cls) -> "FinalCritique":
        return cls(approved=False)
