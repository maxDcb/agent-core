from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _coerce_str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in value:
        candidate = str(item).strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def _normalize_tool_names(value: object) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for name in _normalize_str_list(value):
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _coerce_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    if isinstance(value, str):
        lowered = value.strip().lower()
        mapping = {"low": 0.25, "medium": 0.5, "high": 0.85}
        if lowered in mapping:
            return mapping[lowered]
        try:
            return max(0.0, min(float(lowered), 1.0))
        except ValueError:
            return 0.0
    return 0.0


def _normalize_artifacts(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(slots=True)
class SpecialistProfile:
    profile_id: str
    system_prompt: str
    description: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
    max_tool_calls: int = 8
    max_iterations: int = 6

    def __post_init__(self) -> None:
        self.profile_id = _coerce_str(self.profile_id)
        self.system_prompt = _coerce_str(self.system_prompt)
        self.description = _coerce_str(self.description)
        self.allowed_tools = _normalize_tool_names(self.allowed_tools)
        if self.model is not None:
            self.model = _coerce_str(self.model) or None
        if self.temperature is not None:
            self.temperature = float(self.temperature)
        self.max_tool_calls = max(1, int(self.max_tool_calls))
        self.max_iterations = max(1, int(self.max_iterations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "allowed_tools": list(self.allowed_tools),
            "model": self.model,
            "temperature": self.temperature,
            "max_tool_calls": self.max_tool_calls,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SpecialistProfile | None":
        if not isinstance(payload, dict):
            return None

        profile_id = _coerce_str(payload.get("profile_id"))
        system_prompt = _coerce_str(payload.get("system_prompt"))
        if not profile_id or not system_prompt:
            return None

        return cls(
            profile_id=profile_id,
            system_prompt=system_prompt,
            description=_coerce_str(payload.get("description")),
            allowed_tools=_normalize_tool_names(payload.get("allowed_tools")),
            model=_coerce_str(payload.get("model")) or None,
            temperature=payload.get("temperature") if isinstance(payload.get("temperature"), (int, float)) else None,
            max_tool_calls=payload.get("max_tool_calls", 8) if isinstance(payload.get("max_tool_calls"), int) else 8,
            max_iterations=payload.get("max_iterations", 6) if isinstance(payload.get("max_iterations"), int) else 6,
        )


@dataclass(slots=True)
class SpecialistRunRequest:
    profile_id: str
    objective: str
    context: str = ""
    constraints: list[str] = field(default_factory=list)
    target: str = ""

    def __post_init__(self) -> None:
        self.profile_id = _coerce_str(self.profile_id)
        self.objective = _coerce_str(self.objective)
        self.context = _coerce_str(self.context)
        self.constraints = _normalize_str_list(self.constraints)
        self.target = _coerce_str(self.target)

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "objective": self.objective,
            "target": self.target,
            "context": self.context,
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SpecialistRunRequest | None":
        if not isinstance(payload, dict):
            return None

        profile_id = _coerce_str(payload.get("profile_id"))
        objective = _coerce_str(payload.get("objective"))
        if not profile_id or not objective:
            return None

        return cls(
            profile_id=profile_id,
            objective=objective,
            context=_coerce_str(payload.get("context")),
            constraints=_normalize_str_list(payload.get("constraints")),
            target=_coerce_str(payload.get("target")),
        )


@dataclass(slots=True)
class SpecialistOutput:
    summary: str
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    recommended_next_action: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.summary = _coerce_str(self.summary)
        self.confidence = _coerce_confidence(self.confidence)
        self.evidence = _normalize_str_list(self.evidence)
        self.findings = _normalize_str_list(self.findings)
        self.recommended_next_action = _coerce_str(self.recommended_next_action)
        self.artifacts = _normalize_artifacts(self.artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "findings": list(self.findings),
            "recommended_next_action": self.recommended_next_action,
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SpecialistOutput | None":
        if not isinstance(payload, dict):
            return None

        summary = _coerce_str(payload.get("summary"))
        if not summary:
            return None

        return cls(
            summary=summary,
            confidence=payload.get("confidence", 0.0),
            evidence=_normalize_str_list(payload.get("evidence")),
            findings=_normalize_str_list(payload.get("findings")),
            recommended_next_action=_coerce_str(payload.get("recommended_next_action")),
            artifacts=_normalize_artifacts(payload.get("artifacts")),
        )

    @classmethod
    def create_template(cls) -> dict[str, Any]:
        return cls(
            summary="Short grounded conclusion.",
            confidence=0.0,
            evidence=["Observed proof or concrete evidence."],
            findings=["Confirmed finding or notable result."],
            recommended_next_action="Most useful next step.",
            artifacts={
                "payloads": [],
                "exploit_request": "",
                "script": "",
                "notes": [],
            },
        ).to_dict()


@dataclass(slots=True)
class SpecialistRunResult:
    ok: bool
    profile_id: str
    output: SpecialistOutput | None = None
    raw_content: str = ""
    failure_reason: str = ""
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tool_calls_used: int = 0

    def __post_init__(self) -> None:
        self.profile_id = _coerce_str(self.profile_id)
        self.raw_content = _coerce_str(self.raw_content)
        self.failure_reason = _coerce_str(self.failure_reason)
        self.iterations = max(0, int(self.iterations))
        self.tool_calls_used = max(0, int(self.tool_calls_used))
        self.tool_history = [dict(item) for item in self.tool_history if isinstance(item, dict)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "profile_id": self.profile_id,
            "output": self.output.to_dict() if self.output is not None else None,
            "raw_content": self.raw_content,
            "failure_reason": self.failure_reason,
            "tool_history": [dict(item) for item in self.tool_history],
            "iterations": self.iterations,
            "tool_calls_used": self.tool_calls_used,
        }
