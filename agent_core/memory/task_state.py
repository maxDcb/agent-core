from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.memory.context_block import ContextBlock, estimate_token_count


def _normalize_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _merge_unique(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            merged.append(normalized)
            seen.add(normalized)
    return merged


@dataclass(slots=True)
class TaskState:
    run_id: str
    objective: str
    scope: list[str] = field(default_factory=list)
    source_code_locations: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_action: str | None = None
    stop_conditions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    relevant_artifacts: list[str] = field(default_factory=list)
    status: str = "active"
    domain_extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "scope": self.scope,
            "source_code_locations": self.source_code_locations,
            "open_questions": self.open_questions,
            "next_action": self.next_action,
            "stop_conditions": self.stop_conditions,
            "constraints": self.constraints,
            "relevant_artifacts": self.relevant_artifacts,
            "status": self.status,
            "domain_extensions": self.domain_extensions,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "TaskState | None":
        if not isinstance(payload, dict):
            return None
        objective = payload.get("objective")
        run_id = payload.get("run_id")
        if not isinstance(objective, str) or not isinstance(run_id, str):
            return None
        domain_ext = payload.get("domain_extensions")
        return cls(
            run_id=run_id,
            objective=objective,
            scope=_normalize_str_list(payload.get("scope")),
            source_code_locations=_normalize_str_list(payload.get("source_code_locations")),
            open_questions=_normalize_str_list(payload.get("open_questions")),
            next_action=_normalize_optional_str(payload.get("next_action")),
            stop_conditions=_normalize_str_list(payload.get("stop_conditions")),
            constraints=_normalize_str_list(payload.get("constraints")),
            relevant_artifacts=_normalize_str_list(payload.get("relevant_artifacts")),
            status=payload.get("status") if isinstance(payload.get("status"), str) else "active",
            domain_extensions=dict(domain_ext) if isinstance(domain_ext, dict) else {},
        )

    @classmethod
    def from_any(cls, payload: object) -> "TaskState | None":
        if payload is None:
            return None
        if isinstance(payload, TaskState):
            return payload
        return cls.from_dict(payload)

    @classmethod
    def create_template(
        cls,
        *,
        run_id: str,
        objective: str = "",
        scope: list[str] | None = None,
        source_code_locations: list[str] | None = None,
    ) -> "TaskState":
        return cls(
            run_id=run_id,
            objective=objective,
            scope=list(scope or []),
            source_code_locations=list(source_code_locations or []),
        )

    def with_runtime_context(
        self,
        *,
        run_id: str,
        scope: list[str],
        source_code_locations: list[str],
    ) -> "TaskState":
        # Runtime-provided scope and source locations are authoritative. The
        # synthesizer may add grounded detail, but it should never remove the
        # current application context from TaskState.
        return TaskState(
            run_id=run_id,
            objective=self.objective,
            scope=_merge_unique(scope, self.scope),
            source_code_locations=_merge_unique(source_code_locations, self.source_code_locations),
            open_questions=list(self.open_questions),
            next_action=self.next_action,
            stop_conditions=list(self.stop_conditions),
            constraints=list(self.constraints),
            relevant_artifacts=list(self.relevant_artifacts),
            status=self.status,
            domain_extensions=dict(self.domain_extensions),
        )

    def render_text(self) -> str:
        lines = ["Current task state:"]
        lines.append(f"- Run ID: {self.run_id}")
        lines.append(f"- Objective: {self.objective}")
        lines.append(f"- Status: {self.status}")
        lines.extend(self._render_section("Scope", self.scope))
        lines.extend(self._render_section("Source code locations", self.source_code_locations))
        lines.extend(self._render_section("Open questions", self.open_questions))
        lines.append(f"Next action: {self.next_action or '-'}")
        lines.extend(self._render_section("Stop conditions", self.stop_conditions))
        lines.extend(self._render_section("Constraints", self.constraints))
        lines.extend(self._render_section("Relevant artifacts", self.relevant_artifacts))
        for key, value in self.domain_extensions.items():
            if isinstance(value, list):
                lines.extend(self._render_section(key, [str(v) for v in value]))
            else:
                lines.append(f"- {key}: {value or '-'}")
        return "\n".join(lines)

    def as_context_block(self, *, pinned: bool = True, priority: int = 90, source: str = "runtime") -> ContextBlock:
        return ContextBlock(
            block_id=f"task:{self.run_id}",
            kind="task_state",
            content={"task_state": self.to_dict()},
            token_estimate=estimate_token_count(self.to_dict()),
            pinned=pinned,
            priority=priority,
            source=source,
            metadata={"status": self.status},
        )

    def _render_section(self, label: str, values: list[str]) -> list[str]:
        if not values:
            return [f"{label}: -"]
        lines = [f"{label}:"]
        lines.extend(f"- {value}" for value in values)
        return lines
