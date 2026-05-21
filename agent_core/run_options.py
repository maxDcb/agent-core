from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

AgentRunMode: TypeAlias = Literal["direct", "investigate", "deep_investigate"]


@dataclass(slots=True)
class RunOptions:
    mode: AgentRunMode = "direct"
    max_iterations: int = 1
    max_tool_calls: int = 10
    max_no_progress_iterations: int = 2
    require_initial_plan: bool = False
    require_final_critique: bool = False
    min_confidence_to_answer: float = 0.70
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"direct", "investigate", "deep_investigate"}:
            raise ValueError(f"Unsupported agent run mode: {self.mode}")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative")
        if self.max_no_progress_iterations < 0:
            raise ValueError("max_no_progress_iterations cannot be negative")
        if not 0.0 <= self.min_confidence_to_answer <= 1.0:
            raise ValueError("min_confidence_to_answer must be between 0.0 and 1.0")
        self.metadata = dict(self.metadata)

    @classmethod
    def direct(cls, **overrides: Any) -> "RunOptions":
        return cls(mode="direct", max_iterations=1, **overrides)

    @classmethod
    def investigate(cls, **overrides: Any) -> "RunOptions":
        defaults: dict[str, Any] = {
            "mode": "investigate",
            "max_iterations": 10,
            "max_tool_calls": 50,
            "max_no_progress_iterations": 2,
            "require_initial_plan": True,
            "require_final_critique": False,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def deep_investigate(cls, **overrides: Any) -> "RunOptions":
        defaults: dict[str, Any] = {
            "mode": "deep_investigate",
            "max_iterations": 20,
            "max_tool_calls": 100,
            "max_no_progress_iterations": 3,
            "require_initial_plan": True,
            "require_final_critique": True,
            "min_confidence_to_answer": 0.80,
            "reasoning_effort": "high",
        }
        defaults.update(overrides)
        return cls(**defaults)
