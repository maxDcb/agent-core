from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from agent_core.context_planner import LLMContextPolicy
from agent_core.llm_budget import LLMBudget
from agent_core.output_contracts import FinalOutputMode, StructuredOutputContract
from agent_core.tool_artifacts import ToolArtifactPolicy

AgentRunMode: TypeAlias = Literal["direct", "investigate", "deep_investigate"]


@dataclass(slots=True)
class RunOptions:
    mode: AgentRunMode = "direct"
    max_iterations: int = 1
    max_tool_calls: int = 10
    max_no_progress_iterations: int = 2
    require_initial_plan: bool = False
    require_final_critique: bool = False
    recover_internal_synthesis_errors: bool = False
    final_output_mode: FinalOutputMode = "text"
    final_output_contract: StructuredOutputContract | None = None
    min_confidence_to_answer: float = 0.70
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    llm_budget: LLMBudget | None = None
    llm_context_policy: LLMContextPolicy | None = None
    tool_artifact_policy: ToolArtifactPolicy | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"direct", "investigate", "deep_investigate"}:
            raise ValueError(f"Unsupported agent run mode: {self.mode}")
        if self.final_output_mode not in {"text", "json_schema"}:
            raise ValueError(f"Unsupported final output mode: {self.final_output_mode}")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative")
        if self.max_no_progress_iterations < 0:
            raise ValueError("max_no_progress_iterations cannot be negative")
        if not 0.0 <= self.min_confidence_to_answer <= 1.0:
            raise ValueError("min_confidence_to_answer must be between 0.0 and 1.0")
        self.metadata = dict(self.metadata)
        self.llm_budget = LLMBudget.from_any(self.llm_budget)
        self.llm_context_policy = LLMContextPolicy.from_any(self.llm_context_policy)
        self.tool_artifact_policy = ToolArtifactPolicy.from_any(self.tool_artifact_policy)
        self.final_output_contract = StructuredOutputContract.from_any(self.final_output_contract)
        if self.mode == "direct" and self.final_output_mode != "text":
            raise ValueError("json_schema final output requires investigate or deep_investigate mode")
        if self.final_output_mode == "json_schema" and self.final_output_contract is None:
            raise ValueError("final_output_contract is required when final_output_mode is json_schema")
        if self.final_output_mode == "text" and self.final_output_contract is not None:
            raise ValueError("final_output_contract requires final_output_mode json_schema")

    @classmethod
    def direct(cls, **overrides: Any) -> RunOptions:
        return cls(mode="direct", max_iterations=1, **overrides)

    @classmethod
    def investigate(cls, **overrides: Any) -> RunOptions:
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
    def deep_investigate(cls, **overrides: Any) -> RunOptions:
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
