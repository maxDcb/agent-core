from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any, Literal, Protocol, TypedDict, cast

from agent_core.agent_graph.state import normalize_agent_kernel_backend
from agent_core.context_planner import (
    LLMContextPlanner,
    LLMContextPolicy,
    LLMContextUsage,
    active_llm_context_planner,
    llm_context_scope,
)
from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import (
    BaseLLMProvider,
    LLMCallOptions,
    LLMCallRecord,
    LLMCompletionResult,
    LLMMessage,
    LLMToolCall,
    LLMToolDefinition,
    LLMUsageSummary,
)
from agent_core.llm.errors import LLMProviderError
from agent_core.llm_budget import (
    LLMBudget,
    LLMBudgetController,
    LLMBudgetUsage,
    active_llm_budget_controller,
    llm_budget_scope,
    run_budgeted_llm_call,
)
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.output_contracts import (
    StructuredOutputContract,
    StructuredOutputValidationError,
    parse_json_object,
)
from agent_core.policy_engine import PolicyEngine
from agent_core.settings import CoreSettings
from agent_core.tool_artifacts import (
    READ_ARTIFACT_TOOL_NAME,
    ArtifactStore,
    JsonFileArtifactStore,
    ToolArtifactPolicy,
    ToolArtifactRuntime,
    ToolArtifactUsage,
    active_tool_artifact_runtime,
    artifact_descriptor_from_message,
    message_to_persistence_dict,
    tool_artifact_scope,
)
from agent_core.tool_registry import ToolRegistry
from agent_core.types import ToolExecutionStatus

logger = get_logger(__name__)

StructuredTaskCheckpointPhase = Literal["model_request", "tools", "finalization", "result"]
StructuredToolCallStatus = Literal["prepared", "running", "completed", "budget_exhausted"]
StructuredFinalizationKind = Literal["contract", "budget"]
StructuredResultKind = Literal["direct", "contract", "budget"]


def _clean_string(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _clean_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        item = _clean_string(value)
        if item:
            cleaned.append(item)
    return cleaned


SAFE_TOOL_ARGUMENT_KEYS = {
    "around_line",
    "case_sensitive",
    "context_after",
    "context_before",
    "end_line",
    "glob",
    "identity_id",
    "include_files",
    "max_depth",
    "max_entries",
    "max_results",
    "method",
    "path",
    "query",
    "regex",
    "selector",
    "start_line",
    "timeout_ms",
    "url",
}

SENSITIVE_ARGUMENT_FRAGMENTS = ("answer", "authorization", "body", "cookie", "credential", "email", "field", "pass", "secret", "token", "value")


def _safe_tool_argument_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in sorted(SAFE_TOOL_ARGUMENT_KEYS.intersection(arguments)):
        value = arguments.get(key)
        if value is None:
            continue
        if isinstance(value, (bool, int, float)):
            summary[key] = value
        elif isinstance(value, str):
            summary[key] = safe_preview(value, limit=180)
        else:
            summary[key] = type(value).__name__

    redacted_keys = [
        key
        for key in sorted(arguments)
        if key not in summary and any(fragment in key.lower() for fragment in SENSITIVE_ARGUMENT_FRAGMENTS)
    ]
    if redacted_keys:
        summary["redacted_argument_keys"] = redacted_keys
    return summary


def _approx_token_count_from_chars(char_count: int) -> int:
    return round(max(0, char_count) / 4)


def _jsonish_char_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):
        return len(str(value))


def _message_content_stats(messages: list[LLMMessage]) -> dict[str, int]:
    content_lengths = [len(message.content or "") for message in messages]
    tool_call_chars = sum(_jsonish_char_count(message.tool_calls) for message in messages if message.tool_calls)
    total_chars = sum(content_lengths) + tool_call_chars
    return {
        "message_count": len(messages),
        "transcript_chars": total_chars,
        "transcript_approx_tokens": _approx_token_count_from_chars(total_chars),
        "largest_message_chars": max(content_lengths, default=0),
        "assistant_tool_call_chars": tool_call_chars,
    }


def _response_format_type(response_format: dict[str, Any] | None) -> str | None:
    if not isinstance(response_format, dict):
        return None
    value = response_format.get("type")
    return str(value) if value is not None else "dict"


def _clean_positive_int(value: object, *, default: int, minimum: int = 0) -> int:
    normalized = default
    try:
        if isinstance(value, bool):
            normalized = int(value)
        elif isinstance(value, (int, float, str)):
            normalized = int(value)
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, normalized)


def _clean_optional_positive_int(value: object) -> int | None:
    try:
        if isinstance(value, bool):
            normalized = int(value)
        elif isinstance(value, (int, float, str)):
            normalized = int(value)
        else:
            return None
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _fingerprint_fallback(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value}"


@dataclass(slots=True)
class StructuredTaskSpec:
    """Caller-owned specification for one bounded, tool-using structured task."""

    task_id: str
    system_prompt: str
    objective: str
    context: str = ""
    constraints: list[str] = field(default_factory=list)
    target: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    output_contract: StructuredOutputContract | None = None
    model: str | None = None
    temperature: float | None = None
    max_tool_calls: int = 8
    max_iterations: int = 6
    llm_budget: LLMBudget | None = None
    llm_context_policy: LLMContextPolicy | None = None
    tool_artifact_policy: ToolArtifactPolicy | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_id = _clean_string(self.task_id, default="structured_task")
        self.system_prompt = _clean_string(self.system_prompt)
        self.objective = _clean_string(self.objective)
        self.context = _clean_string(self.context)
        self.target = _clean_string(self.target)
        self.constraints = _clean_string_list(self.constraints)
        self.allowed_tools = _clean_string_list(self.allowed_tools)
        self.output_contract = StructuredOutputContract.from_any(self.output_contract)
        self.llm_budget = LLMBudget.from_any(self.llm_budget)
        self.llm_context_policy = LLMContextPolicy.from_any(self.llm_context_policy)
        if self.tool_artifact_policy is not None:
            self.tool_artifact_policy = ToolArtifactPolicy.from_any(self.tool_artifact_policy)
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        self.model = _clean_string(self.model) or None
        self.max_tool_calls = _clean_positive_int(self.max_tool_calls, default=8, minimum=0)
        self.max_iterations = _clean_positive_int(self.max_iterations, default=6, minimum=1)

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "context": self.context,
            "constraints": list(self.constraints),
            "target": self.target,
            "metadata": dict(self.metadata),
        }

    def fingerprint(self) -> str:
        contract = self.output_contract
        payload = {
            "task_id": self.task_id,
            "system_prompt": self.system_prompt,
            "objective": self.objective,
            "context": self.context,
            "constraints": list(self.constraints),
            "target": self.target,
            "allowed_tools": list(self.allowed_tools),
            "output_contract": (
                {
                    "name": contract.name,
                    "schema": contract.schema,
                    "strict": contract.strict,
                    "instructions": list(contract.instructions),
                }
                if contract is not None
                else None
            ),
            "model": self.model,
            "temperature": self.temperature,
            "max_tool_calls": self.max_tool_calls,
            "max_iterations": self.max_iterations,
            "metadata": self.metadata,
        }
        if self.llm_budget is not None:
            payload["llm_budget"] = self.llm_budget.to_dict()
        if self.llm_context_policy is not None:
            payload["llm_context_policy"] = self.llm_context_policy.to_dict()
        if self.tool_artifact_policy is not None:
            payload["tool_artifact_policy"] = self.tool_artifact_policy.to_dict()
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_fingerprint_fallback,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def _response_format_for_spec(spec: StructuredTaskSpec, *, final_output: bool = True) -> dict[str, Any] | None:
    if spec.output_contract is None or not final_output:
        return None
    return spec.output_contract.response_format()


@dataclass(slots=True)
class StructuredTaskResult:
    ok: bool
    task_id: str
    output: dict[str, Any] | None = None
    raw_content: str = ""
    failure_reason: str = ""
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tool_calls_used: int = 0
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "task_id": self.task_id,
            "output": self.output,
            "raw_content": self.raw_content,
            "failure_reason": self.failure_reason,
            "tool_history": list(self.tool_history),
            "iterations": self.iterations,
            "tool_calls_used": self.tool_calls_used,
            "llm_calls": [call.to_dict() for call in self.llm_calls],
            "usage": LLMUsageSummary.from_calls(self.llm_calls).to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class StructuredToolCallCheckpoint:
    tool_call_id: str
    tool_name: str
    arguments_json: str
    status: StructuredToolCallStatus = "prepared"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments_json": self.arguments_json,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: object) -> StructuredToolCallCheckpoint | None:
        if not isinstance(payload, dict):
            return None
        tool_call_id = payload.get("tool_call_id")
        tool_name = payload.get("tool_name")
        arguments_json = payload.get("arguments_json")
        status = payload.get("status")
        if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
            return None
        if not isinstance(arguments_json, str):
            arguments_json = "{}"
        normalized_status: StructuredToolCallStatus = (
            status if status in {"prepared", "running", "completed", "budget_exhausted"} else "prepared"
        )
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments_json=arguments_json,
            status=normalized_status,
        )


@dataclass(slots=True)
class StructuredTaskCheckpoint:
    spec_fingerprint: str
    phase: StructuredTaskCheckpointPhase
    messages: list[LLMMessage]
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tool_calls_used: int = 0
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    pending_tool_calls: list[StructuredToolCallCheckpoint] = field(default_factory=list)
    next_tool_call_index: int = 0
    finalization_kind: StructuredFinalizationKind | None = None
    finalization_reason: str = ""
    raw_failure_content: str = ""
    result_kind: StructuredResultKind | None = None
    sequence: int = 0
    llm_budget: LLMBudget | None = None
    llm_budget_usage: LLMBudgetUsage = field(default_factory=LLMBudgetUsage)
    llm_context_policy: LLMContextPolicy | None = None
    llm_context_usage: LLMContextUsage = field(default_factory=LLMContextUsage)
    tool_artifact_policy: ToolArtifactPolicy = field(default_factory=ToolArtifactPolicy)
    tool_artifact_usage: ToolArtifactUsage = field(default_factory=ToolArtifactUsage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 6,
            "spec_fingerprint": self.spec_fingerprint,
            "phase": self.phase,
            "messages": [message_to_persistence_dict(message) for message in self.messages],
            "tool_history": [dict(item) for item in self.tool_history],
            "iterations": self.iterations,
            "tool_calls_used": self.tool_calls_used,
            "llm_calls": [call.to_dict() for call in self.llm_calls],
            "pending_tool_calls": [item.to_dict() for item in self.pending_tool_calls],
            "next_tool_call_index": self.next_tool_call_index,
            "finalization_kind": self.finalization_kind,
            "finalization_reason": self.finalization_reason,
            "raw_failure_content": self.raw_failure_content,
            "result_kind": self.result_kind,
            "sequence": self.sequence,
            "llm_budget": self.llm_budget.to_dict() if self.llm_budget is not None else None,
            "llm_budget_usage": self.llm_budget_usage.to_dict(),
            "llm_context_policy": (
                self.llm_context_policy.to_dict() if self.llm_context_policy is not None else None
            ),
            "llm_context_usage": self.llm_context_usage.to_dict(),
            "tool_artifact_policy": self.tool_artifact_policy.to_dict(),
            "tool_artifact_usage": self.tool_artifact_usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> StructuredTaskCheckpoint | None:
        if not isinstance(payload, dict) or payload.get("schema_version") != 6:
            return None
        spec_fingerprint = payload.get("spec_fingerprint")
        phase = payload.get("phase")
        if not isinstance(spec_fingerprint, str) or phase not in {
            "model_request",
            "tools",
            "finalization",
            "result",
        }:
            return None
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            return None
        messages = [LLMMessage.from_history_dict(item) for item in raw_messages if isinstance(item, dict)]
        raw_history = payload.get("tool_history")
        raw_llm_calls = payload.get("llm_calls")
        raw_pending = payload.get("pending_tool_calls")
        pending = (
            [item for value in raw_pending if (item := StructuredToolCallCheckpoint.from_dict(value)) is not None]
            if isinstance(raw_pending, list)
            else []
        )
        finalization_kind = payload.get("finalization_kind")
        normalized_kind: StructuredFinalizationKind | None = (
            finalization_kind if finalization_kind in {"contract", "budget"} else None
        )
        result_kind = payload.get("result_kind")
        normalized_result_kind: StructuredResultKind | None = (
            result_kind if result_kind in {"direct", "contract", "budget"} else None
        )
        try:
            tool_artifact_policy = ToolArtifactPolicy.from_any(payload.get("tool_artifact_policy"))
        except ValueError:
            return None
        return cls(
            spec_fingerprint=spec_fingerprint,
            phase=phase,
            messages=messages,
            tool_history=[dict(item) for item in raw_history if isinstance(item, dict)] if isinstance(raw_history, list) else [],
            iterations=_clean_positive_int(payload.get("iterations"), default=0),
            tool_calls_used=_clean_positive_int(payload.get("tool_calls_used"), default=0),
            llm_calls=(
                [call for item in raw_llm_calls if (call := LLMCallRecord.from_dict(item)) is not None]
                if isinstance(raw_llm_calls, list)
                else []
            ),
            pending_tool_calls=pending,
            next_tool_call_index=_clean_positive_int(payload.get("next_tool_call_index"), default=0),
            finalization_kind=normalized_kind,
            finalization_reason=_clean_string(payload.get("finalization_reason")),
            raw_failure_content=str(payload.get("raw_failure_content") or ""),
            result_kind=normalized_result_kind,
            sequence=_clean_positive_int(payload.get("sequence"), default=0),
            llm_budget=LLMBudget.from_any(payload.get("llm_budget")),
            llm_budget_usage=LLMBudgetUsage.from_any(payload.get("llm_budget_usage")),
            llm_context_policy=LLMContextPolicy.from_any(payload.get("llm_context_policy")),
            llm_context_usage=LLMContextUsage.from_any(payload.get("llm_context_usage")),
            tool_artifact_policy=tool_artifact_policy,
            tool_artifact_usage=ToolArtifactUsage.from_any(payload.get("tool_artifact_usage")),
        )


class StructuredTaskRecoveryError(RuntimeError):
    def __init__(self, *, kind: str, message: str, tool_call_id: str | None = None) -> None:
        self.kind = kind
        self.tool_call_id = tool_call_id
        super().__init__(message)


class StructuredTaskRunner:
    """Run one bounded structured task with a caller-selected tool subset."""

    def __init__(
        self,
        *,
        settings: CoreSettings,
        provider: BaseLLMProvider,
        tool_registry: ToolRegistry,
        policy_engine: PolicyEngine,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.tool_registry = tool_registry
        self.policy_engine = policy_engine
        self.artifact_store = artifact_store or JsonFileArtifactStore(settings.artifacts_directory)
        self._kernel_nodes, self._kernel = build_structured_task_kernel(
            backend=settings.agent_kernel_backend,
            operations=self,
        )

    def run(
        self,
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None = None,
    ) -> StructuredTaskResult:
        budget = spec.llm_budget or self.settings.llm_budget
        context_policy = spec.llm_context_policy or self.settings.llm_context_policy
        artifact_policy = spec.tool_artifact_policy or self.settings.tool_artifact_policy
        checkpoint = StructuredTaskCheckpoint(
            spec_fingerprint=spec.fingerprint(),
            phase="model_request",
            messages=self._build_messages(spec=spec, context=context),
            iterations=1,
            llm_budget=budget,
            llm_context_policy=context_policy,
            tool_artifact_policy=artifact_policy,
        )
        controller = LLMBudgetController(budget) if budget is not None else None
        context_planner = LLMContextPlanner(context_policy) if context_policy is not None else None
        artifact_runtime = self._build_artifact_runtime(
            policy=artifact_policy,
            context=context,
            usage=None,
        )
        with tool_artifact_scope(artifact_runtime):
            with llm_context_scope(context_planner):
                with llm_budget_scope(controller):
                    self._emit_checkpoint(checkpoint, on_checkpoint)
                    result = self._continue_from_checkpoint(
                        spec=spec,
                        context=context,
                        checkpoint=checkpoint,
                        on_checkpoint=on_checkpoint,
                    )
                    return self._attach_runtime_metadata(result)

    def resume(
        self,
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None = None,
    ) -> StructuredTaskResult:
        if checkpoint.spec_fingerprint != spec.fingerprint():
            raise StructuredTaskRecoveryError(
                kind="spec_mismatch",
                message="The structured task specification does not match the persisted run checkpoint.",
            )
        self._validate_checkpoint_shape(checkpoint)
        budget = checkpoint.llm_budget or spec.llm_budget or self.settings.llm_budget
        context_policy = (
            checkpoint.llm_context_policy
            or spec.llm_context_policy
            or self.settings.llm_context_policy
        )
        artifact_policy = (
            checkpoint.tool_artifact_policy
            or spec.tool_artifact_policy
            or self.settings.tool_artifact_policy
        )
        controller = (
            LLMBudgetController(budget, usage=checkpoint.llm_budget_usage)
            if budget is not None
            else None
        )
        context_planner = (
            LLMContextPlanner(context_policy, usage=checkpoint.llm_context_usage)
            if context_policy is not None
            else None
        )
        artifact_runtime = self._build_artifact_runtime(
            policy=artifact_policy,
            context=context,
            usage=checkpoint.tool_artifact_usage,
        )
        with tool_artifact_scope(artifact_runtime):
            with llm_context_scope(context_planner):
                with llm_budget_scope(controller):
                    result = self._continue_from_checkpoint(
                        spec=spec,
                        context=context,
                        checkpoint=checkpoint,
                        on_checkpoint=on_checkpoint,
                    )
                    return self._attach_runtime_metadata(result)

    @staticmethod
    def _validate_checkpoint_shape(checkpoint: StructuredTaskCheckpoint) -> None:
        if not checkpoint.messages or checkpoint.iterations < 1 or checkpoint.tool_calls_used < 0:
            raise StructuredTaskRecoveryError(
                kind="invalid_checkpoint",
                message="Structured task checkpoint is missing required execution state.",
            )
        if checkpoint.next_tool_call_index < 0 or checkpoint.next_tool_call_index > len(
            checkpoint.pending_tool_calls
        ):
            raise StructuredTaskRecoveryError(
                kind="invalid_checkpoint",
                message="Structured task checkpoint has an invalid tool-call cursor.",
            )
        if checkpoint.phase == "finalization" and checkpoint.finalization_kind is None:
            raise StructuredTaskRecoveryError(
                kind="invalid_checkpoint",
                message="Structured task finalization checkpoint has no finalization kind.",
            )
        if checkpoint.phase == "result" and checkpoint.result_kind is None:
            raise StructuredTaskRecoveryError(
                kind="invalid_checkpoint",
                message="Structured task result checkpoint has no result kind.",
            )

    def _continue_from_checkpoint(
        self,
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskResult:
        try:
            registry = self.tool_registry.build_subset(spec.allowed_tools)
        except KeyError as exc:
            return StructuredTaskResult(
                ok=False,
                task_id=spec.task_id,
                failure_reason=str(exc),
            )
        initial_state = self._kernel_nodes.initial_state(
            spec=spec,
            context=context,
            registry=registry,
            checkpoint=checkpoint,
            on_checkpoint=on_checkpoint,
        )
        return self._kernel.run(initial_state)

    def _continue_model_request(
        self,
        *,
        spec: StructuredTaskSpec,
        registry: ToolRegistry,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskResult | None:
        logger.debug(
            "Calling structured task LLM",
            extra={
                "task_id": spec.task_id,
                "iteration": checkpoint.iterations,
                "tool_count": len(registry.list_tool_names()),
            },
        )
        try:
            llm_response = self._call_model_once(
                spec=spec,
                messages=checkpoint.messages,
                registry=registry,
                final_output=not registry.list_tool_names(),
                on_budget_reserved=lambda: self._emit_checkpoint(checkpoint, on_checkpoint),
            )
        except LLMProviderError as exc:
            logger.error(
                "Structured task provider failure",
                extra={"task_id": spec.task_id, "error_kind": exc.kind},
            )
            return StructuredTaskResult(
                ok=False,
                task_id=spec.task_id,
                failure_reason=exc.user_message,
                raw_content=exc.detail or exc.user_message,
                tool_history=checkpoint.tool_history,
                iterations=checkpoint.iterations,
                tool_calls_used=checkpoint.tool_calls_used,
                llm_calls=list(checkpoint.llm_calls),
            )

        self._record_llm_call(
            checkpoint=checkpoint,
            completion=llm_response,
            purpose="structured_direct" if not registry.list_tool_names() else "structured_tool_loop",
        )
        checkpoint.messages.append(
            LLMMessage(
                role="assistant",
                content=llm_response.content,
                tool_calls=list(llm_response.tool_calls),
            )
        )

        if not llm_response.tool_calls:
            if spec.output_contract is not None and registry.list_tool_names():
                checkpoint.phase = "finalization"
                checkpoint.finalization_kind = "contract"
                checkpoint.finalization_reason = "Investigation is complete."
                checkpoint.raw_failure_content = llm_response.content
            else:
                checkpoint.phase = "result"
                checkpoint.result_kind = "direct"
            self._emit_checkpoint(checkpoint, on_checkpoint)
            return None

        checkpoint.phase = "tools"
        checkpoint.pending_tool_calls = [
            StructuredToolCallCheckpoint(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments_json=tool_call.arguments_json,
            )
            for tool_call in llm_response.tool_calls
        ]
        checkpoint.next_tool_call_index = 0
        self._emit_checkpoint(checkpoint, on_checkpoint)
        return None

    def _continue_tools(
        self,
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
        registry: ToolRegistry,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskResult | None:
        blocked = next(
            (item for item in checkpoint.pending_tool_calls if item.status == "running"),
            None,
        )
        if blocked is not None:
            raise StructuredTaskRecoveryError(
                kind="ambiguous_tool_execution",
                message=(
                    "A tool call was running when execution stopped; automatic replay is blocked "
                    f"because its external effect is unknown: {blocked.tool_name} ({blocked.tool_call_id})."
                ),
                tool_call_id=blocked.tool_call_id,
            )
        return self._continue_tool_batch(
            spec=spec,
            context=context,
            registry=registry,
            checkpoint=checkpoint,
            on_checkpoint=on_checkpoint,
        )

    def _continue_tool_batch(
        self,
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
        registry: ToolRegistry,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskResult | None:
        artifact_context_exhausted = False
        while checkpoint.next_tool_call_index < len(checkpoint.pending_tool_calls):
            tool_call = checkpoint.pending_tool_calls[checkpoint.next_tool_call_index]
            artifact_runtime = active_tool_artifact_runtime()
            if artifact_runtime is None:
                raise RuntimeError("Tool artifact runtime is required during tool execution")
            is_internal = artifact_runtime.is_internal_tool(tool_call.tool_name)
            if not is_internal and checkpoint.tool_calls_used >= spec.max_tool_calls:
                self._append_budget_exhausted_tool_responses(
                    messages=checkpoint.messages,
                    tool_calls=[
                        LLMToolCall(
                            id=tool_call.tool_call_id,
                            name=tool_call.tool_name,
                            arguments_json=tool_call.arguments_json,
                        )
                    ],
                    tool_history=checkpoint.tool_history,
                )
                tool_call.status = "budget_exhausted"
                checkpoint.next_tool_call_index += 1
                self._emit_checkpoint(checkpoint, on_checkpoint)
                continue

            tool_call.status = "running"
            if not is_internal:
                checkpoint.tool_calls_used += 1
            self._emit_checkpoint(checkpoint, on_checkpoint)
            remaining_pending_calls = checkpoint.pending_tool_calls[checkpoint.next_tool_call_index + 1 :]
            content_fits: Callable[[str], bool] | None = None
            if is_internal:
                def artifact_content_fits(
                    content: str,
                    current_tool_call_id: str = tool_call.tool_call_id,
                    current_trailing_tool_calls: tuple[StructuredToolCallCheckpoint, ...] = tuple(
                        remaining_pending_calls
                    ),
                ) -> bool:
                    return self._artifact_read_content_fits(
                        spec=spec,
                        registry=registry,
                        messages=checkpoint.messages,
                        tool_call_id=current_tool_call_id,
                        content=content,
                        trailing_tool_calls=current_trailing_tool_calls,
                    )

                content_fits = artifact_content_fits
            tool_message, history_item = self._execute_tool_call(
                registry=registry,
                tool_name=tool_call.tool_name,
                arguments_json=tool_call.arguments_json,
                tool_call_id=tool_call.tool_call_id,
                context=context,
                content_fits=content_fits,
            )
            checkpoint.messages.append(tool_message)
            checkpoint.tool_history.append(history_item)
            tool_call.status = "completed"
            checkpoint.next_tool_call_index += 1
            self._emit_checkpoint(checkpoint, on_checkpoint)
            if history_item.get("artifact_read_context_exhausted") is True:
                artifact_context_exhausted = True
                for skipped in remaining_pending_calls:
                    skipped_content = "Tool call skipped: artifact context capacity reached."
                    checkpoint.messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=skipped.tool_call_id,
                            content=skipped_content,
                        )
                    )
                    checkpoint.tool_history.append(
                        {
                            "tool_name": skipped.tool_name,
                            "arguments": {},
                            "status": "budget_exhausted",
                            "content_preview": skipped_content,
                            "tool_kind": "runtime" if artifact_runtime.is_internal_tool(skipped.tool_name) else "application",
                        }
                    )
                    skipped.status = "budget_exhausted"
                    checkpoint.next_tool_call_index += 1
                break

        tool_budget_exhausted = any(item.status == "budget_exhausted" for item in checkpoint.pending_tool_calls)
        checkpoint.pending_tool_calls = []
        checkpoint.next_tool_call_index = 0
        if artifact_context_exhausted:
            checkpoint.phase = "finalization"
            checkpoint.finalization_kind = "budget"
            checkpoint.finalization_reason = (
                "The remaining model context cannot safely hold another artifact chunk."
            )
            checkpoint.raw_failure_content = checkpoint.messages[-1].content if checkpoint.messages else ""
        elif tool_budget_exhausted:
            checkpoint.phase = "finalization"
            checkpoint.finalization_kind = "budget"
            checkpoint.finalization_reason = "Maximum number of structured task tool calls reached."
            checkpoint.raw_failure_content = checkpoint.messages[-1].content if checkpoint.messages else ""
        elif checkpoint.iterations >= spec.max_iterations:
            checkpoint.phase = "finalization"
            checkpoint.finalization_kind = "budget"
            checkpoint.finalization_reason = "Maximum number of structured task iterations reached."
            checkpoint.raw_failure_content = ""
        else:
            checkpoint.iterations += 1
            checkpoint.phase = "model_request"
        self._emit_checkpoint(checkpoint, on_checkpoint)
        return None

    def _continue_finalization(
        self,
        *,
        spec: StructuredTaskSpec,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskResult | None:
        try:
            llm_response = self._call_model_for_final_output(
                spec=spec,
                messages=checkpoint.messages,
                failure_reason=checkpoint.finalization_reason or "Structured task execution was interrupted.",
                on_budget_reserved=lambda: self._emit_checkpoint(checkpoint, on_checkpoint),
            )
        except LLMProviderError as exc:
            if checkpoint.finalization_kind == "contract":
                return StructuredTaskResult(
                    ok=False,
                    task_id=spec.task_id,
                    raw_content=exc.detail or checkpoint.raw_failure_content,
                    failure_reason=f"Structured output contract finalization failed: {exc.user_message}",
                    tool_history=checkpoint.tool_history,
                    iterations=checkpoint.iterations,
                    tool_calls_used=checkpoint.tool_calls_used,
                    metadata={"contract_finalization": True},
                    llm_calls=list(checkpoint.llm_calls),
                )
            return StructuredTaskResult(
                ok=False,
                task_id=spec.task_id,
                raw_content=exc.detail or checkpoint.raw_failure_content,
                failure_reason=(
                    f"{checkpoint.finalization_reason}; finalization failed: {exc.user_message}"
                ),
                tool_history=checkpoint.tool_history,
                iterations=checkpoint.iterations,
                tool_calls_used=checkpoint.tool_calls_used,
                metadata={"forced_finalization": True},
                llm_calls=list(checkpoint.llm_calls),
            )

        self._record_llm_call(
            checkpoint=checkpoint,
            completion=llm_response,
            purpose="structured_finalization",
        )
        checkpoint.messages.append(
            LLMMessage(
                role="assistant",
                content=llm_response.content,
                tool_calls=list(llm_response.tool_calls),
            )
        )
        checkpoint.phase = "result"
        checkpoint.result_kind = checkpoint.finalization_kind
        self._emit_checkpoint(checkpoint, on_checkpoint)
        return None

    def _continue_persisted_result(
        self,
        *,
        spec: StructuredTaskSpec,
        checkpoint: StructuredTaskCheckpoint,
    ) -> StructuredTaskResult:
        if not checkpoint.messages or checkpoint.messages[-1].role != "assistant":
            raise StructuredTaskRecoveryError(
                kind="invalid_result_checkpoint",
                message="Persisted result checkpoint has no final assistant response.",
            )
        response = checkpoint.messages[-1]
        if checkpoint.result_kind == "direct":
            return self._finalize_result(
                task_id=spec.task_id,
                output_contract=spec.output_contract,
                raw_content=response.content,
                tool_history=checkpoint.tool_history,
                iterations=checkpoint.iterations,
                tool_calls_used=checkpoint.tool_calls_used,
                metadata={
                    "model": spec.model or self.settings.model,
                    "contract_name": spec.output_contract.name if spec.output_contract is not None else None,
                },
                llm_calls=checkpoint.llm_calls,
            )

        failure_reason = checkpoint.finalization_reason or "Structured task execution was interrupted."
        if response.tool_calls:
            if checkpoint.result_kind == "contract":
                return StructuredTaskResult(
                    ok=False,
                    task_id=spec.task_id,
                    raw_content=response.content or checkpoint.raw_failure_content,
                    failure_reason="Structured output contract finalization still requested tools.",
                    tool_history=checkpoint.tool_history,
                    iterations=checkpoint.iterations + 1,
                    tool_calls_used=checkpoint.tool_calls_used,
                    metadata={"contract_finalization": True},
                    llm_calls=list(checkpoint.llm_calls),
                )
            return StructuredTaskResult(
                ok=False,
                task_id=spec.task_id,
                raw_content=response.content or checkpoint.raw_failure_content,
                failure_reason=f"{failure_reason}; finalization still requested tools.",
                tool_history=checkpoint.tool_history,
                iterations=checkpoint.iterations + 1,
                tool_calls_used=checkpoint.tool_calls_used,
                metadata={"forced_finalization": True},
                llm_calls=list(checkpoint.llm_calls),
            )

        metadata: dict[str, Any] = {
            "model": spec.model or self.settings.model,
            "contract_name": spec.output_contract.name if spec.output_contract is not None else None,
        }
        if checkpoint.result_kind == "contract":
            metadata["contract_finalization"] = True
        else:
            metadata.update(
                {
                    "forced_finalization": True,
                    "budget_failure_reason": failure_reason,
                }
            )
        finalized = self._finalize_result(
            task_id=spec.task_id,
            output_contract=spec.output_contract,
            raw_content=response.content,
            tool_history=checkpoint.tool_history,
            iterations=checkpoint.iterations + 1,
            tool_calls_used=checkpoint.tool_calls_used,
            metadata=metadata,
            llm_calls=checkpoint.llm_calls,
        )
        if checkpoint.result_kind == "budget" and not finalized.ok:
            finalized.failure_reason = f"{failure_reason}; {finalized.failure_reason}"
        return finalized

    @staticmethod
    def _attach_runtime_metadata(result: StructuredTaskResult) -> StructuredTaskResult:
        controller = active_llm_budget_controller()
        if controller is not None:
            result.metadata = {**result.metadata, **controller.to_metadata()}
        context_planner = active_llm_context_planner()
        if context_planner is not None:
            result.metadata = {**result.metadata, **context_planner.to_metadata()}
        artifact_runtime = active_tool_artifact_runtime()
        if artifact_runtime is not None:
            result.metadata = {**result.metadata, **artifact_runtime.to_metadata()}
        return result

    def _build_artifact_runtime(
        self,
        *,
        policy: ToolArtifactPolicy,
        context: ExecutionContext,
        usage: ToolArtifactUsage | dict[str, Any] | None,
    ) -> ToolArtifactRuntime:
        return ToolArtifactRuntime(
            policy=policy,
            store=self.artifact_store,
            namespace_id=context.namespace_id,
            run_id=context.run_id,
            usage=usage,
        )

    @staticmethod
    def _emit_checkpoint(
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> None:
        controller = active_llm_budget_controller()
        if controller is not None:
            checkpoint.llm_budget = controller.budget
            checkpoint.llm_budget_usage = LLMBudgetUsage.from_any(controller.usage)
        context_planner = active_llm_context_planner()
        if context_planner is not None:
            checkpoint.llm_context_policy = context_planner.policy
            checkpoint.llm_context_usage = LLMContextUsage.from_any(context_planner.usage)
        artifact_runtime = active_tool_artifact_runtime()
        if artifact_runtime is not None:
            checkpoint.tool_artifact_policy = artifact_runtime.policy
            checkpoint.tool_artifact_usage = ToolArtifactUsage.from_any(artifact_runtime.usage)
        checkpoint.sequence += 1
        if on_checkpoint is not None:
            on_checkpoint(checkpoint)

    @staticmethod
    def _record_llm_call(
        *,
        checkpoint: StructuredTaskCheckpoint,
        completion: LLMCompletionResult,
        purpose: str,
    ) -> None:
        checkpoint.llm_calls.append(
            LLMCallRecord.from_completion(
                completion,
                call_index=len(checkpoint.llm_calls) + 1,
                purpose=purpose,
            )
        )

    def _append_budget_exhausted_tool_responses(
        self,
        *,
        messages: list[LLMMessage],
        tool_calls: list[LLMToolCall],
        tool_history: list[dict[str, Any]],
    ) -> None:
        artifact_runtime = active_tool_artifact_runtime()
        if artifact_runtime is None:
            raise RuntimeError("Tool artifact runtime is required during tool execution")
        for tool_call in tool_calls:
            content = f"Tool call skipped: maximum tool-call budget reached before executing {tool_call.name}."
            tool_message = artifact_runtime.externalize(
                tool_name=tool_call.name,
                content=content,
                tool_call_id=tool_call.id,
                status="budget_exhausted",
                metadata={"status": "budget_exhausted", "synthetic": True},
            )
            messages.append(tool_message)
            history_item = {
                "tool_name": tool_call.name,
                "arguments": {},
                "status": "budget_exhausted",
                "content_preview": content,
            }
            descriptor = artifact_descriptor_from_message(tool_message)
            if descriptor is not None:
                history_item["artifact_id"] = descriptor.artifact_id
            tool_history.append(history_item)

    def _artifact_read_content_fits(
        self,
        *,
        spec: StructuredTaskSpec,
        registry: ToolRegistry,
        messages: list[LLMMessage],
        tool_call_id: str,
        content: str,
        trailing_tool_calls: Sequence[StructuredToolCallCheckpoint],
    ) -> bool:
        context_planner = active_llm_context_planner()
        if context_planner is None or context_planner.policy.mode != "enforce":
            return True
        artifact_runtime = active_tool_artifact_runtime()
        if artifact_runtime is None:
            raise RuntimeError("Tool artifact runtime is required during artifact read planning")

        candidate_messages = [
            *messages,
            LLMMessage(role="tool", tool_call_id=tool_call_id, content=content),
            *[
                LLMMessage(
                    role="tool",
                    tool_call_id=tool_call.tool_call_id,
                    content="Tool call skipped: artifact context capacity reached.",
                )
                for tool_call in trailing_tool_calls
            ],
        ]
        tool_specs = self._structured_tool_specs(registry=registry, messages=candidate_messages)
        tool_options = self._structured_model_options(spec=spec, final_output=False)
        projected_tool_messages = artifact_runtime.project_messages(
            candidate_messages,
            messages_fit=lambda projected: context_planner.can_plan_call(
                messages=projected,
                tools=tool_specs,
                purpose="structured_tool_loop",
                options=tool_options,
            ),
        )
        if not context_planner.can_plan_call(
            messages=projected_tool_messages,
            tools=tool_specs,
            purpose="structured_tool_loop",
            options=tool_options,
        ):
            return False

        final_reason = "The remaining model context cannot safely hold another artifact chunk."
        final_messages = self._build_finalization_messages(
            spec=spec,
            messages=candidate_messages,
            failure_reason=final_reason,
        )
        final_options = self._structured_finalization_options(spec=spec)
        projected_final_messages = artifact_runtime.project_messages(
            final_messages,
            messages_fit=lambda projected: context_planner.can_plan_call(
                messages=projected,
                tools=[],
                purpose="structured_finalization",
                options=final_options,
            ),
        )
        return context_planner.can_plan_call(
            messages=projected_final_messages,
            tools=[],
            purpose="structured_finalization",
            options=final_options,
        )

    def _structured_tool_specs(
        self,
        *,
        registry: ToolRegistry,
        messages: list[LLMMessage],
    ) -> list[LLMToolDefinition]:
        tool_specs = registry.get_tool_specs()
        artifact_runtime = active_tool_artifact_runtime()
        if artifact_runtime is None:
            raise RuntimeError("Tool artifact runtime is required during model execution")
        if registry.get_tool(READ_ARTIFACT_TOOL_NAME) is not None:
            raise ValueError(f"Tool name is reserved by agent-core: {READ_ARTIFACT_TOOL_NAME}")
        if artifact_runtime.has_readable_artifacts(messages):
            tool_specs.extend(artifact_runtime.tool_specs())
        return tool_specs

    def _structured_model_options(
        self,
        *,
        spec: StructuredTaskSpec,
        final_output: bool,
    ) -> LLMCallOptions:
        return LLMCallOptions(
            response_format=_response_format_for_spec(spec, final_output=final_output),
            response_format_fallback=None,
            max_output_tokens=_clean_optional_positive_int(self.settings.llm_max_output_tokens)
            if final_output
            else None,
            metadata={
                "structured_task_id": spec.task_id,
                "llm_call_purpose": "structured_direct" if final_output else "structured_tool_loop",
                **spec.metadata,
            },
        )

    def _structured_finalization_options(self, *, spec: StructuredTaskSpec) -> LLMCallOptions:
        return LLMCallOptions(
            response_format=_response_format_for_spec(spec, final_output=True),
            response_format_fallback=None,
            max_output_tokens=_clean_optional_positive_int(self.settings.llm_max_output_tokens),
            metadata={
                "structured_task_id": spec.task_id,
                "structured_task_finalization": True,
                "llm_call_purpose": "structured_finalization",
                **spec.metadata,
            },
        )

    def _build_finalization_messages(
        self,
        *,
        spec: StructuredTaskSpec,
        messages: list[LLMMessage],
        failure_reason: str,
    ) -> list[LLMMessage]:
        if spec.output_contract is None:
            final_instruction = (
                f"{failure_reason} No more tools are available. "
                "Return the best possible final answer now, using only the evidence already present "
                "in the transcript. Do not request tools."
            )
        else:
            final_instruction = (
                f"{failure_reason} No more tools are available. "
                "Return the best possible final JSON object now, using only the evidence already present "
                "in the transcript. Do not request tools. Return one JSON object only, with no prose, "
                "no markdown fences, and no second JSON object after it."
            )
        return [*messages, LLMMessage(role="system", content=final_instruction)]

    def _call_model_once(
        self,
        *,
        spec: StructuredTaskSpec,
        messages: list[LLMMessage],
        registry: ToolRegistry,
        final_output: bool = False,
        on_budget_reserved: Callable[[], None] | None = None,
    ) -> LLMCompletionResult:
        options = self._structured_model_options(spec=spec, final_output=final_output)
        tool_specs = self._structured_tool_specs(registry=registry, messages=messages)
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tool_specs,
            "model": spec.model or self.settings.model,
            "temperature": spec.temperature if spec.temperature is not None else self.settings.temperature,
        }
        def invoke(effective_options: LLMCallOptions | None) -> LLMCompletionResult:
            logger.info(
                "Structured task LLM request prepared",
                extra={
                    "task_id": spec.task_id,
                    "model": kwargs["model"],
                    "final_output": final_output,
                    "tool_count": len(tool_specs),
                    "response_format_type": _response_format_type(
                        effective_options.response_format if effective_options is not None else None
                    ),
                    "response_format_chars": _jsonish_char_count(
                        effective_options.response_format if effective_options is not None else None
                    ),
                    "response_format_fallback_type": _response_format_type(
                        effective_options.response_format_fallback if effective_options is not None else None
                    ),
                    "max_output_tokens": effective_options.max_output_tokens if effective_options is not None else None,
                    **_message_content_stats(messages),
                },
            )
            effective_kwargs = dict(kwargs)
            if effective_options is not None and self._provider_accepts_options("complete_with_tools"):
                effective_kwargs["options"] = effective_options
            return self.provider.complete_with_tools(**effective_kwargs)

        return run_budgeted_llm_call(
            messages=messages,
            tools=tool_specs,
            purpose="structured_direct" if final_output else "structured_tool_loop",
            options=options,
            invoke=invoke,
            on_reserved=on_budget_reserved,
        )

    def _call_model_for_final_output(
        self,
        *,
        spec: StructuredTaskSpec,
        messages: list[LLMMessage],
        failure_reason: str,
        on_budget_reserved: Callable[[], None] | None = None,
    ) -> LLMCompletionResult:
        options = self._structured_finalization_options(spec=spec)
        final_messages = self._build_finalization_messages(
            spec=spec,
            messages=messages,
            failure_reason=failure_reason,
        )
        kwargs: dict[str, Any] = {
            "messages": final_messages,
            "tools": [],
            "model": spec.model or self.settings.model,
            "temperature": spec.temperature if spec.temperature is not None else self.settings.temperature,
        }

        def invoke(effective_options: LLMCallOptions | None) -> LLMCompletionResult:
            logger.info(
                "Structured task finalization LLM request prepared",
                extra={
                    "task_id": spec.task_id,
                    "model": kwargs["model"],
                    "failure_reason": failure_reason,
                    "response_format_type": _response_format_type(
                        effective_options.response_format if effective_options is not None else None
                    ),
                    "response_format_chars": _jsonish_char_count(
                        effective_options.response_format if effective_options is not None else None
                    ),
                    "response_format_fallback_type": _response_format_type(
                        effective_options.response_format_fallback if effective_options is not None else None
                    ),
                    "max_output_tokens": effective_options.max_output_tokens if effective_options is not None else None,
                    **_message_content_stats(final_messages),
                },
            )
            effective_kwargs = dict(kwargs)
            if effective_options is not None and self._provider_accepts_options("complete_with_tools"):
                effective_kwargs["options"] = effective_options
            return self.provider.complete_with_tools(**effective_kwargs)

        return run_budgeted_llm_call(
            messages=final_messages,
            tools=[],
            purpose="structured_finalization",
            options=options,
            invoke=invoke,
            on_reserved=on_budget_reserved,
        )

    def _build_messages(
        self,
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
    ) -> list[LLMMessage]:
        return [
            LLMMessage(role="system", content=self.settings.base_system_prompt),
            LLMMessage(role="system", content=self._build_task_system_prompt(spec=spec)),
            LLMMessage(role="system", content=self._build_scope_prompt_block(context=context)),
            LLMMessage(role="user", content=json.dumps(spec.to_payload(), ensure_ascii=False, indent=2)),
        ]

    def _build_task_system_prompt(self, *, spec: StructuredTaskSpec) -> str:
        lines = [
            f"Structured task id: {spec.task_id}",
            spec.system_prompt,
            "",
            "Structured task operating rules:",
            "- You are running a bounded task invoked by a higher-level controller.",
            "- Use only the tools exposed in this run. Do not assume hidden tools exist.",
            "- Drive the task to a bounded conclusion within this one run.",
            "- Base conclusions only on observed evidence and tool results.",
        ]
        if spec.output_contract is not None:
            lines.extend(
                [
                    "",
                    "Provider-enforced structured output contract:",
                    f"- Contract name: {spec.output_contract.name}",
                    f"- Strict mode requested: {str(spec.output_contract.strict).lower()}",
                    "- The provider contract is enforced only for the final no-tool output, after investigation is complete.",
                    "- Return exactly one JSON object when you are done.",
                    "- The final assistant message must contain only that one JSON object: no prose before it, no markdown fences, no comments, and no second JSON object after it.",
                    "- Use the schema keys and canonical values exactly when the schema defines them.",
                    "- Put uncertainty, non-standard labels, or unresolved values in note/unknown fields instead of inventing new top-level keys.",
                    "",
                    "Output JSON Schema:",
                    json.dumps(spec.output_contract.schema, ensure_ascii=False, indent=2),
                ]
            )
            if spec.output_contract.instructions:
                lines.append("- Contract-specific instructions:")
                lines.extend(f"  - {instruction}" for instruction in spec.output_contract.instructions)
        return "\n".join(lines)

    def _build_scope_prompt_block(self, *, context: ExecutionContext) -> str:
        allowed_roots = [str(path.resolve()) for path in context.allowed_read_roots()]
        knowledge_root = str(self.settings.knowledge_base_dir.resolve())
        allowed_hosts = context.allowed_http_hosts()
        allowed_methods = context.allowed_http_methods()
        lines = [
            "Execution scope:",
            f"- Namespace ID: {context.namespace_id}",
            f"- Run ID: {context.run_id}",
            "- Allowed local code roots:",
        ]
        if allowed_roots:
            lines.extend(f"  - {root}" for root in allowed_roots)
            lines.append("- For local code tools, use absolute paths inside these roots or paths relative to one of these roots.")
        else:
            lines.append("  - none")

        lines.extend(
            [
                "- Allowed knowledge base root:",
                f"  - {knowledge_root}",
                "- For knowledge tools, use this absolute root for broad searches or paths relative to it; pass exact returned paths to read_knowledge_chunk.",
                "- Allowed web hosts:",
            ]
        )
        if allowed_hosts:
            lines.extend(f"  - {host}" for host in allowed_hosts)
        else:
            lines.append("  - none")

        lines.append("- Allowed HTTP methods:")
        if allowed_methods:
            lines.extend(f"  - {method}" for method in allowed_methods)
        else:
            lines.append("  - none")
        return "\n".join(lines)

    def _execute_tool_call(
        self,
        *,
        registry: ToolRegistry,
        tool_name: str,
        arguments_json: str,
        tool_call_id: str,
        context: ExecutionContext,
        content_fits: Callable[[str], bool] | None = None,
    ) -> tuple[LLMMessage, dict[str, Any]]:
        artifact_runtime = active_tool_artifact_runtime()
        if artifact_runtime is None:
            raise RuntimeError("Tool artifact runtime is required during tool execution")
        arguments: dict[str, Any]
        result_metadata: dict[str, Any] = {}
        try:
            loaded_arguments = json.loads(arguments_json or "{}")
            arguments = loaded_arguments if isinstance(loaded_arguments, dict) else {}
        except json.JSONDecodeError:
            arguments = {}
            tool_content = f"Invalid JSON arguments for tool {tool_name}"
            tool_status: ToolExecutionStatus = "invalid_arguments"
        else:
            logger.info(
                "Structured task tool call started",
                extra={
                    "namespace_id": context.namespace_id,
                    "run_id": context.run_id,
                    "tool_name": tool_name,
                    "argument_keys": sorted(arguments.keys()),
                    "arguments_summary": _safe_tool_argument_summary(arguments),
                },
            )
            is_internal = artifact_runtime.is_internal_tool(tool_name)
            authz = None if is_internal else self.policy_engine.authorize(tool_name, arguments, context)
            if authz is not None and not authz.allowed:
                tool_content = f"Tool denied by policy: {authz.reason}"
                tool_status = "policy_denied"
                logger.info(
                    "Structured task tool call denied",
                    extra={
                        "namespace_id": context.namespace_id,
                        "run_id": context.run_id,
                        "tool_name": tool_name,
                        "reason": authz.reason,
                    },
                )
            else:
                try:
                    result = (
                        artifact_runtime.execute(
                            tool_name=tool_name,
                            arguments=arguments,
                            context=context,
                            content_fits=content_fits,
                        )
                        if is_internal
                        else registry.execute(tool_name, arguments, context)
                    )
                    tool_content = result.content
                    result_metadata = dict(result.metadata)
                    tool_status = "ok" if result.ok else "tool_error"
                    logger.info(
                        "Structured task tool call completed",
                        extra={
                            "namespace_id": context.namespace_id,
                            "run_id": context.run_id,
                            "tool_name": tool_name,
                            "status": tool_status,
                            "content_length": len(tool_content),
                        },
                    )
                except Exception as exc:
                    logger.exception("Structured task tool execution crashed", extra={"tool_name": tool_name})
                    tool_content = f"Tool execution failed: {exc}"
                    tool_status = "execution_failed"

        is_internal = artifact_runtime.is_internal_tool(tool_name)
        tool_message = (
            artifact_runtime.externalize(
                tool_name=tool_name,
                content=tool_content,
                tool_call_id=tool_call_id,
                status=tool_status,
                metadata={"status": tool_status},
            )
            if not is_internal
            else LLMMessage(
                role="tool",
                tool_call_id=tool_call_id,
                content=tool_content,
                metadata=result_metadata,
            )
        )
        history_item: dict[str, Any] = {
            "tool_name": tool_name,
            "arguments": arguments,
            "status": tool_status,
            "content_preview": safe_preview(tool_content, limit=500),
            "tool_kind": "runtime" if is_internal else "application",
        }
        descriptor = artifact_descriptor_from_message(tool_message)
        if descriptor is not None:
            history_item["artifact_id"] = descriptor.artifact_id
        if result_metadata.get("artifact_read_context_limited") is True:
            history_item["artifact_read_context_limited"] = True
        if result_metadata.get("artifact_read_context_exhausted") is True:
            history_item["artifact_read_context_exhausted"] = True
        return tool_message, history_item

    def _finalize_result(
        self,
        *,
        task_id: str,
        output_contract: StructuredOutputContract | None,
        raw_content: str,
        tool_history: list[dict[str, Any]],
        iterations: int,
        tool_calls_used: int,
        metadata: dict[str, Any],
        llm_calls: list[LLMCallRecord],
    ) -> StructuredTaskResult:
        raw_content_chars = len(raw_content or "")
        if metadata.get("contract_name") is None:
            logger.info(
                "Structured task final text output accepted",
                extra={
                    "task_id": task_id,
                    "raw_content_chars": raw_content_chars,
                    "raw_content_approx_tokens": _approx_token_count_from_chars(raw_content_chars),
                    "iterations": iterations,
                    "tool_calls_used": tool_calls_used,
                },
            )
            return StructuredTaskResult(
                ok=True,
                task_id=task_id,
                output=None,
                raw_content=raw_content,
                tool_history=tool_history,
                iterations=iterations,
                tool_calls_used=tool_calls_used,
                metadata={**metadata, "final_output_mode": "text"},
                llm_calls=list(llm_calls),
            )

        try:
            payload = parse_json_object(
                raw_content,
                target_name=task_id,
                contract=output_contract,
            )
        except StructuredOutputValidationError as exc:
            issue_summary = [
                f"{issue.validator}@{issue.instance_path or '/'}"
                for issue in exc.issues
            ]
            logger.warning(
                "Structured task final output failed local JSON Schema validation: %s",
                ", ".join(issue_summary),
                extra={
                    "task_id": task_id,
                    "contract_name": exc.contract_name,
                    "validation_issue_count": len(exc.issues),
                    "validation_issues": [
                        {
                            "validator": issue.validator,
                            "instance_path": issue.instance_path,
                            "schema_path": issue.schema_path,
                        }
                        for issue in exc.issues
                    ],
                    "iterations": iterations,
                    "tool_calls_used": tool_calls_used,
                },
            )
            return StructuredTaskResult(
                ok=False,
                task_id=task_id,
                raw_content=raw_content,
                failure_reason="Structured task output failed local JSON Schema validation.",
                tool_history=tool_history,
                iterations=iterations,
                tool_calls_used=tool_calls_used,
                metadata={**metadata, "validation_error": exc.to_dict()},
                llm_calls=list(llm_calls),
            )
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Structured task final schema output was invalid JSON",
                extra={
                    "task_id": task_id,
                    "raw_content_chars": raw_content_chars,
                    "raw_content_approx_tokens": _approx_token_count_from_chars(raw_content_chars),
                    "iterations": iterations,
                    "tool_calls_used": tool_calls_used,
                },
            )
            return StructuredTaskResult(
                ok=False,
                task_id=task_id,
                raw_content=raw_content,
                failure_reason="Structured task returned invalid JSON Schema output.",
                tool_history=tool_history,
                iterations=iterations,
                tool_calls_used=tool_calls_used,
                metadata=metadata,
                llm_calls=list(llm_calls),
            )

        output_compact_chars = _jsonish_char_count(payload)
        logger.info(
            "Structured task final schema output parsed",
            extra={
                "task_id": task_id,
                "raw_content_chars": raw_content_chars,
                "raw_content_approx_tokens": _approx_token_count_from_chars(raw_content_chars),
                "output_compact_chars": output_compact_chars,
                "output_compact_approx_tokens": _approx_token_count_from_chars(output_compact_chars),
                "output_top_level_key_count": len(payload),
                "iterations": iterations,
                "tool_calls_used": tool_calls_used,
            },
        )
        return StructuredTaskResult(
            ok=True,
            task_id=task_id,
            output=payload,
            raw_content=raw_content,
            tool_history=tool_history,
            iterations=iterations,
            tool_calls_used=tool_calls_used,
            metadata={**metadata, "final_output_mode": "json_schema"},
            llm_calls=list(llm_calls),
        )

    def _provider_accepts_options(self, method_name: str) -> bool:
        method = getattr(self.provider, method_name)
        try:
            parameters = signature(method).parameters.values()
        except (TypeError, ValueError):
            return True
        return any(parameter.kind == Parameter.VAR_KEYWORD or parameter.name == "options" for parameter in parameters)


StructuredTaskKernelBackend = Literal["native", "langgraph"]
StructuredTaskKernelRoute = Literal["model_request", "tools", "finalization", "result", "end"]


class StructuredTaskGraphState(TypedDict):
    """Ephemeral orchestration state; durable state remains in the checkpoint."""

    spec: StructuredTaskSpec
    context: ExecutionContext
    registry: ToolRegistry
    checkpoint: StructuredTaskCheckpoint
    on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None
    result: StructuredTaskResult | None


class StructuredTaskGraphUpdate(TypedDict, total=False):
    checkpoint: StructuredTaskCheckpoint
    result: StructuredTaskResult | None


class StructuredTaskOperations(Protocol):
    settings: CoreSettings

    def _continue_model_request(
        self,
        *,
        spec: StructuredTaskSpec,
        registry: ToolRegistry,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskResult | None: ...

    def _continue_tools(
        self,
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
        registry: ToolRegistry,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskResult | None: ...

    def _continue_finalization(
        self,
        *,
        spec: StructuredTaskSpec,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskResult | None: ...

    def _continue_persisted_result(
        self,
        *,
        spec: StructuredTaskSpec,
        checkpoint: StructuredTaskCheckpoint,
    ) -> StructuredTaskResult: ...


class StructuredTaskNodes:
    """Structured-task behavior shared by the native and LangGraph kernels."""

    def __init__(self, operations: StructuredTaskOperations) -> None:
        self.operations = operations

    @staticmethod
    def initial_state(
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
        registry: ToolRegistry,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskGraphState:
        return StructuredTaskGraphState(
            spec=spec,
            context=context,
            registry=registry,
            checkpoint=checkpoint,
            on_checkpoint=on_checkpoint,
            result=None,
        )

    @staticmethod
    def route(state: StructuredTaskGraphState) -> StructuredTaskKernelRoute:
        if state["result"] is not None:
            return "end"
        return state["checkpoint"].phase

    def model_request(self, state: StructuredTaskGraphState) -> StructuredTaskGraphUpdate:
        checkpoint = state["checkpoint"]
        result = self.operations._continue_model_request(
            spec=state["spec"],
            registry=state["registry"],
            checkpoint=checkpoint,
            on_checkpoint=state["on_checkpoint"],
        )
        return {"checkpoint": checkpoint, "result": result}

    def tools(self, state: StructuredTaskGraphState) -> StructuredTaskGraphUpdate:
        checkpoint = state["checkpoint"]
        result = self.operations._continue_tools(
            spec=state["spec"],
            context=state["context"],
            registry=state["registry"],
            checkpoint=checkpoint,
            on_checkpoint=state["on_checkpoint"],
        )
        return {"checkpoint": checkpoint, "result": result}

    def finalization(self, state: StructuredTaskGraphState) -> StructuredTaskGraphUpdate:
        checkpoint = state["checkpoint"]
        result = self.operations._continue_finalization(
            spec=state["spec"],
            checkpoint=checkpoint,
            on_checkpoint=state["on_checkpoint"],
        )
        return {"checkpoint": checkpoint, "result": result}

    def result(self, state: StructuredTaskGraphState) -> StructuredTaskGraphUpdate:
        return {
            "result": self.operations._continue_persisted_result(
                spec=state["spec"],
                checkpoint=state["checkpoint"],
            )
        }


class StructuredTaskKernel(Protocol):
    backend: StructuredTaskKernelBackend

    def run(self, initial_state: StructuredTaskGraphState) -> StructuredTaskResult: ...


class NativeStructuredTaskKernel:
    backend: StructuredTaskKernelBackend = "native"

    def __init__(self, nodes: StructuredTaskNodes) -> None:
        self.nodes = nodes

    def run(self, initial_state: StructuredTaskGraphState) -> StructuredTaskResult:
        state = initial_state
        while True:
            route = self.nodes.route(state)
            if route == "end":
                break
            node = getattr(self.nodes, route)
            state.update(node(state))

        result = state["result"]
        if result is None:
            raise RuntimeError("Native structured task kernel completed without a result")
        return result


class LangGraphStructuredTaskKernel:
    backend: StructuredTaskKernelBackend = "langgraph"

    def __init__(self, nodes: StructuredTaskNodes) -> None:
        from langgraph.graph import END, START, StateGraph

        self.nodes = nodes
        builder = StateGraph(StructuredTaskGraphState)
        builder.add_node("model_request", nodes.model_request)
        builder.add_node("tools", nodes.tools)
        builder.add_node("finalization", nodes.finalization)
        builder.add_node("result", nodes.result)
        routes: dict[Hashable, str] = {
            "model_request": "model_request",
            "tools": "tools",
            "finalization": "finalization",
            "result": "result",
            "end": END,
        }
        builder.add_conditional_edges(START, nodes.route, routes)
        builder.add_conditional_edges("model_request", nodes.route, routes)
        builder.add_conditional_edges("tools", nodes.route, routes)
        builder.add_conditional_edges("finalization", nodes.route, routes)
        builder.add_edge("result", END)
        self.graph = builder.compile()

    def run(self, initial_state: StructuredTaskGraphState) -> StructuredTaskResult:
        import langsmith as ls

        spec = initial_state["spec"]
        recursion_limit = max(25, (spec.max_iterations * 2) + 8)
        with ls.tracing_context(enabled=self.nodes.operations.settings.langchain_tracing_enabled):
            final_state = cast(
                StructuredTaskGraphState,
                self.graph.invoke(initial_state, {"recursion_limit": recursion_limit}),
            )
        result = final_state["result"]
        if result is None:
            raise RuntimeError("LangGraph structured task kernel completed without a result")
        return result


def build_structured_task_kernel(
    *,
    backend: str,
    operations: StructuredTaskOperations,
) -> tuple[StructuredTaskNodes, StructuredTaskKernel]:
    normalized = normalize_agent_kernel_backend(backend)
    nodes = StructuredTaskNodes(operations)
    if normalized == "native":
        return nodes, NativeStructuredTaskKernel(nodes)
    if normalized == "langgraph":
        return nodes, LangGraphStructuredTaskKernel(nodes)
    raise ValueError(
        f"Unsupported agent kernel backend: {backend!r}. Expected 'native' or 'langgraph'."
    )
