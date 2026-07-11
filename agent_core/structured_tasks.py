from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any, Literal

from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import BaseLLMProvider, LLMCallOptions, LLMCompletionResult, LLMMessage, LLMToolCall
from agent_core.llm.errors import LLMProviderError
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.output_contracts import (
    StructuredOutputContract,
    StructuredOutputValidationError,
    parse_json_object,
)
from agent_core.policy_engine import PolicyEngine
from agent_core.settings import CoreSettings
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
    pending_tool_calls: list[StructuredToolCallCheckpoint] = field(default_factory=list)
    next_tool_call_index: int = 0
    finalization_kind: StructuredFinalizationKind | None = None
    finalization_reason: str = ""
    raw_failure_content: str = ""
    result_kind: StructuredResultKind | None = None
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "spec_fingerprint": self.spec_fingerprint,
            "phase": self.phase,
            "messages": [message.to_history_dict() for message in self.messages],
            "tool_history": [dict(item) for item in self.tool_history],
            "iterations": self.iterations,
            "tool_calls_used": self.tool_calls_used,
            "pending_tool_calls": [item.to_dict() for item in self.pending_tool_calls],
            "next_tool_call_index": self.next_tool_call_index,
            "finalization_kind": self.finalization_kind,
            "finalization_reason": self.finalization_reason,
            "raw_failure_content": self.raw_failure_content,
            "result_kind": self.result_kind,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, payload: object) -> StructuredTaskCheckpoint | None:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
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
        return cls(
            spec_fingerprint=spec_fingerprint,
            phase=phase,
            messages=messages,
            tool_history=[dict(item) for item in raw_history if isinstance(item, dict)] if isinstance(raw_history, list) else [],
            iterations=_clean_positive_int(payload.get("iterations"), default=0),
            tool_calls_used=_clean_positive_int(payload.get("tool_calls_used"), default=0),
            pending_tool_calls=pending,
            next_tool_call_index=_clean_positive_int(payload.get("next_tool_call_index"), default=0),
            finalization_kind=normalized_kind,
            finalization_reason=_clean_string(payload.get("finalization_reason")),
            raw_failure_content=str(payload.get("raw_failure_content") or ""),
            result_kind=normalized_result_kind,
            sequence=_clean_positive_int(payload.get("sequence"), default=0),
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
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.tool_registry = tool_registry
        self.policy_engine = policy_engine

    def run(
        self,
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None = None,
    ) -> StructuredTaskResult:
        checkpoint = StructuredTaskCheckpoint(
            spec_fingerprint=spec.fingerprint(),
            phase="model_request",
            messages=self._build_messages(spec=spec, context=context),
            iterations=1,
        )
        self._emit_checkpoint(checkpoint, on_checkpoint)
        return self._continue_from_checkpoint(
            spec=spec,
            context=context,
            checkpoint=checkpoint,
            on_checkpoint=on_checkpoint,
        )

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
        return self._continue_from_checkpoint(
            spec=spec,
            context=context,
            checkpoint=checkpoint,
            on_checkpoint=on_checkpoint,
        )

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

        while True:
            if checkpoint.phase == "result":
                return self._continue_persisted_result(spec=spec, checkpoint=checkpoint)

            if checkpoint.phase == "finalization":
                finalized = self._continue_finalization(
                    spec=spec,
                    checkpoint=checkpoint,
                    on_checkpoint=on_checkpoint,
                )
                if finalized is not None:
                    return finalized
                continue

            if checkpoint.phase == "tools":
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
                finalized = self._continue_tool_batch(
                    spec=spec,
                    context=context,
                    registry=registry,
                    checkpoint=checkpoint,
                    on_checkpoint=on_checkpoint,
                )
                if finalized is not None:
                    return finalized
                continue

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
                )

            assistant_message = LLMMessage(
                role="assistant",
                content=llm_response.content,
                tool_calls=list(llm_response.tool_calls),
            )
            checkpoint.messages.append(assistant_message)

            if not llm_response.tool_calls:
                if spec.output_contract is not None and registry.list_tool_names():
                    checkpoint.phase = "finalization"
                    checkpoint.finalization_kind = "contract"
                    checkpoint.finalization_reason = "Investigation is complete."
                    checkpoint.raw_failure_content = llm_response.content
                    self._emit_checkpoint(checkpoint, on_checkpoint)
                    continue
                checkpoint.phase = "result"
                checkpoint.result_kind = "direct"
                self._emit_checkpoint(checkpoint, on_checkpoint)
                continue

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

    def _continue_tool_batch(
        self,
        *,
        spec: StructuredTaskSpec,
        context: ExecutionContext,
        registry: ToolRegistry,
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> StructuredTaskResult | None:
        while checkpoint.next_tool_call_index < len(checkpoint.pending_tool_calls):
            if checkpoint.tool_calls_used >= spec.max_tool_calls:
                remaining = checkpoint.pending_tool_calls[checkpoint.next_tool_call_index :]
                self._append_budget_exhausted_tool_responses(
                    messages=checkpoint.messages,
                    tool_calls=[
                        LLMToolCall(
                            id=item.tool_call_id,
                            name=item.tool_name,
                            arguments_json=item.arguments_json,
                        )
                        for item in remaining
                    ],
                    tool_history=checkpoint.tool_history,
                )
                for item in remaining:
                    item.status = "budget_exhausted"
                checkpoint.next_tool_call_index = len(checkpoint.pending_tool_calls)
                checkpoint.phase = "finalization"
                checkpoint.finalization_kind = "budget"
                checkpoint.finalization_reason = "Maximum number of structured task tool calls reached."
                checkpoint.raw_failure_content = checkpoint.messages[-1].content if checkpoint.messages else ""
                self._emit_checkpoint(checkpoint, on_checkpoint)
                return None

            tool_call = checkpoint.pending_tool_calls[checkpoint.next_tool_call_index]
            tool_call.status = "running"
            checkpoint.tool_calls_used += 1
            self._emit_checkpoint(checkpoint, on_checkpoint)
            tool_message, history_item = self._execute_tool_call(
                registry=registry,
                tool_name=tool_call.tool_name,
                arguments_json=tool_call.arguments_json,
                tool_call_id=tool_call.tool_call_id,
                context=context,
            )
            checkpoint.messages.append(tool_message)
            checkpoint.tool_history.append(history_item)
            tool_call.status = "completed"
            checkpoint.next_tool_call_index += 1
            self._emit_checkpoint(checkpoint, on_checkpoint)

        checkpoint.pending_tool_calls = []
        checkpoint.next_tool_call_index = 0
        if checkpoint.iterations >= spec.max_iterations:
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
        )
        if checkpoint.result_kind == "budget" and not finalized.ok:
            finalized.failure_reason = f"{failure_reason}; {finalized.failure_reason}"
        return finalized

    @staticmethod
    def _emit_checkpoint(
        checkpoint: StructuredTaskCheckpoint,
        on_checkpoint: Callable[[StructuredTaskCheckpoint], None] | None,
    ) -> None:
        checkpoint.sequence += 1
        if on_checkpoint is not None:
            on_checkpoint(checkpoint)

    def _append_budget_exhausted_tool_responses(
        self,
        *,
        messages: list[LLMMessage],
        tool_calls: list[LLMToolCall],
        tool_history: list[dict[str, Any]],
    ) -> None:
        for tool_call in tool_calls:
            content = f"Tool call skipped: maximum tool-call budget reached before executing {tool_call.name}."
            messages.append(LLMMessage(role="tool", tool_call_id=tool_call.id, content=content))
            tool_history.append(
                {
                    "tool_name": tool_call.name,
                    "arguments": {},
                    "status": "budget_exhausted",
                    "content_preview": content,
                }
            )

    def _call_model_once(
        self,
        *,
        spec: StructuredTaskSpec,
        messages: list[LLMMessage],
        registry: ToolRegistry,
        final_output: bool = False,
    ) -> LLMCompletionResult:
        options = LLMCallOptions(
            response_format=_response_format_for_spec(spec, final_output=final_output),
            response_format_fallback=None,
            max_output_tokens=_clean_optional_positive_int(self.settings.llm_max_output_tokens) if final_output else None,
            metadata={"structured_task_id": spec.task_id, **spec.metadata},
        )
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": registry.get_tool_specs(),
            "model": spec.model or self.settings.model,
            "temperature": spec.temperature if spec.temperature is not None else self.settings.temperature,
        }
        logger.info(
            "Structured task LLM request prepared",
            extra={
                "task_id": spec.task_id,
                "model": kwargs["model"],
                "final_output": final_output,
                "tool_count": len(kwargs["tools"]),
                "response_format_type": _response_format_type(options.response_format),
                "response_format_chars": _jsonish_char_count(options.response_format),
                "response_format_fallback_type": _response_format_type(options.response_format_fallback),
                "max_output_tokens": options.max_output_tokens,
                **_message_content_stats(messages),
            },
        )
        if self._provider_accepts_options("complete_with_tools"):
            kwargs["options"] = options
        return self.provider.complete_with_tools(**kwargs)

    def _call_model_for_final_output(
        self,
        *,
        spec: StructuredTaskSpec,
        messages: list[LLMMessage],
        failure_reason: str,
    ) -> LLMCompletionResult:
        options = LLMCallOptions(
            response_format=_response_format_for_spec(spec, final_output=True),
            response_format_fallback=None,
            max_output_tokens=_clean_optional_positive_int(self.settings.llm_max_output_tokens),
            metadata={
                "structured_task_id": spec.task_id,
                "structured_task_finalization": True,
                **spec.metadata,
            },
        )
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
        final_messages = [
            *messages,
            LLMMessage(
                role="system",
                content=final_instruction,
            ),
        ]
        kwargs: dict[str, Any] = {
            "messages": final_messages,
            "tools": [],
            "model": spec.model or self.settings.model,
            "temperature": spec.temperature if spec.temperature is not None else self.settings.temperature,
        }
        logger.info(
            "Structured task finalization LLM request prepared",
            extra={
                "task_id": spec.task_id,
                "model": kwargs["model"],
                "failure_reason": failure_reason,
                "response_format_type": _response_format_type(options.response_format),
                "response_format_chars": _jsonish_char_count(options.response_format),
                "response_format_fallback_type": _response_format_type(options.response_format_fallback),
                "max_output_tokens": options.max_output_tokens,
                **_message_content_stats(final_messages),
            },
        )
        if self._provider_accepts_options("complete_with_tools"):
            kwargs["options"] = options
        return self.provider.complete_with_tools(**kwargs)

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
    ) -> tuple[LLMMessage, dict[str, Any]]:
        arguments: dict[str, Any]
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
            authz = self.policy_engine.authorize(tool_name, arguments, context)
            if not authz.allowed:
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
                    result = registry.execute(tool_name, arguments, context)
                    tool_content = result.content
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

        tool_message = LLMMessage(role="tool", tool_call_id=tool_call_id, content=tool_content)
        history_item = {
            "tool_name": tool_name,
            "arguments": arguments,
            "status": tool_status,
            "content_preview": safe_preview(tool_content, limit=500),
        }
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
        )

    def _provider_accepts_options(self, method_name: str) -> bool:
        method = getattr(self.provider, method_name)
        try:
            parameters = signature(method).parameters.values()
        except (TypeError, ValueError):
            return True
        return any(parameter.kind == Parameter.VAR_KEYWORD or parameter.name == "options" for parameter in parameters)
