from __future__ import annotations

import json
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any

from agent_core.execution_context import (
    ExecutionContext,
    effective_allowed_http_hosts,
    effective_allowed_http_methods,
    effective_allowed_read_roots,
)
from agent_core.llm.base import BaseLLMProvider, LLMCallOptions, LLMCompletionResult, LLMMessage
from agent_core.llm.errors import LLMProviderError
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.policy_engine import PolicyEngine
from agent_core.settings import CoreSettings
from agent_core.tool_registry import ToolRegistry
from agent_core.types import ToolExecutionStatus, build_empty_session_state

logger = get_logger(__name__)


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


def _clean_positive_int(value: object, *, default: int, minimum: int = 0) -> int:
    try:
        normalized = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, normalized)


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
    output_schema: dict[str, Any] = field(default_factory=dict)
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
        if not isinstance(self.output_schema, dict):
            self.output_schema = {}
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
        session_id: str = "default",
        session_state: dict[str, Any] | None = None,
    ) -> StructuredTaskResult:
        try:
            registry = self.tool_registry.build_subset(spec.allowed_tools)
        except KeyError as exc:
            return StructuredTaskResult(
                ok=False,
                task_id=spec.task_id,
                failure_reason=str(exc),
            )

        context = ExecutionContext(
            session_id=session_id,
            settings=self.settings,
            session_state=session_state or build_empty_session_state(session_id=session_id),
        )
        messages = self._build_messages(spec=spec, session_id=session_id, session_state=context.session_state)
        tool_history: list[dict[str, Any]] = []
        iterations = 0
        tool_calls_used = 0

        while iterations < spec.max_iterations:
            iterations += 1
            logger.debug(
                "Calling structured task LLM",
                extra={
                    "task_id": spec.task_id,
                    "iteration": iterations,
                    "tool_count": len(registry.list_tool_names()),
                },
            )
            try:
                llm_response = self._call_model_once(
                    spec=spec,
                    messages=messages,
                    registry=registry,
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
                    tool_history=tool_history,
                    iterations=iterations,
                    tool_calls_used=tool_calls_used,
                )

            assistant_message = LLMMessage(
                role="assistant",
                content=llm_response.content,
                tool_calls=list(llm_response.tool_calls),
            )
            messages.append(assistant_message)

            if not llm_response.tool_calls:
                return self._finalize_result(
                    task_id=spec.task_id,
                    raw_content=llm_response.content,
                    tool_history=tool_history,
                    iterations=iterations,
                    tool_calls_used=tool_calls_used,
                    metadata={"model": spec.model or self.settings.model},
                )

            if tool_calls_used >= spec.max_tool_calls:
                return self._finalize_after_budget(
                    spec=spec,
                    messages=messages,
                    raw_failure_content=llm_response.content,
                    failure_reason="Maximum number of structured task tool calls reached.",
                    tool_history=tool_history,
                    iterations=iterations,
                    tool_calls_used=tool_calls_used,
                )

            for tool_call in llm_response.tool_calls:
                if tool_calls_used >= spec.max_tool_calls:
                    return self._finalize_after_budget(
                        spec=spec,
                        messages=messages,
                        raw_failure_content=llm_response.content,
                        failure_reason="Maximum number of structured task tool calls reached.",
                        tool_history=tool_history,
                        iterations=iterations,
                        tool_calls_used=tool_calls_used,
                    )

                tool_calls_used += 1
                tool_message, history_item = self._execute_tool_call(
                    registry=registry,
                    tool_name=tool_call.name,
                    arguments_json=tool_call.arguments_json,
                    tool_call_id=tool_call.id,
                    context=context,
                )
                messages.append(tool_message)
                tool_history.append(history_item)

        return self._finalize_after_budget(
            spec=spec,
            messages=messages,
            raw_failure_content="",
            failure_reason="Maximum number of structured task iterations reached.",
            tool_history=tool_history,
            iterations=iterations,
            tool_calls_used=tool_calls_used,
        )

    def _call_model_once(
        self,
        *,
        spec: StructuredTaskSpec,
        messages: list[LLMMessage],
        registry: ToolRegistry,
    ) -> LLMCompletionResult:
        options = LLMCallOptions(
            response_format={"type": "json_object"},
            metadata={"structured_task_id": spec.task_id, **spec.metadata},
        )
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": registry.get_tool_specs(),
            "model": spec.model or self.settings.model,
            "temperature": spec.temperature if spec.temperature is not None else self.settings.temperature,
        }
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
            response_format={"type": "json_object"},
            metadata={
                "structured_task_id": spec.task_id,
                "structured_task_finalization": True,
                **spec.metadata,
            },
        )
        final_messages = [
            *messages,
            LLMMessage(
                role="system",
                content=(
                    f"{failure_reason} No more tools are available. "
                    "Return the best possible final JSON object now, using only the evidence already present "
                    "in the transcript. Do not request tools. Do not wrap the JSON in markdown fences."
                ),
            ),
        ]
        kwargs: dict[str, Any] = {
            "messages": final_messages,
            "tools": [],
            "model": spec.model or self.settings.model,
            "temperature": spec.temperature if spec.temperature is not None else self.settings.temperature,
        }
        if self._provider_accepts_options("complete_with_tools"):
            kwargs["options"] = options
        return self.provider.complete_with_tools(**kwargs)

    def _build_messages(
        self,
        *,
        spec: StructuredTaskSpec,
        session_id: str,
        session_state: dict[str, Any],
    ) -> list[LLMMessage]:
        return [
            LLMMessage(role="system", content=self.settings.base_system_prompt),
            LLMMessage(role="system", content=self._build_task_system_prompt(spec=spec)),
            LLMMessage(role="system", content=self._build_scope_prompt_block(session_id=session_id, session_state=session_state)),
            LLMMessage(role="user", content=json.dumps(spec.to_payload(), ensure_ascii=False, indent=2)),
        ]

    def _build_task_system_prompt(self, *, spec: StructuredTaskSpec) -> str:
        output_schema = spec.output_schema or {"type": "object", "additionalProperties": True}
        lines = [
            f"Structured task id: {spec.task_id}",
            spec.system_prompt,
            "",
            "Structured task operating rules:",
            "- You are running a bounded task invoked by a higher-level controller.",
            "- Use only the tools exposed in this run. Do not assume hidden tools exist.",
            "- Drive the task to a bounded conclusion within this one run.",
            "- Base conclusions only on observed evidence and tool results.",
            "- Return exactly one JSON object when you are done.",
            "- Do not wrap the final JSON in markdown fences.",
            "",
            "Output schema or caller hint:",
            json.dumps(output_schema, ensure_ascii=False, indent=2),
        ]
        return "\n".join(lines)

    def _build_scope_prompt_block(self, *, session_id: str, session_state: dict[str, Any]) -> str:
        allowed_roots = [str(path.resolve()) for path in effective_allowed_read_roots(self.settings, session_state)]
        knowledge_root = str(self.settings.knowledge_base_dir.resolve())
        allowed_hosts = effective_allowed_http_hosts(self.settings, session_state)
        allowed_methods = effective_allowed_http_methods(self.settings, session_state)
        lines = [
            "Execution scope:",
            f"- Parent session ID: {session_id}",
            "- Allowed local code roots:",
        ]
        if allowed_roots:
            lines.extend(f"  - {root}" for root in allowed_roots)
        else:
            lines.append("  - none")

        lines.extend(
            [
                "- Allowed knowledge base root:",
                f"  - {knowledge_root}",
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
            authz = self.policy_engine.authorize(tool_name, arguments, context)
            if not authz.allowed:
                tool_content = f"Tool denied by policy: {authz.reason}"
                tool_status = "policy_denied"
            else:
                try:
                    result = registry.execute(tool_name, arguments, context)
                    tool_content = result.content
                    tool_status = "ok" if result.ok else "tool_error"
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
        raw_content: str,
        tool_history: list[dict[str, Any]],
        iterations: int,
        tool_calls_used: int,
        metadata: dict[str, Any],
    ) -> StructuredTaskResult:
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            return StructuredTaskResult(
                ok=False,
                task_id=task_id,
                raw_content=raw_content,
                failure_reason="Structured task returned invalid JSON output.",
                tool_history=tool_history,
                iterations=iterations,
                tool_calls_used=tool_calls_used,
                metadata=metadata,
            )

        if not isinstance(payload, dict):
            return StructuredTaskResult(
                ok=False,
                task_id=task_id,
                raw_content=raw_content,
                failure_reason="Structured task returned a non-object JSON output.",
                tool_history=tool_history,
                iterations=iterations,
                tool_calls_used=tool_calls_used,
                metadata=metadata,
            )

        return StructuredTaskResult(
            ok=True,
            task_id=task_id,
            output=payload,
            raw_content=raw_content,
            tool_history=tool_history,
            iterations=iterations,
            tool_calls_used=tool_calls_used,
            metadata=metadata,
        )

    def _finalize_after_budget(
        self,
        *,
        spec: StructuredTaskSpec,
        messages: list[LLMMessage],
        raw_failure_content: str,
        failure_reason: str,
        tool_history: list[dict[str, Any]],
        iterations: int,
        tool_calls_used: int,
    ) -> StructuredTaskResult:
        try:
            llm_response = self._call_model_for_final_output(
                spec=spec,
                messages=messages,
                failure_reason=failure_reason,
            )
        except LLMProviderError as exc:
            return StructuredTaskResult(
                ok=False,
                task_id=spec.task_id,
                raw_content=exc.detail or raw_failure_content,
                failure_reason=f"{failure_reason}; finalization failed: {exc.user_message}",
                tool_history=tool_history,
                iterations=iterations,
                tool_calls_used=tool_calls_used,
                metadata={"forced_finalization": True},
            )

        if llm_response.tool_calls:
            return StructuredTaskResult(
                ok=False,
                task_id=spec.task_id,
                raw_content=llm_response.content or raw_failure_content,
                failure_reason=f"{failure_reason}; finalization still requested tools.",
                tool_history=tool_history,
                iterations=iterations + 1,
                tool_calls_used=tool_calls_used,
                metadata={"forced_finalization": True},
            )

        finalized = self._finalize_result(
            task_id=spec.task_id,
            raw_content=llm_response.content,
            tool_history=tool_history,
            iterations=iterations + 1,
            tool_calls_used=tool_calls_used,
            metadata={
                "model": spec.model or self.settings.model,
                "forced_finalization": True,
                "budget_failure_reason": failure_reason,
            },
        )
        if finalized.ok:
            return finalized
        finalized.failure_reason = f"{failure_reason}; {finalized.failure_reason}"
        return finalized

    def _provider_accepts_options(self, method_name: str) -> bool:
        method = getattr(self.provider, method_name)
        try:
            parameters = signature(method).parameters.values()
        except (TypeError, ValueError):
            return True
        return any(parameter.kind == Parameter.VAR_KEYWORD or parameter.name == "options" for parameter in parameters)
