from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

LLMMessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class LLMToolCall:
    id: str
    name: str
    arguments_json: str

    @classmethod
    def from_history_dict(cls, payload: dict[str, Any]) -> LLMToolCall | None:
        tool_call_id = payload.get("id")
        function_payload = payload.get("function")
        if not isinstance(tool_call_id, str) or not isinstance(function_payload, dict):
            return None

        tool_name = function_payload.get("name")
        arguments_json = function_payload.get("arguments", "{}")
        if not isinstance(tool_name, str):
            return None

        if not isinstance(arguments_json, str):
            arguments_json = "{}"

        return cls(id=tool_call_id, name=tool_name, arguments_json=arguments_json)

    def to_history_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }


@dataclass(slots=True)
class LLMMessage:
    role: LLMMessageRole
    content: str
    tool_call_id: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_history_dict(cls, payload: dict[str, Any]) -> LLMMessage:
        role = str(payload.get("role", "user"))
        content = str(payload.get("content", ""))
        tool_calls = [
            tool_call
            for item in payload.get("tool_calls", []) or []
            if isinstance(item, dict)
            for tool_call in [LLMToolCall.from_history_dict(item)]
            if tool_call is not None
        ]
        tool_call_id = payload.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            tool_call_id = None
        metadata = payload.get("_agent_core")
        if not isinstance(metadata, dict):
            metadata = {}
        normalized_role = cast(LLMMessageRole, role if role in {"system", "user", "assistant", "tool"} else "user")
        return cls(
            role=normalized_role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
            metadata=dict(metadata),
        )

    def to_history_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.to_history_dict() for tool_call in self.tool_calls]
        return payload


@dataclass(slots=True)
class LLMToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class LLMCompletionResult:
    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    usage: LLMTokenUsage | None = None
    provider: str | None = None
    model: str | None = None
    provider_request_id: str | None = None
    duration_seconds: float | None = None
    provider_attempts: int = 1


@dataclass(slots=True)
class LLMTokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    source: Literal["provider"] = "provider"

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: object) -> LLMTokenUsage | None:
        if not isinstance(payload, dict):
            return None
        input_tokens = _non_negative_int(payload.get("input_tokens"))
        output_tokens = _non_negative_int(payload.get("output_tokens"))
        total_tokens = _non_negative_int(payload.get("total_tokens"))
        if input_tokens is None or output_tokens is None:
            return None
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens if total_tokens is not None else input_tokens + output_tokens,
            cached_input_tokens=_non_negative_int(payload.get("cached_input_tokens")),
            cache_creation_input_tokens=_non_negative_int(payload.get("cache_creation_input_tokens")),
            reasoning_output_tokens=_non_negative_int(payload.get("reasoning_output_tokens")),
        )


@dataclass(slots=True)
class LLMCallRecord:
    call_id: str
    call_index: int
    purpose: str
    provider: str | None
    model: str | None
    usage: LLMTokenUsage | None = None
    provider_request_id: str | None = None
    duration_seconds: float | None = None
    provider_attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        usage = self.usage
        return {
            "call_id": self.call_id,
            "call_index": self.call_index,
            "purpose": self.purpose,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": usage.input_tokens if usage is not None else None,
            "output_tokens": usage.output_tokens if usage is not None else None,
            "total_tokens": usage.total_tokens if usage is not None else None,
            "usage": usage.to_dict() if usage is not None else None,
            "usage_source": usage.source if usage is not None else "unavailable",
            "provider_request_id": self.provider_request_id,
            "duration_seconds": self.duration_seconds,
            "provider_attempts": self.provider_attempts,
        }

    @classmethod
    def from_dict(cls, payload: object) -> LLMCallRecord | None:
        if not isinstance(payload, dict):
            return None
        call_id = payload.get("call_id")
        call_index = _non_negative_int(payload.get("call_index"))
        if not isinstance(call_id, str) or not call_id or call_index is None:
            return None
        duration = payload.get("duration_seconds")
        return cls(
            call_id=call_id,
            call_index=call_index,
            purpose=str(payload.get("purpose") or "unspecified"),
            provider=_optional_text(payload.get("provider")),
            model=_optional_text(payload.get("model")),
            usage=LLMTokenUsage.from_dict(payload.get("usage") or payload),
            provider_request_id=_optional_text(payload.get("provider_request_id")),
            duration_seconds=float(duration) if isinstance(duration, (int, float)) and duration >= 0 else None,
            provider_attempts=max(1, _non_negative_int(payload.get("provider_attempts")) or 1),
        )

    @classmethod
    def from_completion(
        cls,
        completion: LLMCompletionResult,
        *,
        call_index: int,
        purpose: str,
    ) -> LLMCallRecord:
        return cls(
            call_id=f"llm-{call_index:04d}",
            call_index=call_index,
            purpose=purpose,
            provider=completion.provider,
            model=completion.model,
            usage=completion.usage,
            provider_request_id=completion.provider_request_id,
            duration_seconds=completion.duration_seconds,
            provider_attempts=max(1, completion.provider_attempts),
        )


@dataclass(slots=True)
class LLMUsageSummary:
    call_count: int = 0
    calls_with_token_usage: int = 0
    token_usage_complete: bool = True
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    total_tokens: int | None = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "calls_with_token_usage": self.calls_with_token_usage,
            "token_usage_complete": self.token_usage_complete,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reported_input_tokens": self.reported_input_tokens,
            "reported_output_tokens": self.reported_output_tokens,
        }

    @classmethod
    def from_calls(cls, calls: list[LLMCallRecord]) -> LLMUsageSummary:
        with_usage = [call for call in calls if call.usage is not None]
        reported_input = sum(call.usage.input_tokens for call in with_usage if call.usage is not None)
        reported_output = sum(call.usage.output_tokens for call in with_usage if call.usage is not None)
        complete = len(with_usage) == len(calls)
        return cls(
            call_count=len(calls),
            calls_with_token_usage=len(with_usage),
            token_usage_complete=complete,
            input_tokens=reported_input if complete else None,
            output_tokens=reported_output if complete else None,
            total_tokens=reported_input + reported_output if complete else None,
            reported_input_tokens=reported_input,
            reported_output_tokens=reported_output,
        )


_LLM_CALL_COLLECTOR: ContextVar[list[LLMCallRecord] | None] = ContextVar("llm_call_collector", default=None)


@contextmanager
def capture_llm_calls() -> Iterator[list[LLMCallRecord]]:
    calls: list[LLMCallRecord] = []
    token = _LLM_CALL_COLLECTOR.set(calls)
    try:
        yield calls
    finally:
        _LLM_CALL_COLLECTOR.reset(token)


def publish_llm_completion(completion: LLMCompletionResult, *, purpose: str = "unspecified") -> LLMCompletionResult:
    collector = _LLM_CALL_COLLECTOR.get()
    if collector is not None:
        collector.append(
            LLMCallRecord.from_completion(completion, call_index=len(collector) + 1, purpose=purpose)
        )
    return completion


def completion_content(value: LLMCompletionResult | str) -> str:
    return value.content if isinstance(value, LLMCompletionResult) else value


def token_usage_from_openai_response(response: object) -> LLMTokenUsage | None:
    usage = getattr(response, "usage", None)
    input_tokens = _non_negative_int(getattr(usage, "prompt_tokens", None))
    output_tokens = _non_negative_int(getattr(usage, "completion_tokens", None))
    if input_tokens is None or output_tokens is None:
        return None
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return LLMTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_non_negative_int(getattr(usage, "total_tokens", None)) or input_tokens + output_tokens,
        cached_input_tokens=_non_negative_int(getattr(prompt_details, "cached_tokens", None)),
        reasoning_output_tokens=_non_negative_int(getattr(completion_details, "reasoning_tokens", None)),
    )


def token_usage_from_anthropic_response(response: object) -> LLMTokenUsage | None:
    usage = getattr(response, "usage", None)
    uncached_input_tokens = _non_negative_int(getattr(usage, "input_tokens", None))
    output_tokens = _non_negative_int(getattr(usage, "output_tokens", None))
    if uncached_input_tokens is None or output_tokens is None:
        return None
    cached_input_tokens = _non_negative_int(getattr(usage, "cache_read_input_tokens", None))
    cache_creation_input_tokens = _non_negative_int(getattr(usage, "cache_creation_input_tokens", None))
    input_tokens = uncached_input_tokens + (cached_input_tokens or 0) + (cache_creation_input_tokens or 0)
    return LLMTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )


def provider_request_id(response: object) -> str | None:
    return _optional_text(getattr(response, "id", None)) or _optional_text(getattr(response, "request_id", None))


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


@dataclass(slots=True)
class LLMCallOptions:
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    response_format: dict[str, Any] | None = None
    response_format_fallback: dict[str, Any] | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(Protocol):
    def complete_text(
        self,
        *,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        ...

    def complete_with_tools(
        self,
        *,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        ...
