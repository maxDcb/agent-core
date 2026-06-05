from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from openai import BadRequestError


TemperatureMode = Literal["supported", "omit"]
TokenLimitParameter = Literal["max_tokens", "max_completion_tokens"]

ALL_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
GPT_5_1_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high"})
PRE_GPT_5_1_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})
GPT_PRO_REASONING_EFFORTS = frozenset({"high"})


@dataclass(frozen=True, slots=True)
class OpenAIModelCapabilities:
    family: str
    known: bool
    temperature_mode: TemperatureMode = "supported"
    supports_reasoning_effort: bool = False
    supported_reasoning_efforts: frozenset[str] = field(default_factory=frozenset)
    token_limit_parameter: TokenLimitParameter = "max_tokens"
    supports_parallel_tool_calls: bool = True


@dataclass(frozen=True, slots=True)
class OpenAIRequestNormalization:
    request: dict[str, Any]
    changes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OpenAIRequestRetryAction:
    parameter: str
    replacement_parameter: str | None = None


class OpenAIModelCapabilityResolver:
    """Resolve static model capabilities and remember runtime parameter rejections."""

    def __init__(self, overrides: dict[str, OpenAIModelCapabilities] | None = None) -> None:
        self._overrides = {_model_key(model): capabilities for model, capabilities in (overrides or {}).items()}
        self._runtime_unsupported_parameters: dict[str, set[str]] = {}

    def resolve(self, model: str) -> OpenAIModelCapabilities:
        key = _model_key(model)
        if key in self._overrides:
            return self._overrides[key]
        return _static_capabilities_for_model(key)

    def unsupported_parameters_for(self, model: str) -> frozenset[str]:
        return frozenset(self._runtime_unsupported_parameters.get(_model_key(model), set()))

    def record_unsupported_parameter(self, model: str, parameter: str) -> None:
        self._runtime_unsupported_parameters.setdefault(_model_key(model), set()).add(parameter)


class OpenAIChatRequestNormalizer:
    def __init__(self, resolver: OpenAIModelCapabilityResolver | None = None) -> None:
        self.resolver = resolver or OpenAIModelCapabilityResolver()

    def normalize(self, request: dict[str, Any]) -> OpenAIRequestNormalization:
        normalized = dict(request)
        model = str(normalized.get("model", ""))
        capabilities = self.resolver.resolve(model)
        changes: list[str] = []

        self._apply_static_capabilities(normalized, capabilities, changes)
        self._apply_runtime_rejections(normalized, model, changes)

        return OpenAIRequestNormalization(request=normalized, changes=tuple(changes))

    def _apply_static_capabilities(
        self,
        request: dict[str, Any],
        capabilities: OpenAIModelCapabilities,
        changes: list[str],
    ) -> None:
        if request.get("temperature") is not None and capabilities.temperature_mode == "omit":
            request.pop("temperature", None)
            changes.append(f"omitted temperature for {capabilities.family}")

        if "reasoning_effort" in request and capabilities.known:
            effort = str(request["reasoning_effort"])
            if not capabilities.supports_reasoning_effort:
                request.pop("reasoning_effort", None)
                changes.append(f"omitted reasoning_effort for {capabilities.family}")
            elif capabilities.supported_reasoning_efforts and effort not in capabilities.supported_reasoning_efforts:
                request.pop("reasoning_effort", None)
                changes.append(f"omitted unsupported reasoning_effort={effort!r} for {capabilities.family}")

        if "max_tokens" in request and capabilities.token_limit_parameter == "max_completion_tokens":
            request["max_completion_tokens"] = request.pop("max_tokens")
            changes.append(f"mapped max_tokens to max_completion_tokens for {capabilities.family}")

        if request.get("parallel_tool_calls") is not None and not capabilities.supports_parallel_tool_calls:
            request.pop("parallel_tool_calls", None)
            changes.append(f"omitted parallel_tool_calls for {capabilities.family}")

    def _apply_runtime_rejections(self, request: dict[str, Any], model: str, changes: list[str]) -> None:
        for parameter in sorted(self.resolver.unsupported_parameters_for(model)):
            if parameter not in request:
                continue

            if parameter == "max_tokens":
                request["max_completion_tokens"] = request.pop("max_tokens")
                changes.append("mapped max_tokens to max_completion_tokens based on runtime rejection")
                continue

            request.pop(parameter, None)
            changes.append(f"omitted {parameter} based on runtime rejection")


def select_bad_request_retry_action(
    exc: BadRequestError,
    request: dict[str, Any],
) -> OpenAIRequestRetryAction | None:
    message = _bad_request_message(exc)
    explicit_param = _bad_request_param(exc)

    if "reasoning_effort" in request and _is_parameter_rejected(
        parameter="reasoning_effort",
        explicit_param=explicit_param,
        message=message,
        fragments=("unrecognized", "unsupported", "unknown", "not supported", "invalid request argument"),
    ):
        return OpenAIRequestRetryAction(parameter="reasoning_effort")

    if "temperature" in request and _is_parameter_rejected(
        parameter="temperature",
        explicit_param=explicit_param,
        message=message,
        fragments=("unsupported", "does not support", "only the default", "unsupported_value"),
    ):
        return OpenAIRequestRetryAction(parameter="temperature")

    if "max_tokens" in request and _is_parameter_rejected(
        parameter="max_tokens",
        explicit_param=explicit_param,
        message=message,
        fragments=("not compatible", "unsupported", "deprecated", "use max_completion_tokens"),
    ):
        return OpenAIRequestRetryAction(parameter="max_tokens", replacement_parameter="max_completion_tokens")

    if "parallel_tool_calls" in request and _is_parameter_rejected(
        parameter="parallel_tool_calls",
        explicit_param=explicit_param,
        message=message,
        fragments=("unrecognized", "unsupported", "unknown", "not supported", "invalid request argument"),
    ):
        return OpenAIRequestRetryAction(parameter="parallel_tool_calls")

    if "response_format" in request and _is_parameter_rejected(
        parameter="response_format",
        explicit_param=explicit_param,
        message=message,
        fragments=(
            "unrecognized",
            "unsupported",
            "unknown",
            "not supported",
            "invalid request argument",
            "json_schema",
            "schema",
        ),
    ):
        return OpenAIRequestRetryAction(parameter="response_format")

    return None


def _static_capabilities_for_model(model_key: str) -> OpenAIModelCapabilities:
    if _looks_like_gpt_pro(model_key):
        return OpenAIModelCapabilities(
            family="gpt-pro",
            known=True,
            temperature_mode="omit",
            supports_reasoning_effort=True,
            supported_reasoning_efforts=GPT_PRO_REASONING_EFFORTS,
            token_limit_parameter="max_completion_tokens",
        )

    if _looks_like_gpt_5(model_key):
        efforts = GPT_5_1_REASONING_EFFORTS if "gpt-5.1" in model_key else ALL_REASONING_EFFORTS
        return OpenAIModelCapabilities(
            family="gpt-5",
            known=True,
            temperature_mode="omit",
            supports_reasoning_effort=True,
            supported_reasoning_efforts=efforts,
            token_limit_parameter="max_completion_tokens",
        )

    if _looks_like_o_series(model_key):
        return OpenAIModelCapabilities(
            family="o-series",
            known=True,
            temperature_mode="omit",
            supports_reasoning_effort=True,
            supported_reasoning_efforts=PRE_GPT_5_1_REASONING_EFFORTS,
            token_limit_parameter="max_completion_tokens",
        )

    if _looks_like_gpt_4_chat(model_key):
        return OpenAIModelCapabilities(
            family="gpt-4-chat",
            known=True,
            temperature_mode="supported",
            supports_reasoning_effort=False,
            token_limit_parameter="max_tokens",
        )

    return OpenAIModelCapabilities(
        family="unknown",
        known=False,
        temperature_mode="supported",
        supports_reasoning_effort=True,
        token_limit_parameter="max_tokens",
    )


def _model_key(model: str) -> str:
    return model.strip().lower().replace("_", "-")


def _looks_like_gpt_pro(model_key: str) -> bool:
    return "gpt-5-pro" in model_key or "gpt-5.2-pro" in model_key or "gpt-5.5-pro" in model_key


def _looks_like_gpt_5(model_key: str) -> bool:
    return "gpt-5" in model_key


def _looks_like_o_series(model_key: str) -> bool:
    return any(fragment in model_key for fragment in ("o1", "o3", "o4"))


def _looks_like_gpt_4_chat(model_key: str) -> bool:
    return "gpt-4.1" in model_key or "gpt-4o" in model_key


def _bad_request_message(exc: BadRequestError) -> str:
    return str(exc).lower()


def _bad_request_param(exc: BadRequestError) -> str | None:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    param = error.get("param")
    if isinstance(param, str) and param:
        return param
    return None


def _is_parameter_rejected(
    *,
    parameter: str,
    explicit_param: str | None,
    message: str,
    fragments: tuple[str, ...],
) -> bool:
    if explicit_param == parameter:
        return True
    return parameter in message and any(fragment in message for fragment in fragments)
