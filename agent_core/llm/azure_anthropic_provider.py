from __future__ import annotations

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import json
import os
import random
import time
from typing import Any

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AnthropicError,
    AnthropicFoundry,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from agent_core.logging_utils import get_logger
from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage, LLMToolCall, LLMToolDefinition
from agent_core.llm.errors import LLMProviderError

logger = get_logger(__name__)

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RETRY_MAX_ATTEMPTS = 5
DEFAULT_RETRY_INITIAL_DELAY_SECONDS = 10.0
DEFAULT_RETRY_MAX_DELAY_SECONDS = 120.0
DEFAULT_RETRY_BACKOFF_MULTIPLIER = 2.0
DEFAULT_RETRY_JITTER_RATIO = 0.1
JSON_OBJECT_SYSTEM_INSTRUCTION = (
    "When a JSON object is requested, return only the raw JSON object. "
    "Do not wrap it in Markdown code fences and do not include prose before or after it."
)
FINAL_USER_TURN_AFTER_ASSISTANT = (
    "Continue from the prior assistant draft and return the requested final answer now. "
    "Follow the current system instructions exactly."
)
UNSUPPORTED_OUTPUT_SCHEMA_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "format",
    }
)


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _elapsed_since(started_at: float | None) -> float | None:
    if started_at is None:
        return None
    return round(time.monotonic() - started_at, 3)


@dataclass(frozen=True, slots=True)
class AzureAnthropicRetryPolicy:
    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS
    initial_delay_seconds: float = DEFAULT_RETRY_INITIAL_DELAY_SECONDS
    max_delay_seconds: float = DEFAULT_RETRY_MAX_DELAY_SECONDS
    backoff_multiplier: float = DEFAULT_RETRY_BACKOFF_MULTIPLIER
    jitter_ratio: float = DEFAULT_RETRY_JITTER_RATIO

    @classmethod
    def from_env(cls, prefix: str = "AGENT_CORE_LLM_RETRY_") -> "AzureAnthropicRetryPolicy":
        return cls(
            max_attempts=_env_int(f"{prefix}MAX_ATTEMPTS", DEFAULT_RETRY_MAX_ATTEMPTS),
            initial_delay_seconds=_env_float(
                f"{prefix}INITIAL_DELAY_SECONDS",
                DEFAULT_RETRY_INITIAL_DELAY_SECONDS,
            ),
            max_delay_seconds=_env_float(f"{prefix}MAX_DELAY_SECONDS", DEFAULT_RETRY_MAX_DELAY_SECONDS),
            backoff_multiplier=_env_float(f"{prefix}BACKOFF_MULTIPLIER", DEFAULT_RETRY_BACKOFF_MULTIPLIER),
            jitter_ratio=_env_float(f"{prefix}JITTER_RATIO", DEFAULT_RETRY_JITTER_RATIO),
        )

    def retry_delay_seconds(self, *, exc: BaseException, attempt_index: int, random_value: float) -> float:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(self.max_delay_seconds, max(0.0, retry_after))

        base_delay = self.initial_delay_seconds * (self.backoff_multiplier ** max(attempt_index - 1, 0))
        capped_delay = min(self.max_delay_seconds, max(0.0, base_delay))
        if self.jitter_ratio <= 0 or capped_delay <= 0:
            return capped_delay

        jitter_window = capped_delay * self.jitter_ratio
        jitter = (random_value * 2.0 - 1.0) * jitter_window
        return max(0.0, min(self.max_delay_seconds, capped_delay + jitter))


class AzureAnthropicProvider:
    """
    Azure Anthropic provider for Claude deployments exposed through the Azure Foundry
    Anthropic Messages API.

    This provider translates the internal, provider-agnostic contract into
    Anthropic-style message payloads so the orchestrator and tools remain unaware of
    backend-specific formats.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        anthropic_version: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.anthropic_version = anthropic_version or DEFAULT_ANTHROPIC_VERSION
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.endpoint_configured = bool(endpoint)
        self.api_key_configured = bool(api_key)
        self.client: AnthropicFoundry | None = None

        logger.debug(
            "Azure Anthropic provider initialized",
            extra={
                "endpoint_configured": self.endpoint_configured,
                "api_key_configured": self.api_key_configured,
                "api_version": self.api_version,
                "anthropic_version": self.anthropic_version,
                "timeout_seconds": self.timeout_seconds,
            },
        )

    def complete_text(
        self,
        *,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> str:
        response = self._create_message(
            messages=messages,
            tools=None,
            model=model,
            temperature=temperature,
            options=options,
        )
        content, tool_calls = self._parse_response(response)
        if tool_calls:
            logger.warning(
                "Azure Anthropic text-only completion returned unexpected tool calls",
                extra={"tool_call_count": len(tool_calls)},
            )
        return content

    def complete_with_tools(
        self,
        *,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        response = self._create_message(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            options=options,
        )
        content, tool_calls = self._parse_response(response)
        return LLMCompletionResult(content=content, tool_calls=tool_calls)

    def _create_message(
        self,
        *,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None,
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> Any:
        self._ensure_configured()

        system_prompt, anthropic_messages = self._to_anthropic_messages(messages)
        output_config = self._output_config_from_options(options)
        if self._uses_prompt_only_json_object(options, output_config=output_config):
            system_prompt = "\n\n".join(part for part in [system_prompt, JSON_OBJECT_SYSTEM_INSTRUCTION] if part)
        request: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "stream": False,
        }
        if options is not None and options.max_output_tokens is not None:
            request["max_tokens"] = options.max_output_tokens
        if system_prompt:
            request["system"] = system_prompt
        if tools:
            request["tools"] = [self._to_anthropic_tool(tool) for tool in tools]
            request["tool_choice"] = {"type": "auto"}

        if output_config is not None:
            request["output_config"] = output_config

        metadata = options.metadata if options is not None else {}
        logger.info(
            "Sending Azure Anthropic messages request",
            extra={
                "model": model,
                "message_count": len(anthropic_messages),
                "tool_count": len(tools or []),
                "api_version": self.api_version,
                "anthropic_version": self.anthropic_version,
                "timeout_seconds": self.timeout_seconds,
                "structured_task_id": metadata.get("structured_task_id"),
                "max_tokens": request.get("max_tokens"),
                "output_config_type": _get_value(_get_value(output_config, "format"), "type"),
            },
        )

        request_started_at: float | None = None
        try:
            client = self._get_client()
            request_started_at = time.monotonic()
            response = self._create_message_with_retry(
                messages_api=client.messages,
                request=request,
            )
            logger.info(
                "Received Azure Anthropic messages response",
                extra={
                    "model": model,
                    "elapsed_seconds": _elapsed_since(request_started_at),
                    "response_block_count": len(_get_value(response, "content", []) or []),
                },
            )
            return response
        except AuthenticationError as exc:
            logger.exception(
                "Azure Anthropic authentication failed",
                extra={"elapsed_seconds": _elapsed_since(request_started_at)},
            )
            raise LLMProviderError(
                kind="configuration_error",
                user_message="Azure Anthropic rejected the credentials. Check AZURE_ANTHROPIC_API_KEY and endpoint access.",
                detail=str(exc),
            ) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            logger.exception(
                "Azure Anthropic request failed due to connectivity or timeout after retries",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="request_error",
                user_message="The assistant could not reach Azure Anthropic. Check network access and try again.",
                detail=str(exc),
            ) from exc
        except RateLimitError as exc:
            logger.warning(
                "Azure Anthropic request was rate limited after retries",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="rate_limit_error",
                user_message="Azure Anthropic rate-limited the request. Wait briefly and try again.",
                detail=str(exc),
            ) from exc
        except BadRequestError as exc:
            logger.exception(
                "Azure Anthropic request was rejected by the API",
                extra={
                    "elapsed_seconds": _elapsed_since(request_started_at),
                    "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                    "provider_error_type": _provider_error_type(exc),
                },
            )
            raise LLMProviderError(
                kind="request_error",
                user_message="Azure Anthropic rejected the request. Review the deployment name, API version, and payload.",
                detail=_provider_error_detail(exc),
            ) from exc
        except APIStatusError as exc:
            status_code = getattr(exc, "status_code", None)
            provider_error_type = _provider_error_type(exc)
            if _is_retryable_transient_error(exc):
                logger.warning(
                    "Azure Anthropic transient API status persisted after retries",
                    extra={
                        "elapsed_seconds": _elapsed_since(request_started_at),
                        "error_type": type(exc).__name__,
                        "status_code": status_code,
                        "provider_error_type": provider_error_type,
                    },
                )
                if _is_overloaded_error(exc):
                    raise LLMProviderError(
                        kind="rate_limit_error",
                        user_message="Azure Anthropic is temporarily overloaded. Wait briefly and retry.",
                        detail=_provider_error_detail(exc),
                    ) from exc
                raise LLMProviderError(
                    kind="request_error",
                    user_message="Azure Anthropic returned a transient server error after retries. Wait briefly and retry.",
                    detail=_provider_error_detail(exc),
                ) from exc
            logger.exception(
                "Azure Anthropic request failed with API status",
                extra={
                    "elapsed_seconds": _elapsed_since(request_started_at),
                    "error_type": type(exc).__name__,
                    "status_code": status_code,
                    "provider_error_type": provider_error_type,
                },
            )
            raise LLMProviderError(
                kind="request_error",
                user_message="Azure Anthropic rejected the request. Review the deployment name, API version, and payload.",
                detail=_provider_error_detail(exc),
            ) from exc
        except AnthropicError as exc:
            logger.exception(
                "Unexpected Azure Anthropic SDK error",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="Azure Anthropic failed unexpectedly. Try again after checking the provider configuration.",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "Unexpected non-SDK Azure Anthropic provider error",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="The assistant encountered an unexpected Azure Anthropic provider failure.",
                detail=str(exc),
            ) from exc

    def _ensure_configured(self) -> None:
        if not self.endpoint_configured:
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The Azure Anthropic endpoint is not configured. Set AZURE_ANTHROPIC_ENDPOINT.",
                detail="Missing AZURE_ANTHROPIC_ENDPOINT for AzureAnthropicProvider",
            )
        if not self.api_key_configured:
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The Azure Anthropic API key is not configured. Set AZURE_ANTHROPIC_API_KEY.",
                detail="Missing AZURE_ANTHROPIC_API_KEY for AzureAnthropicProvider",
            )

    def _create_message_with_retry(self, *, messages_api: Any, request: dict[str, Any]) -> Any:
        policy = AzureAnthropicRetryPolicy.from_env()
        attempt = 1

        while True:
            attempt_started_at = time.monotonic()
            try:
                return messages_api.create(**request)
            except RateLimitError as exc:
                attempt = self._retry_after_transient_error(
                    exc=exc,
                    attempt=attempt,
                    elapsed_seconds=round(time.monotonic() - attempt_started_at, 3),
                    policy=policy,
                    request=request,
                    log_message="Azure Anthropic rate-limited messages request; retrying",
                )
            except (APIConnectionError, APITimeoutError) as exc:
                attempt = self._retry_after_transient_error(
                    exc=exc,
                    attempt=attempt,
                    elapsed_seconds=round(time.monotonic() - attempt_started_at, 3),
                    policy=policy,
                    request=request,
                    log_message="Azure Anthropic connectivity or timeout error; retrying",
                )
            except APIStatusError as exc:
                if not _is_retryable_transient_error(exc):
                    raise
                attempt = self._retry_after_transient_error(
                    exc=exc,
                    attempt=attempt,
                    elapsed_seconds=round(time.monotonic() - attempt_started_at, 3),
                    policy=policy,
                    request=request,
                    log_message="Azure Anthropic transient messages status error; retrying",
                )

    def _retry_after_transient_error(
        self,
        *,
        exc: BaseException,
        attempt: int,
        elapsed_seconds: float,
        policy: AzureAnthropicRetryPolicy,
        request: dict[str, Any],
        log_message: str,
    ) -> int:
        if attempt >= policy.max_attempts:
            raise exc

        delay_seconds = policy.retry_delay_seconds(
            exc=exc,
            attempt_index=attempt,
            random_value=random.random(),
        )
        logger.warning(
            log_message,
            extra={
                "model": request.get("model"),
                "attempt": attempt,
                "max_attempts": policy.max_attempts,
                "elapsed_seconds": elapsed_seconds,
                "delay_seconds": round(delay_seconds, 3),
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
                "provider_error_type": _provider_error_type(exc),
            },
        )
        time.sleep(delay_seconds)
        return attempt + 1

    def _get_client(self) -> AnthropicFoundry:
        if self.client is None:
            assert self.endpoint is not None
            assert self.api_key is not None
            self.client = AnthropicFoundry(
                api_key=self.api_key,
                base_url=self.endpoint,
                timeout=self.timeout_seconds,
                default_headers={"anthropic-version": self.anthropic_version},
                default_query={"api-version": self.api_version} if self.api_version else None,
            )
        return self.client

    def _parse_response(self, response: Any) -> tuple[str, list[LLMToolCall]]:
        content_blocks = _get_value(response, "content")
        if not isinstance(content_blocks, list):
            raise LLMProviderError(
                kind="response_error",
                user_message="Azure Anthropic returned an unusable response.",
                detail="Response content field was not a list",
            )

        text_parts: list[str] = []
        tool_calls: list[LLMToolCall] = []

        for block in content_blocks:
            block_type = _get_value(block, "type")
            if block_type == "text":
                text = _get_value(block, "text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                tool_id = _get_value(block, "id")
                tool_name = _get_value(block, "name")
                tool_input = _get_value(block, "input", {})
                if isinstance(tool_id, str) and isinstance(tool_name, str):
                    tool_calls.append(
                        LLMToolCall(
                            id=tool_id,
                            name=tool_name,
                            arguments_json=json.dumps(
                                tool_input if isinstance(tool_input, dict) else {},
                                ensure_ascii=False,
                            ),
                        )
                    )

        return "\n".join(text_parts).strip(), tool_calls

    def _to_anthropic_tool(self, tool: LLMToolDefinition) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _to_anthropic_messages(self, messages: list[LLMMessage]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue

            anthropic_role = "assistant" if message.role == "assistant" else "user"
            content_blocks = self._message_to_content_blocks(message)
            if not content_blocks:
                continue

            if converted and converted[-1]["role"] == anthropic_role:
                converted[-1]["content"].extend(content_blocks)
            else:
                converted.append({"role": anthropic_role, "content": content_blocks})

        if converted and converted[-1]["role"] == "assistant":
            converted.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": FINAL_USER_TURN_AFTER_ASSISTANT}],
                }
            )

        return "\n\n".join(system_parts).strip(), converted

    def _message_to_content_blocks(self, message: LLMMessage) -> list[dict[str, Any]]:
        if message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for tool_call in message.tool_calls:
                try:
                    tool_input = json.loads(tool_call.arguments_json or "{}")
                except json.JSONDecodeError:
                    tool_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_input if isinstance(tool_input, dict) else {},
                    }
                )
            return blocks

        if message.role == "tool":
            if not message.tool_call_id:
                return []
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
            ]

        if not message.content:
            return []
        return [{"type": "text", "text": message.content}]

    def _output_config_from_options(self, options: LLMCallOptions | None) -> dict[str, Any] | None:
        if options is None:
            return None

        response_format, rejection_reasons = self._response_format_candidate(options.response_format)
        if self._is_json_schema_format(options.response_format) and rejection_reasons:
            self._raise_unenforceable_json_schema(rejection_reasons)

        if response_format is None and options.response_format_fallback is not None:
            response_format, fallback_rejection_reasons = self._response_format_candidate(options.response_format_fallback)
            if self._is_json_schema_format(options.response_format_fallback) and fallback_rejection_reasons:
                self._raise_unenforceable_json_schema(fallback_rejection_reasons)
            rejection_reasons.extend(fallback_rejection_reasons)

        if response_format is None:
            if rejection_reasons and options.response_format_fallback is None:
                self._raise_unenforceable_json_schema(rejection_reasons)
            return None

        return {"format": response_format}

    def _response_format_candidate(self, response_format: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
        if not isinstance(response_format, dict):
            return None, []

        response_format_type = response_format.get("type")
        if response_format_type == "json_object":
            return None, []

        if response_format_type != "json_schema":
            return None, []

        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            return None, ["json_schema response_format is missing a json_schema object"]

        schema = json_schema.get("schema")
        if not isinstance(schema, dict):
            return None, ["json_schema response_format is missing a schema object"]

        rejection_reasons = self._anthropic_output_schema_rejection_reasons(schema)
        if rejection_reasons:
            return None, rejection_reasons

        return {"type": "json_schema", "schema": schema}, []

    def _uses_prompt_only_json_object(self, options: LLMCallOptions | None, *, output_config: dict[str, Any] | None) -> bool:
        if options is None or output_config is not None:
            return False
        return self._is_json_object_format(options.response_format) or self._is_json_object_format(options.response_format_fallback)

    def _is_json_object_format(self, response_format: dict[str, Any] | None) -> bool:
        return isinstance(response_format, dict) and response_format.get("type") == "json_object"

    def _is_json_schema_format(self, response_format: dict[str, Any] | None) -> bool:
        return isinstance(response_format, dict) and response_format.get("type") == "json_schema"

    def _raise_unenforceable_json_schema(self, rejection_reasons: list[str]) -> None:
        detail = "; ".join(rejection_reasons[:20])
        if len(rejection_reasons) > 20:
            detail = f"{detail}; ... {len(rejection_reasons) - 20} more"
        raise LLMProviderError(
            kind="configuration_error",
            user_message="Azure Anthropic cannot enforce the requested structured output schema.",
            detail=detail,
        )

    def _anthropic_output_schema_rejection_reasons(self, schema: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        stack: list[Any] = [schema]
        paths: list[str] = ["$"]
        while stack:
            value = stack.pop()
            path = paths.pop()
            if isinstance(value, dict):
                if value.get("type") == "object" and value.get("additionalProperties") is not False:
                    reasons.append(f"{path}: object schemas must set additionalProperties=false")
                for keyword in sorted(UNSUPPORTED_OUTPUT_SCHEMA_KEYWORDS):
                    if keyword in value:
                        reasons.append(f"{path}: unsupported JSON Schema keyword {keyword}")
                for key, child in value.items():
                    stack.append(child)
                    paths.append(f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    stack.append(child)
                    paths.append(f"{path}[{index}]")
        return reasons


def _is_retryable_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code in {408, 409, 425, 429, 529} or (isinstance(status_code, int) and status_code >= 500)
    return False


def _is_overloaded_error(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) == 529:
        return True
    return _provider_error_type(exc) == "overloaded_error"


def _provider_error_type(exc: BaseException) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            value = error.get("type") or error.get("code")
            return str(value) if value is not None else None
        value = body.get("type") or body.get("code")
        return str(value) if value is not None else None
    return None


def _provider_error_detail(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    status_code = getattr(exc, "status_code", None)
    parts: list[str] = []
    if status_code is not None:
        parts.append(f"status_code={status_code}")
    provider_error_type = _provider_error_type(exc)
    if provider_error_type:
        parts.append(f"provider_error_type={provider_error_type}")
    if isinstance(body, dict):
        request_id = body.get("request_id")
        if request_id:
            parts.append(f"request_id={request_id}")
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                parts.append(f"message={message}")
    detail = "; ".join(parts)
    raw = str(exc)
    if detail and raw:
        return f"{detail}; raw={raw}"
    return detail or raw


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        return max(0.0, float(retry_after))
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError):
        return None
    return max(0.0, parsed.timestamp() - time.time())


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default
