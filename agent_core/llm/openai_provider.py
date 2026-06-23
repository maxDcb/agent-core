from __future__ import annotations

import json
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from agent_core.logging_utils import get_logger
from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage, LLMToolCall, LLMToolDefinition
from agent_core.llm.errors import LLMProviderError
from agent_core.llm.openai_compat import create_chat_completion_with_adaptive_retry
from agent_core.llm.openai_request_policy import OpenAIChatRequestNormalizer, OpenAIModelCapabilityResolver

logger = get_logger(__name__)


def _jsonish_char_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):
        return len(str(value))


def _approx_token_count_from_chars(char_count: int) -> int:
    return round(max(0, char_count) / 4)


def _message_stats(messages: list[dict[str, Any]]) -> dict[str, int]:
    content_lengths = [_jsonish_char_count(message.get("content")) for message in messages]
    tool_call_lengths = [_jsonish_char_count(message.get("tool_calls")) for message in messages if message.get("tool_calls")]
    total_chars = sum(content_lengths) + sum(tool_call_lengths)
    return {
        "request_message_chars": total_chars,
        "request_message_approx_tokens": _approx_token_count_from_chars(total_chars),
        "largest_message_chars": max(content_lengths, default=0),
        "assistant_tool_call_chars": sum(tool_call_lengths),
    }


def _response_format_type(response_format: Any) -> str | None:
    if isinstance(response_format, dict):
        value = response_format.get("type")
        return str(value) if value is not None else "dict"
    if response_format is not None:
        return type(response_format).__name__
    return None


def _elapsed_since(started_at: float | None) -> float | None:
    if started_at is None:
        return None
    return round(time.monotonic() - started_at, 3)


class OpenAIProvider:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        capability_resolver: OpenAIModelCapabilityResolver | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key_configured = bool(api_key)
        self.api_key = api_key
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.client: OpenAI | None = None
        self.capability_resolver = capability_resolver or OpenAIModelCapabilityResolver()
        self.request_normalizer = OpenAIChatRequestNormalizer(self.capability_resolver)
        # Delay client creation so local REPL commands still work when the API key is missing.
        logger.debug(
            "OpenAI provider initialized",
            extra={"api_key_configured": self.api_key_configured, "timeout_seconds": self.timeout_seconds},
        )

    def complete_text(
        self,
        *,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> str:
        # This method is meant for internal, non-agentic completions such as prompt summarization,
        # working-memory synthesis, or report condensation. It does not expose any tools.
        response = self._create_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=None,
            options=options,
        )
        message = response.choices[0].message
        logger.debug(
            "Received text-only completion response",
            extra={"content_length": len(message.content or "")},
        )
        return message.content or ""

    def complete_with_tools(
        self,
        *,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        # This method is the main assistant path. It allows the model to request tools and is used
        # by the interactive runtime loop in the orchestrator.
        response = self._create_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=tools,
            options=options,
        )
        message = response.choices[0].message
        tool_calls: list[LLMToolCall] = []

        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                tool_calls.append(
                    LLMToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments_json=tc.function.arguments or "{}",
                    )
                )

        logger.debug(
            "Received chat completion response",
            extra={"content_length": len(message.content or ""), "tool_call_count": len(tool_calls)},
        )
        return LLMCompletionResult(content=message.content or "", tool_calls=tool_calls)

    def _create_chat_completion(
        self,
        *,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
        tools: list[LLMToolDefinition] | None,
        options: LLMCallOptions | None = None,
    ):
        if not self.api_key_configured:
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The LLM provider is not configured. Set OPENAI_API_KEY before running the assistant.",
                detail="Missing OPENAI_API_KEY for OpenAIProvider",
            )

        request_started_at: float | None = None
        try:
            client = self._get_client()
            request: dict[str, Any] = {
                "model": model,
                "messages": [self._to_openai_message(message) for message in messages],
                "temperature": temperature,
            }
            if tools:
                request["tools"] = [self._to_openai_tool(tool) for tool in tools]
                request["tool_choice"] = "auto"
                request["parallel_tool_calls"] = False
            if options is not None:
                if options.response_format:
                    request["response_format"] = options.response_format
                if options.max_output_tokens is not None:
                    request["max_tokens"] = options.max_output_tokens
                if options.reasoning_effort:
                    request["reasoning_effort"] = options.reasoning_effort
            normalization = self.request_normalizer.normalize(request)
            request = normalization.request
            for change in normalization.changes:
                logger.debug("Adjusted OpenAI chat completion request", extra={"model": model, "change": change})

            request_messages = request.get("messages", [])
            request_message_dicts = [
                message for message in request_messages if isinstance(message, dict)
            ] if isinstance(request_messages, list) else []
            request_tools = request.get("tools", [])
            response_format = request.get("response_format")
            metadata = options.metadata if options is not None else {}
            logger.info(
                "Sending chat completion request",
                extra={
                    "model": model,
                    "message_count": len(request_message_dicts),
                    "tool_count": len(request_tools),
                    "timeout_seconds": self.timeout_seconds,
                    "structured_task_id": metadata.get("structured_task_id"),
                    "max_tokens": request.get("max_tokens"),
                    "max_completion_tokens": request.get("max_completion_tokens"),
                    "response_format_type": _response_format_type(response_format),
                    "response_format_chars": _jsonish_char_count(response_format),
                    "tool_schema_chars": _jsonish_char_count(request_tools),
                    "tool_schema_approx_tokens": _approx_token_count_from_chars(_jsonish_char_count(request_tools)),
                    **_message_stats(request_message_dicts),
                },
            )

            request_started_at = time.monotonic()
            response = create_chat_completion_with_adaptive_retry(
                completions=client.chat.completions,
                request=request,
                provider_name="OpenAI",
                logger=logger,
                capability_resolver=self.capability_resolver,
                response_format_fallback=options.response_format_fallback if options is not None else None,
            )
            choices = getattr(response, "choices", None) or []
            message = choices[0].message if choices else None
            logger.info(
                "Received chat completion response",
                extra={
                    "model": model,
                    "elapsed_seconds": _elapsed_since(request_started_at),
                    "choice_count": len(choices),
                    "response_content_chars": len(getattr(message, "content", None) or "") if message is not None else 0,
                    "response_content_approx_tokens": _approx_token_count_from_chars(
                        len(getattr(message, "content", None) or "") if message is not None else 0
                    ),
                    "response_tool_call_count": len(getattr(message, "tool_calls", None) or []) if message is not None else 0,
                },
            )
        except AuthenticationError as exc:
            logger.exception("OpenAI authentication failed", extra={"elapsed_seconds": _elapsed_since(request_started_at)})
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The LLM provider rejected the credentials. Check OPENAI_API_KEY and provider access.",
                detail=str(exc),
            ) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            logger.exception(
                "OpenAI request failed due to connectivity or timeout",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="request_error",
                user_message="The assistant could not reach the LLM provider. Check network access and try again.",
                detail=str(exc),
            ) from exc
        except RateLimitError as exc:
            logger.exception(
                "OpenAI request was rate limited",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="rate_limit_error",
                user_message="The LLM provider rate-limited the request. Wait briefly and try again.",
                detail=str(exc),
            ) from exc
        except (BadRequestError, APIStatusError) as exc:
            logger.exception(
                "OpenAI request was rejected by the API",
                extra={
                    "elapsed_seconds": _elapsed_since(request_started_at),
                    "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                },
            )
            raise LLMProviderError(
                kind="request_error",
                user_message="The LLM provider rejected the request. Review the model configuration and request payload.",
                detail=str(exc),
            ) from exc
        except OpenAIError as exc:
            logger.exception(
                "Unexpected OpenAI provider error",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="The LLM provider failed unexpectedly. Try again after checking the provider configuration.",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "Unexpected non-OpenAI provider error",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="The assistant encountered an unexpected provider failure.",
                detail=str(exc),
            ) from exc

        if not getattr(response, "choices", None):
            logger.error("OpenAI response did not contain any choices")
            raise LLMProviderError(
                kind="response_error",
                user_message="The LLM provider returned an unusable response.",
                detail="Response contained no choices",
            )
        return response

    def _get_client(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
        return self.client

    def _to_openai_message(self, message: LLMMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.role == "assistant" and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments_json,
                    },
                }
                for tool_call in message.tool_calls
            ]
        if message.role == "tool" and message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        return payload

    def _to_openai_tool(self, tool: LLMToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
