from __future__ import annotations

import json
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
JSON_OBJECT_SYSTEM_INSTRUCTION = (
    "When a JSON object is requested, return only the raw JSON object. "
    "Do not wrap it in Markdown code fences and do not include prose before or after it."
)
FINAL_USER_TURN_AFTER_ASSISTANT = (
    "Continue from the prior assistant draft and return the requested final answer now. "
    "Follow the current system instructions exactly."
)


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _elapsed_since(started_at: float | None) -> float | None:
    if started_at is None:
        return None
    return round(time.monotonic() - started_at, 3)


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
            response = client.messages.create(**request)
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
                "Azure Anthropic request failed due to connectivity or timeout",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="request_error",
                user_message="The assistant could not reach Azure Anthropic. Check network access and try again.",
                detail=str(exc),
            ) from exc
        except RateLimitError as exc:
            logger.exception(
                "Azure Anthropic request was rate limited",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="rate_limit_error",
                user_message="Azure Anthropic rate-limited the request. Wait briefly and try again.",
                detail=str(exc),
            ) from exc
        except (BadRequestError, APIStatusError) as exc:
            logger.exception(
                "Azure Anthropic request was rejected by the API",
                extra={"elapsed_seconds": _elapsed_since(request_started_at), "error_type": type(exc).__name__},
            )
            raise LLMProviderError(
                kind="request_error",
                user_message="Azure Anthropic rejected the request. Review the deployment name, API version, and payload.",
                detail=str(exc),
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

        response_format = self._response_format_candidate(options.response_format)
        if response_format is None:
            response_format = self._response_format_candidate(options.response_format_fallback)
        if response_format is None:
            return None

        return {"format": response_format}

    def _response_format_candidate(self, response_format: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(response_format, dict):
            return None

        response_format_type = response_format.get("type")
        if response_format_type == "json_object":
            return None

        if response_format_type != "json_schema":
            return None

        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            return None

        schema = json_schema.get("schema")
        if not isinstance(schema, dict):
            return None

        if not self._schema_supports_anthropic_output_config(schema):
            return None

        return {"type": "json_schema", "schema": schema}

    def _uses_prompt_only_json_object(self, options: LLMCallOptions | None, *, output_config: dict[str, Any] | None) -> bool:
        if options is None or output_config is not None:
            return False
        return self._is_json_object_format(options.response_format) or self._is_json_object_format(options.response_format_fallback)

    def _is_json_object_format(self, response_format: dict[str, Any] | None) -> bool:
        return isinstance(response_format, dict) and response_format.get("type") == "json_object"

    def _schema_supports_anthropic_output_config(self, schema: dict[str, Any]) -> bool:
        stack: list[Any] = [schema]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if value.get("type") == "object" and value.get("additionalProperties") is not False:
                    return False
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        return True
