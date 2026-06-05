from __future__ import annotations

import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    AzureOpenAI,
    BadRequestError,
    OpenAIError,
    RateLimitError,
)

from agent_core.logging_utils import get_logger
from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage, LLMToolCall, LLMToolDefinition
from agent_core.llm.errors import LLMProviderError
from agent_core.llm.openai_compat import create_chat_completion_with_adaptive_retry
from agent_core.llm.openai_request_policy import OpenAIChatRequestNormalizer, OpenAIModelCapabilityResolver

logger = get_logger(__name__)


class AzureOpenAIProvider:
    """
    Azure OpenAI provider using the OpenAI Python SDK.

    Expected environment/config values:
    - azure_endpoint: e.g. https://my-resource.openai.azure.com/
    - api_key: Azure OpenAI key
    - api_version: preferably configurable, default can be "v1" if your environment supports it
    """

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        capability_resolver: OpenAIModelCapabilityResolver | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.azure_endpoint_configured = bool(azure_endpoint)
        self.api_key_configured = bool(api_key)
        self.api_version = api_version or "v1"
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.client: AzureOpenAI | None = None
        self.capability_resolver = capability_resolver or OpenAIModelCapabilityResolver()
        self.request_normalizer = OpenAIChatRequestNormalizer(self.capability_resolver)

        logger.debug(
            "Azure OpenAI provider initialized",
            extra={
                "azure_endpoint_configured": self.azure_endpoint_configured,
                "api_key_configured": self.api_key_configured,
                "api_version": self.api_version,
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
        # This method is for internal summarization-style completions. It intentionally avoids tool
        # exposure so components like a working-memory synthesizer cannot trigger tool calls.
        response = self._create_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=None,
            options=options,
        )
        message = response.choices[0].message
        logger.debug(
            "Received Azure text-only completion response",
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
        # This is the primary agent loop method. It keeps tool-calling enabled for the assistant's
        # main investigation turn and should not be used for internal synthesis tasks.
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
                if not getattr(tc, "function", None):
                    continue

                tool_calls.append(
                    LLMToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments_json=tc.function.arguments or "{}",
                    )
                )

        logger.debug(
            "Received Azure OpenAI chat completion response",
            extra={
                "content_length": len(message.content or ""),
                "tool_call_count": len(tool_calls),
            },
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
        if not self.azure_endpoint_configured:
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The Azure OpenAI endpoint is not configured. Set AZURE_OPENAI_ENDPOINT.",
                detail="Missing AZURE_OPENAI_ENDPOINT for AzureOpenAIProvider",
            )

        if not self.api_key_configured:
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The Azure OpenAI API key is not configured. Set AZURE_OPENAI_API_KEY.",
                detail="Missing AZURE_OPENAI_API_KEY for AzureOpenAIProvider",
            )

        logger.info(
            "Sending Azure OpenAI chat completion request",
            extra={
                "model": model,
                "message_count": len(messages),
                "tool_count": len(tools or []),
                "api_version": self.api_version,
                "timeout_seconds": self.timeout_seconds,
            },
        )

        try:
            client = self._get_client()
            request: dict[str, Any] = {
                "model": model,  # In Azure, this is the deployment name in most setups.
                "messages": [self._to_openai_message(message) for message in messages],
                "temperature": temperature,
            }
            if tools:
                request["tools"] = [self._to_openai_tool(tool) for tool in tools]
                request["tool_choice"] = "auto"
                request["parallel_tool_calls"] = True
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
                logger.debug("Adjusted Azure OpenAI chat completion request", extra={"model": model, "change": change})

            started_at = time.monotonic()
            response = create_chat_completion_with_adaptive_retry(
                completions=client.chat.completions,
                request=request,
                provider_name="Azure OpenAI",
                logger=logger,
                capability_resolver=self.capability_resolver,
                response_format_fallback=options.response_format_fallback if options is not None else None,
            )
            logger.info(
                "Received Azure OpenAI chat completion response",
                extra={"model": model, "elapsed_seconds": round(time.monotonic() - started_at, 3)},
            )
        except AuthenticationError as exc:
            logger.exception("Azure OpenAI authentication failed")
            raise LLMProviderError(
                kind="configuration_error",
                user_message="Azure OpenAI rejected the credentials. Check AZURE_OPENAI_API_KEY and endpoint access.",
                detail=str(exc),
            ) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            logger.exception("Azure OpenAI request failed due to connectivity or timeout")
            raise LLMProviderError(
                kind="request_error",
                user_message="The assistant could not reach Azure OpenAI. Check network access and try again.",
                detail=str(exc),
            ) from exc
        except RateLimitError as exc:
            logger.exception("Azure OpenAI request was rate limited")
            raise LLMProviderError(
                kind="rate_limit_error",
                user_message="Azure OpenAI rate-limited the request. Wait briefly and try again.",
                detail=str(exc),
            ) from exc
        except (BadRequestError, APIStatusError) as exc:
            logger.exception("Azure OpenAI request was rejected by the API")
            raise LLMProviderError(
                kind="request_error",
                user_message="Azure OpenAI rejected the request. Review the deployment name, API version, and payload.",
                detail=str(exc),
            ) from exc
        except OpenAIError as exc:
            logger.exception("Unexpected Azure OpenAI SDK error")
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="Azure OpenAI failed unexpectedly. Try again after checking the provider configuration.",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected non-SDK provider error")
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="The assistant encountered an unexpected Azure provider failure.",
                detail=str(exc),
            ) from exc

        if not getattr(response, "choices", None):
            logger.error("Azure OpenAI response did not contain any choices")
            raise LLMProviderError(
                kind="response_error",
                user_message="Azure OpenAI returned an unusable response.",
                detail="Response contained no choices",
            )
        return response

    def _get_client(self) -> AzureOpenAI:
        if self.client is None:
            assert self.azure_endpoint is not None
            assert self.api_key is not None
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                timeout=self.timeout_seconds,
            )
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
