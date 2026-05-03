from __future__ import annotations

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
from agent_core.llm.base import LLMCompletionResult, LLMMessage, LLMToolCall, LLMToolDefinition
from agent_core.llm.errors import LLMProviderError

logger = get_logger(__name__)


class OpenAIProvider:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key_configured = bool(api_key)
        self.api_key = api_key
        self.client: OpenAI | None = None
        # Delay client creation so local REPL commands still work when the API key is missing.
        logger.debug("OpenAI provider initialized", extra={"api_key_configured": self.api_key_configured})

    def complete_text(
        self,
        *,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
    ) -> str:
        # This method is meant for internal, non-agentic completions such as prompt summarization,
        # working-memory synthesis, or report condensation. It does not expose any tools.
        response = self._create_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=None,
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
    ) -> LLMCompletionResult:
        # This method is the main assistant path. It allows the model to request tools and is used
        # by the interactive runtime loop in the orchestrator.
        response = self._create_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=tools,
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
    ):
        if not self.api_key_configured:
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The LLM provider is not configured. Set OPENAI_API_KEY before running the assistant.",
                detail="Missing OPENAI_API_KEY for OpenAIProvider",
            )

        logger.debug(
            "Sending chat completion request",
            extra={"model": model, "message_count": len(messages), "tool_count": len(tools or [])},
        )
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
            response = client.chat.completions.create(**request)
        except AuthenticationError as exc:
            logger.exception("OpenAI authentication failed")
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The LLM provider rejected the credentials. Check OPENAI_API_KEY and provider access.",
                detail=str(exc),
            ) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            logger.exception("OpenAI request failed due to connectivity or timeout")
            raise LLMProviderError(
                kind="request_error",
                user_message="The assistant could not reach the LLM provider. Check network access and try again.",
                detail=str(exc),
            ) from exc
        except RateLimitError as exc:
            logger.exception("OpenAI request was rate limited")
            raise LLMProviderError(
                kind="rate_limit_error",
                user_message="The LLM provider rate-limited the request. Wait briefly and try again.",
                detail=str(exc),
            ) from exc
        except (BadRequestError, APIStatusError) as exc:
            logger.exception("OpenAI request was rejected by the API")
            raise LLMProviderError(
                kind="request_error",
                user_message="The LLM provider rejected the request. Review the model configuration and request payload.",
                detail=str(exc),
            ) from exc
        except OpenAIError as exc:
            logger.exception("Unexpected OpenAI provider error")
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="The LLM provider failed unexpectedly. Try again after checking the provider configuration.",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected non-OpenAI provider error")
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
            self.client = OpenAI(api_key=self.api_key)
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
