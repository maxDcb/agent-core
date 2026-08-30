from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, cast

import langsmith as ls
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAIError,
    RateLimitError,
)

from agent_core.llm.base import (
    LLMCallOptions,
    LLMCompletionResult,
    LLMMessage,
    LLMTokenUsage,
    LLMToolCall,
    LLMToolDefinition,
    publish_llm_completion,
)
from agent_core.llm.errors import LLMProviderError
from agent_core.llm.openai_compat import invoke_openai_request_with_adaptive_retry
from agent_core.llm.openai_request_policy import OpenAIChatRequestNormalizer, OpenAIModelCapabilityResolver
from agent_core.logging_utils import get_logger

logger = get_logger(__name__)

ChatModelFactory = Callable[[str], Any]


def _call_purpose(options: LLMCallOptions | None, *, default: str) -> str:
    value = options.metadata.get("llm_call_purpose") if options is not None else None
    return value.strip() if isinstance(value, str) and value.strip() else default


class LangChainAzureOpenAIProvider:
    """Azure OpenAI adapter implemented with LangChain's model integration.

    The public agent-core contract deliberately remains independent from
    LangChain. This adapter is therefore replaceable and can run beside the
    existing native SDK provider during the migration.
    """

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        capability_resolver: OpenAIModelCapabilityResolver | None = None,
        timeout_seconds: float = 120.0,
        chat_model_factory: ChatModelFactory | None = None,
        tracing_enabled: bool = False,
    ) -> None:
        self.azure_endpoint_configured = bool(azure_endpoint)
        self.api_key_configured = bool(api_key)
        self.api_version = api_version or "v1"
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.tracing_enabled = bool(tracing_enabled)
        self.capability_resolver = capability_resolver or OpenAIModelCapabilityResolver()
        self.request_normalizer = OpenAIChatRequestNormalizer(self.capability_resolver)
        self._chat_model_factory = chat_model_factory or self._build_chat_model
        self._chat_models: dict[str, Any] = {}

        logger.debug(
            "LangChain Azure OpenAI provider initialized",
            extra={
                "azure_endpoint_configured": self.azure_endpoint_configured,
                "api_key_configured": self.api_key_configured,
                "api_version": self.api_version,
                "timeout_seconds": self.timeout_seconds,
                "tracing_enabled": self.tracing_enabled,
            },
        )

    def complete_text(
        self,
        *,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        return self._complete(
            messages=messages,
            tools=None,
            model=model,
            temperature=temperature,
            options=options,
            purpose="text",
        )

    def complete_with_tools(
        self,
        *,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        return self._complete(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            options=options,
            purpose="tool_loop",
        )

    def _complete(
        self,
        *,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None,
        model: str,
        temperature: float,
        options: LLMCallOptions | None,
        purpose: str,
    ) -> LLMCompletionResult:
        started_at = time.monotonic()
        response, provider_attempts = self._invoke(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            options=options,
        )
        tool_calls = self._tool_calls_from_message(response) if tools is not None else []
        result = LLMCompletionResult(
            content=self._message_text(response),
            tool_calls=tool_calls,
            usage=self._token_usage_from_message(response),
            provider="azure_openai",
            model_backend="langchain",
            model=model,
            provider_request_id=self._provider_request_id(response),
            duration_seconds=round(time.monotonic() - started_at, 3),
            provider_attempts=provider_attempts,
        )
        logger.debug(
            "Received LangChain Azure OpenAI completion response",
            extra={"content_length": len(result.content), "tool_call_count": len(result.tool_calls)},
        )
        return publish_llm_completion(result, purpose=_call_purpose(options, default=purpose))

    def _invoke(
        self,
        *,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None,
        model: str,
        temperature: float,
        options: LLMCallOptions | None,
    ) -> tuple[AIMessage, int]:
        self._validate_configuration()
        request: dict[str, Any] = {
            "model": model,
            "messages": [self._to_langchain_message(message) for message in messages],
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
            logger.debug(
                "Adjusted LangChain Azure OpenAI request",
                extra={"model": model, "change": change},
            )

        logger.info(
            "Sending LangChain Azure OpenAI chat completion request",
            extra={
                "model": model,
                "message_count": len(messages),
                "tool_count": len(tools or []),
                "api_version": self.api_version,
                "timeout_seconds": self.timeout_seconds,
            },
        )
        provider_attempts = 0

        def count_attempt() -> None:
            nonlocal provider_attempts
            provider_attempts += 1

        try:
            response = invoke_openai_request_with_adaptive_retry(
                invoke=self._invoke_request,
                request=request,
                provider_name="Azure OpenAI via LangChain",
                logger=logger,
                capability_resolver=self.capability_resolver,
                response_format_fallback=options.response_format_fallback if options is not None else None,
                on_attempt=count_attempt,
            )
        except AuthenticationError as exc:
            logger.exception("LangChain Azure OpenAI authentication failed")
            raise LLMProviderError(
                kind="configuration_error",
                user_message="Azure OpenAI rejected the credentials. Check AZURE_OPENAI_API_KEY and endpoint access.",
                detail=str(exc),
            ) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            logger.exception("LangChain Azure OpenAI request failed due to connectivity or timeout")
            raise LLMProviderError(
                kind="request_error",
                user_message="The assistant could not reach Azure OpenAI. Check network access and try again.",
                detail=str(exc),
            ) from exc
        except RateLimitError as exc:
            logger.exception("LangChain Azure OpenAI request was rate limited")
            raise LLMProviderError(
                kind="rate_limit_error",
                user_message="Azure OpenAI rate-limited the request. Wait briefly and try again.",
                detail=str(exc),
            ) from exc
        except (BadRequestError, APIStatusError) as exc:
            logger.exception("LangChain Azure OpenAI request was rejected by the API")
            raise LLMProviderError(
                kind="request_error",
                user_message="Azure OpenAI rejected the request. Review the deployment name, API version, and payload.",
                detail=str(exc),
            ) from exc
        except OpenAIError as exc:
            logger.exception("Unexpected LangChain Azure OpenAI SDK error")
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="Azure OpenAI failed unexpectedly. Try again after checking the provider configuration.",
                detail=str(exc),
            ) from exc
        except LLMProviderError:
            raise
        except Exception as exc:
            logger.exception("Unexpected LangChain Azure OpenAI provider error")
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="The assistant encountered an unexpected Azure provider failure.",
                detail=str(exc),
            ) from exc

        if not isinstance(response, AIMessage):
            raise LLMProviderError(
                kind="response_error",
                user_message="Azure OpenAI returned an unusable response.",
                detail=f"LangChain returned {type(response).__name__}, expected AIMessage",
            )
        return response, provider_attempts

    def _invoke_request(self, request: dict[str, Any]) -> AIMessage:
        deployment = str(request.pop("model"))
        messages = request.pop("messages")
        tools = request.pop("tools", None)
        tool_choice = request.pop("tool_choice", None)
        parallel_tool_calls = request.pop("parallel_tool_calls", None)
        runnable = self._get_chat_model(deployment)
        if tools:
            binding_options: dict[str, Any] = {"tool_choice": tool_choice}
            if parallel_tool_calls is not None:
                binding_options["parallel_tool_calls"] = parallel_tool_calls
            runnable = runnable.bind_tools(tools, **binding_options)
        # Model prompts and tool results can contain sensitive application data.
        # Disable LangSmith export unless the host explicitly opts in, even if
        # LANGSMITH_TRACING=true is inherited from the process environment.
        with ls.tracing_context(enabled=self.tracing_enabled):
            return cast(AIMessage, runnable.invoke(messages, **request))

    def _validate_configuration(self) -> None:
        if not self.azure_endpoint_configured:
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The Azure OpenAI endpoint is not configured. Set AZURE_OPENAI_ENDPOINT.",
                detail="Missing AZURE_OPENAI_ENDPOINT for LangChainAzureOpenAIProvider",
            )
        if not self.api_key_configured:
            raise LLMProviderError(
                kind="configuration_error",
                user_message="The Azure OpenAI API key is not configured. Set AZURE_OPENAI_API_KEY.",
                detail="Missing AZURE_OPENAI_API_KEY for LangChainAzureOpenAIProvider",
            )

    def _get_chat_model(self, deployment: str) -> Any:
        if deployment not in self._chat_models:
            self._chat_models[deployment] = self._chat_model_factory(deployment)
        return self._chat_models[deployment]

    def _build_chat_model(self, deployment: str) -> AzureChatOpenAI:
        assert self.azure_endpoint is not None
        assert self.api_key is not None
        return AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=cast(Any, self.api_key),
            api_version=self.api_version,
            azure_deployment=deployment,
            timeout=self.timeout_seconds,
            # agent-core owns retries and attempt telemetry for parity with the native adapter.
            max_retries=0,
            cache=False,
        )

    @staticmethod
    def _to_langchain_message(message: LLMMessage) -> Any:
        if message.role == "system":
            return SystemMessage(content=message.content)
        if message.role == "user":
            return HumanMessage(content=message.content)
        if message.role == "tool":
            if message.tool_call_id:
                return ToolMessage(content=message.content, tool_call_id=message.tool_call_id)
            return ChatMessage(role="tool", content=message.content)
        additional_kwargs: dict[str, Any] = {}
        if message.tool_calls:
            additional_kwargs["tool_calls"] = [tool_call.to_history_dict() for tool_call in message.tool_calls]
        return AIMessage(content=message.content, additional_kwargs=additional_kwargs)

    @staticmethod
    def _to_openai_tool(tool: LLMToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _message_text(message: AIMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        return message.text

    @classmethod
    def _tool_calls_from_message(cls, message: AIMessage) -> list[LLMToolCall]:
        raw_tool_calls = message.additional_kwargs.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            converted = [
                tool_call
                for item in raw_tool_calls
                if isinstance(item, dict)
                for tool_call in [LLMToolCall.from_history_dict(item)]
                if tool_call is not None
            ]
            if converted:
                return converted

        converted = []
        for item in [*message.tool_calls, *message.invalid_tool_calls]:
            name = item.get("name")
            call_id = item.get("id")
            arguments = item.get("args")
            if not isinstance(name, str) or not isinstance(call_id, str):
                continue
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments if isinstance(arguments, dict) else {}, separators=(",", ":"))
            converted.append(LLMToolCall(id=call_id, name=name, arguments_json=arguments))
        return converted

    @staticmethod
    def _token_usage_from_message(message: AIMessage) -> LLMTokenUsage | None:
        usage = message.usage_metadata
        if not isinstance(usage, dict):
            return None
        input_tokens = _non_negative_int(usage.get("input_tokens"))
        output_tokens = _non_negative_int(usage.get("output_tokens"))
        if input_tokens is None or output_tokens is None:
            return None
        input_details = usage.get("input_token_details")
        output_details = usage.get("output_token_details")
        return LLMTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=_non_negative_int(usage.get("total_tokens")) or input_tokens + output_tokens,
            cached_input_tokens=_detail_token_count(input_details, "cache_read"),
            cache_creation_input_tokens=_detail_token_count(input_details, "cache_creation"),
            reasoning_output_tokens=_detail_token_count(output_details, "reasoning"),
        )

    @staticmethod
    def _provider_request_id(message: AIMessage) -> str | None:
        for key in ("id", "request_id", "response_id"):
            value = message.response_metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _detail_token_count(details: object, key: str) -> int | None:
    if not isinstance(details, dict):
        return None
    direct = _non_negative_int(details.get(key))
    if direct is not None:
        return direct
    return next(
        (
            parsed
            for detail_key, value in details.items()
            if detail_key.endswith(f"_{key}")
            for parsed in [_non_negative_int(value)]
            if parsed is not None
        ),
        None,
    )
