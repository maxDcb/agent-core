from __future__ import annotations

import json
from typing import Any

import requests
from requests import Response, Session
from requests.exceptions import RequestException, Timeout

from agent_core.logging_utils import get_logger
from agent_core.llm.base import LLMCompletionResult, LLMMessage, LLMToolCall, LLMToolDefinition
from agent_core.llm.errors import LLMProviderError

logger = get_logger(__name__)

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 60


class AzureAnthropicProvider:
    """
    Azure Anthropic provider for Claude deployments exposed through the Azure Foundry Anthropic
    Messages API.

    This provider translates the internal, provider-agnostic contract into Anthropic-style message
    payloads so the orchestrator and tools remain unaware of backend-specific formats.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version or DEFAULT_ANTHROPIC_VERSION
        self.timeout_seconds = timeout_seconds
        self.endpoint_configured = bool(endpoint)
        self.api_key_configured = bool(api_key)
        self.session: Session = requests.Session()

        logger.debug(
            "Azure Anthropic provider initialized",
            extra={
                "endpoint_configured": self.endpoint_configured,
                "api_key_configured": self.api_key_configured,
                "api_version": self.api_version,
            },
        )

    def complete_text(
        self,
        *,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
    ) -> str:
        response_payload = self._post_messages(
            messages=messages,
            tools=None,
            model=model,
            temperature=temperature,
        )
        content, tool_calls = self._parse_response(response_payload)
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
    ) -> LLMCompletionResult:
        response_payload = self._post_messages(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
        )
        content, tool_calls = self._parse_response(response_payload)
        return LLMCompletionResult(content=content, tool_calls=tool_calls)

    def _post_messages(
        self,
        *,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None,
        model: str,
        temperature: float,
    ) -> dict[str, Any]:
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

        system_prompt, anthropic_messages = self._to_anthropic_messages(messages)
        request_payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "stream": False,
        }
        if system_prompt:
            request_payload["system"] = system_prompt
        if tools:
            request_payload["tools"] = [self._to_anthropic_tool(tool) for tool in tools]

        logger.debug(
            "Sending Azure Anthropic messages request",
            extra={
                "model": model,
                "message_count": len(anthropic_messages),
                "tool_count": len(tools or []),
                "api_version": self.api_version,
            },
        )

        try:
            response = self.session.post(
                self._messages_endpoint(),
                headers=self._build_headers(),
                json=request_payload,
                timeout=self.timeout_seconds,
            )
        except Timeout as exc:
            logger.exception("Azure Anthropic request timed out")
            raise LLMProviderError(
                kind="request_error",
                user_message="The assistant could not reach Azure Anthropic before the request timed out.",
                detail=str(exc),
            ) from exc
        except RequestException as exc:
            logger.exception("Azure Anthropic request failed")
            raise LLMProviderError(
                kind="request_error",
                user_message="The assistant could not reach Azure Anthropic. Check network access and endpoint configuration.",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected Azure Anthropic transport failure")
            raise LLMProviderError(
                kind="unexpected_error",
                user_message="The assistant encountered an unexpected Azure Anthropic transport failure.",
                detail=str(exc),
            ) from exc

        return self._handle_http_response(response)

    def _messages_endpoint(self) -> str:
        assert self.endpoint is not None
        base = self.endpoint.rstrip("/")
        if base.endswith("/v1/messages"):
            return base
        return f"{base}/v1/messages"

    def _build_headers(self) -> dict[str, str]:
        assert self.api_key is not None
        return {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
        }

    def _handle_http_response(self, response: Response) -> dict[str, Any]:
        status_code = response.status_code
        if status_code == 401 or status_code == 403:
            raise LLMProviderError(
                kind="configuration_error",
                user_message="Azure Anthropic rejected the credentials. Check AZURE_ANTHROPIC_API_KEY and endpoint access.",
                detail=self._safe_response_text(response),
            )
        if status_code == 429:
            raise LLMProviderError(
                kind="rate_limit_error",
                user_message="Azure Anthropic rate-limited the request. Wait briefly and try again.",
                detail=self._safe_response_text(response),
            )
        if status_code >= 400:
            raise LLMProviderError(
                kind="request_error",
                user_message="Azure Anthropic rejected the request. Review the deployment name, API version, and payload.",
                detail=self._safe_response_text(response),
            )

        try:
            payload = response.json()
        except ValueError as exc:
            logger.exception("Azure Anthropic returned non-JSON content")
            raise LLMProviderError(
                kind="response_error",
                user_message="Azure Anthropic returned an unusable response.",
                detail="Response body was not valid JSON",
            ) from exc

        if not isinstance(payload, dict):
            raise LLMProviderError(
                kind="response_error",
                user_message="Azure Anthropic returned an unusable response.",
                detail="Response body was not a JSON object",
            )
        return payload

    def _parse_response(self, payload: dict[str, Any]) -> tuple[str, list[LLMToolCall]]:
        content_blocks = payload.get("content")
        if not isinstance(content_blocks, list):
            raise LLMProviderError(
                kind="response_error",
                user_message="Azure Anthropic returned an unusable response.",
                detail="Response content field was not a list",
            )

        text_parts: list[str] = []
        tool_calls: list[LLMToolCall] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                tool_id = block.get("id")
                tool_name = block.get("name")
                tool_input = block.get("input", {})
                if isinstance(tool_id, str) and isinstance(tool_name, str):
                    tool_calls.append(
                        LLMToolCall(
                            id=tool_id,
                            name=tool_name,
                            arguments_json=json.dumps(tool_input if isinstance(tool_input, dict) else {}, ensure_ascii=False),
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

    def _safe_response_text(self, response: Response) -> str:
        try:
            return response.text[:1000]
        except Exception:
            return f"HTTP {response.status_code}"
