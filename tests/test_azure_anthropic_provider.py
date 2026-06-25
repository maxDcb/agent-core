from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
from anthropic import APIStatusError

from agent_core.llm import azure_anthropic_provider as provider_module
from agent_core.llm.azure_anthropic_provider import AzureAnthropicProvider
from agent_core.llm.base import LLMCallOptions, LLMMessage, LLMToolDefinition
from agent_core.llm.errors import LLMProviderError


class ScriptedMessages:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        step = self.responses.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class FakeFoundryClient:
    def __init__(self, messages: ScriptedMessages) -> None:
        self.messages = messages


def _provider_with_scripted_messages(messages: ScriptedMessages) -> AzureAnthropicProvider:
    provider = AzureAnthropicProvider(
        endpoint="https://example.services.ai.azure.com/anthropic",
        api_key="test-key",
        api_version="2023-06-01",
        timeout_seconds=42,
    )
    provider.client = FakeFoundryClient(messages)  # type: ignore[assignment]
    return provider


def _text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=content)])


def _tool_response(*, tool_id: str, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id=tool_id,
                name=name,
                input=arguments,
            )
        ]
    )


def _api_status_error(
    status_code: int,
    *,
    body: dict[str, Any] | None = None,
    retry_after: str | None = None,
) -> APIStatusError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return APIStatusError(
        f"HTTP {status_code}",
        response=httpx.Response(
            status_code,
            headers=headers,
            request=httpx.Request("POST", "https://example.services.ai.azure.com/anthropic/v1/messages"),
        ),
        body=body or {"error": {"type": "server_error", "message": f"HTTP {status_code}"}},
    )


def test_azure_anthropic_provider_builds_anthropic_foundry_client(monkeypatch) -> None:
    created: dict[str, Any] = {}
    scripted_messages = ScriptedMessages(_text_response("OK"))

    class FakeAnthropicFoundry:
        def __init__(self, **kwargs: Any) -> None:
            created.update(kwargs)
            self.messages = scripted_messages

    monkeypatch.setattr(provider_module, "AnthropicFoundry", FakeAnthropicFoundry)
    provider = AzureAnthropicProvider(
        endpoint="https://example.services.ai.azure.com/anthropic",
        api_key="test-key",
        anthropic_version="2024-10-01",
        timeout_seconds=123,
    )

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="Return exactly OK")],
        model="claude-opus-4-6",
        temperature=0.0,
    )

    assert result == "OK"
    assert created == {
        "api_key": "test-key",
        "base_url": "https://example.services.ai.azure.com/anthropic",
        "timeout": 123.0,
        "default_headers": {"anthropic-version": "2024-10-01"},
        "default_query": None,
    }
    assert scripted_messages.requests[0]["model"] == "claude-opus-4-6"


def test_azure_anthropic_provider_can_pass_optional_api_version_query(monkeypatch) -> None:
    created: dict[str, Any] = {}
    scripted_messages = ScriptedMessages(_text_response("OK"))

    class FakeAnthropicFoundry:
        def __init__(self, **kwargs: Any) -> None:
            created.update(kwargs)
            self.messages = scripted_messages

    monkeypatch.setattr(provider_module, "AnthropicFoundry", FakeAnthropicFoundry)
    provider = AzureAnthropicProvider(
        endpoint="https://example.services.ai.azure.com/anthropic",
        api_key="test-key",
        api_version="2025-01-01-preview",
        anthropic_version="2023-06-01",
        timeout_seconds=123,
    )

    provider.complete_text(
        messages=[LLMMessage(role="user", content="Return exactly OK")],
        model="claude-opus-4-6",
        temperature=0.0,
    )

    assert created["default_query"] == {"api-version": "2025-01-01-preview"}


def test_azure_anthropic_provider_plain_chat_matches_quickstart_shape() -> None:
    scripted_messages = ScriptedMessages(_text_response("OK"))
    provider = _provider_with_scripted_messages(scripted_messages)

    result = provider.complete_text(
        messages=[
            LLMMessage(role="system", content="You are a compatibility checker."),
            LLMMessage(role="user", content="Return exactly: OK"),
        ],
        model="claude-opus-4-6",
        temperature=0.0,
    )

    request = scripted_messages.requests[0]
    assert result == "OK"
    assert request["model"] == "claude-opus-4-6"
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 4096
    assert request["stream"] is False
    assert request["system"] == "You are a compatibility checker."
    assert request["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Return exactly: OK"}],
        }
    ]


def test_azure_anthropic_provider_tool_call_matches_quickstart_shape() -> None:
    scripted_messages = ScriptedMessages(
        _tool_response(tool_id="toolu_1", name="echo", arguments={"text": "compatibility-check"})
    )
    provider = _provider_with_scripted_messages(scripted_messages)

    result = provider.complete_with_tools(
        messages=[
            LLMMessage(
                role="user",
                content="Call the echo tool once with text set to compatibility-check. Do not answer directly.",
            )
        ],
        tools=[
            LLMToolDefinition(
                name="echo",
                description="Return the provided text.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            )
        ],
        model="claude-opus-4-6",
        temperature=0.0,
    )

    request = scripted_messages.requests[0]
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "toolu_1"
    assert result.tool_calls[0].name == "echo"
    assert result.tool_calls[0].arguments_json == '{"text": "compatibility-check"}'
    assert request["tools"] == [
        {
            "name": "echo",
            "description": "Return the provided text.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        }
    ]
    assert request["tool_choice"] == {"type": "auto"}


def test_azure_anthropic_provider_ignores_json_object_response_format() -> None:
    scripted_messages = ScriptedMessages(_text_response('{"ok": true, "mode": "json_object"}'))
    provider = _provider_with_scripted_messages(scripted_messages)

    result = provider.complete_text(
        messages=[
            LLMMessage(
                role="user",
                content='Return one JSON object with fields "ok": true and "mode": "json_object". No prose.',
            )
        ],
        model="claude-opus-4-6",
        temperature=0.0,
        options=LLMCallOptions(response_format={"type": "json_object"}),
    )

    request = scripted_messages.requests[0]
    assert result == '{"ok": true, "mode": "json_object"}'
    assert "raw JSON object" in request["system"]
    assert "output_config" not in request
    assert "response_format" not in request


def test_azure_anthropic_provider_maps_json_schema_response_format_to_output_config() -> None:
    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["json_schema"]},
        },
        "required": ["ok", "mode"],
        "additionalProperties": False,
    }
    scripted_messages = ScriptedMessages(_text_response('{"ok": true, "mode": "json_schema"}'))
    provider = _provider_with_scripted_messages(scripted_messages)

    result = provider.complete_text(
        messages=[
            LLMMessage(
                role="user",
                content='Return one JSON object with fields "ok": true and "mode": "json_schema". No prose.',
            )
        ],
        model="claude-opus-4-6",
        temperature=0.0,
        options=LLMCallOptions(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "compatibility_check",
                    "schema": schema,
                    "strict": True,
                },
            },
            max_output_tokens=1024,
        ),
    )

    request = scripted_messages.requests[0]
    assert result == '{"ok": true, "mode": "json_schema"}'
    assert request["max_tokens"] == 1024
    assert request["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    assert "response_format" not in request


def test_azure_anthropic_provider_rejects_open_object_json_schema_even_with_fallback() -> None:
    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "metadata": {
                "type": "object",
                "additionalProperties": True,
            },
        },
        "required": ["ok", "metadata"],
        "additionalProperties": False,
    }
    scripted_messages = ScriptedMessages(_text_response('{"ok": true, "metadata": {"dynamic": "value"}}'))
    provider = _provider_with_scripted_messages(scripted_messages)

    try:
        provider.complete_text(
            messages=[LLMMessage(role="user", content="Return the final object.")],
            model="claude-opus-4-6",
            temperature=0.0,
            options=LLMCallOptions(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "open_object_check",
                        "schema": schema,
                        "strict": True,
                    },
                },
                response_format_fallback={"type": "json_object"},
            ),
        )
    except LLMProviderError as exc:
        assert exc.kind == "configuration_error"
        assert "cannot enforce" in exc.user_message
        assert "additionalProperties=false" in exc.detail
    else:  # pragma: no cover
        raise AssertionError("expected Azure Anthropic to reject the unenforceable schema")

    assert scripted_messages.requests == []


def test_azure_anthropic_provider_rejects_unenforceable_schema_without_fallback() -> None:
    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 0},
            "metadata": {
                "type": "object",
                "additionalProperties": True,
            },
        },
        "required": ["count", "metadata"],
        "additionalProperties": False,
    }
    scripted_messages = ScriptedMessages(_text_response('{"count": 1, "metadata": {"dynamic": "value"}}'))
    provider = _provider_with_scripted_messages(scripted_messages)

    try:
        provider.complete_text(
            messages=[LLMMessage(role="user", content="Return the final object.")],
            model="claude-opus-4-6",
            temperature=0.0,
            options=LLMCallOptions(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "strict_open_object_check",
                        "schema": schema,
                        "strict": True,
                    },
                },
            ),
        )
    except LLMProviderError as exc:
        assert exc.kind == "configuration_error"
        assert "cannot enforce" in exc.user_message
        assert "additionalProperties=false" in exc.detail
        assert "minimum" in exc.detail
    else:  # pragma: no cover - keeps the assertion message useful
        raise AssertionError("expected Azure Anthropic to reject the unenforceable strict schema")

    assert scripted_messages.requests == []


def test_azure_anthropic_provider_retries_overloaded_status(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CORE_LLM_RETRY_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("AGENT_CORE_LLM_RETRY_INITIAL_DELAY_SECONDS", "0")
    monkeypatch.setenv("AGENT_CORE_LLM_RETRY_JITTER_RATIO", "0")
    scripted_messages = ScriptedMessages(
        _api_status_error(
            529,
            body={
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
                "request_id": "req_overloaded",
            },
        ),
        _text_response("OK"),
    )
    provider = _provider_with_scripted_messages(scripted_messages)

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="Return exactly OK")],
        model="claude-opus-4-6",
        temperature=0.0,
    )

    assert result == "OK"
    assert len(scripted_messages.requests) == 2


def test_azure_anthropic_provider_reports_overloaded_after_retries(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CORE_LLM_RETRY_MAX_ATTEMPTS", "1")
    scripted_messages = ScriptedMessages(
        _api_status_error(
            529,
            body={
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
                "request_id": "req_overloaded",
            },
        )
    )
    provider = _provider_with_scripted_messages(scripted_messages)

    try:
        provider.complete_text(
            messages=[LLMMessage(role="user", content="Return exactly OK")],
            model="claude-opus-4-6",
            temperature=0.0,
        )
    except LLMProviderError as exc:
        assert exc.kind == "rate_limit_error"
        assert "overloaded" in exc.user_message
        assert "status_code=529" in exc.detail
        assert "provider_error_type=overloaded_error" in exc.detail
        assert "request_id=req_overloaded" in exc.detail
    else:  # pragma: no cover
        raise AssertionError("expected overloaded API status to be reported")


def test_azure_anthropic_provider_appends_user_turn_after_trailing_assistant_message() -> None:
    scripted_messages = ScriptedMessages(_text_response('{"ok": true}'))
    provider = _provider_with_scripted_messages(scripted_messages)

    result = provider.complete_with_tools(
        messages=[
            LLMMessage(role="system", content="Return JSON only."),
            LLMMessage(role="user", content="Investigate the target."),
            LLMMessage(role="assistant", content='Draft: {"ok": true}'),
            LLMMessage(role="system", content="No more tools are available. Return final JSON now."),
        ],
        tools=[],
        model="claude-opus-4-6",
        temperature=0.0,
        options=LLMCallOptions(response_format={"type": "json_object"}),
    )

    request = scripted_messages.requests[0]
    assert result.content == '{"ok": true}'
    assert request["system"].startswith("Return JSON only.")
    assert "No more tools are available" in request["system"]
    assert request["messages"][-2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": 'Draft: {"ok": true}'}],
    }
    assert request["messages"][-1]["role"] == "user"
    assert "final answer" in request["messages"][-1]["content"][0]["text"]
