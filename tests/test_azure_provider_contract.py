from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from openai import APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

import agent_core.llm.langchain_azure_openai_provider as langchain_provider_module
from agent_core.llm.azure_openai_provider import AzureOpenAIProvider
from agent_core.llm.base import LLMCallOptions, LLMMessage, LLMToolCall, LLMToolDefinition
from agent_core.llm.errors import LLMProviderError
from agent_core.llm.langchain_azure_openai_provider import LangChainAzureOpenAIProvider


class ScriptedNativeCompletions:
    def __init__(self, steps: list[object]) -> None:
        self.steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.requests.append(kwargs)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class ScriptedLangChainModel:
    def __init__(self, steps: list[object]) -> None:
        self.steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> BoundLangChainModel:
        return BoundLangChainModel(self, tools=tools, bind_options=kwargs)

    def invoke(self, messages: list[object], **kwargs: Any) -> object:
        return self._invoke(messages, tools=None, bind_options={}, invoke_options=kwargs)

    def _invoke(
        self,
        messages: list[object],
        *,
        tools: list[dict[str, Any]] | None,
        bind_options: dict[str, Any],
        invoke_options: dict[str, Any],
    ) -> object:
        self.requests.append(
            {
                "messages": [_langchain_message_to_history(message) for message in messages],
                "tools": tools,
                **bind_options,
                **invoke_options,
            }
        )
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@dataclass
class RawLangChainResponse:
    payload: dict[str, Any]
    headers: dict[str, str]

    def parse(self) -> dict[str, Any]:
        return self.payload


class FakeLangChainOpenAIClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[dict[str, Any]] = []
        self.with_raw_response = self

    def create(self, **kwargs: Any) -> RawLangChainResponse:
        self.requests.append(kwargs)
        return RawLangChainResponse(payload=self.payload, headers={})

    def parse(self, **kwargs: Any) -> RawLangChainResponse:
        self.requests.append(kwargs)
        return RawLangChainResponse(payload=self.payload, headers={})


@dataclass
class BoundLangChainModel:
    model: ScriptedLangChainModel
    tools: list[dict[str, Any]]
    bind_options: dict[str, Any]

    def invoke(self, messages: list[object], **kwargs: Any) -> object:
        return self.model._invoke(
            messages,
            tools=self.tools,
            bind_options=self.bind_options,
            invoke_options=kwargs,
        )


@dataclass
class AzureProviderHarness:
    provider: Any
    requests: list[dict[str, Any]]


def _provider_harness(backend: str, steps: list[object]) -> AzureProviderHarness:
    if backend == "native":
        completions = ScriptedNativeCompletions(steps)
        provider = AzureOpenAIProvider(
            azure_endpoint="https://example.openai.azure.com",
            api_key="test-key",
            api_version="2025-01-01-preview",
        )
        provider.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        return AzureProviderHarness(provider=provider, requests=completions.requests)

    model = ScriptedLangChainModel(steps)
    provider = LangChainAzureOpenAIProvider(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        api_version="2025-01-01-preview",
        chat_model_factory=lambda deployment: model,
    )
    return AzureProviderHarness(provider=provider, requests=model.requests)


def _text_response(backend: str, content: str) -> object:
    if backend == "native":
        return SimpleNamespace(
            id="chatcmpl-contract",
            usage=SimpleNamespace(
                prompt_tokens=21,
                completion_tokens=7,
                total_tokens=28,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        )
    return AIMessage(
        content=content,
        response_metadata={"id": "chatcmpl-contract"},
        usage_metadata={
            "input_tokens": 21,
            "output_tokens": 7,
            "total_tokens": 28,
            "input_token_details": {"cache_read": 4},
            "output_token_details": {"reasoning": 2},
        },
    )


def _tool_response(backend: str) -> object:
    raw_tool_call = {
        "id": "call-contract",
        "type": "function",
        "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
    }
    if backend == "native":
        function = SimpleNamespace(name="shell", arguments='{"command":"pwd"}')
        message = SimpleNamespace(
            content="checking",
            tool_calls=[SimpleNamespace(id="call-contract", function=function)],
        )
        return SimpleNamespace(id="chatcmpl-tool", usage=None, choices=[SimpleNamespace(message=message)])
    return AIMessage(
        content="checking",
        additional_kwargs={"tool_calls": [raw_tool_call]},
        response_metadata={"id": "chatcmpl-tool"},
    )


def _multiple_tool_response(backend: str) -> object:
    calls = [
        ("call-valid", "shell", '{"command":"pwd"}'),
        ("call-invalid", "shell", '{"command":'),
    ]
    if backend == "native":
        return SimpleNamespace(
            id="chatcmpl-multiple-tools",
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id=call_id,
                                function=SimpleNamespace(name=name, arguments=arguments),
                            )
                            for call_id, name, arguments in calls
                        ],
                    )
                )
            ],
        )
    return AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
                for call_id, name, arguments in calls
            ]
        },
        response_metadata={"id": "chatcmpl-multiple-tools"},
    )


def _unsupported_reasoning_effort_error() -> BadRequestError:
    return BadRequestError(
        "Unrecognized request argument supplied: reasoning_effort",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "https://example.openai.azure.com/openai/deployments/test/chat/completions"),
        ),
        body={"error": {"message": "Unrecognized request argument supplied: reasoning_effort"}},
    )


def _authentication_error() -> AuthenticationError:
    return AuthenticationError(
        "Invalid API key",
        response=httpx.Response(
            401,
            request=httpx.Request("POST", "https://example.openai.azure.com/chat/completions"),
        ),
        body={"error": {"message": "Invalid API key"}},
    )


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(
        request=httpx.Request("POST", "https://example.openai.azure.com/chat/completions")
    )


def _rate_limit_error() -> RateLimitError:
    return RateLimitError(
        "Rate limit reached",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://example.openai.azure.com/chat/completions"),
        ),
        body={"error": {"message": "Rate limit reached"}},
    )


@pytest.mark.parametrize("backend", ["native", "langchain"])
def test_azure_provider_contract_preserves_text_usage_and_request_id(backend: str) -> None:
    harness = _provider_harness(backend, [_text_response(backend, "same answer")])

    result = harness.provider.complete_text(
        messages=[LLMMessage(role="system", content="be concise"), LLMMessage(role="user", content="hello")],
        model="deployment-name",
        temperature=0.2,
    )

    assert result.content == "same answer"
    assert result.tool_calls == []
    assert result.provider == "azure_openai"
    assert result.model_backend == backend
    assert result.model == "deployment-name"
    assert result.provider_request_id == "chatcmpl-contract"
    assert result.provider_attempts == 1
    assert result.usage is not None
    assert result.usage.to_dict() == {
        "input_tokens": 21,
        "output_tokens": 7,
        "total_tokens": 28,
        "cached_input_tokens": 4,
        "cache_creation_input_tokens": None,
        "reasoning_output_tokens": 2,
        "source": "provider",
    }
    assert harness.requests[0]["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hello"},
    ]
    assert harness.requests[0]["temperature"] == 0.2


@pytest.mark.parametrize("backend", ["native", "langchain"])
def test_azure_provider_contract_preserves_tool_history_schema_and_result(backend: str) -> None:
    harness = _provider_harness(backend, [_tool_response(backend)])
    messages = [
        LLMMessage(role="user", content="where am I?"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(id="call-before", name="shell", arguments_json='{"command":"whoami"}')],
        ),
        LLMMessage(role="tool", content="tester", tool_call_id="call-before"),
    ]
    tools = [
        LLMToolDefinition(
            name="shell",
            description="Run a shell command",
            parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        )
    ]

    result = harness.provider.complete_with_tools(
        messages=messages,
        tools=tools,
        model="deployment-name",
        temperature=0.0,
        options=LLMCallOptions(max_output_tokens=250),
    )

    assert result.content == "checking"
    assert result.tool_calls == [
        LLMToolCall(id="call-contract", name="shell", arguments_json='{"command":"pwd"}')
    ]
    assert harness.requests[0]["messages"] == [message.to_history_dict() for message in messages]
    assert harness.requests[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    assert harness.requests[0]["tool_choice"] == "auto"
    assert harness.requests[0]["parallel_tool_calls"] is True
    assert harness.requests[0]["max_tokens"] == 250


@pytest.mark.parametrize("backend", ["native", "langchain"])
def test_azure_provider_contract_preserves_multiple_and_invalid_tool_calls(backend: str) -> None:
    harness = _provider_harness(backend, [_multiple_tool_response(backend)])

    result = harness.provider.complete_with_tools(
        messages=[LLMMessage(role="user", content="run both")],
        tools=[
            LLMToolDefinition(
                name="shell",
                description="Run a shell command",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            )
        ],
        model="deployment-name",
        temperature=0.0,
    )

    assert result.tool_calls == [
        LLMToolCall(id="call-valid", name="shell", arguments_json='{"command":"pwd"}'),
        LLMToolCall(id="call-invalid", name="shell", arguments_json='{"command":'),
    ]


@pytest.mark.parametrize("backend", ["native", "langchain"])
def test_azure_provider_contract_combines_tools_and_structured_response_format(backend: str) -> None:
    harness = _provider_harness(backend, [_tool_response(backend)])
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "tool_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    }

    harness.provider.complete_with_tools(
        messages=[LLMMessage(role="user", content="check")],
        tools=[
            LLMToolDefinition(
                name="shell",
                description="Run a shell command",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            )
        ],
        model="deployment-name",
        temperature=0.0,
        options=LLMCallOptions(response_format=response_format),
    )

    assert harness.requests[0]["tools"]
    assert harness.requests[0]["response_format"] == response_format


@pytest.mark.parametrize("backend", ["native", "langchain"])
def test_azure_provider_contract_learns_rejected_reasoning_parameter(backend: str) -> None:
    harness = _provider_harness(
        backend,
        [_unsupported_reasoning_effort_error(), _text_response(backend, "ok"), _text_response(backend, "again")],
    )

    first_result = harness.provider.complete_text(
        messages=[LLMMessage(role="user", content="hello")],
        model="custom-deployment",
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="high"),
    )
    second_result = harness.provider.complete_text(
        messages=[LLMMessage(role="user", content="again")],
        model="custom-deployment",
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="high"),
    )

    assert first_result.provider_attempts == 2
    assert second_result.provider_attempts == 1
    assert harness.requests[0]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in harness.requests[1]
    assert "reasoning_effort" not in harness.requests[2]


@pytest.mark.parametrize("backend", ["native", "langchain"])
@pytest.mark.parametrize(
    ("endpoint", "api_key", "expected_detail"),
    [(None, "key", "ENDPOINT"), ("https://example.openai.azure.com", None, "API_KEY")],
)
def test_azure_provider_contract_rejects_missing_configuration(
    backend: str,
    endpoint: str | None,
    api_key: str | None,
    expected_detail: str,
) -> None:
    if backend == "native":
        provider: Any = AzureOpenAIProvider(azure_endpoint=endpoint, api_key=api_key)
    else:
        provider = LangChainAzureOpenAIProvider(azure_endpoint=endpoint, api_key=api_key)

    with pytest.raises(LLMProviderError) as exc_info:
        provider.complete_text(
            messages=[LLMMessage(role="user", content="hello")],
            model="deployment-name",
            temperature=0.0,
        )

    assert exc_info.value.kind == "configuration_error"
    assert expected_detail in exc_info.value.detail


@pytest.mark.parametrize("backend", ["native", "langchain"])
@pytest.mark.parametrize(
    ("error", "expected_kind"),
    [(_authentication_error(), "configuration_error"), (_timeout_error(), "request_error")],
)
def test_azure_provider_contract_maps_common_sdk_errors(
    backend: str,
    error: Exception,
    expected_kind: str,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENT_CORE_LLM_RETRY_MAX_ATTEMPTS", "1")
    harness = _provider_harness(backend, [error])

    with pytest.raises(LLMProviderError) as exc_info:
        harness.provider.complete_text(
            messages=[LLMMessage(role="user", content="hello")],
            model="deployment-name",
            temperature=0.0,
        )

    assert exc_info.value.kind == expected_kind


@pytest.mark.parametrize("backend", ["native", "langchain"])
def test_azure_provider_contract_maps_exhausted_rate_limit(backend: str, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CORE_LLM_RETRY_MAX_ATTEMPTS", "1")
    harness = _provider_harness(backend, [_rate_limit_error()])

    with pytest.raises(LLMProviderError) as exc_info:
        harness.provider.complete_text(
            messages=[LLMMessage(role="user", content="hello")],
            model="deployment-name",
            temperature=0.0,
        )

    assert exc_info.value.kind == "rate_limit_error"
    assert len(harness.requests) == 1


@pytest.mark.parametrize("backend", ["native", "langchain"])
def test_azure_provider_contract_rejects_unusable_response(backend: str) -> None:
    response: object = (
        SimpleNamespace(choices=[])
        if backend == "native"
        else HumanMessage(content="not an assistant response")
    )
    harness = _provider_harness(backend, [response])

    with pytest.raises(LLMProviderError) as exc_info:
        harness.provider.complete_text(
            messages=[LLMMessage(role="user", content="hello")],
            model="deployment-name",
            temperature=0.0,
        )

    assert exc_info.value.kind == "response_error"


def test_langchain_adapter_flattens_text_content_blocks() -> None:
    model = ScriptedLangChainModel(
        [
            AIMessage(
                content=[
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                    {"type": "reasoning", "reasoning": "hidden"},
                ]
            )
        ]
    )
    provider = LangChainAzureOpenAIProvider(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        chat_model_factory=lambda deployment: model,
    )

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="hello")],
        model="deployment-name",
        temperature=0.0,
    )

    assert result.content == "hello world"


@pytest.mark.parametrize("tracing_enabled", [False, True])
def test_langchain_adapter_scopes_langsmith_tracing(monkeypatch, tracing_enabled: bool) -> None:
    observed: list[bool | None] = []

    @contextmanager
    def fake_tracing_context(*, enabled=None, **kwargs):
        observed.append(enabled)
        yield

    monkeypatch.setattr(langchain_provider_module.ls, "tracing_context", fake_tracing_context)
    model = ScriptedLangChainModel([AIMessage(content="ok")])
    provider = LangChainAzureOpenAIProvider(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        chat_model_factory=lambda deployment: model,
        tracing_enabled=tracing_enabled,
    )

    provider.complete_text(
        messages=[LLMMessage(role="user", content="hello")],
        model="deployment-name",
        temperature=0.0,
    )

    assert observed == [tracing_enabled]


def test_langchain_adapter_integrates_with_real_azure_chat_model_conversion() -> None:
    client = FakeLangChainOpenAIClient(
        {
            "id": "chatcmpl-langchain-integration",
            "model": "deployment-name",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "checking",
                        "tool_calls": [
                            {
                                "id": "call-integration",
                                "type": "function",
                                "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
    )
    chat_model = AzureChatOpenAI(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",  # type: ignore[arg-type]
        api_version="2025-01-01-preview",
        azure_deployment="deployment-name",
        max_retries=0,
    )
    chat_model.client = client
    chat_model.root_client = SimpleNamespace(chat=SimpleNamespace(completions=client))
    provider = LangChainAzureOpenAIProvider(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        chat_model_factory=lambda deployment: chat_model,
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "strict": True,
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
    result = provider.complete_with_tools(
        messages=[LLMMessage(role="user", content="where am I?")],
        tools=[
            LLMToolDefinition(
                name="shell",
                description="Run a command",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            )
        ],
        model="deployment-name",
        temperature=0.0,
        options=LLMCallOptions(max_output_tokens=120, response_format=response_format),
    )

    assert result.content == "checking"
    assert result.provider_request_id == "chatcmpl-langchain-integration"
    assert result.tool_calls == [
        LLMToolCall(id="call-integration", name="shell", arguments_json='{"command":"pwd"}')
    ]
    assert result.usage is not None
    assert result.usage.cached_input_tokens == 3
    assert result.usage.reasoning_output_tokens == 1
    assert client.requests[0]["messages"] == [{"content": "where am I?", "role": "user"}]
    assert client.requests[0]["parallel_tool_calls"] is True
    assert client.requests[0]["max_tokens"] == 120
    assert client.requests[0]["response_format"] == response_format


def _langchain_message_to_history(message: object) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolMessage):
        return {"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id}
    if isinstance(message, AIMessage):
        payload: dict[str, Any] = {"role": "assistant", "content": message.content}
        if raw_tool_calls := message.additional_kwargs.get("tool_calls"):
            payload["tool_calls"] = raw_tool_calls
        return payload
    raise AssertionError(f"Unexpected LangChain message: {type(message).__name__}")
