from __future__ import annotations

from types import SimpleNamespace

import httpx
from openai import BadRequestError

from agent_core.llm.azure_openai_provider import AzureOpenAIProvider
from agent_core.llm.base import LLMCallOptions, LLMMessage
from agent_core.llm.openai_provider import OpenAIProvider


class ScriptedCompletions:
    def __init__(self, *steps) -> None:
        self.steps = list(steps)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class FakeClient:
    def __init__(self, completions: ScriptedCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _text_response(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=None,
                )
            )
        ]
    )


def _unsupported_reasoning_effort_error() -> BadRequestError:
    return BadRequestError(
        "Unrecognized request argument supplied: reasoning_effort",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.test/v1/chat/completions")),
        body={"error": {"message": "Unrecognized request argument supplied: reasoning_effort"}},
    )


def _unsupported_temperature_error() -> BadRequestError:
    return BadRequestError(
        "Unsupported value: 'temperature' does not support 0.0 with this model. Only the default (1) value is supported.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.test/v1/chat/completions")),
        body={
            "error": {
                "message": "Unsupported value: 'temperature' does not support 0.0 with this model. Only the default (1) value is supported.",
                "param": "temperature",
                "code": "unsupported_value",
            }
        },
    )


def test_openai_provider_retries_without_unsupported_reasoning_effort() -> None:
    completions = ScriptedCompletions(_unsupported_reasoning_effort_error(), _text_response("ok"))
    provider = OpenAIProvider(api_key="test-key")
    provider.client = FakeClient(completions)  # type: ignore[assignment]

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="hello")],
        model="gpt-4.1-mini",
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="high"),
    )

    assert result == "ok"
    assert completions.requests[0]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in completions.requests[1]


def test_azure_openai_provider_retries_without_unsupported_reasoning_effort() -> None:
    completions = ScriptedCompletions(_unsupported_reasoning_effort_error(), _text_response("ok"))
    provider = AzureOpenAIProvider(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        api_version="2024-10-21",
    )
    provider.client = FakeClient(completions)  # type: ignore[assignment]

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="hello")],
        model="deployment-name",
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="high"),
    )

    assert result == "ok"
    assert completions.requests[0]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in completions.requests[1]


def test_azure_openai_provider_retries_without_unsupported_temperature() -> None:
    completions = ScriptedCompletions(_unsupported_temperature_error(), _text_response("ok"))
    provider = AzureOpenAIProvider(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        api_version="2024-10-21",
    )
    provider.client = FakeClient(completions)  # type: ignore[assignment]

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="hello")],
        model="gpt-5-deployment",
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="high"),
    )

    assert result == "ok"
    assert completions.requests[0]["temperature"] == 0.0
    assert "temperature" not in completions.requests[1]
    assert completions.requests[1]["reasoning_effort"] == "high"


def test_azure_openai_provider_can_retry_without_reasoning_then_temperature() -> None:
    completions = ScriptedCompletions(
        _unsupported_reasoning_effort_error(),
        _unsupported_temperature_error(),
        _text_response("ok"),
    )
    provider = AzureOpenAIProvider(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        api_version="2024-10-21",
    )
    provider.client = FakeClient(completions)  # type: ignore[assignment]

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="hello")],
        model="gpt-5-deployment",
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="high"),
    )

    assert result == "ok"
    assert "reasoning_effort" not in completions.requests[1]
    assert "temperature" not in completions.requests[2]
