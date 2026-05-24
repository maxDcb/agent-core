from __future__ import annotations

from types import SimpleNamespace

import httpx
from openai import BadRequestError, RateLimitError

from agent_core.llm.azure_openai_provider import AzureOpenAIProvider
from agent_core.llm.base import LLMCallOptions, LLMMessage
from agent_core.llm.openai_compat import OpenAIRateLimitRetryPolicy, create_chat_completion_with_adaptive_retry
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


def _unsupported_max_tokens_error() -> BadRequestError:
    return BadRequestError(
        "Unsupported parameter: 'max_tokens' is not compatible with this model. Use 'max_completion_tokens' instead.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.test/v1/chat/completions")),
        body={
            "error": {
                "message": "Unsupported parameter: 'max_tokens' is not compatible with this model. Use 'max_completion_tokens' instead.",
                "param": "max_tokens",
            }
        },
    )


def _rate_limit_error(*, retry_after: str | None = None) -> RateLimitError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return RateLimitError(
        "Rate limit reached",
        response=httpx.Response(
            429,
            headers=headers,
            request=httpx.Request("POST", "https://api.openai.test/v1/chat/completions"),
        ),
        body={"error": {"message": "Rate limit reached"}},
    )


def test_openai_provider_omits_reasoning_effort_for_known_non_reasoning_model() -> None:
    completions = ScriptedCompletions(_text_response("ok"))
    provider = OpenAIProvider(api_key="test-key")
    provider.client = FakeClient(completions)  # type: ignore[assignment]

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="hello")],
        model="gpt-4.1-mini",
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="high"),
    )

    assert result == "ok"
    assert len(completions.requests) == 1
    assert completions.requests[0]["temperature"] == 0.0
    assert "reasoning_effort" not in completions.requests[0]


def test_azure_openai_provider_learns_unknown_deployment_reasoning_rejection() -> None:
    completions = ScriptedCompletions(
        _unsupported_reasoning_effort_error(),
        _text_response("ok"),
        _text_response("ok-again"),
    )
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

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="hello again")],
        model="deployment-name",
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="high"),
    )

    assert result == "ok-again"
    assert "reasoning_effort" not in completions.requests[2]


def test_azure_openai_provider_omits_temperature_and_maps_tokens_for_known_reasoning_deployment() -> None:
    completions = ScriptedCompletions(_text_response("ok"))
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
        options=LLMCallOptions(reasoning_effort="high", max_output_tokens=250),
    )

    assert result == "ok"
    assert len(completions.requests) == 1
    assert "temperature" not in completions.requests[0]
    assert completions.requests[0]["reasoning_effort"] == "high"
    assert "max_tokens" not in completions.requests[0]
    assert completions.requests[0]["max_completion_tokens"] == 250


def test_azure_openai_provider_can_retry_without_reasoning_then_temperature_for_unknown_deployment() -> None:
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
        model="deployment-name",
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="high"),
    )

    assert result == "ok"
    assert "reasoning_effort" not in completions.requests[1]
    assert "temperature" not in completions.requests[2]


def test_azure_openai_provider_retries_with_max_completion_tokens_and_caches_rejection() -> None:
    completions = ScriptedCompletions(
        _unsupported_max_tokens_error(),
        _text_response("ok"),
        _text_response("ok-again"),
    )
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
        options=LLMCallOptions(max_output_tokens=120),
    )

    assert result == "ok"
    assert completions.requests[0]["max_tokens"] == 120
    assert "max_tokens" not in completions.requests[1]
    assert completions.requests[1]["max_completion_tokens"] == 120

    result = provider.complete_text(
        messages=[LLMMessage(role="user", content="hello again")],
        model="deployment-name",
        temperature=0.0,
        options=LLMCallOptions(max_output_tokens=80),
    )

    assert result == "ok-again"
    assert "max_tokens" not in completions.requests[2]
    assert completions.requests[2]["max_completion_tokens"] == 80


def test_openai_compat_retries_rate_limit_with_retry_after() -> None:
    completions = ScriptedCompletions(_rate_limit_error(retry_after="3"), _text_response("ok"))
    sleeps: list[float] = []

    response = create_chat_completion_with_adaptive_retry(
        completions=completions,
        request={"model": "gpt-test", "messages": []},
        provider_name="OpenAI",
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        rate_limit_policy=OpenAIRateLimitRetryPolicy(max_attempts=2, jitter_ratio=0.0),
        sleeper=sleeps.append,
        random_fn=lambda: 0.5,
    )

    assert response.choices[0].message.content == "ok"
    assert sleeps == [3.0]
    assert len(completions.requests) == 2


def test_openai_compat_preserves_bad_request_fallback_across_rate_limit_retry() -> None:
    completions = ScriptedCompletions(
        _unsupported_reasoning_effort_error(),
        _rate_limit_error(),
        _text_response("ok"),
    )
    sleeps: list[float] = []

    response = create_chat_completion_with_adaptive_retry(
        completions=completions,
        request={"model": "deployment-name", "messages": [], "reasoning_effort": "high"},
        provider_name="Azure OpenAI",
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        rate_limit_policy=OpenAIRateLimitRetryPolicy(max_attempts=2, initial_delay_seconds=1.0, jitter_ratio=0.0),
        sleeper=sleeps.append,
        random_fn=lambda: 0.5,
    )

    assert response.choices[0].message.content == "ok"
    assert "reasoning_effort" not in completions.requests[1]
    assert "reasoning_effort" not in completions.requests[2]
    assert sleeps == [1.0]
