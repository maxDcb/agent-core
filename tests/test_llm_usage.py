from __future__ import annotations

from types import SimpleNamespace

from agent_core.llm.base import (
    LLMCallRecord,
    LLMCompletionResult,
    LLMTokenUsage,
    LLMUsageSummary,
    capture_llm_calls,
    publish_llm_completion,
    token_usage_from_anthropic_response,
    token_usage_from_openai_response,
)


def test_openai_usage_is_normalized_with_cache_and_reasoning_details() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
        )
    )

    usage = token_usage_from_openai_response(response)

    assert usage is not None
    assert usage.to_dict() == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
        "cached_input_tokens": 40,
        "cache_creation_input_tokens": None,
        "reasoning_output_tokens": 12,
        "source": "provider",
    }


def test_anthropic_usage_is_normalized_with_cache_details() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=80,
            output_tokens=20,
            cache_read_input_tokens=50,
            cache_creation_input_tokens=10,
        )
    )

    usage = token_usage_from_anthropic_response(response)

    assert usage is not None
    assert usage.input_tokens == 140
    assert usage.output_tokens == 20
    assert usage.total_tokens == 160
    assert usage.cached_input_tokens == 50
    assert usage.cache_creation_input_tokens == 10


def test_usage_summary_never_presents_partial_usage_as_an_exact_total() -> None:
    calls = [
        LLMCallRecord(
            call_id="llm-0001",
            call_index=1,
            purpose="tool_loop",
            provider="openai",
            model="gpt-test",
            usage=LLMTokenUsage(input_tokens=10, output_tokens=2, total_tokens=12),
        ),
        LLMCallRecord(
            call_id="llm-0002",
            call_index=2,
            purpose="finalization",
            provider="openai",
            model="gpt-test",
        ),
    ]

    summary = LLMUsageSummary.from_calls(calls)

    assert summary.calls_with_token_usage == 1
    assert summary.token_usage_complete is False
    assert summary.input_tokens is None
    assert summary.output_tokens is None
    assert summary.total_tokens is None
    assert summary.reported_input_tokens == 10
    assert summary.reported_output_tokens == 2


def test_completion_capture_records_conversation_provider_calls() -> None:
    completion = LLMCompletionResult(
        content="done",
        usage=LLMTokenUsage(input_tokens=7, output_tokens=3, total_tokens=10),
        provider="azure_openai",
        model_backend="langchain",
        model="deployment",
        provider_request_id="request-1",
    )

    with capture_llm_calls() as calls:
        publish_llm_completion(completion, purpose="conversation_tool_loop")

    assert len(calls) == 1
    assert calls[0].purpose == "conversation_tool_loop"
    assert calls[0].usage is not None
    assert calls[0].usage.total_tokens == 10
    assert calls[0].provider_request_id == "request-1"
    assert calls[0].model_backend == "langchain"
    assert LLMCallRecord.from_dict(calls[0].to_dict()) == calls[0]


def test_call_record_loads_legacy_payload_without_model_backend() -> None:
    record = LLMCallRecord.from_dict(
        {
            "call_id": "llm-0001",
            "call_index": 1,
            "purpose": "tool_loop",
            "provider": "azure_openai",
            "model": "deployment",
        }
    )

    assert record is not None
    assert record.model_backend is None
