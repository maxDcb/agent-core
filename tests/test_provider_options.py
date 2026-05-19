from __future__ import annotations

import json

from agent_core.llm.base import LLMCallOptions, LLMMessage
from agent_core.settings import CoreSettings
from agent_core.structured_synthesizer import StructuredSynthesisRequest, StructuredSynthesizer


class RecordingProvider:
    def __init__(self) -> None:
        self.options_seen = []

    def complete_text(self, *, messages, model, temperature, options=None):
        self.options_seen.append(options)
        return json.dumps({"value": "ok"})

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        raise AssertionError("not used")


class LegacyProvider:
    def complete_text(self, *, messages, model, temperature):
        return json.dumps({"value": "ok"})

    def complete_with_tools(self, *, messages, tools, model, temperature):
        raise AssertionError("not used")


def test_structured_synthesizer_passes_llm_call_options_when_supported(tmp_path) -> None:
    provider = RecordingProvider()
    synthesizer = StructuredSynthesizer(
        settings=CoreSettings(session_file=tmp_path / "session.json", memory_model="fake"),
        provider=provider,
    )
    options = LLMCallOptions(reasoning_effort="low", response_format={"type": "json_object"})

    result = synthesizer.synthesize(
        request=StructuredSynthesisRequest(
            target_name="value",
            instructions="Return JSON only.",
            output_format={"value": ""},
            payload={"input": "x"},
            parser=lambda payload: payload if isinstance(payload, dict) else None,
            options=options,
        )
    )

    assert result == {"value": "ok"}
    assert provider.options_seen == [options]


def test_structured_synthesizer_keeps_legacy_provider_working(tmp_path) -> None:
    synthesizer = StructuredSynthesizer(
        settings=CoreSettings(session_file=tmp_path / "session.json", memory_model="fake"),
        provider=LegacyProvider(),
    )

    result = synthesizer.synthesize(
        request=StructuredSynthesisRequest(
            target_name="value",
            instructions="Return JSON only.",
            output_format={"value": ""},
            payload={"messages": [LLMMessage(role="user", content="x").to_history_dict()]},
            parser=lambda payload: payload if isinstance(payload, dict) else None,
            options=LLMCallOptions(response_format={"type": "json_object"}),
        )
    )

    assert result == {"value": "ok"}
