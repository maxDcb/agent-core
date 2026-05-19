from __future__ import annotations

from dataclasses import fields

import pytest

from agent_core.investigation_prompts import (
    FINAL_CRITIQUE_PROMPT,
    INITIAL_PLAN_PROMPT,
    INVESTIGATION_DECISION_PROMPT,
    STEP_REFLECTION_PROMPT,
)
from agent_core.investigation_state import InvestigationState
from agent_core.run_options import RunOptions


def test_run_options_defaults_and_presets() -> None:
    assert RunOptions().mode == "direct"

    investigate = RunOptions.investigate()
    assert investigate.mode == "investigate"
    assert investigate.max_iterations == 10
    assert investigate.max_tool_calls > 0

    deep = RunOptions.deep_investigate()
    assert deep.mode == "deep_investigate"
    assert deep.require_final_critique is True
    assert deep.max_iterations == 20


def test_run_options_reject_invalid_budgets_and_confidence() -> None:
    with pytest.raises(ValueError):
        RunOptions(max_iterations=0)
    with pytest.raises(ValueError):
        RunOptions(max_tool_calls=-1)
    with pytest.raises(ValueError):
        RunOptions(max_no_progress_iterations=-1)
    with pytest.raises(ValueError):
        RunOptions(min_confidence_to_answer=1.2)


def test_investigation_state_has_no_private_reasoning_fields() -> None:
    forbidden = {"thought", "thoughts", "chain_of_thought", "reasoning_trace", "reasoning"}
    field_names = {field.name for field in fields(InvestigationState)}
    assert field_names.isdisjoint(forbidden)
    assert set(InvestigationState.create_template(objective="check").to_dict()).isdisjoint(forbidden)


def test_investigation_prompts_prohibit_chain_of_thought() -> None:
    prompts = [
        INITIAL_PLAN_PROMPT,
        STEP_REFLECTION_PROMPT,
        INVESTIGATION_DECISION_PROMPT,
        FINAL_CRITIQUE_PROMPT,
    ]
    for prompt in prompts:
        assert "Do not output chain-of-thought" in prompt
        assert "Return JSON only" in prompt
        lowered = prompt.lower()
        assert "credential" not in lowered
        assert "exploit" not in lowered
        assert "lateral movement" not in lowered
