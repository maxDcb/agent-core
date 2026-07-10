from __future__ import annotations

from dataclasses import fields

import pytest

from agent_core.investigation_prompts import (
    DEFAULT_INVESTIGATION_PROMPTS,
    FINAL_CRITIQUE_PROMPT,
    INITIAL_PLAN_PROMPT,
    INVESTIGATION_DECISION_PROMPT,
    RUN_GUIDANCE_PROMPT,
    STEP_REFLECTION_PROMPT,
)
from agent_core.investigation_state import InvestigationState
from agent_core.output_contracts import StructuredOutputContract
from agent_core.run_options import RunOptions


def test_run_options_defaults_and_presets() -> None:
    assert RunOptions().mode == "direct"

    investigate = RunOptions.investigate()
    assert investigate.mode == "investigate"
    assert investigate.max_iterations == 10
    assert investigate.max_tool_calls > 0
    assert investigate.recover_internal_synthesis_errors is False

    deep = RunOptions.deep_investigate()
    assert deep.mode == "deep_investigate"
    assert deep.require_final_critique is True
    assert deep.max_iterations == 20
    assert deep.final_output_mode == "text"


def test_run_options_reject_invalid_budgets_and_confidence() -> None:
    with pytest.raises(ValueError):
        RunOptions(max_iterations=0)
    with pytest.raises(ValueError):
        RunOptions(max_tool_calls=-1)
    with pytest.raises(ValueError):
        RunOptions(max_no_progress_iterations=-1)
    with pytest.raises(ValueError):
        RunOptions(min_confidence_to_answer=1.2)
    with pytest.raises(ValueError):
        RunOptions.direct(
            final_output_mode="json_schema",
            final_output_contract=StructuredOutputContract(name="x", schema={"type": "object"}),
        )
    with pytest.raises(ValueError):
        RunOptions.investigate(final_output_mode="json_schema")
    with pytest.raises(ValueError):
        RunOptions.investigate(
            final_output_contract=StructuredOutputContract(name="x", schema={"type": "object"}),
        )


def test_run_options_accepts_explicit_internal_synthesis_recovery() -> None:
    options = RunOptions.investigate(
        recover_internal_synthesis_errors=True,
        metadata={"surface": "conversation_api"},
    )

    assert options.recover_internal_synthesis_errors is True
    assert options.metadata == {"surface": "conversation_api"}


def test_run_options_accepts_json_schema_final_output_contract() -> None:
    options = RunOptions.investigate(
        final_output_mode="json_schema",
        final_output_contract={
            "name": "final_answer",
            "schema": {"type": "object", "additionalProperties": False, "properties": {}},
            "strict": True,
            "instructions": ["Return final answer fields only."],
        },
    )

    assert options.final_output_mode == "json_schema"
    assert options.final_output_contract is not None
    assert options.final_output_contract.name == "final_answer"
    assert options.final_output_contract.strict is True


def test_investigation_state_has_no_private_reasoning_fields() -> None:
    forbidden = {"thought", "thoughts", "chain_of_thought", "reasoning_trace", "reasoning"}
    field_names = {field.name for field in fields(InvestigationState)}
    assert field_names.isdisjoint(forbidden)
    assert set(InvestigationState.create_template(objective="check").to_dict()).isdisjoint(forbidden)


def test_default_investigation_prompts_are_available() -> None:
    prompts = [
        INITIAL_PLAN_PROMPT,
        STEP_REFLECTION_PROMPT,
        INVESTIGATION_DECISION_PROMPT,
        FINAL_CRITIQUE_PROMPT,
        RUN_GUIDANCE_PROMPT,
        DEFAULT_INVESTIGATION_PROMPTS.initial_plan,
        DEFAULT_INVESTIGATION_PROMPTS.step_reflection,
        DEFAULT_INVESTIGATION_PROMPTS.decision,
        DEFAULT_INVESTIGATION_PROMPTS.final_critique,
        DEFAULT_INVESTIGATION_PROMPTS.run_guidance,
    ]
    for prompt in prompts:
        assert isinstance(prompt, str)
        assert prompt.strip()

    assert DEFAULT_INVESTIGATION_PROMPTS.render_run_guidance(mode="investigate").strip()
