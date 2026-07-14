from __future__ import annotations

import json
from typing import Any

import pytest

from agent_core import LLMBudget
from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage, LLMTokenUsage, LLMToolCall
from agent_core.llm_budget import (
    LLMBudgetController,
    LLMBudgetExceededError,
    estimate_llm_input_tokens,
    llm_budget_scope,
    run_budgeted_llm_call,
)
from agent_core.orchestrator import AgentOrchestrator
from agent_core.policy_engine import PolicyEngine
from agent_core.run_options import RunOptions
from agent_core.session_manager import SessionManager
from agent_core.session_repo import SessionRepository
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskCheckpoint, StructuredTaskRunner, StructuredTaskSpec
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult
from tests.run_helpers import execution_context, resume_turn, run_structured, run_turn


def _completion(*, input_tokens: int = 10, output_tokens: int = 2) -> LLMCompletionResult:
    return LLMCompletionResult(
        content="done",
        usage=LLMTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def test_budget_enforces_call_limit_before_invoking_provider() -> None:
    controller = LLMBudgetController(LLMBudget(max_calls=1))
    invoked = 0

    def invoke(options: LLMCallOptions | None) -> LLMCompletionResult:
        nonlocal invoked
        invoked += 1
        return _completion()

    with llm_budget_scope(controller):
        run_budgeted_llm_call(
            messages=[LLMMessage(role="user", content="first")],
            purpose="first",
            options=None,
            invoke=invoke,
        )
        with pytest.raises(LLMBudgetExceededError) as captured:
            run_budgeted_llm_call(
                messages=[LLMMessage(role="user", content="second")],
                purpose="second",
                options=None,
                invoke=invoke,
            )

    assert captured.value.dimension == "calls"
    assert invoked == 1
    assert controller.usage.calls_started == 1
    assert controller.usage.calls_rejected == 1
    assert controller.usage.calls_by_purpose == {"first": 1}
    assert controller.usage.rejected_calls_by_purpose == {"second": 1}
    assert controller.usage.exhausted_dimension == "calls"


def test_budget_caps_output_to_remaining_total_tokens() -> None:
    messages = [LLMMessage(role="user", content="bounded output")]
    estimated_input = estimate_llm_input_tokens(messages=messages)
    controller = LLMBudgetController(LLMBudget(max_total_tokens=estimated_input + 7))
    observed_options: list[LLMCallOptions | None] = []

    with llm_budget_scope(controller):
        run_budgeted_llm_call(
            messages=messages,
            purpose="bounded",
            options=None,
            invoke=lambda options: observed_options.append(options) or _completion(input_tokens=estimated_input),
        )

    assert observed_options[0] is not None
    assert observed_options[0].max_output_tokens == 7
    assert controller.usage.accounted_total_tokens == estimated_input + 2


def test_observe_mode_records_violation_without_blocking() -> None:
    controller = LLMBudgetController(LLMBudget(max_calls=0, mode="observe"))

    with llm_budget_scope(controller):
        result = run_budgeted_llm_call(
            messages=[LLMMessage(role="user", content="observe")],
            purpose="observe",
            options=None,
            invoke=lambda options: _completion(),
        )

    assert result.content == "done"
    assert controller.usage.calls_started == 1
    assert controller.usage.observed_violations == ["calls"]
    assert controller.usage.exhausted_dimension == "calls"


def test_budget_rejects_estimated_input_over_limit() -> None:
    messages = [LLMMessage(role="user", content="input too large")]
    estimated_input = estimate_llm_input_tokens(messages=messages)
    controller = LLMBudgetController(LLMBudget(max_input_tokens=estimated_input - 1))
    invoked = False

    def invoke(options: LLMCallOptions | None) -> LLMCompletionResult:
        nonlocal invoked
        invoked = True
        return _completion()

    with llm_budget_scope(controller):
        with pytest.raises(LLMBudgetExceededError) as captured:
            run_budgeted_llm_call(
                messages=messages,
                purpose="oversized",
                options=None,
                invoke=invoke,
            )

    assert captured.value.dimension == "input_tokens"
    assert invoked is False


def test_budget_configuration_is_normalized_from_dicts() -> None:
    settings = CoreSettings(llm_budget={"max_calls": 2, "mode": "observe"})  # type: ignore[arg-type]
    options = RunOptions.direct(llm_budget={"max_total_tokens": 500})
    spec = StructuredTaskSpec(
        task_id="dict_budget",
        system_prompt="",
        objective="",
        llm_budget={"max_output_tokens": 20},  # type: ignore[arg-type]
    )

    assert settings.llm_budget == LLMBudget(max_calls=2, mode="observe")
    assert options.llm_budget == LLMBudget(max_total_tokens=500)
    assert spec.llm_budget == LLMBudget(max_output_tokens=20)


class EchoTool:
    name = "echo"
    description = "Echo a value."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

    def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        return ToolResult(ok=True, content=str(arguments["value"]))


class PendingTool(EchoTool):
    name = "pending"

    def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        return ToolResult.pending_result("waiting", metadata={"job_id": arguments["value"]})


class ToolThenFinalProvider:
    def __init__(self, *, tool_name: str = "echo") -> None:
        self.tool_name = tool_name
        self.chat_calls = 0

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        self.chat_calls += 1
        if self.chat_calls == 1:
            return LLMCompletionResult(
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="call-1",
                        name=self.tool_name,
                        arguments_json=json.dumps({"value": "value-1"}),
                    )
                ],
            )
        return LLMCompletionResult(content="final")

    def complete_text(self, *, messages, model, temperature, options=None):
        return json.dumps(
            {
                "run_id": "run-0000",
                "objective": "test",
                "scope": [],
                "source_code_locations": [],
                "open_questions": [],
                "next_action": None,
                "stop_conditions": [],
                "constraints": [],
                "relevant_artifacts": [],
                "status": "active",
                "domain_extensions": {},
            }
        )


class FinalProvider(ToolThenFinalProvider):
    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        self.chat_calls += 1
        return _completion()


def _settings(tmp_path) -> CoreSettings:
    return CoreSettings(
        openai_api_key="test",
        model="fake",
        memory_model="fake",
        session_file=tmp_path / "session.json",
        base_system_prompt="system",
        task_state_synthesis_prompt="task",
        session_summary_synthesis_prompt="summary",
        session_summary_merge_prompt="merge",
        max_active_context_tokens=100_000,
    )


def _orchestrator(tmp_path, provider: ToolThenFinalProvider, tool) -> AgentOrchestrator:
    settings = _settings(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    return AgentOrchestrator(
        settings=settings,
        provider=provider,
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )


def test_structured_task_budget_stops_before_second_model_call(tmp_path) -> None:
    provider = ToolThenFinalProvider()
    settings = _settings(tmp_path)
    registry = ToolRegistry()
    registry.register(EchoTool())
    runner = StructuredTaskRunner(
        settings=settings,
        provider=provider,
        tool_registry=registry,
        policy_engine=PolicyEngine(),
    )

    result = run_structured(
        runner,
        spec=StructuredTaskSpec(
            task_id="budgeted",
            system_prompt="Use the tool.",
            objective="Echo once, then answer.",
            allowed_tools=["echo"],
            max_iterations=2,
            llm_budget=LLMBudget(max_calls=1),
        ),
    )

    assert result.ok is False
    assert provider.chat_calls == 1
    assert "budget was exhausted" in result.failure_reason
    assert result.metadata["llm_budget_usage"]["calls_started"] == 1
    assert result.metadata["llm_budget_usage"]["exhausted_dimension"] == "calls"


def test_structured_checkpoint_persists_call_reservation_before_dispatch(tmp_path) -> None:
    provider = FinalProvider()
    settings = _settings(tmp_path)
    checkpoints: list[dict[str, Any]] = []
    runner = StructuredTaskRunner(
        settings=settings,
        provider=provider,
        tool_registry=ToolRegistry(),
        policy_engine=PolicyEngine(),
    )

    result = runner.run(
        spec=StructuredTaskSpec(
            task_id="reserved_checkpoint",
            system_prompt="Answer.",
            objective="Answer once.",
            llm_budget=LLMBudget(max_calls=2),
        ),
        context=execution_context(settings),
        on_checkpoint=lambda checkpoint: checkpoints.append(checkpoint.to_dict()),
    )

    assert result.ok is True
    assert provider.chat_calls == 1
    assert any(
        checkpoint["llm_budget_usage"]["calls_started"] == 1 and checkpoint["llm_calls"] == []
        for checkpoint in checkpoints
    )


def test_pending_resume_restores_budget_usage(tmp_path) -> None:
    provider = ToolThenFinalProvider(tool_name="pending")
    orchestrator = _orchestrator(tmp_path, provider, PendingTool())

    pending = run_turn(
        orchestrator,
        "start pending work",
        options=RunOptions.direct(llm_budget=LLMBudget(max_calls=1)),
    )
    assert pending.status == "pending_tool_result"
    assert pending.metadata["llm_budget_usage"]["calls_started"] == 1

    completed = resume_turn(
        orchestrator,
        pending_id=pending.pending_id or "",
        tool_content="external result",
    )

    assert provider.chat_calls == 1
    assert completed.metadata["stop_reason"] == "llm_budget_exhausted"
    assert completed.metadata["llm_budget_usage"]["calls_started"] == 1
    assert completed.metadata["llm_budget_usage"]["exhausted_dimension"] == "calls"


def test_structured_checkpoint_round_trip_preserves_budget() -> None:
    checkpoint = StructuredTaskCheckpoint(
        spec_fingerprint="fingerprint",
        phase="model_request",
        messages=[LLMMessage(role="user", content="resume")],
        iterations=1,
        llm_budget=LLMBudget(max_calls=3, max_total_tokens=500),
    )
    checkpoint.llm_budget_usage.calls_started = 1
    checkpoint.llm_budget_usage.accounted_input_tokens = 40

    restored = StructuredTaskCheckpoint.from_dict(checkpoint.to_dict())

    assert restored is not None
    assert restored.llm_budget == LLMBudget(max_calls=3, max_total_tokens=500)
    assert restored.llm_budget_usage.calls_started == 1
    assert restored.llm_budget_usage.accounted_input_tokens == 40
