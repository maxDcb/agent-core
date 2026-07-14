from __future__ import annotations

from typing import Any

import pytest

from agent_core import LLMContextPolicy
from agent_core.context_planner import (
    LLMContextOverflowError,
    LLMContextPlanner,
    LLMContextUsage,
    estimate_llm_input_tokens,
    llm_context_scope,
)
from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage, LLMToolCall
from agent_core.llm_budget import LLMBudget, LLMBudgetController, llm_budget_scope, run_budgeted_llm_call
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
from tests.run_helpers import execution_context, resume_turn, run_turn


def _tool_exchange() -> list[LLMMessage]:
    return [
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(id="call-1", name="lookup", arguments_json='{"q":"recent"}')],
        ),
        LLMMessage(role="tool", content="recent evidence", tool_call_id="call-1"),
    ]


def test_planner_removes_old_history_and_keeps_tool_exchange_atomic() -> None:
    system = LLMMessage(role="system", content="system")
    current_user = LLMMessage(role="user", content="current request")
    exchange = _tool_exchange()
    expected = [system, current_user, *exchange]
    reserve = 40
    max_input = estimate_llm_input_tokens(messages=expected)
    messages = [
        system,
        LLMMessage(role="user", content="old question " * 100),
        LLMMessage(role="assistant", content="old answer " * 100),
        current_user,
        *exchange,
    ]
    planner = LLMContextPlanner(
        LLMContextPolicy(
            max_context_tokens=max_input + reserve,
            reserved_output_tokens=reserve,
            safety_margin_tokens=0,
        )
    )

    planned, options, plan = planner.plan_call(
        messages=messages,
        tools=None,
        purpose="tool_loop",
        options=None,
    )

    assert planned == expected
    assert [message.role for message in planned[-2:]] == ["assistant", "tool"]
    assert options is not None and options.max_output_tokens == reserve
    assert plan.fits is True
    assert plan.compacted is True
    assert plan.removed_message_count == 2
    assert planner.usage.calls_compacted == 1


def test_planner_rejects_mandatory_context_before_provider_call() -> None:
    planner = LLMContextPlanner(
        LLMContextPolicy(
            max_context_tokens=120,
            reserved_output_tokens=20,
            safety_margin_tokens=0,
        )
    )
    messages = [
        LLMMessage(role="system", content="system " * 200),
        LLMMessage(role="user", content="request " * 200),
    ]
    invoked = False

    def invoke(options: LLMCallOptions | None) -> LLMCompletionResult:
        nonlocal invoked
        invoked = True
        return LLMCompletionResult(content="unexpected")

    with llm_context_scope(planner):
        with pytest.raises(LLMContextOverflowError) as captured:
            run_budgeted_llm_call(
                messages=messages,
                purpose="mandatory_overflow",
                options=None,
                invoke=invoke,
            )

    assert invoked is False
    assert captured.value.kind == "context_overflow"
    assert planner.usage.calls_overflowed == 1
    assert planner.usage.plans_by_purpose == {"mandatory_overflow": 1}


def test_planner_never_drops_current_turn_tool_results_to_make_room() -> None:
    messages = [
        LLMMessage(role="system", content="system"),
        LLMMessage(role="user", content="current request"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(id="call-large", name="lookup", arguments_json="{}")],
        ),
        LLMMessage(role="tool", content="evidence " * 500, tool_call_id="call-large"),
    ]
    mandatory_without_tool_result = estimate_llm_input_tokens(messages=messages[:2])
    planner = LLMContextPlanner(
        LLMContextPolicy(
            max_context_tokens=mandatory_without_tool_result + 40,
            reserved_output_tokens=40,
            safety_margin_tokens=0,
        )
    )

    with pytest.raises(LLMContextOverflowError):
        planner.plan_call(
            messages=messages,
            tools=None,
            purpose="current_tool_result",
            options=None,
        )


def test_observe_mode_reports_overflow_without_changing_request() -> None:
    messages = [
        LLMMessage(role="system", content="system " * 100),
        LLMMessage(role="user", content="request " * 100),
    ]
    planner = LLMContextPlanner(
        LLMContextPolicy(
            max_context_tokens=100,
            reserved_output_tokens=20,
            safety_margin_tokens=0,
            mode="observe",
        )
    )
    seen_messages: list[LLMMessage] = []
    seen_options: list[LLMCallOptions | None] = []

    with llm_context_scope(planner):
        run_budgeted_llm_call(
            messages=messages,
            purpose="observe",
            options=None,
            invoke=lambda options: (
                seen_messages.extend(messages),
                seen_options.append(options),
                LLMCompletionResult(content="ok"),
            )[-1],
        )

    assert seen_messages == messages
    assert seen_options == [None]
    assert planner.usage.calls_overflowed == 1
    assert planner.usage.calls_compacted == 0


def test_planner_accounts_for_tools_and_response_schema() -> None:
    messages = [LLMMessage(role="system", content="system"), LLMMessage(role="user", content="request")]
    tools = [{"name": "large", "description": "tool " * 100, "parameters": {"type": "object"}}]
    options = LLMCallOptions(
        response_format={"type": "json_schema", "json_schema": {"description": "schema " * 100}},
        max_output_tokens=20,
    )
    without_fixed_payload = estimate_llm_input_tokens(messages=messages)
    with_fixed_payload = estimate_llm_input_tokens(messages=messages, tools=tools, options=options)
    assert with_fixed_payload > without_fixed_payload

    planner = LLMContextPlanner(
        LLMContextPolicy(
            max_context_tokens=without_fixed_payload + 40,
            reserved_output_tokens=20,
            safety_margin_tokens=0,
        )
    )
    with pytest.raises(LLMContextOverflowError):
        planner.plan_call(
            messages=messages,
            tools=tools,
            purpose="fixed_payload",
            options=options,
        )


def test_budget_accounts_for_context_after_planning() -> None:
    system = LLMMessage(role="system", content="system")
    current = LLMMessage(role="user", content="current")
    messages = [
        system,
        LLMMessage(role="user", content="old " * 200),
        LLMMessage(role="assistant", content="answer " * 200),
        current,
    ]
    planned_estimate = estimate_llm_input_tokens(messages=[system, current])
    planner = LLMContextPlanner(
        LLMContextPolicy(
            max_context_tokens=planned_estimate + 20,
            reserved_output_tokens=20,
            safety_margin_tokens=0,
        )
    )
    budget = LLMBudgetController(LLMBudget(max_input_tokens=planned_estimate))
    provider_messages: list[LLMMessage] = []

    with llm_context_scope(planner):
        with llm_budget_scope(budget):
            run_budgeted_llm_call(
                messages=messages,
                purpose="planned_then_budgeted",
                options=None,
                invoke=lambda options: provider_messages.extend(messages) or LLMCompletionResult(content="ok"),
            )

    assert provider_messages == [system, current]
    assert budget.usage.accounted_input_tokens == planned_estimate


def test_context_configuration_is_normalized_from_dicts() -> None:
    policy = LLMContextPolicy(max_context_tokens=4096, reserved_output_tokens=512)
    settings = CoreSettings(  # type: ignore[arg-type]
        llm_context_policy={"max_context_tokens": 4096, "reserved_output_tokens": 512}
    )
    options = RunOptions.direct(  # type: ignore[arg-type]
        llm_context_policy={"max_context_tokens": 4096, "reserved_output_tokens": 512}
    )
    spec = StructuredTaskSpec(  # type: ignore[arg-type]
        task_id="context",
        system_prompt="",
        objective="",
        llm_context_policy={"max_context_tokens": 4096, "reserved_output_tokens": 512},
    )

    assert settings.llm_context_policy == policy
    assert options.llm_context_policy == policy
    assert spec.llm_context_policy == policy


class _FinalProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_with_tools(self, **kwargs: Any) -> LLMCompletionResult:
        self.calls += 1
        return LLMCompletionResult(content="done")

    def complete_text(self, **kwargs: Any) -> LLMCompletionResult:
        return LLMCompletionResult(content="done")


def test_structured_checkpoint_persists_context_plan_before_dispatch(tmp_path) -> None:
    provider = _FinalProvider()
    settings = CoreSettings(base_system_prompt="system", session_file=tmp_path / "session.json")
    checkpoints: list[dict[str, Any]] = []
    runner = StructuredTaskRunner(
        settings=settings,
        provider=provider,
        tool_registry=ToolRegistry(),
        policy_engine=PolicyEngine(),
    )
    policy = LLMContextPolicy(max_context_tokens=4096, reserved_output_tokens=128)

    result = runner.run(
        spec=StructuredTaskSpec(
            task_id="planned_checkpoint",
            system_prompt="Answer.",
            objective="Answer once.",
            llm_context_policy=policy,
        ),
        context=execution_context(settings),
        on_checkpoint=lambda checkpoint: checkpoints.append(checkpoint.to_dict()),
    )

    assert result.ok is True
    assert provider.calls == 1
    assert any(
        checkpoint["llm_context_usage"]["plans_created"] == 1 and checkpoint["llm_calls"] == []
        for checkpoint in checkpoints
    )
    assert result.metadata["llm_context_usage"]["plans_created"] == 1


def test_context_checkpoint_round_trip_preserves_usage() -> None:
    policy = LLMContextPolicy(max_context_tokens=4096, reserved_output_tokens=256)
    checkpoint = StructuredTaskCheckpoint(
        spec_fingerprint="fingerprint",
        phase="model_request",
        messages=[LLMMessage(role="user", content="resume")],
        iterations=1,
        llm_context_policy=policy,
        llm_context_usage=LLMContextUsage(plans_created=2, calls_compacted=1),
    )

    restored = StructuredTaskCheckpoint.from_dict(checkpoint.to_dict())

    assert restored is not None
    assert restored.llm_context_policy == policy
    assert restored.llm_context_usage.plans_created == 2
    assert restored.llm_context_usage.calls_compacted == 1


class _PendingTool:
    name = "pending"
    description = "Start pending work."

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
        return ToolResult.pending_result("waiting", metadata={"value": arguments["value"]})


class _PendingThenFinalProvider:
    def __init__(self) -> None:
        self.chat_calls = 0

    def complete_with_tools(self, **kwargs: Any) -> LLMCompletionResult:
        self.chat_calls += 1
        if self.chat_calls == 1:
            return LLMCompletionResult(
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="call-pending",
                        name="pending",
                        arguments_json='{"value":"job-1"}',
                    )
                ],
            )
        return LLMCompletionResult(content="final")

    def complete_text(self, **kwargs: Any) -> LLMCompletionResult:
        return LLMCompletionResult(
            content=(
                '{"run_id":"run-0000","objective":"test","scope":[],'
                '"source_code_locations":[],"open_questions":[],"next_action":null,'
                '"stop_conditions":[],"constraints":[],"relevant_artifacts":[],'
                '"status":"active","domain_extensions":{}}'
            )
        )


def test_pending_conversation_resume_restores_context_usage(tmp_path) -> None:
    policy = LLMContextPolicy(max_context_tokens=8_192, reserved_output_tokens=256)
    settings = CoreSettings(
        model="fake",
        memory_model="fake",
        session_file=tmp_path / "session.json",
        base_system_prompt="system",
        task_state_synthesis_prompt="task state",
        llm_context_policy=policy,
    )
    provider = _PendingThenFinalProvider()
    registry = ToolRegistry()
    registry.register(_PendingTool())
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )

    pending = run_turn(
        orchestrator,
        "start work",
        options=RunOptions.direct(llm_context_policy=policy),
    )
    assert pending.status == "pending_tool_result"
    assert pending.metadata["llm_context_usage"]["plans_created"] == 1

    completed = resume_turn(
        orchestrator,
        pending_id=pending.pending_id or "",
        tool_content="external result",
    )

    assert completed.status == "completed"
    assert provider.chat_calls == 2
    assert completed.metadata["llm_context_usage"]["plans_created"] >= 2


def test_conversation_reports_irreducible_context_overflow_without_provider_call(tmp_path) -> None:
    policy = LLMContextPolicy(
        max_context_tokens=160,
        reserved_output_tokens=20,
        safety_margin_tokens=0,
    )
    settings = CoreSettings(
        model="fake",
        memory_model="fake",
        session_file=tmp_path / "session.json",
        base_system_prompt="system " * 100,
    )
    provider = _PendingThenFinalProvider()
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        registry=ToolRegistry(),
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )

    result = run_turn(
        orchestrator,
        "request " * 100,
        options=RunOptions.direct(llm_context_policy=policy),
    )

    assert result.status == "completed"
    assert result.metadata["stop_reason"] == "llm_context_overflow"
    assert result.metadata["provider_error_kind"] == "context_overflow"
    assert result.metadata["llm_context_usage"]["calls_overflowed"] >= 1
    assert provider.chat_calls == 0
