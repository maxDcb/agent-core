from __future__ import annotations

import json
from typing import Any

from agent_core.llm.base import LLMCompletionResult, LLMMessage, LLMToolCall
from agent_core.orchestrator import AgentOrchestrator
from agent_core.policy_engine import PolicyEngine
from agent_core.run_options import RunOptions
from agent_core.run_trace import PromptSnapshot, RunTrace
from agent_core.session_manager import SessionManager
from agent_core.session_repo import SessionRepository
from agent_core.settings import CoreSettings
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult


def task_state_payload() -> dict[str, Any]:
    return {
        "run_id": "run-0000",
        "objective": "Trace test",
        "scope": [],
        "source_code_locations": [],
        "domain_extensions": {},
        "open_questions": [],
        "next_action": None,
        "stop_conditions": [],
        "constraints": [],
        "relevant_artifacts": [],
        "status": "active",
    }


def reflection_payload() -> dict[str, Any]:
    return {
        "observation_summary": "Observed tool output",
        "new_facts": ["echo returned fact"],
        "updated_hypotheses": [],
        "rejected_hypotheses": [],
        "remaining_gaps": [],
        "recommended_next_actions": [],
        "risk_notes": [],
        "confidence": 0.9,
        "should_continue": False,
        "stop_reason": "enough evidence",
    }


def decision_payload() -> dict[str, Any]:
    return {
        "kind": "final",
        "reason_summary": "enough evidence",
        "next_action": None,
        "question": None,
        "required_approval": False,
    }


class ScriptedProvider:
    def __init__(self, *, chat: list[LLMCompletionResult]) -> None:
        self.chat = list(chat)

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        if not self.chat:
            raise AssertionError("No scripted chat response left")
        return self.chat.pop(0)

    def complete_text(self, *, messages, model, temperature, options=None):
        target = (options.metadata or {}).get("target") if options is not None else None
        if target == "investigation_step_reflection":
            return json.dumps(reflection_payload())
        if target == "investigation_decision":
            return json.dumps(decision_payload())
        return json.dumps(task_state_payload())


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
                "additionalProperties": False,
            },
        )

    def execute(self, arguments, context):
        return ToolResult(ok=True, content=f"echo:{arguments['value']}")


def tool_call(*, value: str = "hello") -> LLMCompletionResult:
    return LLMCompletionResult(
        content="",
        tool_calls=[
            LLMToolCall(
                id="call-1",
                name="echo",
                arguments_json=json.dumps({"value": value}),
            )
        ],
    )


def build_orchestrator(tmp_path, provider: ScriptedProvider) -> AgentOrchestrator:
    settings = CoreSettings(
        openai_api_key="test",
        model="fake",
        memory_model="fake",
        session_file=tmp_path / "session.json",
        base_system_prompt="system",
        task_state_synthesis_prompt="task",
        session_summary_synthesis_prompt="summary",
        session_summary_merge_prompt="merge",
        max_active_context_tokens=100000,
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    return AgentOrchestrator(
        settings=settings,
        provider=provider,
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )


def event_types(trace_payload: dict[str, object]) -> set[str]:
    events = trace_payload.get("events")
    assert isinstance(events, list)
    return {event["type"] for event in events if isinstance(event, dict)}


def test_prompt_snapshot_tracks_messages_and_context_budget() -> None:
    snapshot = PromptSnapshot.from_messages(
        messages=[
            LLMMessage(role="system", content="system"),
            LLMMessage(role="user", content="hello"),
        ],
        context_window_tokens=100,
    )

    assert snapshot.message_count == 2
    assert snapshot.estimated_tokens > 0
    assert snapshot.context_budget.usage_percent is not None
    assert [block.type for block in snapshot.blocks] == ["system", "user_query"]


def test_session_repository_persists_run_trace_summaries(tmp_path) -> None:
    repo = SessionRepository(tmp_path / "session.json")
    trace = RunTrace.start(
        run_id="run-test",
        session_id="default",
        mode="direct",
        turn_index=0,
        prompt_snapshot=PromptSnapshot.from_messages(
            messages=[LLMMessage(role="user", content="hello")],
            context_window_tokens=1000,
        ),
    )
    trace.complete(status="completed", final_metadata={"stop_reason": "completed"})

    repo.save_run_trace("default", trace.to_dict())

    assert (tmp_path / "default" / "traces" / "run-test.json").exists()
    loaded = repo.load_run_trace("default", "run-test")
    assert loaded is not None
    assert loaded["run_id"] == "run-test"
    summaries = repo.list_run_traces("default")
    assert summaries == [trace.to_summary_dict()]


def test_session_repository_stores_state_under_session_directory(tmp_path) -> None:
    repo = SessionRepository(tmp_path / "session.json")
    state = repo.load("workspace-a")
    state["tool_history"].append({"tool_name": "search_code", "status": "ok"})

    repo.save("workspace-a", state)

    assert (tmp_path / "workspace-a" / "workspace-a.json").exists()
    assert not (tmp_path / "workspace-a.json").exists()
    loaded = repo.load("workspace-a")
    assert loaded["tool_history"] == [{"tool_name": "search_code", "status": "ok"}]


def test_direct_run_persists_tool_audit_trace_without_changing_default_metadata(tmp_path) -> None:
    provider = ScriptedProvider(chat=[tool_call(), LLMCompletionResult(content="final")])
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result("echo")

    assert result.status == "completed"
    assert result.metadata == {}
    summaries = orchestrator.session_manager.list_run_traces()
    assert len(summaries) == 1
    trace_payload = orchestrator.session_manager.load_run_trace(str(summaries[0]["run_id"]))
    assert trace_payload is not None
    assert trace_payload["status"] == "completed"
    assert {
        "run_started",
        "prompt_snapshot_created",
        "llm_call_started",
        "assistant_response_received",
        "tool_call_started",
        "tool_call_completed",
        "tool_exchange_completed",
        "run_completed",
    }.issubset(event_types(trace_payload))


def test_investigation_run_persists_process_events_and_trace_id(tmp_path) -> None:
    provider = ScriptedProvider(chat=[tool_call(value="fact")])
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result(
        "investigate",
        options=RunOptions(mode="investigate", max_iterations=2, require_initial_plan=False),
    )

    run_trace_id = result.metadata["run_trace_id"]
    assert isinstance(run_trace_id, str)
    trace_payload = orchestrator.session_manager.load_run_trace(run_trace_id)
    assert trace_payload is not None
    assert trace_payload["mode"] == "investigate"
    assert trace_payload["final_metadata"]["run_trace_id"] == run_trace_id
    prompt_snapshot = trace_payload["prompt_snapshot"]
    assert "run_mode_guidance" in [block["type"] for block in prompt_snapshot["blocks"]]
    assert {
        "investigation_iteration_started",
        "assistant_step_completed",
        "tool_step_completed",
        "reflection_completed",
        "decision_completed",
        "investigation_completed",
        "run_completed",
    }.issubset(event_types(trace_payload))
