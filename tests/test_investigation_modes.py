from __future__ import annotations

import json
from typing import Any

from agent_core.llm.base import LLMCompletionResult, LLMToolCall
from agent_core.orchestrator import AgentOrchestrator
from agent_core.policy_engine import PolicyEngine
from agent_core.run_options import RunOptions
from agent_core.session_manager import SessionManager
from agent_core.session_repo import SessionRepository
from agent_core.settings import CoreSettings
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import AuthorizationResult, ToolResult


def task_state_payload() -> dict[str, Any]:
    return {
        "run_id": "run-0000",
        "objective": "Test objective",
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


def reflection_payload(
    *,
    observation_summary: str = "Observed tool output",
    new_facts: list[str] | None = None,
    remaining_gaps: list[str] | None = None,
    recommended_next_actions: list[str] | None = None,
    confidence: float = 0.8,
    should_continue: bool = True,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "observation_summary": observation_summary,
        "new_facts": new_facts or [],
        "updated_hypotheses": [],
        "rejected_hypotheses": [],
        "remaining_gaps": remaining_gaps or [],
        "recommended_next_actions": recommended_next_actions or [],
        "risk_notes": [],
        "confidence": confidence,
        "should_continue": should_continue,
        "stop_reason": stop_reason,
    }


def decision_payload(kind: str, *, reason_summary: str = "continue", question: str | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "reason_summary": reason_summary,
        "next_action": "continue" if kind == "continue" else None,
        "question": question,
        "required_approval": False,
    }


def critique_payload(
    *,
    approved: bool,
    unsupported_claims: list[str] | None = None,
    missing_evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "approved": approved,
        "unsupported_claims": unsupported_claims or [],
        "missing_evidence": missing_evidence or [],
        "scope_or_safety_issues": [],
        "suggested_followup_actions": ["collect more evidence"] if not approved else [],
    }


class ScriptedProvider:
    def __init__(
        self,
        *,
        chat: list[LLMCompletionResult],
        plans: list[dict[str, Any]] | None = None,
        reflections: list[dict[str, Any]] | None = None,
        decisions: list[dict[str, Any]] | None = None,
        critiques: list[dict[str, Any]] | None = None,
    ) -> None:
        self.chat = list(chat)
        self.plans = list(plans or [])
        self.reflections = list(reflections or [])
        self.decisions = list(decisions or [])
        self.critiques = list(critiques or [])
        self.tool_options = []
        self.text_options = []

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        self.tool_options.append(options)
        if not self.chat:
            raise AssertionError("No scripted chat response left")
        return self.chat.pop(0)

    def complete_text(self, *, messages, model, temperature, options=None):
        self.text_options.append(options)
        system_prompt = messages[0].content
        if "preparing a generic bounded investigation plan" in system_prompt:
            return json.dumps(self.plans.pop(0))
        if "updating a generic investigation state" in system_prompt:
            return json.dumps(self.reflections.pop(0))
        if "choosing the next generic investigation action" in system_prompt:
            return json.dumps(self.decisions.pop(0))
        if "critiquing a final draft" in system_prompt:
            return json.dumps(self.critiques.pop(0))
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


class PendingTool:
    name = "pending"
    description = "Return pending."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        )

    def execute(self, arguments, context):
        return ToolResult.pending_result("waiting", metadata={"job_id": arguments["value"]})


def tool_call(name: str = "echo", *, value: str = "hello", call_id: str = "call-1") -> LLMCompletionResult:
    return LLMCompletionResult(
        content="",
        tool_calls=[
            LLMToolCall(
                id=call_id,
                name=name,
                arguments_json=json.dumps({"value": value}),
            )
        ],
    )


def build_orchestrator(tmp_path, provider, *, policy_engine: PolicyEngine | None = None, pending: bool = False):
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
    registry.register(PendingTool() if pending else EchoTool())
    return AgentOrchestrator(
        settings=settings,
        provider=provider,
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=policy_engine or PolicyEngine(),
    )


def test_direct_mode_without_options_preserves_tool_loop_shape(tmp_path) -> None:
    provider = ScriptedProvider(chat=[tool_call(), LLMCompletionResult(content="final")])
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result("echo")

    assert result.status == "completed"
    assert result.content == "final"
    assert result.metadata == {}
    blocks = orchestrator.session_manager.get_context_blocks()
    assert [block.kind for block in blocks] == ["tool_exchange", "conversation_turn"]
    assert len(blocks[0].content["tool_messages"]) == 1


def test_investigation_no_tool_no_critique_returns_final(tmp_path) -> None:
    provider = ScriptedProvider(chat=[LLMCompletionResult(content="plain final")])
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result(
        "answer",
        options=RunOptions(mode="investigate", max_iterations=1, require_initial_plan=False),
    )

    assert result.content == "plain final"
    assert result.metadata["mode"] == "investigate"
    assert result.metadata["iterations_used"] == 1
    assert "reasoning" not in json.dumps(result.metadata)


def test_investigation_tool_result_updates_state_and_returns_state_answer(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[tool_call(value="fact")],
        reflections=[
            reflection_payload(
                new_facts=["echo returned fact"],
                remaining_gaps=["need second source"],
                recommended_next_actions=["verify independently"],
                should_continue=False,
            )
        ],
        decisions=[decision_payload("final", reason_summary="enough evidence")],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result(
        "investigate",
        options=RunOptions(mode="investigate", max_iterations=2, require_initial_plan=False),
    )

    assert "echo returned fact" in result.content
    assert "need second source" in result.content
    assert result.metadata["stop_reason"] == "enough evidence"
    assert result.metadata["investigation_state"]["facts"] == ["echo returned fact"]


def test_investigation_stops_at_max_iterations_with_budget_answer(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[tool_call()],
        reflections=[
            reflection_payload(
                new_facts=["first fact"],
                remaining_gaps=["open gap"],
                recommended_next_actions=["continue checking"],
            )
        ],
        decisions=[decision_payload("continue")],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result(
        "investigate",
        options=RunOptions(mode="investigate", max_iterations=1, require_initial_plan=False),
    )

    assert result.metadata["stop_reason"] == "max_iterations"
    assert "first fact" in result.content
    assert "open gap" in result.content


def test_investigation_stops_after_no_progress(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[tool_call()],
        reflections=[reflection_payload(observation_summary="", confidence=0.0)],
        decisions=[decision_payload("continue")],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result(
        "investigate",
        options=RunOptions(
            mode="investigate",
            max_iterations=3,
            max_no_progress_iterations=1,
            require_initial_plan=False,
        ),
    )

    assert result.metadata["stop_reason"] == "no_progress"


def test_investigation_stops_at_max_tool_calls(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[tool_call()],
        reflections=[reflection_payload(new_facts=["used final tool"])],
        decisions=[decision_payload("continue")],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result(
        "investigate",
        options=RunOptions(mode="investigate", max_iterations=3, max_tool_calls=1, require_initial_plan=False),
    )

    assert result.metadata["stop_reason"] == "max_tool_calls"


def test_investigation_returns_ask_user_question(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[tool_call()],
        reflections=[reflection_payload(new_facts=["partial fact"])],
        decisions=[decision_payload("ask_user", reason_summary="need input", question="Which target should I use?")],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result(
        "investigate",
        options=RunOptions(mode="investigate", max_iterations=2, require_initial_plan=False),
    )

    assert result.content == "Which target should I use?"
    assert result.metadata["stop_reason"] == "ask_user"


def test_investigation_policy_denial_can_block_safely(tmp_path) -> None:
    def deny(arguments, context):
        return AuthorizationResult(False, "denied for test")

    provider = ScriptedProvider(
        chat=[tool_call()],
        reflections=[reflection_payload(remaining_gaps=["tool was denied"], confidence=0.2)],
        decisions=[decision_payload("blocked", reason_summary="required action was denied")],
    )
    orchestrator = build_orchestrator(tmp_path, provider, policy_engine=PolicyEngine(validators={"echo": deny}))

    result = orchestrator.run_turn_result(
        "investigate",
        options=RunOptions(mode="investigate", max_iterations=2, require_initial_plan=False),
    )

    assert result.content == "Investigation blocked: required action was denied"
    assert result.metadata["stop_reason"] == "blocked"


def test_final_critique_approved_returns_draft(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[LLMCompletionResult(content="approved draft")],
        critiques=[critique_payload(approved=True)],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result(
        "answer",
        options=RunOptions(
            mode="investigate",
            max_iterations=1,
            require_initial_plan=False,
            require_final_critique=True,
        ),
    )

    assert result.content == "approved draft"
    assert result.metadata["stop_reason"] == "final_critique_approved"


def test_final_critique_rejected_continues_when_budget_remains(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[LLMCompletionResult(content="unsupported draft"), LLMCompletionResult(content="revised draft")],
        critiques=[
            critique_payload(approved=False, unsupported_claims=["unsupported claim"]),
            critique_payload(approved=True),
        ],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = orchestrator.run_turn_result(
        "answer",
        options=RunOptions(
            mode="investigate",
            max_iterations=2,
            require_initial_plan=False,
            require_final_critique=True,
        ),
    )

    assert result.content == "revised draft"
    assert result.metadata["iterations_used"] == 2


def test_investigation_pending_resume_continues_same_mode(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[tool_call(name="pending", value="job-1")],
        reflections=[reflection_payload(new_facts=["external result arrived"], should_continue=False)],
        decisions=[decision_payload("final", reason_summary="pending result resolved")],
    )
    orchestrator = build_orchestrator(tmp_path, provider, pending=True)

    pending = orchestrator.run_turn_result(
        "start pending",
        options=RunOptions(mode="investigate", max_iterations=2, require_initial_plan=False),
    )

    assert pending.status == "pending_tool_result"
    assert pending.metadata["job_id"] == "job-1"
    assert pending.metadata["mode"] == "investigate"

    completed = orchestrator.resume_turn(pending_id=pending.pending_id or "", tool_content="done")

    assert completed.status == "completed"
    assert "external result arrived" in completed.content
    assert completed.metadata["mode"] == "investigate"
    assert [block.kind for block in orchestrator.session_manager.get_context_blocks()] == [
        "tool_exchange",
        "conversation_turn",
    ]
    assert orchestrator.session_manager.get_state()["meta"].get(AgentOrchestrator.PENDING_TURN_META_KEY) is None
