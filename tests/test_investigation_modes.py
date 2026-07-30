from __future__ import annotations

import json
from typing import Any

import pytest

from agent_core.domain_hooks import DomainHooks
from agent_core.investigation_controller import INVESTIGATION_STATE_MESSAGE_PREFIX
from agent_core.investigation_prompts import InvestigationPromptSet
from agent_core.llm.base import LLMCompletionResult, LLMMessage, LLMToolCall
from agent_core.llm.errors import LLMProviderError
from agent_core.memory.thread_state import render_context_blocks_to_messages
from agent_core.orchestrator import AgentOrchestrator
from agent_core.output_contracts import StructuredOutputContract
from agent_core.policy_engine import PolicyEngine
from agent_core.run_options import RunOptions
from agent_core.session_manager import SessionManager
from agent_core.session_repo import SessionRepository
from agent_core.settings import CoreSettings
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import AuthorizationResult, ToolResult
from tests.run_helpers import resume_turn, run_turn, turn_memory_payload


def reflection_payload(
    *,
    observation_summary: str = "Observed tool output",
    new_facts: list[str] | None = None,
    remaining_gaps: list[str] | None = None,
    resolved_gaps: list[str] | None = None,
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
        "resolved_gaps": resolved_gaps or [],
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
        finals: list[LLMCompletionResult | Exception] | None = None,
    ) -> None:
        self.chat = list(chat)
        self.plans = list(plans or [])
        self.reflections = list(reflections or [])
        self.decisions = list(decisions or [])
        self.critiques = list(critiques or [])
        self.finals = list(finals or [])
        self.tool_options = []
        self.chat_tools = []
        self.text_options = []
        self.chat_messages = []
        self.text_prompts = []

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        self.tool_options.append(options)
        self.chat_tools.append(list(tools))
        self.chat_messages.append(list(messages))
        target = (options.metadata or {}).get("target") if options is not None else None
        if target == "investigation_final_response":
            if self.finals:
                scripted = self.finals.pop(0)
                if isinstance(scripted, Exception):
                    raise scripted
                return scripted
            payload = json.loads(messages[-1].content)
            candidate = payload.get("candidate_answer")
            if candidate:
                return LLMCompletionResult(content=candidate)
            state = payload["investigation_state"]
            facts = [item["summary"] for item in state.get("facts", [])]
            gaps = list(state.get("evidence_gaps", []))
            rendered = "\n".join([*facts, *gaps]).strip() or "No conclusive result was established."
            return LLMCompletionResult(content=rendered)
        if not self.chat:
            raise AssertionError("No scripted chat response left")
        return self.chat.pop(0)

    def complete_text(self, *, messages, model, temperature, options=None):
        self.text_options.append(options)
        system_prompt = messages[0].content
        self.text_prompts.append(system_prompt)
        target = (options.metadata or {}).get("target") if options is not None else None
        if target == "investigation_initial_plan":
            return json.dumps(self.plans.pop(0))
        if target == "investigation_step_reflection":
            return json.dumps(self.reflections.pop(0))
        if target == "investigation_decision":
            return json.dumps(self.decisions.pop(0))
        if target == "investigation_final_critique":
            return json.dumps(self.critiques.pop(0))
        return json.dumps(turn_memory_payload())


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


class CustomInvestigationHooks(DomainHooks):
    def __init__(self) -> None:
        self.modes: list[str] = []

    def customize_investigation_prompts(self, *, prompt_set, settings, options) -> InvestigationPromptSet:
        self.modes.append(options.mode)
        return prompt_set.append_domain_guidance(
            "Domain investigation guidance: preserve source provenance and follow-up gaps."
        )


def build_orchestrator(
    tmp_path,
    provider,
    *,
    memory_provider=None,
    policy_engine: PolicyEngine | None = None,
    pending: bool = False,
    domain_hooks: DomainHooks | None = None,
):
    settings = CoreSettings(
        openai_api_key="test",
        model="fake",
        memory_model="fake",
        session_file=tmp_path / "session.json",
        base_system_prompt="system",
        turn_memory_synthesis_prompt="memory",
        max_active_context_tokens=100000,
    )
    registry = ToolRegistry()
    registry.register(PendingTool() if pending else EchoTool())
    return AgentOrchestrator(
        settings=settings,
        provider=provider,
        memory_provider=memory_provider,
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=policy_engine or PolicyEngine(),
        domain_hooks=domain_hooks,
    )


def test_direct_mode_without_options_preserves_tool_loop_shape(tmp_path) -> None:
    provider = ScriptedProvider(chat=[tool_call(), LLMCompletionResult(content="final")])
    orchestrator = build_orchestrator(tmp_path, provider)

    result = run_turn(orchestrator, "echo")

    assert result.status == "completed"
    assert result.content == "final"
    assert result.metadata["run_trace_id"].startswith("test-run-")
    blocks = orchestrator.session_manager.get_context_blocks()
    assert [block.kind for block in blocks] == ["tool_exchange", "conversation_turn"]
    assert len(blocks[0].content["tool_messages"]) == 1


def test_direct_mode_budget_exhaustion_persists_tool_responses_for_skipped_calls(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[
            LLMCompletionResult(
                content="",
                tool_calls=[
                    LLMToolCall(id="call-1", name="echo", arguments_json=json.dumps({"value": "first"})),
                    LLMToolCall(id="call-2", name="echo", arguments_json=json.dumps({"value": "second"})),
                ],
            )
        ]
    )
    orchestrator = build_orchestrator(tmp_path, provider)
    orchestrator.settings.max_tool_calls_per_turn = 1

    result = run_turn(orchestrator, "echo twice")
    blocks = orchestrator.session_manager.get_context_blocks()
    tool_messages = blocks[0].content["tool_messages"]
    history = orchestrator.session_manager.get_state()["tool_history"]

    assert result.content == "Maximum number of tool calls reached for this turn."
    assert [item["status"] for item in history] == ["ok", "budget_exhausted"]
    assert [message["tool_call_id"] for message in tool_messages] == ["call-1", "call-2"]

    pending_tool_calls: set[str] = set()
    for message in render_context_blocks_to_messages(blocks):
        if message.role == "assistant" and message.tool_calls:
            assert not pending_tool_calls
            pending_tool_calls = {tool_call.id for tool_call in message.tool_calls}
        elif message.role == "tool":
            pending_tool_calls.discard(message.tool_call_id or "")
        else:
            assert not pending_tool_calls
    assert not pending_tool_calls


def test_prompt_sanitizer_drops_legacy_unanswered_tool_call_messages(tmp_path) -> None:
    provider = ScriptedProvider(chat=[])
    orchestrator = build_orchestrator(tmp_path, provider)

    sanitized = orchestrator.prompt_builder._sanitize_messages(
        [
            LLMMessage(role="user", content="old turn"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[LLMToolCall(id="call-orphan", name="echo", arguments_json='{"value":"x"}')],
            ),
            LLMMessage(role="assistant", content="final answer"),
        ]
    )

    assert [(message.role, message.content) for message in sanitized] == [
        ("user", "old turn"),
        ("assistant", "final answer"),
    ]


def test_investigation_no_tool_no_critique_returns_final(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[LLMCompletionResult(content='{"raw":"draft"}')],
        finals=[LLMCompletionResult(content="Polished conversational answer.")],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = run_turn(
        orchestrator,
        "answer",
        options=RunOptions(mode="investigate", max_iterations=1, require_initial_plan=False),
    )

    assert result.content == "Polished conversational answer."
    assert result.metadata["mode"] == "investigate"
    assert result.metadata["iterations_used"] == 1
    assert result.metadata["final_response_origin"] == "model"
    assert provider.chat_tools[0]
    assert provider.chat_tools[1] == []
    assert (provider.tool_options[1].metadata or {})["target"] == "investigation_final_response"
    final_payload = json.loads(provider.chat_messages[1][-1].content)
    assert final_payload["original_user_request"] == "answer"
    assert final_payload["candidate_answer"] == '{"raw":"draft"}'
    assert "reasoning" not in json.dumps(result.metadata)


@pytest.mark.parametrize(
    "invalid_final",
    [
        LLMCompletionResult(content='{"answer":"raw"}'),
        LLMCompletionResult(content='```json\n{"answer":"raw"}\n```'),
        LLMCompletionResult(content=""),
        LLMCompletionResult(
            content="",
            tool_calls=[LLMToolCall(id="unexpected", name="echo", arguments_json='{"value":"x"}')],
        ),
        RuntimeError("provider unavailable"),
    ],
)
def test_investigation_conversational_finalization_uses_state_renderer_only_as_fallback(
    tmp_path,
    invalid_final,
) -> None:
    provider = ScriptedProvider(
        chat=[LLMCompletionResult(content="candidate answer")],
        finals=[invalid_final],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = run_turn(
        orchestrator,
        "answer",
        options=RunOptions(mode="investigate", max_iterations=1, require_initial_plan=False),
    )

    assert result.content.startswith("Investigation complete.")
    assert "Established facts:" in result.content
    assert result.metadata["final_response_origin"] == "fallback"
    assert result.metadata["final_response_error_type"]


def test_investigation_json_schema_final_output_uses_last_no_tool_turn(tmp_path) -> None:
    contract_schema = {
        "type": "object",
        "required": ["summary", "confidence"],
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }
    provider = ScriptedProvider(
        chat=[
            LLMCompletionResult(content="plain final draft"),
            LLMCompletionResult(content=json.dumps({"summary": "plain final draft", "confidence": 0.82})),
        ]
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = run_turn(
        orchestrator,
        "answer as schema",
        options=RunOptions.investigate(
            max_iterations=1,
            require_initial_plan=False,
            final_output_mode="json_schema",
            final_output_contract=StructuredOutputContract(
                name="investigation_final",
                schema=contract_schema,
                strict=True,
                instructions=["Keep confidence as a number."],
            ),
            reasoning_effort="high",
        ),
    )

    assert json.loads(result.content) == {"summary": "plain final draft", "confidence": 0.82}
    assert result.metadata["final_output_mode"] == "json_schema"
    assert result.metadata["final_output_contract"] == "investigation_final"
    assert provider.chat_tools[0]
    assert provider.chat_tools[1] == []
    assert provider.tool_options[0].response_format is None
    assert provider.tool_options[1].response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "investigation_final",
            "schema": contract_schema,
            "strict": True,
        },
    }
    assert provider.tool_options[1].reasoning_effort == "high"
    assert provider.chat_messages[1][-1].role == "user"
    assert "candidate_final_answer" in provider.chat_messages[1][-1].content


def test_investigation_json_schema_final_output_is_validated_locally(tmp_path) -> None:
    sensitive_value = "jwt.sensitive-investigation-payload.signature"
    provider = ScriptedProvider(
        chat=[
            LLMCompletionResult(content="plain final draft"),
            LLMCompletionResult(content=json.dumps({"status": "unexpected", "token": sensitive_value})),
        ]
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    with pytest.raises(LLMProviderError) as captured:
        run_turn(
            orchestrator,
            "answer as schema",
            options=RunOptions.investigate(
                max_iterations=1,
                require_initial_plan=False,
                final_output_mode="json_schema",
                final_output_contract=StructuredOutputContract(
                    name="investigation_final",
                    schema={
                        "type": "object",
                        "required": ["status"],
                        "additionalProperties": False,
                        "properties": {"status": {"type": "string", "enum": ["accepted"]}},
                    },
                ),
            ),
        )

    assert "violated the output contract" in captured.value.user_message
    assert sensitive_value not in captured.value.detail


def test_internal_synthesis_recovery_option_recovers_invalid_initial_plan_json(tmp_path) -> None:
    class InvalidInitialPlanProvider(ScriptedProvider):
        def complete_text(self, *, messages, model, temperature, options=None):
            self.text_options.append(options)
            target = (options.metadata or {}).get("target") if options is not None else None
            if target == "investigation_initial_plan":
                return "not-json"
            return super().complete_text(messages=messages, model=model, temperature=temperature, options=options)

    provider = InvalidInitialPlanProvider(chat=[LLMCompletionResult(content="final after internal synthesis failure")])
    orchestrator = build_orchestrator(tmp_path, provider)

    result = run_turn(
        orchestrator,
        "recover from invalid initial plan",
        options=RunOptions.investigate(
            max_iterations=1,
            recover_internal_synthesis_errors=True,
        ),
    )

    assert result.content == "final after internal synthesis failure"
    assert result.metadata["stop_reason"] == "final"
    assert provider.text_options[0].response_format == {"type": "json_object"}


def test_domain_hooks_customize_investigation_prompts_and_guidance(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[LLMCompletionResult(content="domain-guided final")],
        plans=[
            {
                "objective": "investigate with domain hooks",
                "plan": ["use the domain guidance"],
                "facts": [],
                "hypotheses": [],
                "evidence_gaps": [],
                "completed_actions": [],
                "next_actions": [],
                "risk_notes": [],
                "confidence": 0.2,
                "stop_reason": None,
                "metadata": {},
            }
        ],
    )
    hooks = CustomInvestigationHooks()
    orchestrator = build_orchestrator(tmp_path, provider, domain_hooks=hooks)

    result = run_turn(
        orchestrator,
        "investigate with domain hooks",
        options=RunOptions.investigate(max_iterations=1),
    )

    assert result.content == "domain-guided final"
    assert hooks.modes == ["investigate"]
    assert any("Domain investigation guidance" in prompt for prompt in provider.text_prompts)
    first_chat_system_messages = [message.content for message in provider.chat_messages[0] if message.role == "system"]
    assert any("Domain investigation guidance" in content for content in first_chat_system_messages)
    assert sum(content.startswith("Run mode: investigate.") for content in first_chat_system_messages) == 1
    assert sum(content.startswith(INVESTIGATION_STATE_MESSAGE_PREFIX) for content in first_chat_system_messages) == 1


def test_investigation_internal_synthesis_can_use_dedicated_memory_provider(tmp_path) -> None:
    class AgentOnlyProvider(ScriptedProvider):
        def complete_text(self, *, messages, model, temperature, options=None):
            raise AssertionError("main provider should not be used for internal synthesis")

    primary_provider = AgentOnlyProvider(chat=[LLMCompletionResult(content="primary final")])
    memory_provider = ScriptedProvider(
        chat=[],
        plans=[
            {
                "objective": "investigate with memory provider",
                "plan": ["collect the answer"],
                "facts": [],
                "hypotheses": [],
                "evidence_gaps": [],
                "completed_actions": [],
                "next_actions": [],
                "risk_notes": [],
                "confidence": 0.2,
                "stop_reason": None,
                "metadata": {},
            }
        ],
    )
    orchestrator = build_orchestrator(tmp_path, primary_provider, memory_provider=memory_provider)

    result = run_turn(
        orchestrator,
        "investigate with memory provider",
        options=RunOptions.investigate(max_iterations=1),
    )

    assert result.content == "primary final"
    assert primary_provider.chat_messages
    assert not primary_provider.text_options
    assert len(memory_provider.text_options) >= 2
    assert (memory_provider.text_options[0].metadata or {}).get("target") == "investigation_initial_plan"


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

    result = run_turn(
        orchestrator,
        "investigate",
        options=RunOptions(mode="investigate", max_iterations=2, require_initial_plan=False),
    )

    assert "echo returned fact" in result.content
    assert "need second source" in result.content
    assert result.metadata["stop_reason"] == "enough evidence"
    assert result.metadata["investigation_state"]["facts"] == ["echo returned fact"]
    journal = orchestrator.session_manager.get_memory_journal()
    assert [item.kind for item in journal.exchanges] == [
        "tool_exchange",
        "reflection",
        "final_response",
    ]
    assert journal.exchanges[1].confirmed_facts == ["echo returned fact"]
    assert journal.turns[0].turn_summary
    assert journal.session_view is not None
    assert "Test objective" in journal.session_view.content


def test_investigation_reflection_can_resolve_previous_evidence_gap(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[tool_call(value="first pass"), tool_call(value="snapshot artifact")],
        reflections=[
            reflection_payload(
                new_facts=["navigation succeeded"],
                remaining_gaps=["browser_snapshot not yet performed"],
                recommended_next_actions=["run browser_snapshot"],
                should_continue=True,
            ),
            reflection_payload(
                new_facts=[{"summary": "browser_snapshot produced artifact browser-000005"}],
                remaining_gaps=[],
                resolved_gaps=["browser_snapshot not yet performed"],
                recommended_next_actions=[],
                should_continue=False,
                confidence=0.95,
            ),
        ],
        decisions=[
            decision_payload("continue"),
            decision_payload("final", reason_summary="snapshot evidence collected"),
        ],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = run_turn(
        orchestrator,
        "investigate browser target",
        options=RunOptions(mode="investigate", max_iterations=3, require_initial_plan=False),
    )

    assert "browser_snapshot produced artifact browser-000005" in result.content
    assert "browser_snapshot not yet performed" not in result.content
    assert result.metadata["investigation_state"]["evidence_gaps"] == []
    second_iteration_state_messages = [
        message.content
        for message in provider.chat_messages[1]
        if message.role == "system" and message.content.startswith(INVESTIGATION_STATE_MESSAGE_PREFIX)
    ]
    assert len(second_iteration_state_messages) == 1
    second_iteration_state = json.loads(second_iteration_state_messages[0].splitlines()[-1])
    assert second_iteration_state["facts"][0]["summary"] == "navigation succeeded"
    assert second_iteration_state["evidence_gaps"] == ["browser_snapshot not yet performed"]
    assert second_iteration_state["next_actions"] == ["run browser_snapshot"]


def test_investigation_does_not_finalize_when_reflection_requires_continuation(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[tool_call(value="partial")],
        reflections=[
            reflection_payload(
                new_facts=["partial evidence saved"],
                remaining_gaps=["validated finding missing"],
                recommended_next_actions=["save validated finding"],
                should_continue=True,
            )
        ],
        decisions=[decision_payload("final", reason_summary="inconsistent final")],
    )
    orchestrator = build_orchestrator(tmp_path, provider)

    result = run_turn(
        orchestrator,
        "investigate",
        options=RunOptions(mode="investigate", max_iterations=1, require_initial_plan=False),
    )

    assert result.metadata["stop_reason"] == "max_iterations"
    assert "partial evidence saved" in result.content
    assert "validated finding missing" in result.content


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

    result = run_turn(
        orchestrator,
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

    result = run_turn(
        orchestrator,
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

    result = run_turn(
        orchestrator,
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

    result = run_turn(
        orchestrator,
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

    result = run_turn(
        orchestrator,
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

    result = run_turn(
        orchestrator,
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

    result = run_turn(
        orchestrator,
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
    revised_iteration_state_messages = [
        message.content
        for message in provider.chat_messages[1]
        if message.role == "system" and message.content.startswith(INVESTIGATION_STATE_MESSAGE_PREFIX)
    ]
    assert len(revised_iteration_state_messages) == 1
    revised_iteration_state = json.loads(revised_iteration_state_messages[0].splitlines()[-1])
    assert revised_iteration_state["evidence_gaps"] == ["unsupported claim"]
    assert revised_iteration_state["next_actions"] == ["collect more evidence"]


def test_investigation_pending_resume_continues_same_mode(tmp_path) -> None:
    provider = ScriptedProvider(
        chat=[tool_call(name="pending", value="job-1")],
        reflections=[reflection_payload(new_facts=["external result arrived"], should_continue=False)],
        decisions=[decision_payload("final", reason_summary="pending result resolved")],
    )
    orchestrator = build_orchestrator(tmp_path, provider, pending=True)

    pending = run_turn(
        orchestrator,
        "start pending",
        options=RunOptions(mode="investigate", max_iterations=2, require_initial_plan=False),
    )

    assert pending.status == "pending_tool_result"
    assert pending.metadata["job_id"] == "job-1"
    assert pending.metadata["mode"] == "investigate"

    completed = resume_turn(orchestrator, pending_id=pending.pending_id or "", tool_content="done")

    assert completed.status == "completed"
    assert "external result arrived" in completed.content
    assert completed.metadata["mode"] == "investigate"
    assert [block.kind for block in orchestrator.session_manager.get_context_blocks()] == [
        "tool_exchange",
        "conversation_turn",
    ]
    assert orchestrator.session_manager.get_state()["meta"].get(AgentOrchestrator.PENDING_TURN_META_KEY) is None
