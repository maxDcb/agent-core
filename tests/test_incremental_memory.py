from __future__ import annotations

import json

import pytest

from agent_core.context_assembler import ContextAssembler
from agent_core.llm.base import LLMCompletionResult, LLMMessage
from agent_core.memory.committer import (
    DEFAULT_TURN_MEMORY_SYNTHESIS_PROMPT,
    MemorySynthesisResult,
    TurnMemoryCommitter,
    TurnMemorySynthesisInput,
)
from agent_core.memory.context_block import estimate_token_count
from agent_core.memory.derivation import derive_final_response_memory
from agent_core.memory.journal import (
    ExchangeMemory,
    IncrementalMemoryJournal,
    SessionView,
    TurnMemory,
)
from agent_core.memory.thread_state import (
    create_conversation_turn_block,
    create_tool_exchange_block,
)
from agent_core.orchestrator import AgentOrchestrator
from agent_core.policy_engine import PolicyEngine
from agent_core.session_manager import SessionManager
from agent_core.session_repo import SessionRepository
from agent_core.settings import CoreSettings
from agent_core.tool_registry import ToolRegistry
from tests.run_helpers import run_turn, turn_memory_payload


def _turn_memory(
    turn_index: int,
    *,
    handoff: str,
    summary: str | None = None,
    degraded: bool = False,
) -> TurnMemory:
    return TurnMemory(
        memory_id=f"turn-{turn_index:04d}-memory",
        thread_id="thread",
        turn_index=turn_index,
        user_intent=f"request {turn_index}",
        assistant_outcome=f"outcome {turn_index}",
        turn_summary=summary or f"turn {turn_index} completed",
        handoff_after_turn=handoff,
        degraded=degraded,
    )


def test_session_view_is_the_latest_rebuildable_handoff() -> None:
    journal = IncrementalMemoryJournal(thread_id="thread")
    journal.commit_turn(
        _turn_memory(0, handoff="Old operational handoff"),
        max_handoff_chars=100,
    )
    journal.commit_turn(
        _turn_memory(1, handoff="Replacement operational handoff"),
        max_handoff_chars=100,
    )

    assert journal.session_view is not None
    assert journal.session_view.generation == 2
    assert journal.session_view.content == "Replacement operational handoff"
    assert journal.session_view.through_turn_index == 1

    payload = journal.to_dict()
    payload["session_view"] = SessionView.empty(thread_id="thread").to_dict()
    restored = IncrementalMemoryJournal.from_any(payload, thread_id="thread")

    assert restored.session_view is not None
    assert restored.session_view.generation == journal.session_view.generation
    assert restored.session_view.content == journal.session_view.content


def test_default_memory_prompt_requires_conservative_retention() -> None:
    prompt = " ".join(DEFAULT_TURN_MEMORY_SYNTHESIS_PROMPT.split())

    assert "current turn as a delta" in prompt
    assert "do not drop a previous fact merely because the current turn does not repeat it" in prompt
    assert "Silence in the current turn is not evidence that a fact became irrelevant" in prompt
    assert "rules an approach in or out" in prompt
    assert "preserve the conflict and its provenance" in prompt
    assert "fresh model given only the next handoff" in prompt


def test_journal_rejects_an_oversized_turn_summary() -> None:
    journal = IncrementalMemoryJournal(thread_id="thread")

    with pytest.raises(ValueError, match="Turn summary exceeds"):
        journal.commit_turn(
            _turn_memory(0, handoff="Continue", summary="s" * 101),
            max_handoff_chars=100,
            max_turn_summary_chars=100,
        )

    assert journal.turns == []


def test_legacy_memory_schema_is_not_loaded() -> None:
    restored = IncrementalMemoryJournal.from_any(
        {
            "schema_version": "1",
            "thread_id": "thread",
            "exchanges": [{"memory_id": "legacy-exchange"}],
            "turns": [{"memory_id": "legacy-turn"}],
        },
        thread_id="thread",
    )

    assert restored.exchanges == []
    assert restored.turns == []
    assert restored.session_view is not None
    assert restored.session_view.generation == 0
    assert restored.session_view.content == ""


def test_session_manager_recovers_uncommitted_raw_turn_without_llm(tmp_path) -> None:
    manager = SessionManager(SessionRepository(tmp_path / "session.json"))
    assistant_tool_message = LLMMessage(role="assistant", content="Inspecting", tool_calls=[])
    tool_message = LLMMessage(role="tool", content="result", tool_call_id="call-1")
    manager.append_context_block(
        create_tool_exchange_block(
            turn_index=0,
            exchange_index=0,
            assistant_message=assistant_tool_message.to_history_dict(),
            tool_messages=[tool_message.to_history_dict()],
        )
    )
    manager.append_context_block(
        create_conversation_turn_block(
            turn_index=0,
            user_message=LLMMessage(role="user", content="inspect").to_history_dict(),
            assistant_message=LLMMessage(role="assistant", content="inspection complete").to_history_dict(),
        )
    )

    changed = manager.reconcile_memory(
        max_handoff_chars=6_000,
        max_turn_summary_chars=4_000,
    )
    journal = manager.get_memory_journal()

    assert changed is True
    assert [item.kind for item in journal.exchanges] == ["tool_exchange", "final_response"]
    assert len(journal.turns) == 1
    assert journal.turns[0].origin == "recovery"
    assert journal.session_view is not None and journal.session_view.generation == 1

    reloaded = SessionManager(SessionRepository(tmp_path / "session.json"))
    assert reloaded.get_memory_journal().to_dict() == journal.to_dict()


def test_memory_transaction_rolls_back_in_memory_state_when_storage_fails(tmp_path, monkeypatch) -> None:
    manager = SessionManager(SessionRepository(tmp_path / "session.json"))
    before = json.loads(json.dumps(manager.get_state()))
    memory = derive_final_response_memory(
        thread_id="default",
        turn_index=0,
        exchange_index=0,
        assistant_content="answer",
        source_block_id="turn-0000-conversation",
    )

    def fail_save(session_id: str, state: object) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(manager.repo, "save", fail_save)
    with pytest.raises(OSError, match="disk unavailable"):
        manager.append_exchange_memory(memory)

    assert manager.get_state() == before


class _IncrementalProvider:
    def __init__(
        self,
        *,
        fail_memory: bool = False,
        oversized_handoff: bool = False,
        serialized_handoff: bool = False,
    ) -> None:
        self.fail_memory = fail_memory
        self.oversized_handoff = oversized_handoff
        self.serialized_handoff = serialized_handoff
        self.memory_payloads: list[dict[str, object]] = []
        self.chat_calls = 0

    def complete_with_tools(self, **kwargs) -> LLMCompletionResult:
        self.chat_calls += 1
        return LLMCompletionResult(content=f"answer {self.chat_calls}")

    def complete_text(self, *, messages, model, temperature, options=None):
        payload = json.loads(messages[1].content)
        self.memory_payloads.append(payload)
        if self.fail_memory:
            raise RuntimeError("memory provider unavailable")
        turn_index = int(payload["turn_index"])
        response = turn_memory_payload(
            objective="Stable objective",
            user_intent=str(payload["user_request"]),
            assistant_outcome=str(payload["assistant_outcome"]),
            confirmed_facts=[f"fact {turn_index}"],
        )
        if self.oversized_handoff:
            response["next_handoff"] = "x" * 6_001
        if self.serialized_handoff:
            response["next_handoff"] = json.dumps(
                {"current_objective": "This must not become visible prompt memory"}
            )
        return json.dumps(response)


def _orchestrator(
    tmp_path,
    provider: _IncrementalProvider,
    *,
    memory_max_turn_input_chars: int = 64_000,
) -> AgentOrchestrator:
    settings = CoreSettings(
        model="fake",
        memory_model="fake",
        session_file=tmp_path / "session.json",
        base_system_prompt="system",
        turn_memory_synthesis_prompt="memory",
        memory_max_turn_input_chars=memory_max_turn_input_chars,
        max_active_context_tokens=100_000,
    )
    return AgentOrchestrator(
        settings=settings,
        provider=provider,
        registry=ToolRegistry(),
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )


def test_orchestrator_synthesizes_only_the_current_turn_increment(tmp_path) -> None:
    provider = _IncrementalProvider()
    orchestrator = _orchestrator(tmp_path, provider)

    run_turn(orchestrator, "first request")
    run_turn(orchestrator, "second request")

    assert len(provider.memory_payloads) == 2
    assert provider.memory_payloads[0]["turn_index"] == 0
    assert provider.memory_payloads[1]["turn_index"] == 1
    assert "recent_history" not in provider.memory_payloads[1]
    assert "context_blocks" not in provider.memory_payloads[1]
    assert all(item["memory_id"].startswith("turn-0001-") for item in provider.memory_payloads[1]["exchange_memories"])
    assert "fact 0" in str(provider.memory_payloads[1]["previous_handoff"])
    journal = orchestrator.session_manager.get_memory_journal()
    assert [turn.origin for turn in journal.turns] == ["model", "model"]
    assert journal.session_view is not None
    assert journal.session_view.generation == 2
    assert "fact 1" in journal.session_view.content
    assert "fact 0" not in journal.session_view.content


def test_memory_synthesis_failure_commits_fallback_and_does_not_retry_old_turn(tmp_path) -> None:
    provider = _IncrementalProvider(fail_memory=True)
    orchestrator = _orchestrator(tmp_path, provider)

    first = run_turn(orchestrator, "first request")
    second = run_turn(orchestrator, "second request")

    assert first.status == second.status == "completed"
    assert len(provider.memory_payloads) == 2
    journal = orchestrator.session_manager.get_memory_journal()
    assert [turn.origin for turn in journal.turns] == ["fallback", "fallback"]
    assert journal.session_view is not None and journal.session_view.generation == 2
    assert journal.session_view.degraded is True


def test_oversized_turn_memory_payload_falls_back_before_provider_call(tmp_path) -> None:
    provider = _IncrementalProvider()
    orchestrator = _orchestrator(
        tmp_path,
        provider,
        memory_max_turn_input_chars=1_000,
    )

    result = run_turn(orchestrator, "large request " * 1_000)

    assert result.status == "completed"
    assert provider.memory_payloads == []
    journal = orchestrator.session_manager.get_memory_journal()
    assert journal.turns[0].origin == "fallback"


def test_turn_memory_projects_long_controller_state_and_exchanges_before_provider_call() -> None:
    class CapturingSynthesizer:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def synthesize(self, *, request):
            self.payload = request.payload
            return MemorySynthesisResult(
                turn_summary="Completed a long investigation turn.",
                next_handoff="Continue with the retained HTTP evidence and next hypothesis.",
            )

    synthesizer = CapturingSynthesizer()
    committer = TurnMemoryCommitter(
        synthesizer=synthesizer,  # type: ignore[arg-type]
        max_input_chars=64_000,
    )
    exchanges = [
        ExchangeMemory(
            memory_id=f"turn-0000-exchange-{index:02d}",
            thread_id="thread",
            turn_index=0,
            exchange_index=index,
            kind="reflection" if index % 2 else "tool_exchange",
            summary=(
                f"HTTP 200 evidence for payload {index}: "
                + ("response-body " * 180)
            ),
            confirmed_facts=[f"Constraint {index}: " + ("fact " * 180)],
            open_hypotheses=[f"Next payload hypothesis {index}: " + ("candidate " * 120)],
            completed_actions=[f"Tested payload {index}: " + ("tested " * 100)],
            next_actions=[f"Try follow-up {index}: " + ("next " * 100)],
            relevant_artifacts=[f"http-response-{index:03d}"],
        )
        for index in range(61)
    ]
    controller_state = {
        "objective": "Retrieve /flag.txt through the confirmed file-read endpoint.",
        "facts": [f"Established constraint {index}: " + ("detail " * 180) for index in range(40)],
        "hypotheses": [
            {"statement": f"Payload family {index}: " + ("candidate " * 160), "status": "open"}
            for index in range(40)
        ],
        "completed_actions": [f"Already tested {index}: " + ("payload " * 180) for index in range(40)],
        "next_actions": ["Preserve and test the strongest remaining payload family."],
    }
    raw_variable_chars = len(
        json.dumps(
            {
                "controller_state": controller_state,
                "exchange_memories": [exchange.to_dict() for exchange in exchanges],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )

    result = committer.synthesize(
        TurnMemorySynthesisInput(
            thread_id="thread",
            turn_index=0,
            user_intent="Continue the authorized file-read investigation.",
            assistant_outcome="The turn tested many payloads and retained live HTTP evidence.",
            exchange_memories=exchanges,
            source_block_ids=["turn-0000-conversation"],
            previous_handoff="Keep the established parser and path-normalization constraints.",
            runtime_context={"source_code_locations": ["/data/workspace/nextpath/app"]},
            controller_state=controller_state,
            domain_payload={"domain": "authorized_pentest"},
            domain_guidance="Keep evidence, tested payloads, constraints, and next hypotheses.",
        )
    )

    assert raw_variable_chars > 64_000
    assert result.origin == "model"
    assert synthesizer.payload is not None
    serialized_payload = json.dumps(
        synthesizer.payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert len(serialized_payload) <= 64_000
    assert len(synthesizer.payload["exchange_memories"]) == 61  # type: ignore[arg-type]
    assert "Retrieve /flag.txt" in serialized_payload
    assert "HTTP 200 evidence" in serialized_payload
    assert "http-response-060" in serialized_payload
    assert "Next payload hypothesis 60" in serialized_payload


def test_oversized_model_handoff_is_rejected_and_falls_back(tmp_path) -> None:
    provider = _IncrementalProvider(oversized_handoff=True)
    orchestrator = _orchestrator(tmp_path, provider)

    result = run_turn(orchestrator, "bounded handoff request")

    assert result.status == "completed"
    assert len(provider.memory_payloads) == 1
    journal = orchestrator.session_manager.get_memory_journal()
    assert journal.turns[0].origin == "fallback"
    assert journal.turns[0].degraded is True
    assert len(journal.session_view.content) <= 6_000


def test_serialized_json_handoff_is_rejected_and_falls_back(tmp_path) -> None:
    provider = _IncrementalProvider(serialized_handoff=True)
    orchestrator = _orchestrator(tmp_path, provider)

    result = run_turn(orchestrator, "reject raw JSON memory")

    assert result.status == "completed"
    journal = orchestrator.session_manager.get_memory_journal()
    assert journal.turns[0].origin == "fallback"
    assert journal.session_view is not None
    assert "current_objective" not in journal.session_view.content


def test_context_assembly_reserves_handoff_before_selecting_history(tmp_path) -> None:
    manager = SessionManager(SessionRepository(tmp_path / "session.json"))
    for turn_index in range(3):
        manager.append_context_block(
            create_conversation_turn_block(
                turn_index=turn_index,
                user_message=LLMMessage(role="user", content=f"request {turn_index} " + "u" * 300).to_history_dict(),
                assistant_message=LLMMessage(
                    role="assistant",
                    content=f"outcome {turn_index} " + "a" * 300,
                ).to_history_dict(),
            )
        )
    manager.commit_turn_memory(
        TurnMemory(
            memory_id="turn-0002-memory",
            thread_id="default",
            turn_index=2,
            user_intent="request 2",
            assistant_outcome="outcome 2",
            turn_summary="Turn 2 completed.",
            handoff_after_turn="Current objective:\n" + "h" * 600,
        ),
        max_handoff_chars=1_000,
        max_turn_summary_chars=1_000,
    )
    settings = CoreSettings(
        session_file=tmp_path / "session.json",
        base_system_prompt="system",
        max_active_context_tokens=500,
        memory_max_handoff_chars=1_000,
    )

    assembly = ContextAssembler(settings=settings, session_manager=manager).assemble(
        base_messages=[LLMMessage(role="system", content="s" * 400)],
        user_input="continue",
    )

    fixed_tokens = (
        estimate_token_count({"role": "system", "content": "s" * 400})
        + manager.get_thread_state().session_view.as_context_block().token_estimate
        + estimate_token_count({"role": "user", "content": "continue"})
    )
    selected_tokens = sum(block.token_estimate for block in assembly.selected_blocks)
    assert assembly.overflow_blocks
    assert selected_tokens <= max(1, 500 - fixed_tokens) or {
        block.metadata.get("turn_index") for block in assembly.selected_blocks
    } == {2}
