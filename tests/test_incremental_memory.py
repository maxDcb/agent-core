from __future__ import annotations

import json

import pytest

from agent_core.llm.base import LLMCompletionResult, LLMMessage
from agent_core.memory.derivation import derive_final_response_memory
from agent_core.memory.journal import (
    ActiveTask,
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
    facts: list[str] | None = None,
    superseded_facts: list[str] | None = None,
    open_questions: list[str] | None = None,
    resolved_questions: list[str] | None = None,
    next_actions: list[str] | None = None,
    completed_actions: list[str] | None = None,
) -> TurnMemory:
    return TurnMemory(
        memory_id=f"turn-{turn_index:04d}-memory",
        thread_id="thread",
        turn_index=turn_index,
        user_intent=f"request {turn_index}",
        assistant_outcome=f"outcome {turn_index}",
        active_task=ActiveTask(
            objective="Stable objective",
            next_action=next_actions[0] if next_actions else None,
        ),
        confirmed_facts=facts or [],
        superseded_facts=superseded_facts or [],
        open_questions=open_questions or [],
        resolved_questions=resolved_questions or [],
        next_actions=next_actions or [],
        completed_actions=completed_actions or [],
    )


def test_session_view_is_a_rebuildable_bounded_projection() -> None:
    journal = IncrementalMemoryJournal(thread_id="thread")
    journal.commit_turn(
        _turn_memory(
            0,
            facts=["old fact"],
            open_questions=["which route?"],
            next_actions=["inspect route"],
        ),
        max_session_items=10,
        max_recent_outcomes=2,
    )
    journal.commit_turn(
        _turn_memory(
            1,
            facts=["new fact"],
            superseded_facts=["old fact"],
            resolved_questions=["which route?"],
            completed_actions=["inspect route"],
        ),
        max_session_items=10,
        max_recent_outcomes=2,
    )

    assert journal.session_view is not None
    assert journal.session_view.generation == 2
    assert journal.session_view.confirmed_facts == ["new fact"]
    assert journal.session_view.open_questions == []
    assert journal.session_view.next_actions == []

    payload = journal.to_dict()
    payload["session_view"] = SessionView.empty(thread_id="thread").to_dict()
    restored = IncrementalMemoryJournal.from_any(payload, thread_id="thread")

    assert restored.session_view is not None
    assert restored.session_view.generation == journal.session_view.generation
    assert restored.session_view.confirmed_facts == journal.session_view.confirmed_facts
    assert restored.session_view.open_questions == journal.session_view.open_questions


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
        max_session_items=20,
        max_recent_outcomes=5,
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
    def __init__(self, *, fail_memory: bool = False) -> None:
        self.fail_memory = fail_memory
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
        response["turn_index"] = turn_index
        response["memory_id"] = f"turn-{turn_index:04d}-memory"
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
    journal = orchestrator.session_manager.get_memory_journal()
    assert [turn.origin for turn in journal.turns] == ["model", "model"]
    assert journal.session_view is not None
    assert journal.session_view.generation == 2
    assert journal.session_view.confirmed_facts == ["fact 0", "fact 1"]


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
