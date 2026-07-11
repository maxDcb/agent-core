from __future__ import annotations

import pytest

from agent_core.conversation import ConversationAgent
from agent_core.run_context import RunContext
from agent_core.run_store import JsonFileRunStore
from agent_core.types import AgentTurnResult


class ScriptedConversationOrchestrator:
    def __init__(self) -> None:
        self.run_calls = 0
        self.resume_calls = 0

    def run_turn_result(self, **kwargs) -> AgentTurnResult:
        _ = kwargs
        self.run_calls += 1
        return AgentTurnResult(
            status="pending_tool_result",
            content="waiting",
            pending_id="pending-1",
            tool_name="delayed",
            tool_arguments={"value": "safe"},
            metadata={"iterations_used": 1, "tool_calls_used": 1},
        )

    def resume_turn(self, **kwargs) -> AgentTurnResult:
        self.resume_calls += 1
        assert kwargs["pending_id"] == "pending-1"
        assert kwargs["tool_content"] == "tool result"
        return AgentTurnResult(
            status="completed",
            content="finished",
            metadata={"iterations_used": 2, "tool_calls_used": 1},
        )


def _agent(tmp_path):
    orchestrator = ScriptedConversationOrchestrator()
    store = JsonFileRunStore(tmp_path / "runs")
    agent = ConversationAgent(orchestrator=orchestrator, run_store=store)  # type: ignore[arg-type]
    return agent, orchestrator, store


def test_conversation_pending_resume_records_complete_attempt_lifecycle(tmp_path) -> None:
    agent, orchestrator, store = _agent(tmp_path)
    context = RunContext(namespace_id="assessment", parent_id="job-1")

    pending = agent.execute_turn(
        thread_id="thread-1",
        context=context,
        user_input="start",
        run_id="run-conversation",
    )
    completed = agent.resume(
        namespace_id="assessment",
        run_id="run-conversation",
        pending_id="pending-1",
        tool_content="tool result",
    )

    assert pending.status == "pending"
    assert completed.status == "completed"
    assert completed.raw_content == "finished"
    state = store.load(namespace_id="assessment", run_id="run-conversation")
    assert state is not None
    assert state.status == "completed"
    assert state.checkpoint is None
    assert [attempt.status for attempt in state.attempts] == ["pending", "completed"]
    assert state.attempts[1].resumed_from_sequence == 1
    assert orchestrator.run_calls == 1
    assert orchestrator.resume_calls == 1


def test_conversation_wrong_pending_id_is_rejected_without_state_change(tmp_path) -> None:
    agent, orchestrator, store = _agent(tmp_path)
    agent.execute_turn(
        thread_id="thread-1",
        context=RunContext(namespace_id="assessment"),
        user_input="start",
        run_id="run-conversation",
    )
    before = store.load(namespace_id="assessment", run_id="run-conversation")
    assert before is not None

    with pytest.raises(ValueError, match="Pending id does not match"):
        agent.resume(
            namespace_id="assessment",
            run_id="run-conversation",
            pending_id="wrong",
            tool_content="tool result",
        )

    after = store.load(namespace_id="assessment", run_id="run-conversation")
    assert after is not None
    assert after.to_dict() == before.to_dict()
    assert orchestrator.resume_calls == 0


def test_conversation_terminal_execute_and_resume_are_idempotent(tmp_path) -> None:
    agent, orchestrator, store = _agent(tmp_path)
    context = RunContext(namespace_id="assessment")
    pending = agent.execute_turn(
        thread_id="thread-1",
        context=context,
        user_input="start",
        run_id="run-conversation",
    )
    repeated_pending = agent.execute_turn(
        thread_id="thread-1",
        context=context,
        user_input="start",
        run_id="run-conversation",
    )
    completed = agent.resume(
        namespace_id="assessment",
        run_id="run-conversation",
        pending_id="pending-1",
        tool_content="tool result",
    )
    repeated_completed = agent.resume(
        namespace_id="assessment",
        run_id="run-conversation",
        pending_id="pending-1",
        tool_content="tool result",
    )

    assert repeated_pending.to_dict() == pending.to_dict()
    assert repeated_completed.to_dict() == completed.to_dict()
    assert orchestrator.run_calls == 1
    assert orchestrator.resume_calls == 1
    state = store.load(namespace_id="assessment", run_id="run-conversation")
    assert state is not None
    assert len(state.attempts) == 2
