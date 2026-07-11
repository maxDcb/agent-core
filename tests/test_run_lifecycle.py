from __future__ import annotations

import pytest

from agent_core.run_context import RunContext
from agent_core.run_models import AgentRunAttempt, AgentRunResult, AgentRunState, RunCheckpoint


def _state(*, status: str = "created") -> AgentRunState:
    state = AgentRunState(
        run_id="run-lifecycle",
        strategy="structured",
        spec_id="task-lifecycle",
        context=RunContext(namespace_id="assessment", run_id="run-lifecycle"),
    )
    if status != "created":
        state.transition("running")
        if status != "running":
            state.transition(status)  # type: ignore[arg-type]
    return state


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("created", "completed"),
        ("created", "blocked"),
        ("interrupted", "completed"),
        ("completed", "running"),
        ("failed", "running"),
        ("cancelled", "running"),
    ],
)
def test_run_state_rejects_invalid_lifecycle_transitions(source: str, target: str) -> None:
    state = _state(status=source)

    with pytest.raises(ValueError, match=f"{source} -> {target}"):
        state.transition(target)  # type: ignore[arg-type]

    assert state.status == source


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("created", "running"),
        ("running", "pending"),
        ("running", "interrupted"),
        ("running", "completed"),
        ("running", "failed"),
        ("running", "blocked"),
        ("pending", "running"),
        ("interrupted", "running"),
        ("blocked", "running"),
    ],
)
def test_run_state_accepts_declared_lifecycle_transitions(source: str, target: str) -> None:
    state = _state(status=source)

    state.transition(target)  # type: ignore[arg-type]

    assert state.status == target


def test_run_state_round_trip_preserves_checkpoint_attempts_and_terminal_result() -> None:
    state = _state(status="running")
    first_attempt = AgentRunAttempt(attempt_id="attempt-1")
    first_attempt.finish("interrupted", failure_reason="worker lost")
    second_attempt = AgentRunAttempt(attempt_id="attempt-2", resumed_from_sequence=7)
    second_attempt.finish("completed")
    state.attempts = [first_attempt, second_attempt]
    state.checkpoint = RunCheckpoint(
        kind="structured_task",
        sequence=9,
        payload={"phase": "result", "nested": {"kept": True}},
    )
    state.result = AgentRunResult(
        run_id=state.run_id,
        status="completed",
        raw_content="done",
        metadata={"spec_id": state.spec_id},
    )
    state.transition("completed")

    restored = AgentRunState.from_dict(state.to_dict())

    assert restored is not None
    assert restored.to_dict() == state.to_dict()
    assert [attempt.status for attempt in restored.attempts] == ["interrupted", "completed"]
    assert restored.attempts[1].resumed_from_sequence == 7
    assert restored.checkpoint is not None
    assert restored.checkpoint.payload["nested"] == {"kept": True}


def test_malformed_attempts_do_not_destroy_an_otherwise_valid_run_state() -> None:
    payload = _state().to_dict()
    payload["attempts"] = [None, {}, {"attempt_id": "kept", "status": "unknown"}]

    restored = AgentRunState.from_dict(payload)

    assert restored is not None
    assert len(restored.attempts) == 1
    assert restored.attempts[0].attempt_id == "kept"
    assert restored.attempts[0].status == "interrupted"
