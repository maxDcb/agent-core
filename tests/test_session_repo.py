from __future__ import annotations

import json

from agent_core.session_repo import SessionRepository
from agent_core.types import SessionState, build_empty_session_state


class InMemorySessionStore:
    storage_backend = "memory"

    def __init__(self) -> None:
        self.states: dict[str, SessionState] = {}
        self.traces: dict[str, dict[str, dict[str, object]]] = {}

    def load(self, session_id: str) -> SessionState:
        return self.states.get(session_id, build_empty_session_state(session_id=session_id, storage_backend=self.storage_backend))

    def save(self, session_id: str, state: SessionState) -> None:
        self.states[session_id] = state

    def save_run_trace(self, session_id: str, trace_payload: dict[str, object]) -> None:
        self.traces.setdefault(session_id, {})[str(trace_payload["run_id"])] = trace_payload

    def load_run_trace(self, session_id: str, run_id: str) -> dict[str, object] | None:
        return self.traces.get(session_id, {}).get(run_id)

    def list_run_traces(self, session_id: str) -> list[dict[str, object]]:
        return list(self.traces.get(session_id, {}).values())

    def list_session_ids(self) -> list[str]:
        return sorted(self.states)

    def describe(self) -> dict[str, object]:
        return {"backend": self.storage_backend}


def test_json_file_session_repository_lists_and_describes_sessions(tmp_path) -> None:
    repo = SessionRepository(tmp_path / "session.json")
    state = repo.load("default")
    state["tool_history"].append({"tool_name": "search_code", "status": "ok"})

    repo.save("default", state)

    assert repo.storage_backend == "json"
    assert repo.list_session_ids() == ["default"]
    assert repo.describe()["backend"] == "json"
    assert (tmp_path / "default" / "default.json").exists()


def test_json_file_session_repository_quarantines_corrupt_state(tmp_path) -> None:
    session_file = tmp_path / "default" / "default.json"
    session_file.parent.mkdir(parents=True)
    session_file.write_text("{not-json", encoding="utf-8")

    state = SessionRepository(tmp_path / "session.json").load("default")

    assert state["context_blocks"] == []
    assert not session_file.exists()
    assert len(list(session_file.parent.glob("default.json.corrupt.*"))) == 1


def test_json_file_session_repository_normalizes_malformed_state(tmp_path) -> None:
    session_file = tmp_path / "default" / "default.json"
    session_file.parent.mkdir(parents=True)
    session_file.write_text(json.dumps({"tool_history": None, "context_blocks": "bad", "meta": "bad"}), encoding="utf-8")

    state = SessionRepository(tmp_path / "session.json").load("default")

    assert state["tool_history"] == []
    assert state["context_blocks"] == []
    assert state["meta"]["session_id"] == "default"


def test_session_repository_can_delegate_to_custom_store() -> None:
    store = InMemorySessionStore()
    repo = SessionRepository(store=store)
    state = repo.load("memory-session")
    state["tool_history"].append({"tool_name": "noop", "status": "ok"})

    repo.save("memory-session", state)
    repo.save_run_trace("memory-session", {"run_id": "run-1", "session_id": "memory-session"})

    assert repo.storage_backend == "memory"
    assert repo.list_session_ids() == ["memory-session"]
    assert repo.load("memory-session")["tool_history"][0]["tool_name"] == "noop"
    assert repo.load_run_trace("memory-session", "run-1")["run_id"] == "run-1"
    assert repo.describe() == {"backend": "memory"}
