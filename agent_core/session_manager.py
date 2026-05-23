from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock, RLock
from typing import Any

from agent_core.logging_utils import get_logger
from agent_core.memory.context_block import ContextBlock
from agent_core.memory.history_compactor import CompactionPolicy, HistoryCompactor
from agent_core.memory.session_summary import SessionSummary
from agent_core.memory.task_state import TaskState
from agent_core.run_trace import RunTrace
from agent_core.session_repo import SessionRepository
from agent_core.memory.thread_state import ThreadState
from agent_core.types import SESSION_SCHEMA_VERSION, SessionState, build_empty_session_state, utc_now_iso

logger = get_logger(__name__)


class SessionManager:
    """Own the mutable session state between orchestration and storage.

    The manager exposes a small write API around the canonical session payload:
    context blocks, synthesized memory objects, domain state, and metadata.
    It deliberately sits between the orchestrator and repository so storage
    normalization stays outside the runtime flow.
    """

    def __init__(self, repo: SessionRepository, *, default_session_id: str = "default") -> None:
        self.repo = repo
        self._session_id_var: ContextVar[str] = ContextVar("session_id", default=default_session_id)
        self._state_var: ContextVar[SessionState | None] = ContextVar("session_state", default=None)
        self._scope_depth_var: ContextVar[int] = ContextVar("session_scope_depth", default=0)
        self._locks_guard = Lock()
        self._session_locks: dict[str, RLock] = {}
        self._active_session_id = default_session_id
        self.activate_session(default_session_id)

    @property
    def session_id(self) -> str:
        if self._scope_depth_var.get() > 0:
            return self._session_id_var.get()
        return self._active_session_id

    @property
    def state(self) -> SessionState:
        session_id = self.session_id
        state = self._state_var.get()
        if state is None or self._session_id_var.get() != session_id:
            self._session_id_var.set(session_id)
            state = self.repo.load(session_id)
            self._state_var.set(state)
        return state

    @state.setter
    def state(self, value: SessionState) -> None:
        self._state_var.set(value)

    def activate_session(self, session_id: str) -> None:
        with self._locks_guard:
            self._active_session_id = session_id

        if session_id == self._session_id_var.get() and self._state_var.get() is not None:
            logger.trace("Session already active", extra={"session_id": session_id})
            return

        # Session switching is explicit so the storage layer can later move from JSON files to SQLite cleanly.
        self._session_id_var.set(session_id)
        self.state = self.repo.load(session_id)
        logger.info(
            "Activated session",
            extra={"session_id": session_id, "context_block_count": len(self.state.get("context_blocks", []))},
        )

    @contextmanager
    def session_scope(self, session_id: str) -> Iterator[None]:
        """Bind the active session to the current execution context.

        A shared SessionManager can be used by concurrent requests. Each
        request receives its own context-local state, while same-session turns
        are serialized by a per-session lock. Nested scopes for the same
        session reuse the already-loaded state so API handlers can wrap a full
        request and the orchestrator can still protect direct callers.
        """

        depth = self._scope_depth_var.get()
        if depth > 0 and self.session_id == session_id:
            depth_token = self._scope_depth_var.set(depth + 1)
            try:
                yield
            finally:
                self._scope_depth_var.reset(depth_token)
            return

        previous_context_session_id = self._session_id_var.get()
        previous_context_state = self._state_var.get()
        lock = self._lock_for_session(session_id)
        lock.acquire()
        session_token = self._session_id_var.set(session_id)
        state_token = self._state_var.set(self.repo.load(session_id))
        depth_token = self._scope_depth_var.set(1)
        try:
            logger.trace("Entered session scope", extra={"session_id": session_id})
            yield
        finally:
            scoped_state = self._state_var.get()
            self._scope_depth_var.reset(depth_token)
            self._state_var.reset(state_token)
            self._session_id_var.reset(session_token)
            if previous_context_session_id == session_id and previous_context_state is not None and scoped_state is not None:
                self._session_id_var.set(session_id)
                self._state_var.set(scoped_state)
            lock.release()
            logger.trace("Exited session scope", extra={"session_id": session_id})

    def reset_session(self, session_id: str) -> None:
        with self.session_scope(session_id):
            self.reset()

    def load_run_trace_for_session(self, session_id: str, run_id: str) -> dict[str, object] | None:
        with self.session_scope(session_id):
            return self.load_run_trace(run_id)

    def list_run_traces_for_session(self, session_id: str) -> list[dict[str, object]]:
        with self.session_scope(session_id):
            return self.list_run_traces()

    def get_state(self) -> SessionState:
        logger.trace("Returning current session state")
        return self.state

    def set_meta_value(self, key: str, value: Any) -> None:
        meta = self.state.setdefault("meta", {})
        if not isinstance(meta, dict):
            self.state["meta"] = {}
            meta = self.state["meta"]
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
        logger.debug("Persisting session meta value", extra={"key": key, "has_value": value is not None})
        self._save()

    def append_tool_history(self, item: dict[str, Any]) -> None:
        self.state.setdefault("tool_history", []).append(item)
        logger.debug("Persisting tool history item", extra={"tool_name": item.get("tool_name")})
        self._save()

    def get_domain_state(self, namespace: str) -> dict[str, Any] | None:
        logger.trace("Returning domain state namespace", extra={"namespace": namespace})
        domain_state = self.state.setdefault("domain_state", {})
        if not isinstance(domain_state, dict):
            self.state["domain_state"] = {}
            domain_state = self.state["domain_state"]
        payload = domain_state.get(namespace)
        return payload if isinstance(payload, dict) else None

    def set_domain_state(self, namespace: str, payload: dict[str, Any]) -> None:
        domain_state = self.state.setdefault("domain_state", {})
        if not isinstance(domain_state, dict):
            self.state["domain_state"] = {}
            domain_state = self.state["domain_state"]
        domain_state[namespace] = payload
        logger.debug("Persisting domain state namespace", extra={"namespace": namespace})
        self._save()

    def get_context_blocks(self) -> list[ContextBlock]:
        return self._normalize_context_blocks(self.state.get("context_blocks", []))

    def set_context_blocks(self, blocks: list[ContextBlock | dict[str, Any]]) -> None:
        # Replacing the canonical block list invalidates the previous compaction
        # split, so active/overflow pointers are rebuilt on the next compaction.
        normalized_blocks = self._normalize_context_blocks(blocks)
        self._store_context_blocks(normalized_blocks)
        self.state["active_block_ids"] = []
        self.state["overflow_block_ids"] = []

        logger.debug("Persisting context blocks", extra={"block_count": len(normalized_blocks)})
        self._save()

    def append_context_block(self, block: ContextBlock | dict[str, Any]) -> ContextBlock:
        normalized = self._coerce_context_block(block)
        if normalized is None:
            raise ValueError("Invalid context block payload")

        blocks = self.get_context_blocks()
        blocks.append(normalized)
        self.set_context_blocks(list(blocks))
        return normalized

    def get_next_turn_index(self) -> int:
        turn_indices = [
            turn_index
            for block in self.get_context_blocks()
            if isinstance((turn_index := block.metadata.get("turn_index")), int)
        ]
        return (max(turn_indices) + 1) if turn_indices else 0

    def set_summary(self, summary: SessionSummary | dict[str, Any] | str | None) -> None:
        normalized = SessionSummary.from_any(summary, thread_id=self.session_id)
        self.state["summary"] = normalized.to_dict() if normalized is not None else None
        logger.debug("Persisting session summary", extra={"has_summary": normalized is not None})
        self._save()

    def set_task_state(self, task_state: TaskState | dict[str, Any] | None) -> None:
        normalized = TaskState.from_any(task_state)
        self.state["task_state"] = normalized.to_dict() if normalized is not None else None
        logger.debug("Persisting task state", extra={"has_task_state": normalized is not None})
        self._save()

    def get_thread_state(self) -> ThreadState:
        return ThreadState.from_session_state(self.state, thread_id=self.session_id or "default")

    def compact_history(self, *, max_active_tokens: int) -> ThreadState:
        # Compaction does not rewrite history; it only updates which blocks are
        # considered active versus overflow for prompt construction.
        thread_state = self.get_thread_state()
        compacted = HistoryCompactor(CompactionPolicy(max_active_tokens=max_active_tokens)).compact(thread_state)
        self.state["active_block_ids"] = [block.block_id for block in compacted.active_blocks]
        self.state["overflow_block_ids"] = [block.block_id for block in compacted.overflow_blocks]
        logger.debug(
            "Compacted thread history",
            extra={
                "active_block_count": len(compacted.active_blocks),
                "overflow_block_count": len(compacted.overflow_blocks),
                "max_active_tokens": max_active_tokens,
            },
        )
        self._save()
        return compacted

    def reset(self) -> None:
        logger.info("Resetting session state", extra={"session_id": self.session_id})
        self.state = build_empty_session_state(
            session_id=self.session_id,
            storage_backend=self.repo.storage_backend,
        )
        self._save()

    def save_run_trace(self, trace: RunTrace | dict[str, object]) -> None:
        payload = trace.to_dict() if isinstance(trace, RunTrace) else dict(trace)
        session_id = payload.get("session_id")
        target_session_id = session_id if isinstance(session_id, str) and session_id else self.session_id
        self.repo.save_run_trace(target_session_id, payload)

    def load_run_trace(self, run_id: str) -> dict[str, object] | None:
        return self.repo.load_run_trace(self.session_id, run_id)

    def list_run_traces(self) -> list[dict[str, object]]:
        return self.repo.list_run_traces(self.session_id)

    def _save(self) -> None:
        # Session metadata is maintained centrally here so callers do not need
        # to coordinate schema/version/timestamp updates.
        meta = self.state.setdefault("meta", {})
        meta["session_id"] = self.session_id
        meta["storage_backend"] = self.repo.storage_backend
        meta["schema_version"] = SESSION_SCHEMA_VERSION
        meta["updated_at"] = utc_now_iso()
        self.repo.save(self.session_id, self.state)

    def _coerce_context_block(self, block: ContextBlock | dict[str, Any]) -> ContextBlock | None:
        return block if isinstance(block, ContextBlock) else ContextBlock.from_dict(block)

    def _normalize_context_blocks(self, blocks: object) -> list[ContextBlock]:
        if not isinstance(blocks, list):
            return []
        return [block for item in blocks if (block := self._coerce_context_block(item)) is not None]

    def _store_context_blocks(self, blocks: list[ContextBlock]) -> None:
        self.state["context_blocks"] = [block.to_dict() for block in blocks]

    def _lock_for_session(self, session_id: str) -> RLock:
        with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = RLock()
                self._session_locks[session_id] = lock
            return lock
