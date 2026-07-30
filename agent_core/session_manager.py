from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from threading import Lock, RLock
from typing import Any

from agent_core.llm.base import LLMMessage
from agent_core.logging_utils import get_logger
from agent_core.memory.context_block import ContextBlock
from agent_core.memory.derivation import (
    derive_final_response_memory,
    derive_tool_exchange_memory,
)
from agent_core.memory.history_compactor import CompactionPolicy, HistoryCompactor
from agent_core.memory.journal import (
    ExchangeMemory,
    IncrementalMemoryJournal,
    TurnMemory,
    build_fallback_turn_memory,
)
from agent_core.memory.thread_state import ThreadState
from agent_core.run_trace import RunTrace
from agent_core.session_repo import SessionRepository
from agent_core.types import SESSION_SCHEMA_VERSION, SessionState, build_empty_session_state, utc_now_iso

logger = get_logger(__name__)


class SessionManager:
    """Own the mutable session state between orchestration and storage.

    The manager exposes a small write API around the canonical session payload:
    context blocks, incremental memory journals, domain state, and metadata.
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
            if (
                previous_context_session_id == session_id
                and previous_context_state is not None
                and scoped_state is not None
            ):
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

    def get_memory_journal(self) -> IncrementalMemoryJournal:
        return IncrementalMemoryJournal.from_any(
            self.state.get("memory"),
            thread_id=self.session_id or "default",
        )

    def commit_tool_exchange(
        self,
        *,
        block: ContextBlock,
        memory: ExchangeMemory,
    ) -> bool:
        return self._commit_context_block_with_exchange(block=block, memory=memory)

    def commit_conversation_turn(
        self,
        *,
        block: ContextBlock,
        memory: ExchangeMemory,
    ) -> bool:
        return self._commit_context_block_with_exchange(block=block, memory=memory)

    def append_exchange_memory(self, memory: ExchangeMemory) -> bool:
        journal = self.get_memory_journal()
        changed = journal.append_exchange(memory)
        if not changed:
            return False
        self._save_memory_transaction(journal)
        logger.debug(
            "Persisted exchange memory",
            extra={"memory_id": memory.memory_id, "turn_index": memory.turn_index},
        )
        return True

    def commit_turn_memory(
        self,
        memory: TurnMemory,
        *,
        max_handoff_chars: int,
        max_turn_summary_chars: int,
    ) -> bool:
        journal = self.get_memory_journal()
        changed = journal.commit_turn(
            memory,
            max_handoff_chars=max_handoff_chars,
            max_turn_summary_chars=max_turn_summary_chars,
        )
        if not changed:
            return False
        self._save_memory_transaction(journal)
        logger.debug(
            "Committed turn memory",
            extra={
                "memory_id": memory.memory_id,
                "turn_index": memory.turn_index,
                "origin": memory.origin,
                "degraded": memory.degraded,
                "turn_summary_chars": len(memory.turn_summary),
                "handoff_chars": len(memory.handoff_after_turn),
            },
        )
        return True

    def reconcile_memory(
        self,
        *,
        max_handoff_chars: int,
        max_turn_summary_chars: int,
    ) -> bool:
        """Recover journal entries after interruption without calling an LLM."""

        journal = self.get_memory_journal()
        context_blocks = self.get_context_blocks()
        policy_changed = (
            journal.max_handoff_chars != max_handoff_chars
            or journal.max_turn_summary_chars != max_turn_summary_chars
        )
        changed = policy_changed
        recovered = False
        if policy_changed:
            journal.max_turn_summary_chars = max_turn_summary_chars
            journal.rebuild_session_view(
                max_handoff_chars=max_handoff_chars,
            )

        for block in context_blocks:
            turn_index = block.metadata.get("turn_index")
            if not isinstance(turn_index, int):
                continue
            if block.kind == "tool_exchange":
                exchange_index = block.metadata.get("exchange_index")
                if not isinstance(exchange_index, int):
                    continue
                memory_id = f"turn-{turn_index:04d}-exchange-{exchange_index:02d}-runtime"
                if any(item.memory_id == memory_id for item in journal.exchanges):
                    continue
                assistant_payload = block.content.get("assistant_message")
                assistant_message = (
                    LLMMessage.from_history_dict(assistant_payload)
                    if isinstance(assistant_payload, dict)
                    else LLMMessage(role="assistant", content="")
                )
                raw_tool_messages = block.content.get("tool_messages")
                tool_messages = (
                    [LLMMessage.from_history_dict(item) for item in raw_tool_messages if isinstance(item, dict)]
                    if isinstance(raw_tool_messages, list)
                    else []
                )
                appended = journal.append_exchange(
                    derive_tool_exchange_memory(
                        thread_id=self.session_id,
                        turn_index=turn_index,
                        exchange_index=exchange_index,
                        assistant_message=assistant_message,
                        tool_messages=tool_messages,
                        source_block_id=block.block_id,
                        origin="recovery",
                    )
                )
                changed = appended or changed
                recovered = appended or recovered

        conversation_blocks = [
            block
            for block in context_blocks
            if block.kind == "conversation_turn" and isinstance(block.metadata.get("turn_index"), int)
        ]
        for block in sorted(conversation_blocks, key=lambda item: int(item.metadata["turn_index"])):
            turn_index = int(block.metadata["turn_index"])
            exchange_index = (
                max(
                    (
                        int(candidate.metadata["exchange_index"])
                        for candidate in context_blocks
                        if candidate.kind == "tool_exchange"
                        and candidate.metadata.get("turn_index") == turn_index
                        and isinstance(candidate.metadata.get("exchange_index"), int)
                    ),
                    default=-1,
                )
                + 1
            )
            final_memory_id = f"turn-{turn_index:04d}-final-response"
            assistant_content = self._message_content(block.content.get("assistant_message"))
            user_intent = self._message_content(block.content.get("user_message"))
            if not any(item.memory_id == final_memory_id for item in journal.exchanges):
                appended = journal.append_exchange(
                    derive_final_response_memory(
                        thread_id=self.session_id,
                        turn_index=turn_index,
                        exchange_index=exchange_index,
                        assistant_content=assistant_content,
                        source_block_id=block.block_id,
                        origin="recovery",
                    )
                )
                changed = appended or changed
                recovered = appended or recovered

            if journal.turn_for_index(turn_index) is not None:
                continue
            exchanges = journal.exchanges_for_turn(turn_index)
            fallback = build_fallback_turn_memory(
                thread_id=self.session_id,
                turn_index=turn_index,
                user_intent=user_intent,
                assistant_outcome=assistant_content,
                exchanges=exchanges,
                source_block_ids=[
                    candidate.block_id
                    for candidate in context_blocks
                    if candidate.metadata.get("turn_index") == turn_index
                ],
                previous_handoff=journal.handoff_before_turn(turn_index),
                max_handoff_chars=max_handoff_chars,
                max_turn_summary_chars=max_turn_summary_chars,
                origin="recovery",
            )
            committed = journal.commit_turn(
                fallback,
                max_handoff_chars=max_handoff_chars,
                max_turn_summary_chars=max_turn_summary_chars,
            )
            changed = committed or changed
            recovered = committed or recovered

        if not changed:
            return False
        self._save_memory_transaction(journal)
        log = logger.warning if recovered else logger.info
        log(
            "Recovered incomplete incremental memory journal"
            if recovered
            else "Updated incremental memory view policy",
            extra={
                "session_id": self.session_id,
                "exchange_count": len(journal.exchanges),
                "turn_count": len(journal.turns),
            },
        )
        return True

    def get_thread_state(self) -> ThreadState:
        return ThreadState.from_session_state(self.state, thread_id=self.session_id or "default")

    def compact_history(self, *, max_active_tokens: int) -> ThreadState:
        # Compaction does not rewrite history; it only updates which blocks are
        # considered active versus overflow for prompt construction.
        state_before = deepcopy(self.state)
        thread_state = self.get_thread_state()
        compacted = HistoryCompactor(CompactionPolicy(max_active_tokens=max_active_tokens)).compact(thread_state)
        try:
            self.state["active_block_ids"] = [block.block_id for block in compacted.active_blocks]
            self.state["overflow_block_ids"] = [block.block_id for block in compacted.overflow_blocks]
            self._save()
        except Exception:
            self.state = state_before
            raise
        logger.debug(
            "Compacted thread history",
            extra={
                "active_block_count": len(compacted.active_blocks),
                "overflow_block_count": len(compacted.overflow_blocks),
                "max_active_tokens": max_active_tokens,
            },
        )
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

    def _commit_context_block_with_exchange(
        self,
        *,
        block: ContextBlock,
        memory: ExchangeMemory,
    ) -> bool:
        state_before = deepcopy(self.state)
        try:
            blocks = self.get_context_blocks()
            existing_block = next((item for item in blocks if item.block_id == block.block_id), None)
            block_changed = existing_block is None
            if existing_block is not None and existing_block.to_dict() != block.to_dict():
                raise ValueError(f"Context block id already exists with different content: {block.block_id}")
            if block_changed:
                blocks.append(block)

            journal = self.get_memory_journal()
            memory_changed = journal.append_exchange(memory)
            if not block_changed and not memory_changed:
                return False

            self._store_context_blocks(blocks)
            self.state["active_block_ids"] = []
            self.state["overflow_block_ids"] = []
            self.state["memory"] = journal.to_dict()
            self._save()
        except Exception:
            self.state = state_before
            raise
        logger.debug(
            "Persisted atomic context and exchange memory",
            extra={"block_id": block.block_id, "memory_id": memory.memory_id},
        )
        return True

    def _save_memory_transaction(self, journal: IncrementalMemoryJournal) -> None:
        state_before = deepcopy(self.state)
        try:
            self.state["memory"] = journal.to_dict()
            self._save()
        except Exception:
            self.state = state_before
            raise

    @staticmethod
    def _message_content(payload: object) -> str:
        if not isinstance(payload, dict):
            return ""
        content = payload.get("content")
        if isinstance(content, str):
            return content
        return "" if content is None else str(content)

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
