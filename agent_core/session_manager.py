from __future__ import annotations

from typing import Any

from agent_core.logging_utils import get_logger
from agent_core.memory.context_block import ContextBlock
from agent_core.memory.history_compactor import CompactionPolicy, HistoryCompactor
from agent_core.memory.session_summary import SessionSummary
from agent_core.memory.task_state import TaskState
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
        self.session_id = ""
        self.state: SessionState = build_empty_session_state(
            session_id=default_session_id,
            storage_backend=self.repo.storage_backend,
        )
        self.activate_session(default_session_id)

    def activate_session(self, session_id: str) -> None:
        if session_id == self.session_id and self.state:
            logger.trace("Session already active", extra={"session_id": session_id})
            return

        # Session switching is explicit so the storage layer can later move from JSON files to SQLite cleanly.
        self.session_id = session_id
        self.state = self.repo.load(session_id)
        logger.info(
            "Activated session",
            extra={"session_id": session_id, "context_block_count": len(self.state.get("context_blocks", []))},
        )

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
        self.set_context_blocks(blocks)
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
