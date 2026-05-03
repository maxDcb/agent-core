from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agent_core.logging_utils import get_logger
from agent_core.memory.context_block import ContextBlock
from agent_core.memory.session_summary import SessionSummary
from agent_core.memory.task_state import TaskState
from agent_core.types import SessionState, build_empty_session_state

logger = get_logger(__name__)


class SessionRepository:
    def __init__(self, session_file: Path) -> None:
        self.default_session_file = session_file
        self.default_session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_directory = self.default_session_file.parent
        self.session_directory.mkdir(parents=True, exist_ok=True)
        self.storage_backend = "json"
        logger.debug(
            "Session repository ready",
            extra={
                "default_session_file": str(self.default_session_file),
                "session_directory": str(self.session_directory),
            },
        )

    def load(self, session_id: str) -> SessionState:
        session_file = self._resolve_session_file(session_id)
        if not session_file.exists():
            logger.info("Session file does not exist yet; using empty state", extra={"session_file": str(session_file)})
            return build_empty_session_state(session_id=session_id, storage_backend=self.storage_backend)

        logger.debug("Loading session file", extra={"session_file": str(session_file), "session_id": session_id})
        try:
            with session_file.open("r", encoding="utf-8") as fh:
                state = json.load(fh)
        except json.JSONDecodeError:
            backup_path = self._quarantine_corrupt_file(session_file)
            logger.error(
                "Session file is corrupt; quarantined and replaced with empty state",
                extra={"session_file": str(session_file), "backup_path": str(backup_path)},
            )
            return build_empty_session_state(session_id=session_id, storage_backend=self.storage_backend)
        normalized = self._normalize_state(state)
        normalized_meta = normalized.setdefault("meta", {})
        normalized_meta["session_id"] = session_id
        normalized_meta["storage_backend"] = self.storage_backend
        summary = SessionSummary.from_any(
            normalized.get("summary"),
            thread_id=session_id,
            covers_blocks_until=normalized.get("context_blocks", [])[-1]["block_id"] if normalized.get("context_blocks") else "",
        )
        normalized["summary"] = summary.to_dict() if summary is not None else None
        logger.info(
            "Loaded session state",
            extra={
                "tool_history_count": len(normalized.get("tool_history", [])),
                "domain_state_count": len(normalized.get("domain_state", {})),
                "context_block_count": len(normalized.get("context_blocks", [])),
            },
        )
        return normalized

    def save(self, session_id: str, state: SessionState) -> None:
        session_file = self._resolve_session_file(session_id)
        temp_file = session_file.with_suffix(f"{session_file.suffix}.tmp")
        # Replace-on-write avoids leaving a partially written session file behind on normal save paths.
        logger.trace("Saving session state", extra={"session_file": str(session_file), "session_id": session_id})
        with temp_file.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
        temp_file.replace(session_file)

    def _normalize_state(self, state: object) -> SessionState:
        normalized = build_empty_session_state(storage_backend=self.storage_backend)
        if not isinstance(state, dict):
            logger.info("Loaded session state is not a dictionary; falling back to empty state")
            return normalized

        tool_history = state.get("tool_history", [])
        if isinstance(tool_history, list):
            normalized["tool_history"] = [item for item in tool_history if isinstance(item, dict)]
        elif "tool_history" in state:
            logger.info("Loaded session field is not a list; resetting field", extra={"field": "tool_history"})

        for key in ("active_block_ids", "overflow_block_ids"):
            value = state.get(key, [])
            if isinstance(value, list):
                normalized[key] = [item for item in value if isinstance(item, str)]
            elif key in state:
                logger.info("Loaded session field is not a list; resetting field", extra={"field": key})

        context_blocks = state.get("context_blocks", [])
        normalized_blocks: list[dict[str, object]] = []
        if isinstance(context_blocks, list):
            normalized_blocks = [
                block.to_dict()
                for item in context_blocks
                if (block := ContextBlock.from_dict(item)) is not None
            ]
        elif "context_blocks" in state:
            logger.info("Loaded session context_blocks field is not a list; resetting context_blocks")

        domain_state = state.get("domain_state", {})
        if isinstance(domain_state, dict):
            normalized["domain_state"] = domain_state
        else:
            logger.info("Loaded session domain_state field is not a dictionary; resetting domain_state")

        meta = state.get("meta", {})
        if isinstance(meta, dict):
            normalized["meta"].update(meta)
        else:
            logger.info("Loaded session meta field is not a dictionary; resetting meta")

        normalized["context_blocks"] = normalized_blocks

        last_block_id = normalized_blocks[-1]["block_id"] if normalized_blocks else ""
        summary = SessionSummary.from_any(
            state.get("summary"),
            thread_id=str(normalized["meta"].get("session_id", "")),
            covers_blocks_until=last_block_id,
        )
        normalized["summary"] = summary.to_dict() if summary is not None else None

        task_state = TaskState.from_any(state.get("task_state"))
        normalized["task_state"] = task_state.to_dict() if task_state is not None else None

        return normalized

    def _resolve_session_file(self, session_id: str) -> Path:
        # The default session keeps the configured file name, while named sessions live alongside it.
        if session_id == "default":
            return self.default_session_file
        return self.session_directory / f"{self._sanitize_session_id(session_id)}.json"

    def _sanitize_session_id(self, session_id: str) -> str:
        sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id).strip("._")
        return sanitized or "default"

    def _quarantine_corrupt_file(self, session_file: Path) -> Path:
        suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        backup_path = session_file.with_suffix(f"{session_file.suffix}.corrupt.{suffix}")
        session_file.replace(backup_path)
        return backup_path
