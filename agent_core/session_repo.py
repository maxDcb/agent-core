from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from agent_core.logging_utils import get_logger
from agent_core.memory.context_block import ContextBlock
from agent_core.memory.session_summary import SessionSummary
from agent_core.memory.task_state import TaskState
from agent_core.run_trace import RunTrace
from agent_core.types import SessionState, build_empty_session_state

logger = get_logger(__name__)


class SessionStore(Protocol):
    storage_backend: str

    def load(self, session_id: str) -> SessionState:
        ...

    def save(self, session_id: str, state: SessionState) -> None:
        ...

    def save_run_trace(self, session_id: str, trace_payload: dict[str, object]) -> None:
        ...

    def load_run_trace(self, session_id: str, run_id: str) -> dict[str, object] | None:
        ...

    def list_run_traces(self, session_id: str) -> list[dict[str, object]]:
        ...

    def list_session_ids(self) -> list[str]:
        ...

    def describe(self) -> dict[str, object]:
        ...


class SessionRepository:
    """Facade around the configured session storage implementation."""

    def __init__(self, session_file: Path | None = None, *, store: SessionStore | None = None) -> None:
        if store is None:
            if session_file is None:
                raise ValueError("SessionRepository requires either a session_file or a store")
            store = JsonFileSessionStore(session_file)
        elif session_file is not None:
            raise ValueError("Provide either session_file or store, not both")

        self.store = store

    @property
    def storage_backend(self) -> str:
        return self.store.storage_backend

    @property
    def session_directory(self) -> Path:
        session_directory = getattr(self.store, "session_directory", None)
        if isinstance(session_directory, Path):
            return session_directory
        raise AttributeError("The configured session store does not expose a filesystem session_directory")

    def load(self, session_id: str) -> SessionState:
        return self.store.load(session_id)

    def save(self, session_id: str, state: SessionState) -> None:
        self.store.save(session_id, state)

    def save_run_trace(self, session_id: str, trace_payload: dict[str, object]) -> None:
        self.store.save_run_trace(session_id, trace_payload)

    def load_run_trace(self, session_id: str, run_id: str) -> dict[str, object] | None:
        return self.store.load_run_trace(session_id, run_id)

    def list_run_traces(self, session_id: str) -> list[dict[str, object]]:
        return self.store.list_run_traces(session_id)

    def list_session_ids(self) -> list[str]:
        return self.store.list_session_ids()

    def describe(self) -> dict[str, object]:
        return self.store.describe()


class JsonFileSessionStore:
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
        # Replace-on-write avoids leaving a partially written session file behind on normal save paths.
        session_file = self._resolve_session_file(session_id)
        logger.trace("Saving session state", extra={"session_file": str(session_file), "session_id": session_id})
        self._atomic_write_json(session_file, state)

    def save_run_trace(self, session_id: str, trace_payload: dict[str, object]) -> None:
        run_id = trace_payload.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("Run trace payload must include a non-empty run_id")

        trace_file = self._resolve_trace_file(session_id=session_id, run_id=run_id)
        logger.trace(
            "Saving run trace",
            extra={"trace_file": str(trace_file), "session_id": session_id, "run_id": run_id},
        )
        self._atomic_write_json(trace_file, trace_payload)

    def load_run_trace(self, session_id: str, run_id: str) -> dict[str, object] | None:
        trace_file = self._resolve_trace_file(session_id=session_id, run_id=run_id)
        if not trace_file.exists():
            return None
        try:
            with trace_file.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError:
            backup_path = self._quarantine_corrupt_file(trace_file)
            logger.error(
                "Run trace file is corrupt; quarantined",
                extra={"trace_file": str(trace_file), "backup_path": str(backup_path)},
            )
            return None
        return payload if isinstance(payload, dict) else None

    def list_run_traces(self, session_id: str) -> list[dict[str, object]]:
        trace_dir = self._resolve_trace_directory(session_id)
        if not trace_dir.exists():
            return []

        summaries: list[dict[str, object]] = []
        for trace_file in trace_dir.glob("*.json"):
            try:
                with trace_file.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt run trace in list operation", extra={"trace_file": str(trace_file)})
                continue
            trace = RunTrace.from_any(payload)
            if trace is not None:
                summaries.append(trace.to_summary_dict())

        return sorted(
            summaries,
            key=lambda item: str(item.get("started_at") or ""),
            reverse=True,
        )

    def list_session_ids(self) -> list[str]:
        session_ids: set[str] = set()
        if not self.session_directory.exists():
            return []
        for session_dir in self.session_directory.iterdir():
            if not session_dir.is_dir():
                continue
            session_file = session_dir / f"{session_dir.name}.json"
            if session_file.exists():
                session_ids.add(session_dir.name)
        return sorted(session_ids)

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.storage_backend,
            "session_directory": str(self.session_directory),
            "default_session_file": str(self.default_session_file),
        }

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

        last_block_id = str(normalized_blocks[-1]["block_id"]) if normalized_blocks else ""
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
        sanitized_session_id = self._sanitize_session_id(session_id)
        return self._resolve_session_directory(session_id) / f"{sanitized_session_id}.json"

    def _resolve_session_directory(self, session_id: str) -> Path:
        return self.session_directory / self._sanitize_session_id(session_id)

    def _resolve_trace_directory(self, session_id: str) -> Path:
        return self._resolve_session_directory(session_id) / "traces"

    def _resolve_trace_file(self, *, session_id: str, run_id: str) -> Path:
        return self._resolve_trace_directory(session_id) / f"{self._sanitize_session_id(run_id)}.json"

    def _sanitize_session_id(self, session_id: str) -> str:
        sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id).strip("._")
        return sanitized or "default"

    def _quarantine_corrupt_file(self, session_file: Path) -> Path:
        suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        backup_path = session_file.with_suffix(f"{session_file.suffix}.corrupt.{suffix}")
        session_file.replace(backup_path)
        return backup_path

    def _atomic_write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
        self._fsync_directory(path.parent)

    def _fsync_directory(self, directory: Path) -> None:
        try:
            directory_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)
