from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from agent_core.run_models import AgentRunState


class RunStore(Protocol):
    def create(self, state: AgentRunState) -> None: ...

    def save(self, state: AgentRunState) -> None: ...

    def load(self, *, namespace_id: str, run_id: str) -> AgentRunState | None: ...

    def list(self, *, namespace_id: str, parent_id: str | None = None) -> list[AgentRunState]: ...


class JsonFileRunStore:
    """Lossless single-process JSON store for autonomous run state."""

    def __init__(self, root_directory: Path) -> None:
        self.root_directory = root_directory.resolve()
        self.root_directory.mkdir(parents=True, exist_ok=True)

    def create(self, state: AgentRunState) -> None:
        path = self._run_path(namespace_id=state.context.namespace_id, run_id=state.run_id)
        if path.exists():
            existing = self.load(namespace_id=state.context.namespace_id, run_id=state.run_id)
            if existing is not None and existing.to_dict() == state.to_dict():
                return
            raise FileExistsError(f"Run already exists: {state.run_id}")
        self._atomic_write(path, state.to_dict())

    def save(self, state: AgentRunState) -> None:
        self._atomic_write(
            self._run_path(namespace_id=state.context.namespace_id, run_id=state.run_id),
            state.to_dict(),
        )

    def load(self, *, namespace_id: str, run_id: str) -> AgentRunState | None:
        path = self._run_path(namespace_id=namespace_id, run_id=run_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return AgentRunState.from_dict(payload)

    def list(self, *, namespace_id: str, parent_id: str | None = None) -> list[AgentRunState]:
        directory = self._namespace_directory(namespace_id)
        if not directory.exists():
            return []
        states: list[AgentRunState] = []
        for path in directory.glob("*/state.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    state = AgentRunState.from_dict(json.load(handle))
            except (OSError, json.JSONDecodeError):
                continue
            if state is not None and (parent_id is None or state.context.parent_id == parent_id):
                states.append(state)
        return sorted(states, key=lambda item: item.created_at)

    def _namespace_directory(self, namespace_id: str) -> Path:
        return self.root_directory / _storage_key(namespace_id)

    def _run_path(self, *, namespace_id: str, run_id: str) -> Path:
        return self._namespace_directory(namespace_id) / _storage_key(run_id) / "state.json"

    def _atomic_write(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise


def _storage_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
