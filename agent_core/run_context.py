from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    """Explicit execution boundary for one run.

    ``None`` inherits the corresponding application setting. An empty tuple is
    an explicit deny-all boundary.
    """

    allowed_read_roots: tuple[Path, ...] | None = None
    allowed_http_hosts: tuple[str, ...] | None = None
    allowed_http_methods: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_read_roots": (
                [str(path) for path in self.allowed_read_roots]
                if self.allowed_read_roots is not None
                else None
            ),
            "allowed_http_hosts": list(self.allowed_http_hosts) if self.allowed_http_hosts is not None else None,
            "allowed_http_methods": (
                list(self.allowed_http_methods) if self.allowed_http_methods is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: object) -> ExecutionScope:
        if not isinstance(payload, dict):
            return cls()
        roots = payload.get("allowed_read_roots")
        hosts = payload.get("allowed_http_hosts")
        methods = payload.get("allowed_http_methods")
        return cls(
            allowed_read_roots=(
                tuple(Path(item).resolve() for item in roots if isinstance(item, str) and item.strip())
                if isinstance(roots, list)
                else None
            ),
            allowed_http_hosts=(
                tuple(item.strip() for item in hosts if isinstance(item, str) and item.strip())
                if isinstance(hosts, list)
                else None
            ),
            allowed_http_methods=(
                tuple(item.strip().upper() for item in methods if isinstance(item, str) and item.strip())
                if isinstance(methods, list)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class RunContext:
    """Application-supplied context for one autonomous agent run."""

    namespace_id: str
    run_id: str | None = None
    parent_id: str | None = None
    thread_id: str | None = None
    scope: ExecutionScope = field(default_factory=ExecutionScope)
    correlation: dict[str, Any] = field(default_factory=dict)
    application_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.namespace_id.strip():
            raise ValueError("namespace_id must be non-empty")
        object.__setattr__(self, "namespace_id", self.namespace_id.strip())
        object.__setattr__(self, "correlation", dict(self.correlation))
        object.__setattr__(self, "application_context", dict(self.application_context))

    def with_run_id(self, run_id: str) -> RunContext:
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        return replace(self, run_id=run_id.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace_id": self.namespace_id,
            "run_id": self.run_id,
            "parent_id": self.parent_id,
            "thread_id": self.thread_id,
            "scope": self.scope.to_dict(),
            "correlation": dict(self.correlation),
            "application_context": dict(self.application_context),
        }

