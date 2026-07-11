from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core.run_context import ExecutionScope, RunContext
from agent_core.settings import CoreSettings


def effective_allowed_read_roots(settings: CoreSettings, scope: ExecutionScope) -> list[Path]:
    roots = scope.allowed_read_roots
    return [path.resolve() for path in (roots if roots is not None else tuple(settings.allowed_read_roots))]


def effective_allowed_http_hosts(settings: CoreSettings, scope: ExecutionScope) -> list[str]:
    hosts = scope.allowed_http_hosts
    return list(hosts if hosts is not None else tuple(settings.allowed_http_hosts))


def effective_allowed_http_methods(settings: CoreSettings, scope: ExecutionScope) -> list[str]:
    methods = scope.allowed_http_methods
    selected = methods if methods is not None else tuple(settings.allowed_http_methods)
    return [method.upper() for method in selected]


@dataclass(slots=True)
class ExecutionContext:
    namespace_id: str
    run_id: str
    settings: CoreSettings
    scope: ExecutionScope
    correlation: dict[str, Any]
    application_context: dict[str, Any]

    @classmethod
    def from_run_context(cls, *, context: RunContext, settings: CoreSettings) -> ExecutionContext:
        if context.run_id is None:
            raise ValueError("RunContext must have a run_id before tool execution")
        return cls(
            namespace_id=context.namespace_id,
            run_id=context.run_id,
            settings=settings,
            scope=context.scope,
            correlation=dict(context.correlation),
            application_context=dict(context.application_context),
        )

    def allowed_read_roots(self) -> list[Path]:
        return effective_allowed_read_roots(self.settings, self.scope)

    def allowed_http_hosts(self) -> list[str]:
        return effective_allowed_http_hosts(self.settings, self.scope)

    def allowed_http_methods(self) -> list[str]:
        return effective_allowed_http_methods(self.settings, self.scope)

    def is_path_allowed(self, candidate: Path) -> bool:
        candidate = candidate.resolve()
        for root in self.allowed_read_roots():
            try:
                candidate.relative_to(root)
                return True
            except ValueError:
                continue
        return False
