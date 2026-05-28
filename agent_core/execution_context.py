from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core.settings import CoreSettings


EXECUTION_SCOPE_STATE_KEY = "execution_scope"


def _scope_from_state(session_state: dict[str, Any]) -> dict[str, Any]:
    scope = session_state.get(EXECUTION_SCOPE_STATE_KEY)
    return scope if isinstance(scope, dict) else {}


def _clean_string_list(raw_value: object) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    cleaned: list[str] = []
    for item in raw_value:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
    return cleaned


def effective_allowed_read_roots(settings: CoreSettings, session_state: dict[str, Any]) -> list[Path]:
    scope_roots = _clean_string_list(_scope_from_state(session_state).get("allowed_read_roots"))
    if scope_roots:
        return [Path(root).resolve() for root in scope_roots]
    return [root.resolve() for root in settings.allowed_read_roots]


def effective_allowed_http_hosts(settings: CoreSettings, session_state: dict[str, Any]) -> list[str]:
    scope_hosts = _clean_string_list(_scope_from_state(session_state).get("allowed_http_hosts"))
    return scope_hosts or list(settings.allowed_http_hosts)


def effective_allowed_http_methods(settings: CoreSettings, session_state: dict[str, Any]) -> list[str]:
    scope_methods = _clean_string_list(_scope_from_state(session_state).get("allowed_http_methods"))
    return [method.upper() for method in (scope_methods or settings.allowed_http_methods)]


@dataclass(slots=True)
class ExecutionContext:
    session_id: str
    settings: CoreSettings
    session_state: dict[str, Any]

    def allowed_read_roots(self) -> list[Path]:
        return effective_allowed_read_roots(self.settings, self.session_state)

    def allowed_http_hosts(self) -> list[str]:
        return effective_allowed_http_hosts(self.settings, self.session_state)

    def allowed_http_methods(self) -> list[str]:
        return effective_allowed_http_methods(self.settings, self.session_state)

    def is_path_allowed(self, candidate: Path) -> bool:
        candidate = candidate.resolve()
        for root in self.allowed_read_roots():
            try:
                candidate.relative_to(root)
                return True
            except ValueError:
                continue
        return False
