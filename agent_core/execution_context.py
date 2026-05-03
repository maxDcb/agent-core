from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core.settings import CoreSettings


@dataclass(slots=True)
class ExecutionContext:
    session_id: str
    settings: CoreSettings
    session_state: dict[str, Any]

    def is_path_allowed(self, candidate: Path) -> bool:
        candidate = candidate.resolve()
        for root in self.settings.allowed_read_roots:
            try:
                candidate.relative_to(root)
                return True
            except ValueError:
                continue
        return False
