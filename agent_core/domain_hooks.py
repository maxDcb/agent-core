from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_core.memory.context_block import ContextBlock
    from agent_core.settings import CoreSettings
    from agent_core.memory.thread_state import ThreadState
    from agent_core.session_manager import SessionManager


class DomainHooks:
    """Runtime extension points for domain-specific prompt and memory data."""

    def build_system_prompt_blocks(
        self,
        *,
        settings: CoreSettings,
        session_manager: SessionManager,
    ) -> list[str]:
        return []

    def extend_task_state_payload(
        self,
        *,
        thread_state: ThreadState,
        turn_index: int,
    ) -> dict[str, Any]:
        return {}

    def extend_session_summary_delta_payload(
        self,
        *,
        thread_state: ThreadState,
        new_overflow_blocks: list[ContextBlock],
    ) -> dict[str, Any]:
        return {}

    def task_state_extensions_template(
        self,
        *,
        thread_state: ThreadState,
        turn_index: int,
    ) -> dict[str, Any]:
        return {}

    def session_summary_extensions_template(
        self,
        *,
        thread_state: ThreadState,
    ) -> dict[str, Any]:
        return {}

    def after_turn(
        self,
        *,
        session_manager: SessionManager,
        thread_state: ThreadState,
        turn_index: int,
    ) -> None:
        return None