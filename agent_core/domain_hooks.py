from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_core.conversation_state import ConversationStateView
    from agent_core.investigation_prompts import InvestigationPromptSet
    from agent_core.run_options import RunOptions
    from agent_core.session_manager import SessionManager
    from agent_core.settings import CoreSettings


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
        thread_state: ConversationStateView,
        turn_index: int,
    ) -> dict[str, Any]:
        return {}

    def extend_session_summary_delta_payload(
        self,
        *,
        thread_state: ConversationStateView,
        new_overflow_blocks: tuple[Mapping[str, Any], ...],
    ) -> dict[str, Any]:
        return {}

    def task_state_extensions_template(
        self,
        *,
        thread_state: ConversationStateView,
        turn_index: int,
    ) -> dict[str, Any]:
        return {}

    def session_summary_extensions_template(
        self,
        *,
        thread_state: ConversationStateView,
    ) -> dict[str, Any]:
        return {}

    def customize_investigation_prompts(
        self,
        *,
        prompt_set: InvestigationPromptSet,
        settings: CoreSettings,
        options: RunOptions,
    ) -> InvestigationPromptSet:
        return prompt_set

    def after_turn(
        self,
        *,
        session_manager: SessionManager,
        thread_state: ConversationStateView,
        turn_index: int,
    ) -> None:
        return None
