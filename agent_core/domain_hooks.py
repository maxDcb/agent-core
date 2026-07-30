from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_core.conversation_state import ConversationStateView, TurnMemoryContextView
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

    def extend_turn_memory_payload(
        self,
        *,
        memory_context: TurnMemoryContextView,
    ) -> dict[str, Any]:
        return {}

    def turn_memory_guidance(
        self,
        *,
        memory_context: TurnMemoryContextView,
    ) -> str:
        return ""

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
