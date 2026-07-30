from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.llm.base import LLMMessage
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.memory.context_block import ContextBlock
from agent_core.memory.history_compactor import CompactionPolicy, HistoryCompactor
from agent_core.memory.thread_state import render_context_blocks_to_messages
from agent_core.session_manager import SessionManager
from agent_core.settings import CoreSettings

logger = get_logger(__name__)


@dataclass(slots=True)
class ContextAssembly:
    """Result of prompt-context selection before the provider call.

    `messages` is the flattened transcript sent to the model. The block lists
    remain available so callers can reason about what was kept, overflowed, or
    injected while building that transcript.
    """

    messages: list[LLMMessage]
    selected_blocks: list[ContextBlock] = field(default_factory=list)
    overflow_blocks: list[ContextBlock] = field(default_factory=list)
    injected_blocks: list[ContextBlock] = field(default_factory=list)


class ContextAssembler:
    """Choose which memory blocks enter the prompt for the current turn.

    The assembler works on `ContextBlock` objects instead of raw chat messages
    so a whole conversation turn or tool exchange can stay atomic. The active
    versus overflow split is computed elsewhere and this assembler flattens
    the selected blocks alongside the materialized SessionView.
    """

    def __init__(self, *, settings: CoreSettings, session_manager: SessionManager) -> None:
        self.settings = settings
        self.session_manager = session_manager

    def assemble(
        self,
        *,
        base_messages: list[LLMMessage],
        user_input: str,
        retrieved_blocks: list[ContextBlock] | None = None,
    ) -> ContextAssembly:
        """Select atomic blocks first, then flatten them at the provider boundary."""

        try:
            self.session_manager.reconcile_memory(
                max_session_items=self.settings.memory_max_session_items,
                max_recent_outcomes=self.settings.memory_max_recent_outcomes,
            )
        except Exception as exc:
            logger.warning(
                "Incremental memory reconciliation failed; continuing with raw history",
                extra={
                    "exception_type": type(exc).__name__,
                    "error_preview": safe_preview(str(exc), limit=200),
                },
            )
        thread_state = self.session_manager.get_thread_state()
        if not thread_state.active_blocks and thread_state.context_blocks:
            # Fresh or legacy-loaded sessions may not have compaction pointers
            # yet. Rebuild the active/overflow split on demand before prompt
            # selection.
            thread_state = HistoryCompactor(
                CompactionPolicy(max_active_tokens=self.settings.max_active_context_tokens)
            ).compact(thread_state)

        selected_blocks = list(thread_state.active_blocks)
        overflow_blocks = list(thread_state.overflow_blocks)

        injected_blocks: list[ContextBlock] = []
        session_view = thread_state.session_view
        if session_view.generation > 0:
            injected_blocks.append(session_view.as_context_block())

        if retrieved_blocks:
            injected_blocks.extend(retrieved_blocks)

        messages = list(base_messages)
        messages.extend(render_context_blocks_to_messages(injected_blocks))
        messages.extend(render_context_blocks_to_messages(selected_blocks))
        messages.append(LLMMessage(role="user", content=user_input))

        return ContextAssembly(
            messages=messages,
            selected_blocks=selected_blocks,
            overflow_blocks=overflow_blocks,
            injected_blocks=injected_blocks,
        )
