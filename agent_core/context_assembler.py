from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.llm.base import LLMMessage
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.memory.context_block import ContextBlock, estimate_token_count
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
    so a whole conversation turn or tool exchange can stay atomic. It reserves
    space for fixed prompt layers and the operational handoff before selecting
    bounded history, then flattens those blocks at the provider boundary.
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
                max_handoff_chars=self.settings.memory_max_handoff_chars,
                max_turn_summary_chars=self.settings.memory_max_turn_summary_chars,
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
        injected_blocks: list[ContextBlock] = []
        session_view = thread_state.session_view
        if session_view.generation > 0:
            injected_blocks.append(session_view.as_context_block())

        if retrieved_blocks:
            injected_blocks.extend(retrieved_blocks)

        fixed_tokens = (
            sum(estimate_token_count(message.to_history_dict()) for message in base_messages)
            + sum(block.token_estimate for block in injected_blocks)
            + estimate_token_count({"role": "user", "content": user_input})
        )
        history_budget = max(1, self.settings.max_active_context_tokens - fixed_tokens)
        thread_state = HistoryCompactor(
            CompactionPolicy(max_active_tokens=history_budget)
        ).compact(thread_state)
        selected_blocks = list(thread_state.active_blocks)
        overflow_blocks = list(thread_state.overflow_blocks)
        logger.debug(
            "Assembled bounded conversation context",
            extra={
                "fixed_token_estimate": fixed_tokens,
                "history_token_budget": history_budget,
                "selected_history_blocks": len(selected_blocks),
                "overflow_history_blocks": len(overflow_blocks),
                "injected_blocks": len(injected_blocks),
            },
        )

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
