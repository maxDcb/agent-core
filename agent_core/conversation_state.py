from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_core.memory.thread_state import ThreadState


@dataclass(frozen=True, slots=True)
class ConversationStateView:
    """Detached, serializable view exposed to conversation-domain hooks."""

    thread_id: str
    meta: Mapping[str, Any]
    session_view: Mapping[str, Any]
    exchange_memories: tuple[Mapping[str, Any], ...]
    turn_memories: tuple[Mapping[str, Any], ...]
    context_blocks: tuple[Mapping[str, Any], ...]
    active_blocks: tuple[Mapping[str, Any], ...]
    overflow_blocks: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class TurnMemoryContextView:
    """Bounded hook input for one turn-memory commit."""

    thread_id: str
    turn_index: int
    meta: Mapping[str, Any]
    previous_handoff: str
    exchange_memories: tuple[Mapping[str, Any], ...]


def build_conversation_state_view(thread_state: ThreadState) -> ConversationStateView:
    journal = thread_state.memory_journal
    return ConversationStateView(
        thread_id=thread_state.thread_id,
        meta=deepcopy(thread_state.meta),
        session_view=deepcopy(thread_state.session_view.to_dict()),
        exchange_memories=tuple(
            deepcopy(item.to_dict()) for item in (journal.exchanges if journal is not None else [])
        ),
        turn_memories=tuple(deepcopy(item.to_dict()) for item in (journal.turns if journal is not None else [])),
        context_blocks=tuple(deepcopy(block.to_dict()) for block in thread_state.context_blocks),
        active_blocks=tuple(deepcopy(block.to_dict()) for block in thread_state.active_blocks),
        overflow_blocks=tuple(deepcopy(block.to_dict()) for block in thread_state.overflow_blocks),
    )


def build_turn_memory_context_view(
    thread_state: ThreadState,
    *,
    turn_index: int,
    exchange_memories: tuple[Mapping[str, Any], ...],
) -> TurnMemoryContextView:
    return TurnMemoryContextView(
        thread_id=thread_state.thread_id,
        turn_index=turn_index,
        meta=deepcopy(thread_state.meta),
        previous_handoff=thread_state.session_view.content,
        exchange_memories=deepcopy(exchange_memories),
    )
