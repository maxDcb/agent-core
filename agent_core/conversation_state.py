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
    summary: Mapping[str, Any] | None
    task_state: Mapping[str, Any] | None
    context_blocks: tuple[Mapping[str, Any], ...]
    active_blocks: tuple[Mapping[str, Any], ...]
    overflow_blocks: tuple[Mapping[str, Any], ...]


def build_conversation_state_view(thread_state: ThreadState) -> ConversationStateView:
    return ConversationStateView(
        thread_id=thread_state.thread_id,
        meta=deepcopy(thread_state.meta),
        summary=deepcopy(thread_state.summary.to_dict()) if thread_state.summary is not None else None,
        task_state=deepcopy(thread_state.task_state.to_dict()) if thread_state.task_state is not None else None,
        context_blocks=tuple(deepcopy(block.to_dict()) for block in thread_state.context_blocks),
        active_blocks=tuple(deepcopy(block.to_dict()) for block in thread_state.active_blocks),
        overflow_blocks=tuple(deepcopy(block.to_dict()) for block in thread_state.overflow_blocks),
    )
