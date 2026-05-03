from __future__ import annotations

from dataclasses import dataclass

from agent_core.memory.context_block import ContextBlock
from agent_core.memory.thread_state import ThreadState, group_context_blocks


@dataclass(slots=True)
class CompactionPolicy:
    max_active_tokens: int


class HistoryCompactor:
    """Deterministically split history into active and overflow windows.

    The compactor works on whole turn groups, never on individual messages. That
    keeps block boundaries stable and makes the same input state always produce
    the same active window.
    """

    def __init__(self, policy: CompactionPolicy) -> None:
        self.policy = policy

    def compact(self, thread_state: ThreadState) -> ThreadState:
        history_blocks = [block for block in thread_state.context_blocks if block.kind in {"conversation_turn", "tool_exchange"}]
        groups = group_context_blocks(history_blocks)
        if not groups:
            thread_state.active_blocks = []
            thread_state.overflow_blocks = []
            return thread_state

        selected_indices: set[int] = set()
        used_tokens = 0
        latest_index = len(groups) - 1

        for index, group in enumerate(groups):
            if not any(block.pinned for block in group):
                continue
            selected_indices.add(index)
            used_tokens += self._token_count(group)

        for index in range(len(groups) - 1, -1, -1):
            if index in selected_indices:
                continue
            group = groups[index]
            group_tokens = self._token_count(group)
            if index == latest_index or not selected_indices or used_tokens + group_tokens <= self.policy.max_active_tokens:
                selected_indices.add(index)
                used_tokens += group_tokens

        active_blocks: list[ContextBlock] = []
        overflow_blocks: list[ContextBlock] = []
        for index, group in enumerate(groups):
            target = active_blocks if index in selected_indices else overflow_blocks
            target.extend(group)

        thread_state.active_blocks = active_blocks
        thread_state.overflow_blocks = overflow_blocks
        return thread_state

    def _token_count(self, blocks: list[ContextBlock]) -> int:
        return sum(max(1, block.token_estimate) for block in blocks)
