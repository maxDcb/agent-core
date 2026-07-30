from __future__ import annotations

from dataclasses import dataclass

from agent_core.memory.context_block import ContextBlock
from agent_core.memory.thread_state import ThreadState, group_context_blocks


@dataclass(slots=True)
class CompactionPolicy:
    max_active_tokens: int


class HistoryCompactor:
    """Split history into a chronological overflow prefix and active suffix.

    The compactor works on whole turn groups, never on individual messages. That
    keeps block boundaries stable and guarantees that replayed raw history never
    contains holes. Important facts from older turns belong in the session view
    or explicit retrieved context, not in isolated pinned history groups.
    """

    def __init__(self, policy: CompactionPolicy) -> None:
        self.policy = policy

    def compact(self, thread_state: ThreadState) -> ThreadState:
        history_blocks = [
            block for block in thread_state.context_blocks if block.kind in {"conversation_turn", "tool_exchange"}
        ]
        groups = group_context_blocks(history_blocks)
        if not groups:
            thread_state.active_blocks = []
            thread_state.overflow_blocks = []
            return thread_state

        active_start = len(groups) - 1
        used_tokens = 0

        for index in range(len(groups) - 1, -1, -1):
            group = groups[index]
            group_tokens = self._token_count(group)
            if index != len(groups) - 1 and used_tokens + group_tokens > self.policy.max_active_tokens:
                break
            active_start = index
            used_tokens += group_tokens

        thread_state.overflow_blocks = [block for group in groups[:active_start] for block in group]
        thread_state.active_blocks = [block for group in groups[active_start:] for block in group]
        return thread_state

    def _token_count(self, blocks: list[ContextBlock]) -> int:
        return sum(max(1, block.token_estimate) for block in blocks)
