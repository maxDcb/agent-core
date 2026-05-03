from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from agent_core.memory.context_block import ContextBlock, estimate_token_count
from agent_core.memory.session_summary import SessionSummary
from agent_core.memory.task_state import TaskState
from agent_core.llm.base import LLMMessage


def render_context_blocks_to_messages(blocks: list[ContextBlock]) -> list[LLMMessage]:
    """Project stored context blocks back into a flat chat transcript.

    Summary-like blocks are emitted as system messages first. Conversation and
    tool-exchange blocks are then replayed in turn order so the provider sees a
    chronological transcript even though storage keeps richer grouped objects.
    """

    rendered: list[LLMMessage] = []
    history_groups = _group_blocks_by_turn(blocks)

    for block in blocks:
        if block.kind in {"summary", "task_state", "retrieved_memory"}:
            rendered.extend(block.to_llm_messages())

    for group_blocks in history_groups.values():
        rendered.extend(_render_history_group(group_blocks))

    return rendered


def render_context_blocks_to_history_dicts(blocks: list[ContextBlock]) -> list[dict[str, Any]]:
    return [message.to_history_dict() for message in render_context_blocks_to_messages(blocks) if message.role != "system"]


def group_context_blocks(blocks: list[ContextBlock]) -> list[list[ContextBlock]]:
    return list(_group_blocks_by_turn(blocks).values())


@dataclass(slots=True)
class ThreadState:
    """In-memory view of one session after loading persisted state.

    `context_blocks` is the canonical stored history. `active_blocks` and
    `overflow_blocks` are the compaction split used by the prompt stack:
    active blocks are eligible for immediate replay, overflow blocks are kept in
    storage and may be summarized.
    """

    thread_id: str
    context_blocks: list[ContextBlock] = field(default_factory=list)
    summary: SessionSummary | None = None
    task_state: TaskState | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    active_blocks: list[ContextBlock] = field(default_factory=list)
    overflow_blocks: list[ContextBlock] = field(default_factory=list)

    @classmethod
    def from_session_state(cls, state: dict[str, Any], *, thread_id: str) -> "ThreadState":
        """Normalize persisted session payloads into runtime objects."""

        raw_context_blocks = state.get("context_blocks", [])
        context_blocks = (
            [block for item in raw_context_blocks if (block := ContextBlock.from_dict(item)) is not None]
            if isinstance(raw_context_blocks, list)
            else []
        )

        last_block_id = context_blocks[-1].block_id if context_blocks else ""
        summary = SessionSummary.from_any(state.get("summary"), thread_id=thread_id, covers_blocks_until=last_block_id)
        task_state = TaskState.from_any(state.get("task_state"))

        meta = state.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}

        active_block_ids = _normalize_block_ids(state.get("active_block_ids"))
        overflow_block_ids = _normalize_block_ids(state.get("overflow_block_ids"))
        block_index = {block.block_id: block for block in context_blocks}
        active_blocks = [block_index[block_id] for block_id in active_block_ids if block_id in block_index]
        overflow_blocks = [block_index[block_id] for block_id in overflow_block_ids if block_id in block_index]

        return cls(
            thread_id=thread_id,
            context_blocks=context_blocks,
            summary=summary,
            task_state=task_state,
            meta=meta,
            active_blocks=active_blocks,
            overflow_blocks=overflow_blocks,
        )


def _render_history_group(group_blocks: list[ContextBlock]) -> list[LLMMessage]:
    """Replay one turn-sized group in the order expected by the provider."""

    conversation_block: ContextBlock | None = None
    exchanges: list[tuple[int, int, ContextBlock]] = []
    fallback: list[LLMMessage] = []

    for order, block in enumerate(group_blocks):
        if block.kind == "conversation_turn" and conversation_block is None:
            conversation_block = block
            continue
        if block.kind == "tool_exchange":
            exchange_index = block.metadata.get("exchange_index")
            exchanges.append((exchange_index if isinstance(exchange_index, int) else order, order, block))
            continue
        fallback.extend(block.to_llm_messages())

    rendered: list[LLMMessage] = []
    if conversation_block is not None:
        user_message = conversation_block.content.get("user_message")
        if isinstance(user_message, dict):
            rendered.append(LLMMessage.from_history_dict(user_message))

    for _, _, exchange in sorted(exchanges, key=lambda item: (item[0], item[1])):
        rendered.extend(exchange.to_llm_messages())

    if conversation_block is not None:
        assistant_message = conversation_block.content.get("assistant_message")
        if isinstance(assistant_message, dict):
            rendered.append(LLMMessage.from_history_dict(assistant_message))

    rendered.extend(fallback)
    return rendered


def _group_blocks_by_turn(blocks: list[ContextBlock]) -> "OrderedDict[tuple[str, str], list[ContextBlock]]":
    """Group persisted blocks so a turn can be replayed atomically."""

    groups: "OrderedDict[tuple[str, str], list[ContextBlock]]" = OrderedDict()
    for block in blocks:
        if block.kind in {"summary", "task_state", "retrieved_memory"}:
            continue
        turn_index = block.metadata.get("turn_index")
        if isinstance(turn_index, int):
            groups.setdefault(("turn", str(turn_index)), []).append(block)
        else:
            groups.setdefault(("block", block.block_id), []).append(block)
    return groups


def _normalize_block_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [block_id for block_id in value if isinstance(block_id, str) and block_id]


def create_conversation_turn_block(
    *,
    turn_index: int,
    user_message: dict[str, Any] | None,
    assistant_message: dict[str, Any] | None,
) -> ContextBlock:
    """Persist the final user/assistant exchange for one turn as one block."""

    content = {
        "user_message": user_message,
        "assistant_message": assistant_message,
    }
    return ContextBlock(
        block_id=f"turn-{turn_index:04d}-conversation",
        kind="conversation_turn",
        content=content,
        token_estimate=estimate_token_count(content),
        metadata={"turn_index": turn_index},
    )


def create_tool_exchange_block(
    *,
    turn_index: int,
    exchange_index: int,
    assistant_message: dict[str, Any] | None,
    tool_messages: list[dict[str, Any]],
    orphan: bool = False,
) -> ContextBlock:
    """Persist one assistant tool-call step plus its tool responses."""

    content = {
        "assistant_message": assistant_message,
        "tool_messages": tool_messages,
    }
    metadata = {"turn_index": turn_index, "exchange_index": exchange_index}
    if orphan:
        metadata["orphan"] = True
    return ContextBlock(
        block_id=f"turn-{turn_index:04d}-exchange-{exchange_index:02d}",
        kind="tool_exchange",
        content=content,
        token_estimate=estimate_token_count(content),
        metadata=metadata,
    )
