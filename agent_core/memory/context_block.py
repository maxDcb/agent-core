from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_core.llm.base import LLMMessage

ContextBlockKind = Literal[
    "conversation_turn",
    "tool_exchange",
    "summary",
    "task_state",
    "retrieved_memory",
]


def estimate_token_count(value: Any) -> int:
    """Use a simple heuristic instead of a tokenizer dependency."""

    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        raw = str(value)
    return max(1, (len(raw) + 3) // 4)


def _string_or_default(value: object, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _normalize_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


@dataclass(slots=True)
class ContextBlock:
    """Atomic persisted unit used to rebuild the prompt stack later.

    A block is richer than a flat chat message: one block can represent a full
    user/assistant turn, a tool-exchange phase, or synthesized memory injected
    back into the prompt as a system message.
    """

    block_id: str
    kind: ContextBlockKind
    content: dict[str, Any]
    token_estimate: int
    pinned: bool = False
    priority: int = 0
    source: str = "runtime"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "kind": self.kind,
            "content": self.content,
            "token_estimate": self.token_estimate,
            "pinned": self.pinned,
            "priority": self.priority,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ContextBlock | None":
        if not isinstance(payload, dict):
            return None

        kind = payload.get("kind")
        if kind not in {"conversation_turn", "tool_exchange", "summary", "task_state", "retrieved_memory"}:
            return None

        content = payload.get("content")
        if not isinstance(content, dict):
            content = {}

        token_estimate = payload.get("token_estimate")
        if not isinstance(token_estimate, int) or token_estimate <= 0:
            token_estimate = estimate_token_count(content)

        return cls(
            block_id=_string_or_default(payload.get("block_id"), "block"),
            kind=kind,
            content=content,
            token_estimate=token_estimate,
            pinned=bool(payload.get("pinned", False)),
            priority=payload.get("priority", 0) if isinstance(payload.get("priority"), int) else 0,
            source=_string_or_default(payload.get("source"), "runtime"),
            metadata=_normalize_metadata(payload.get("metadata")),
        )

    def render_system_text(self) -> str:
        if self.kind == "summary":
            from agent_core.memory.session_summary import SessionSummary

            summary = SessionSummary.from_any(self.content.get("summary"))
            return summary.render_text() if summary is not None else ""

        if self.kind == "task_state":
            from agent_core.memory.task_state import TaskState

            task_state = TaskState.from_any(self.content.get("task_state"))
            return task_state.render_text() if task_state is not None else ""

        if self.kind == "retrieved_memory":
            return self._render_retrieved_memory_text()

        return ""

    def to_llm_messages(self) -> list[LLMMessage]:
        """Render a single block into provider-facing flat messages.

        Sequence-level rendering still lives outside the block because
        `conversation_turn` intentionally groups a user message and the final
        assistant answer, while `tool_exchange` stores the intermediate
        assistant/tool loop. Rendering a full turn in chronological order needs
        both block types to be considered together.
        """

        if self.kind == "conversation_turn":
            return self._render_conversation_turn()
        if self.kind == "tool_exchange":
            return self._render_tool_exchange()

        system_text = self.render_system_text()
        if not system_text:
            return []
        return [LLMMessage(role="system", content=system_text)]

    def message_count(self) -> int:
        return len(self.to_llm_messages())

    def _render_conversation_turn(self) -> list[LLMMessage]:
        # Stored turns keep the final assistant answer attached to the user
        # prompt. Intermediate tool steps live in separate tool-exchange blocks.
        rendered: list[LLMMessage] = []
        user_payload = self.content.get("user_message")
        assistant_payload = self.content.get("assistant_message")
        if isinstance(user_payload, dict):
            rendered.append(LLMMessage.from_history_dict(user_payload))
        if isinstance(assistant_payload, dict):
            rendered.append(LLMMessage.from_history_dict(assistant_payload))
        return rendered

    def _render_tool_exchange(self) -> list[LLMMessage]:
        # Replay the assistant tool-call message first, then the matching tool
        # outputs, so the provider sees the same call/response shape it expects.
        rendered: list[LLMMessage] = []
        assistant_payload = self.content.get("assistant_message")
        if isinstance(assistant_payload, dict):
            rendered.append(LLMMessage.from_history_dict(assistant_payload))

        tool_payloads = self.content.get("tool_messages", [])
        if isinstance(tool_payloads, list):
            rendered.extend(
                LLMMessage.from_history_dict(payload)
                for payload in tool_payloads
                if isinstance(payload, dict)
            )
        return rendered

    def _render_retrieved_memory_text(self) -> str:
        text = self.content.get("text")
        items = self.content.get("items")

        lines = ["Retrieved memory:"]
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())

        if isinstance(items, list):
            for item in items:
                if isinstance(item, str) and item.strip():
                    lines.append(f"- {item.strip()}")
                elif isinstance(item, dict):
                    label = item.get("label") or item.get("title") or "memory"
                    value = item.get("value") or item.get("content") or item
                    lines.append(f"- {label}: {value}")

        return "\n".join(lines) if len(lines) > 1 else ""
