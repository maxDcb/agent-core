from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


LLMMessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class LLMToolCall:
    id: str
    name: str
    arguments_json: str

    @classmethod
    def from_history_dict(cls, payload: dict[str, Any]) -> "LLMToolCall | None":
        tool_call_id = payload.get("id")
        function_payload = payload.get("function")
        if not isinstance(tool_call_id, str) or not isinstance(function_payload, dict):
            return None

        tool_name = function_payload.get("name")
        arguments_json = function_payload.get("arguments", "{}")
        if not isinstance(tool_name, str):
            return None

        if not isinstance(arguments_json, str):
            arguments_json = "{}"

        return cls(id=tool_call_id, name=tool_name, arguments_json=arguments_json)

    def to_history_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }


@dataclass(slots=True)
class LLMMessage:
    role: LLMMessageRole
    content: str
    tool_call_id: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)

    @classmethod
    def from_history_dict(cls, payload: dict[str, Any]) -> "LLMMessage":
        role = str(payload.get("role", "user"))
        content = str(payload.get("content", ""))
        tool_calls = [
            tool_call
            for item in payload.get("tool_calls", []) or []
            if isinstance(item, dict)
            for tool_call in [LLMToolCall.from_history_dict(item)]
            if tool_call is not None
        ]
        tool_call_id = payload.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            tool_call_id = None
        return cls(
            role=role if role in {"system", "user", "assistant", "tool"} else "user",
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )

    def to_history_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.to_history_dict() for tool_call in self.tool_calls]
        return payload


@dataclass(slots=True)
class LLMToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class LLMCompletionResult:
    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)


class BaseLLMProvider(Protocol):
    def complete_text(
        self,
        *,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
    ) -> str:
        ...

    def complete_with_tools(
        self,
        *,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition],
        model: str,
        temperature: float,
    ) -> LLMCompletionResult:
        ...
