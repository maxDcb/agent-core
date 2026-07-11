from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

FinalOutputMode: TypeAlias = Literal["text", "json_schema"]


def _clean_string(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _clean_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        item = _clean_string(value)
        if item:
            cleaned.append(item)
    return cleaned


def _clean_contract_name(value: object) -> str:
    cleaned = _clean_string(value, default="structured_output")
    normalized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in cleaned)
    normalized = normalized.strip("_-") or "structured_output"
    return normalized[:64]


@dataclass(slots=True)
class StructuredOutputContract:
    """Provider-facing JSON Schema output contract.

    Contracts are hard requirements. If a provider cannot enforce the schema,
    the caller should receive a failure instead of a looser JSON response.
    """

    name: str
    schema: dict[str, Any]
    strict: bool = True
    instructions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = _clean_contract_name(self.name)
        if not isinstance(self.schema, dict):
            self.schema = {"type": "object", "additionalProperties": False}
        self.instructions = _clean_string_list(self.instructions)

    @classmethod
    def from_any(cls, value: object) -> StructuredOutputContract | None:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return None
        raw_schema = value.get("schema")
        return cls(
            name=value.get("name") or "structured_output",
            schema=dict(raw_schema) if isinstance(raw_schema, dict) else {},
            strict=bool(value.get("strict", True)),
            instructions=_clean_string_list(value.get("instructions")),
        )

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.schema,
                "strict": self.strict,
            },
        }


def parse_json_object(raw_content: str, *, target_name: str) -> dict[str, Any]:
    payload = json.loads(raw_content)
    if not isinstance(payload, dict):
        raise ValueError(f"{target_name} returned a non-object JSON payload")
    return payload


def render_json_object(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
