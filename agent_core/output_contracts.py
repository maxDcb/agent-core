from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError

FinalOutputMode: TypeAlias = Literal["text", "json_schema"]
ValidationPhase: TypeAlias = Literal["schema", "payload"]
JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"


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


def _json_pointer(path: Iterable[object]) -> str:
    parts = list(path)
    if not parts:
        return ""
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _safe_validation_message(*, validator: object, phase: ValidationPhase) -> str:
    keyword = str(validator) if validator is not None else "unknown"
    if phase == "schema":
        return f"Schema is invalid for the '{keyword}' constraint."
    messages = {
        "required": "A required property is missing.",
        "additionalProperties": "The object contains properties that are not allowed.",
        "type": "The value has an invalid JSON type.",
        "enum": "The value is not one of the allowed enum values.",
        "const": "The value does not match the required constant.",
        "pattern": "The string does not match the required pattern.",
        "format": "The string does not match the required format.",
        "minimum": "The number is below the allowed minimum.",
        "maximum": "The number is above the allowed maximum.",
        "exclusiveMinimum": "The number is not above the exclusive minimum.",
        "exclusiveMaximum": "The number is not below the exclusive maximum.",
        "minLength": "The string is shorter than allowed.",
        "maxLength": "The string is longer than allowed.",
        "minItems": "The array contains too few items.",
        "maxItems": "The array contains too many items.",
        "uniqueItems": "The array contains duplicate items.",
        "minProperties": "The object contains too few properties.",
        "maxProperties": "The object contains too many properties.",
        "oneOf": "The value does not match exactly one allowed schema.",
        "anyOf": "The value does not match any allowed schema.",
        "allOf": "The value does not satisfy every required schema.",
        "not": "The value matches a forbidden schema.",
    }
    return messages.get(keyword, f"The value does not satisfy the '{keyword}' constraint.")


@dataclass(frozen=True, slots=True)
class StructuredOutputValidationIssue:
    validator: str
    instance_path: str
    schema_path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "validator": self.validator,
            "instance_path": self.instance_path,
            "schema_path": self.schema_path,
            "message": self.message,
        }


class StructuredOutputValidationError(ValueError):
    """Safe structured diagnostics for an invalid schema or output payload."""

    def __init__(
        self,
        *,
        contract_name: str,
        phase: ValidationPhase,
        issues: list[StructuredOutputValidationIssue],
    ) -> None:
        self.contract_name = contract_name
        self.phase = phase
        self.issues = tuple(issues)
        super().__init__(f"Structured output contract '{contract_name}' failed {phase} validation ({len(issues)} issue(s)).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "structured_output_validation_error",
            "contract_name": self.contract_name,
            "phase": self.phase,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _schema_issue(error: SchemaError) -> StructuredOutputValidationIssue:
    validator = str(error.validator) if error.validator is not None else "unknown"
    return StructuredOutputValidationIssue(
        validator=validator,
        instance_path=_json_pointer(error.path),
        schema_path=_json_pointer(error.schema_path),
        message=_safe_validation_message(validator=error.validator, phase="schema"),
    )


def _payload_issue(error: ValidationError) -> StructuredOutputValidationIssue:
    validator = str(error.validator) if error.validator is not None else "unknown"
    return StructuredOutputValidationIssue(
        validator=validator,
        instance_path=_json_pointer(error.absolute_path),
        schema_path=_json_pointer(error.absolute_schema_path),
        message=_safe_validation_message(validator=error.validator, phase="payload"),
    )


@dataclass(slots=True)
class StructuredOutputContract:
    """Provider-enforced and locally validated JSON Schema contract."""

    name: str
    schema: dict[str, Any]
    strict: bool = True
    instructions: list[str] = field(default_factory=list)
    _validator: Draft202012Validator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.name = _clean_contract_name(self.name)
        if not isinstance(self.schema, dict):
            raise TypeError("StructuredOutputContract.schema must be a JSON Schema object")
        self.schema = copy.deepcopy(self.schema)
        self.instructions = _clean_string_list(self.instructions)
        try:
            Draft202012Validator.check_schema(self.schema)
        except SchemaError as exc:
            raise StructuredOutputValidationError(
                contract_name=self.name,
                phase="schema",
                issues=[_schema_issue(exc)],
            ) from exc
        self._validator = Draft202012Validator(self.schema, format_checker=FormatChecker())

    @classmethod
    def from_any(cls, value: object) -> StructuredOutputContract | None:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return None
        raw_schema = value.get("schema")
        if not isinstance(raw_schema, dict):
            raise TypeError("StructuredOutputContract.schema must be a JSON Schema object")
        return cls(
            name=value.get("name") or "structured_output",
            schema=dict(raw_schema),
            strict=bool(value.get("strict", True)),
            instructions=_clean_string_list(value.get("instructions")),
        )

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": copy.deepcopy(self.schema),
                "strict": self.strict,
            },
        }

    def validate_payload(self, payload: object) -> None:
        errors = sorted(
            self._validator.iter_errors(payload),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                tuple(str(part) for part in error.absolute_schema_path),
                str(error.validator),
            ),
        )
        if errors:
            raise StructuredOutputValidationError(
                contract_name=self.name,
                phase="payload",
                issues=[_payload_issue(error) for error in errors],
            )


def parse_json_object(
    raw_content: str,
    *,
    target_name: str,
    contract: StructuredOutputContract | None = None,
) -> dict[str, Any]:
    payload = json.loads(raw_content)
    if not isinstance(payload, dict):
        raise ValueError(f"{target_name} returned a non-object JSON payload")
    if contract is not None:
        contract.validate_payload(payload)
    return payload


def render_json_object(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
