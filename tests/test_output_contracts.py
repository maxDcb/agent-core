from __future__ import annotations

import json

import pytest

from agent_core.output_contracts import (
    JSON_SCHEMA_DRAFT,
    StructuredOutputContract,
    StructuredOutputValidationError,
    parse_json_object,
)


def _contract() -> StructuredOutputContract:
    return StructuredOutputContract(
        name="security_result",
        schema={
            "$schema": JSON_SCHEMA_DRAFT,
            "type": "object",
            "required": ["status", "details"],
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["safe", "unsafe"]},
                "details": {
                    "type": "object",
                    "required": ["endpoint", "confidence"],
                    "additionalProperties": False,
                    "properties": {
                        "endpoint": {"type": "string", "format": "uri"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
        },
    )


def test_contract_rejects_invalid_schema_at_construction() -> None:
    with pytest.raises(StructuredOutputValidationError) as captured:
        StructuredOutputContract(
            name="invalid",
            schema={"type": "definitely-not-a-json-type"},
        )

    error = captured.value
    assert error.phase == "schema"
    assert error.contract_name == "invalid"
    assert error.issues[0].instance_path == "/type"
    assert error.to_dict()["issues"][0]["message"].startswith("Schema is invalid")


@pytest.mark.parametrize(
    ("payload", "validator", "instance_path"),
    [
        ({"details": {"endpoint": "https://example.test", "confidence": 0.5}}, "required", ""),
        (
            {"status": "unknown", "details": {"endpoint": "https://example.test", "confidence": 0.5}},
            "enum",
            "/status",
        ),
        (
            {
                "status": "safe",
                "details": {"endpoint": "https://example.test", "confidence": 0.5},
                "unexpected": True,
            },
            "additionalProperties",
            "",
        ),
        (
            {"status": "safe", "details": {"endpoint": "not a uri", "confidence": 0.5}},
            "format",
            "/details/endpoint",
        ),
        (
            {"status": "safe", "details": {"endpoint": "https://example.test", "confidence": 2}},
            "maximum",
            "/details/confidence",
        ),
    ],
)
def test_contract_returns_structured_nested_validation_issues(
    payload: dict,
    validator: str,
    instance_path: str,
) -> None:
    with pytest.raises(StructuredOutputValidationError) as captured:
        _contract().validate_payload(payload)

    matching = [issue for issue in captured.value.issues if issue.validator == validator]
    assert matching
    assert matching[0].instance_path == instance_path
    assert matching[0].schema_path.startswith("/")


def test_validation_diagnostics_never_include_sensitive_payload_values() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.super-secret-payload.signature"
    contract = StructuredOutputContract(
        name="secret_boundary",
        schema={
            "type": "object",
            "required": ["token"],
            "additionalProperties": False,
            "properties": {"token": {"type": "string", "enum": ["expected-placeholder"]}},
        },
    )

    with pytest.raises(StructuredOutputValidationError) as captured:
        contract.validate_payload({"token": jwt})

    serialized = json.dumps(captured.value.to_dict())
    assert jwt not in str(captured.value)
    assert jwt not in serialized
    assert "super-secret-payload" not in serialized


def test_parse_json_object_validates_provider_output_locally() -> None:
    valid = parse_json_object(
        json.dumps(
            {
                "status": "safe",
                "details": {"endpoint": "https://example.test/api", "confidence": 0.8},
            }
        ),
        target_name="provider_response",
        contract=_contract(),
    )

    assert valid["status"] == "safe"
