from __future__ import annotations

from typing import Any

from openai import BadRequestError


def create_chat_completion_with_reasoning_fallback(
    *,
    completions: Any,
    request: dict[str, Any],
    provider_name: str,
    logger: Any,
) -> Any:
    fallback_request = dict(request)
    retried_without: set[str] = set()

    while True:
        try:
            return completions.create(**fallback_request)
        except BadRequestError as exc:
            unsupported_parameter = _unsupported_fallback_parameter(exc, fallback_request)
            if unsupported_parameter is None or unsupported_parameter in retried_without:
                raise

            logger.warning(
                "%s rejected %s; retrying without it",
                provider_name,
                unsupported_parameter,
                extra={"model": fallback_request.get("model")},
            )
            fallback_request.pop(unsupported_parameter, None)
            retried_without.add(unsupported_parameter)


def _unsupported_fallback_parameter(exc: BadRequestError, request: dict[str, Any]) -> str | None:
    if "reasoning_effort" in request and _is_unsupported_reasoning_effort_error(exc):
        return "reasoning_effort"
    if "temperature" in request and _is_unsupported_temperature_error(exc):
        return "temperature"
    return None


def _is_unsupported_reasoning_effort_error(exc: BadRequestError) -> bool:
    message = str(exc).lower()
    return "reasoning_effort" in message and any(
        fragment in message
        for fragment in (
            "unrecognized",
            "unsupported",
            "unknown",
            "not supported",
            "invalid request argument",
        )
    )


def _is_unsupported_temperature_error(exc: BadRequestError) -> bool:
    message = str(exc).lower()
    return "temperature" in message and any(
        fragment in message
        for fragment in (
            "unsupported",
            "does not support",
            "only the default",
            "unsupported_value",
        )
    )
