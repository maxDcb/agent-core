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
    try:
        return completions.create(**request)
    except BadRequestError as exc:
        if "reasoning_effort" not in request or not _is_unsupported_reasoning_effort_error(exc):
            raise

        logger.warning(
            "%s rejected reasoning_effort; retrying without it",
            provider_name,
            extra={"model": request.get("model")},
        )
        fallback_request = dict(request)
        fallback_request.pop("reasoning_effort", None)
        return completions.create(**fallback_request)


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
