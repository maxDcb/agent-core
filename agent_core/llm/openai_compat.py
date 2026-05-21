from __future__ import annotations

from typing import Any

from openai import BadRequestError

from agent_core.llm.openai_request_policy import OpenAIModelCapabilityResolver, select_bad_request_retry_action


def create_chat_completion_with_adaptive_retry(
    *,
    completions: Any,
    request: dict[str, Any],
    provider_name: str,
    logger: Any,
    capability_resolver: OpenAIModelCapabilityResolver | None = None,
) -> Any:
    fallback_request = dict(request)
    retried_without: set[str] = set()

    while True:
        try:
            return completions.create(**fallback_request)
        except BadRequestError as exc:
            retry_action = select_bad_request_retry_action(exc, fallback_request)
            if retry_action is None or retry_action.parameter in retried_without:
                raise

            if capability_resolver is not None:
                capability_resolver.record_unsupported_parameter(
                    model=str(fallback_request.get("model", "")),
                    parameter=retry_action.parameter,
                )

            if retry_action.replacement_parameter is not None and retry_action.replacement_parameter not in fallback_request:
                fallback_request[retry_action.replacement_parameter] = fallback_request[retry_action.parameter]

            logger.warning(
                "%s rejected %s; retrying with adjusted request",
                provider_name,
                retry_action.parameter,
                extra={"model": fallback_request.get("model")},
            )
            fallback_request.pop(retry_action.parameter, None)
            retried_without.add(retry_action.parameter)


def create_chat_completion_with_reasoning_fallback(
    *,
    completions: Any,
    request: dict[str, Any],
    provider_name: str,
    logger: Any,
) -> Any:
    return create_chat_completion_with_adaptive_retry(
        completions=completions,
        request=request,
        provider_name=provider_name,
        logger=logger,
    )
