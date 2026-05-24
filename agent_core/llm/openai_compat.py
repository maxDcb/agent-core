from __future__ import annotations

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import os
import random
import time
from typing import Any

from openai import BadRequestError, RateLimitError

from agent_core.llm.openai_request_policy import OpenAIModelCapabilityResolver, select_bad_request_retry_action


@dataclass(frozen=True, slots=True)
class OpenAIRateLimitRetryPolicy:
    max_attempts: int = 5
    initial_delay_seconds: float = 10.0
    max_delay_seconds: float = 120.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.1

    @classmethod
    def from_env(cls, prefix: str = "AGENT_CORE_LLM_RETRY_") -> "OpenAIRateLimitRetryPolicy":
        return cls(
            max_attempts=_env_int(f"{prefix}MAX_ATTEMPTS", cls.max_attempts),
            initial_delay_seconds=_env_float(f"{prefix}INITIAL_DELAY_SECONDS", cls.initial_delay_seconds),
            max_delay_seconds=_env_float(f"{prefix}MAX_DELAY_SECONDS", cls.max_delay_seconds),
            backoff_multiplier=_env_float(f"{prefix}BACKOFF_MULTIPLIER", cls.backoff_multiplier),
            jitter_ratio=_env_float(f"{prefix}JITTER_RATIO", cls.jitter_ratio),
        )

    def retry_delay_seconds(self, *, exc: RateLimitError, attempt_index: int, random_value: float) -> float:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(self.max_delay_seconds, max(0.0, retry_after))

        base_delay = self.initial_delay_seconds * (self.backoff_multiplier ** max(attempt_index - 1, 0))
        capped_delay = min(self.max_delay_seconds, max(0.0, base_delay))
        if self.jitter_ratio <= 0 or capped_delay <= 0:
            return capped_delay

        jitter_window = capped_delay * self.jitter_ratio
        jitter = (random_value * 2.0 - 1.0) * jitter_window
        return max(0.0, min(self.max_delay_seconds, capped_delay + jitter))


def create_chat_completion_with_adaptive_retry(
    *,
    completions: Any,
    request: dict[str, Any],
    provider_name: str,
    logger: Any,
    capability_resolver: OpenAIModelCapabilityResolver | None = None,
    rate_limit_policy: OpenAIRateLimitRetryPolicy | None = None,
    sleeper: Any = time.sleep,
    random_fn: Any = random.random,
) -> Any:
    fallback_request = dict(request)
    retried_without: set[str] = set()
    policy = rate_limit_policy or OpenAIRateLimitRetryPolicy.from_env()
    attempt = 1

    while True:
        try:
            return completions.create(**fallback_request)
        except RateLimitError as exc:
            if attempt >= policy.max_attempts:
                raise

            delay_seconds = policy.retry_delay_seconds(
                exc=exc,
                attempt_index=attempt,
                random_value=float(random_fn()),
            )
            logger.warning(
                "%s rate limited chat completion request; retrying",
                provider_name,
                extra={
                    "model": fallback_request.get("model"),
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "delay_seconds": round(delay_seconds, 3),
                },
            )
            sleeper(delay_seconds)
            attempt += 1
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


def _retry_after_seconds(exc: RateLimitError) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        return max(0.0, float(retry_after))
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError):
        return None
    return max(0.0, parsed.timestamp() - time.time())


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default
