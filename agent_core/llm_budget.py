from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeVar

from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage, LLMTokenUsage
from agent_core.llm.errors import LLMProviderError

LLMBudgetMode = Literal["observe", "enforce"]
LLMBudgetDimension = Literal["calls", "input_tokens", "output_tokens", "total_tokens", "duration"]


def _optional_non_negative_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer or None")
    return value


def _optional_positive_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{field_name} must be a positive number or None")
    return float(value)


@dataclass(frozen=True, slots=True)
class LLMBudget:
    """Run-level limits shared by every LLM call made inside one execution."""

    max_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_duration_seconds: float | None = None
    mode: LLMBudgetMode = "enforce"

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_calls", _optional_non_negative_int(self.max_calls, field_name="max_calls"))
        object.__setattr__(
            self,
            "max_input_tokens",
            _optional_non_negative_int(self.max_input_tokens, field_name="max_input_tokens"),
        )
        object.__setattr__(
            self,
            "max_output_tokens",
            _optional_non_negative_int(self.max_output_tokens, field_name="max_output_tokens"),
        )
        object.__setattr__(
            self,
            "max_total_tokens",
            _optional_non_negative_int(self.max_total_tokens, field_name="max_total_tokens"),
        )
        object.__setattr__(
            self,
            "max_duration_seconds",
            _optional_positive_float(self.max_duration_seconds, field_name="max_duration_seconds"),
        )
        if self.mode not in {"observe", "enforce"}:
            raise ValueError(f"Unsupported LLM budget mode: {self.mode}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_calls": self.max_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_duration_seconds": self.max_duration_seconds,
            "mode": self.mode,
        }

    @classmethod
    def from_any(cls, payload: object) -> LLMBudget | None:
        if payload is None:
            return None
        if isinstance(payload, LLMBudget):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("LLM budget must be an LLMBudget, dictionary, or None")
        return cls(
            max_calls=payload.get("max_calls"),
            max_input_tokens=payload.get("max_input_tokens"),
            max_output_tokens=payload.get("max_output_tokens"),
            max_total_tokens=payload.get("max_total_tokens"),
            max_duration_seconds=payload.get("max_duration_seconds"),
            mode=payload.get("mode", "enforce"),
        )


@dataclass(slots=True)
class LLMBudgetUsage:
    calls_started: int = 0
    calls_completed: int = 0
    calls_failed: int = 0
    calls_rejected: int = 0
    calls_with_token_usage: int = 0
    accounted_input_tokens: int = 0
    accounted_output_tokens: int = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0
    duration_seconds: float = 0.0
    exhausted_dimension: LLMBudgetDimension | None = None
    observed_violations: list[LLMBudgetDimension] = field(default_factory=list)
    calls_by_purpose: dict[str, int] = field(default_factory=dict)
    rejected_calls_by_purpose: dict[str, int] = field(default_factory=dict)

    @property
    def accounted_total_tokens(self) -> int:
        return self.accounted_input_tokens + self.accounted_output_tokens

    @property
    def token_usage_complete(self) -> bool:
        return self.calls_completed == self.calls_with_token_usage and self.calls_failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls_started": self.calls_started,
            "calls_completed": self.calls_completed,
            "calls_failed": self.calls_failed,
            "calls_rejected": self.calls_rejected,
            "calls_with_token_usage": self.calls_with_token_usage,
            "token_usage_complete": self.token_usage_complete,
            "accounted_input_tokens": self.accounted_input_tokens,
            "accounted_output_tokens": self.accounted_output_tokens,
            "accounted_total_tokens": self.accounted_total_tokens,
            "reported_input_tokens": self.reported_input_tokens,
            "reported_output_tokens": self.reported_output_tokens,
            "duration_seconds": round(self.duration_seconds, 3),
            "exhausted_dimension": self.exhausted_dimension,
            "observed_violations": list(self.observed_violations),
            "calls_by_purpose": dict(self.calls_by_purpose),
            "rejected_calls_by_purpose": dict(self.rejected_calls_by_purpose),
        }

    @classmethod
    def from_any(cls, payload: object) -> LLMBudgetUsage:
        if isinstance(payload, LLMBudgetUsage):
            return cls(
                calls_started=payload.calls_started,
                calls_completed=payload.calls_completed,
                calls_failed=payload.calls_failed,
                calls_rejected=payload.calls_rejected,
                calls_with_token_usage=payload.calls_with_token_usage,
                accounted_input_tokens=payload.accounted_input_tokens,
                accounted_output_tokens=payload.accounted_output_tokens,
                reported_input_tokens=payload.reported_input_tokens,
                reported_output_tokens=payload.reported_output_tokens,
                duration_seconds=payload.duration_seconds,
                exhausted_dimension=payload.exhausted_dimension,
                observed_violations=list(payload.observed_violations),
                calls_by_purpose=dict(payload.calls_by_purpose),
                rejected_calls_by_purpose=dict(payload.rejected_calls_by_purpose),
            )
        if not isinstance(payload, dict):
            return cls()

        def non_negative_int(key: str) -> int:
            value = payload.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

        raw_duration = payload.get("duration_seconds")
        raw_exhausted = payload.get("exhausted_dimension")
        raw_violations = payload.get("observed_violations")
        raw_calls_by_purpose = payload.get("calls_by_purpose")
        raw_rejected_by_purpose = payload.get("rejected_calls_by_purpose")
        valid_dimensions = {"calls", "input_tokens", "output_tokens", "total_tokens", "duration"}

        def purpose_counts(value: object) -> dict[str, int]:
            if not isinstance(value, dict):
                return {}
            return {
                str(key): count
                for key, count in value.items()
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0
            }

        return cls(
            calls_started=non_negative_int("calls_started"),
            calls_completed=non_negative_int("calls_completed"),
            calls_failed=non_negative_int("calls_failed"),
            calls_rejected=non_negative_int("calls_rejected"),
            calls_with_token_usage=non_negative_int("calls_with_token_usage"),
            accounted_input_tokens=non_negative_int("accounted_input_tokens"),
            accounted_output_tokens=non_negative_int("accounted_output_tokens"),
            reported_input_tokens=non_negative_int("reported_input_tokens"),
            reported_output_tokens=non_negative_int("reported_output_tokens"),
            duration_seconds=float(raw_duration) if isinstance(raw_duration, (int, float)) and raw_duration >= 0 else 0.0,
            exhausted_dimension=raw_exhausted if raw_exhausted in valid_dimensions else None,
            observed_violations=[item for item in raw_violations if item in valid_dimensions]
            if isinstance(raw_violations, list)
            else [],
            calls_by_purpose=purpose_counts(raw_calls_by_purpose),
            rejected_calls_by_purpose=purpose_counts(raw_rejected_by_purpose),
        )


class LLMBudgetExceededError(LLMProviderError):
    def __init__(
        self,
        *,
        dimension: LLMBudgetDimension,
        detail: str,
        budget_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.dimension = dimension
        self.budget_metadata = dict(budget_metadata or {})
        super().__init__(
            kind="budget_exhausted",
            user_message=f"The run stopped because its LLM {dimension.replace('_', ' ')} budget was exhausted.",
            detail=detail,
        )


@dataclass(slots=True)
class _CallReservation:
    purpose: str
    estimated_input_tokens: int
    reserved_output_tokens: int
    started_at: float


class LLMBudgetController:
    """Account for and optionally enforce one budget across nested LLM calls."""

    def __init__(self, budget: LLMBudget, *, usage: LLMBudgetUsage | dict[str, Any] | None = None) -> None:
        self.budget = budget
        self.usage = LLMBudgetUsage.from_any(usage)

    def prepare_call(
        self,
        *,
        purpose: str,
        estimated_input_tokens: int,
        options: LLMCallOptions | None,
    ) -> tuple[_CallReservation, LLMCallOptions | None]:
        purpose = purpose.strip() or "unspecified"
        estimated_input_tokens = max(1, estimated_input_tokens)
        self._check_limit(
            dimension="duration",
            proposed=self.usage.duration_seconds,
            maximum=self.budget.max_duration_seconds,
            purpose=purpose,
        )
        self._check_limit(
            dimension="calls",
            proposed=self.usage.calls_started + 1,
            maximum=self.budget.max_calls,
            purpose=purpose,
        )
        self._check_limit(
            dimension="input_tokens",
            proposed=self.usage.accounted_input_tokens + estimated_input_tokens,
            maximum=self.budget.max_input_tokens,
            purpose=purpose,
        )
        self._check_limit(
            dimension="total_tokens",
            proposed=self.usage.accounted_total_tokens + estimated_input_tokens,
            maximum=self.budget.max_total_tokens,
            purpose=purpose,
        )

        effective_options = replace(options) if options is not None else LLMCallOptions()
        output_caps = [cap for cap in [effective_options.max_output_tokens] if cap is not None]
        if self.budget.max_output_tokens is not None:
            output_caps.append(max(0, self.budget.max_output_tokens - self.usage.accounted_output_tokens))
        if self.budget.max_total_tokens is not None:
            output_caps.append(
                max(
                    0,
                    self.budget.max_total_tokens
                    - self.usage.accounted_total_tokens
                    - estimated_input_tokens,
                )
            )

        reserved_output_tokens = min(output_caps) if output_caps else 0
        if output_caps:
            self._check_limit(
                dimension="output_tokens",
                proposed=1,
                maximum=reserved_output_tokens,
                purpose=purpose,
            )
            if self.budget.mode == "enforce":
                effective_options.max_output_tokens = reserved_output_tokens

        self.usage.calls_started += 1
        self.usage.calls_by_purpose[purpose] = self.usage.calls_by_purpose.get(purpose, 0) + 1
        self.usage.accounted_input_tokens += estimated_input_tokens
        self.usage.accounted_output_tokens += reserved_output_tokens
        return (
            _CallReservation(
                purpose=purpose,
                estimated_input_tokens=estimated_input_tokens,
                reserved_output_tokens=reserved_output_tokens,
                started_at=time.monotonic(),
            ),
            effective_options if options is not None or output_caps else None,
        )

    def complete_call(self, reservation: _CallReservation, usage: LLMTokenUsage | None) -> None:
        self.usage.calls_completed += 1
        self.usage.duration_seconds += max(0.0, time.monotonic() - reservation.started_at)
        if usage is not None:
            self.usage.calls_with_token_usage += 1
            self.usage.reported_input_tokens += usage.input_tokens
            self.usage.reported_output_tokens += usage.output_tokens
            self.usage.accounted_input_tokens += usage.input_tokens - reservation.estimated_input_tokens
            self.usage.accounted_output_tokens += usage.output_tokens - reservation.reserved_output_tokens
        self._observe_current_usage()

    def fail_call(self, reservation: _CallReservation) -> None:
        self.usage.calls_failed += 1
        self.usage.duration_seconds += max(0.0, time.monotonic() - reservation.started_at)
        self._observe_current_usage()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "llm_budget": self.budget.to_dict(),
            "llm_budget_usage": self.usage.to_dict(),
        }

    def _check_limit(
        self,
        *,
        dimension: LLMBudgetDimension,
        proposed: int | float,
        maximum: int | float | None,
        purpose: str,
    ) -> None:
        if maximum is None or proposed <= maximum:
            return
        self._record_violation(dimension)
        if self.budget.mode == "enforce":
            self.usage.calls_rejected += 1
            self.usage.rejected_calls_by_purpose[purpose] = (
                self.usage.rejected_calls_by_purpose.get(purpose, 0) + 1
            )
            self.usage.exhausted_dimension = dimension
            raise LLMBudgetExceededError(
                dimension=dimension,
                detail=f"LLM budget exhausted for {dimension}: proposed={proposed}, maximum={maximum}",
                budget_metadata=self.to_metadata(),
            )

    def _observe_current_usage(self) -> None:
        limits: tuple[tuple[LLMBudgetDimension, int | float, int | float | None], ...] = (
            ("calls", self.usage.calls_started, self.budget.max_calls),
            ("input_tokens", self.usage.accounted_input_tokens, self.budget.max_input_tokens),
            ("output_tokens", self.usage.accounted_output_tokens, self.budget.max_output_tokens),
            ("total_tokens", self.usage.accounted_total_tokens, self.budget.max_total_tokens),
            ("duration", self.usage.duration_seconds, self.budget.max_duration_seconds),
        )
        for dimension, consumed, maximum in limits:
            if maximum is not None and consumed > maximum:
                self._record_violation(dimension)
                if self.usage.exhausted_dimension is None:
                    self.usage.exhausted_dimension = dimension

    def _record_violation(self, dimension: LLMBudgetDimension) -> None:
        if dimension not in self.usage.observed_violations:
            self.usage.observed_violations.append(dimension)


_ACTIVE_LLM_BUDGET: ContextVar[LLMBudgetController | None] = ContextVar("active_llm_budget", default=None)


@contextmanager
def llm_budget_scope(controller: LLMBudgetController | None) -> Iterator[LLMBudgetController | None]:
    token = _ACTIVE_LLM_BUDGET.set(controller)
    try:
        yield controller
    finally:
        _ACTIVE_LLM_BUDGET.reset(token)


def active_llm_budget_controller() -> LLMBudgetController | None:
    return _ACTIVE_LLM_BUDGET.get()


def estimate_llm_input_tokens(*, messages: list[LLMMessage], tools: object = None) -> int:
    payload: dict[str, Any] = {"messages": [message.to_history_dict() for message in messages]}
    if tools:
        payload["tools"] = tools
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return max(1, (len(raw) + 3) // 4)


T = TypeVar("T")


def run_budgeted_llm_call(
    *,
    messages: list[LLMMessage],
    tools: object = None,
    purpose: str,
    options: LLMCallOptions | None,
    invoke: Callable[[LLMCallOptions | None], T],
    on_reserved: Callable[[], None] | None = None,
) -> T:
    controller = active_llm_budget_controller()
    if controller is None:
        return invoke(options)

    reservation, effective_options = controller.prepare_call(
        purpose=purpose,
        estimated_input_tokens=estimate_llm_input_tokens(messages=messages, tools=tools),
        options=options,
    )
    try:
        if on_reserved is not None:
            on_reserved()
        result = invoke(effective_options)
    except Exception:
        controller.fail_call(reservation)
        raise
    completion_usage = result.usage if isinstance(result, LLMCompletionResult) else None
    controller.complete_call(reservation, completion_usage)
    return result
