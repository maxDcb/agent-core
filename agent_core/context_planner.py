from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from agent_core.llm.base import LLMCallOptions, LLMMessage
from agent_core.llm.errors import LLMProviderError

LLMContextMode = Literal["observe", "enforce"]


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    normalized = _non_negative_int(value, field_name=field_name)
    if normalized == 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return normalized


@dataclass(frozen=True, slots=True)
class LLMContextPolicy:
    """Provider context-window policy applied immediately before each LLM call."""

    max_context_tokens: int
    reserved_output_tokens: int = 1024
    safety_margin_tokens: int = 128
    mode: LLMContextMode = "enforce"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_context_tokens",
            _positive_int(self.max_context_tokens, field_name="max_context_tokens"),
        )
        object.__setattr__(
            self,
            "reserved_output_tokens",
            _positive_int(self.reserved_output_tokens, field_name="reserved_output_tokens"),
        )
        object.__setattr__(
            self,
            "safety_margin_tokens",
            _non_negative_int(self.safety_margin_tokens, field_name="safety_margin_tokens"),
        )
        if self.mode not in {"observe", "enforce"}:
            raise ValueError(f"Unsupported LLM context mode: {self.mode}")
        if self.reserved_output_tokens + self.safety_margin_tokens >= self.max_context_tokens:
            raise ValueError(
                "reserved_output_tokens plus safety_margin_tokens must be smaller than max_context_tokens"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "mode": self.mode,
        }

    @classmethod
    def from_any(cls, payload: object) -> LLMContextPolicy | None:
        if payload is None:
            return None
        if isinstance(payload, LLMContextPolicy):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("LLM context policy must be an LLMContextPolicy, dictionary, or None")
        return cls(
            max_context_tokens=cast(int, payload.get("max_context_tokens")),
            reserved_output_tokens=payload.get("reserved_output_tokens", 1024),
            safety_margin_tokens=payload.get("safety_margin_tokens", 128),
            mode=payload.get("mode", "enforce"),
        )


@dataclass(frozen=True, slots=True)
class LLMContextPlan:
    purpose: str
    original_input_tokens: int
    planned_input_tokens: int
    max_input_tokens: int
    reserved_output_tokens: int
    original_message_count: int
    planned_message_count: int
    removed_message_count: int
    removed_group_count: int
    fits: bool
    compacted: bool

    @property
    def removed_input_tokens(self) -> int:
        return max(0, self.original_input_tokens - self.planned_input_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "original_input_tokens": self.original_input_tokens,
            "planned_input_tokens": self.planned_input_tokens,
            "max_input_tokens": self.max_input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "original_message_count": self.original_message_count,
            "planned_message_count": self.planned_message_count,
            "removed_message_count": self.removed_message_count,
            "removed_group_count": self.removed_group_count,
            "removed_input_tokens": self.removed_input_tokens,
            "fits": self.fits,
            "compacted": self.compacted,
        }


@dataclass(slots=True)
class LLMContextUsage:
    plans_created: int = 0
    calls_compacted: int = 0
    calls_overflowed: int = 0
    messages_removed: int = 0
    groups_removed: int = 0
    estimated_input_tokens_removed: int = 0
    largest_original_input_tokens: int = 0
    largest_planned_input_tokens: int = 0
    plans_by_purpose: dict[str, int] = field(default_factory=dict)
    last_plan: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plans_created": self.plans_created,
            "calls_compacted": self.calls_compacted,
            "calls_overflowed": self.calls_overflowed,
            "messages_removed": self.messages_removed,
            "groups_removed": self.groups_removed,
            "estimated_input_tokens_removed": self.estimated_input_tokens_removed,
            "largest_original_input_tokens": self.largest_original_input_tokens,
            "largest_planned_input_tokens": self.largest_planned_input_tokens,
            "plans_by_purpose": dict(self.plans_by_purpose),
            "last_plan": dict(self.last_plan) if self.last_plan is not None else None,
        }

    @classmethod
    def from_any(cls, payload: object) -> LLMContextUsage:
        if isinstance(payload, LLMContextUsage):
            return cls(
                plans_created=payload.plans_created,
                calls_compacted=payload.calls_compacted,
                calls_overflowed=payload.calls_overflowed,
                messages_removed=payload.messages_removed,
                groups_removed=payload.groups_removed,
                estimated_input_tokens_removed=payload.estimated_input_tokens_removed,
                largest_original_input_tokens=payload.largest_original_input_tokens,
                largest_planned_input_tokens=payload.largest_planned_input_tokens,
                plans_by_purpose=dict(payload.plans_by_purpose),
                last_plan=dict(payload.last_plan) if payload.last_plan is not None else None,
            )
        if not isinstance(payload, dict):
            return cls()

        def count(key: str) -> int:
            value = payload.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

        raw_by_purpose = payload.get("plans_by_purpose")
        by_purpose = (
            {
                str(key): value
                for key, value in raw_by_purpose.items()
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            }
            if isinstance(raw_by_purpose, dict)
            else {}
        )
        raw_last_plan = payload.get("last_plan")
        return cls(
            plans_created=count("plans_created"),
            calls_compacted=count("calls_compacted"),
            calls_overflowed=count("calls_overflowed"),
            messages_removed=count("messages_removed"),
            groups_removed=count("groups_removed"),
            estimated_input_tokens_removed=count("estimated_input_tokens_removed"),
            largest_original_input_tokens=count("largest_original_input_tokens"),
            largest_planned_input_tokens=count("largest_planned_input_tokens"),
            plans_by_purpose=by_purpose,
            last_plan=dict(raw_last_plan) if isinstance(raw_last_plan, dict) else None,
        )


class LLMContextOverflowError(LLMProviderError):
    def __init__(self, *, detail: str, context_metadata: dict[str, Any]) -> None:
        self.context_metadata = dict(context_metadata)
        super().__init__(
            kind="context_overflow",
            user_message="The mandatory LLM context does not fit inside the configured context window.",
            detail=detail,
        )


@dataclass(slots=True)
class _MessageGroup:
    index: int
    messages: list[LLMMessage]
    mandatory: bool = False


class LLMContextPlanner:
    """Build a deterministic, provider-safe context plan for every model call."""

    def __init__(
        self,
        policy: LLMContextPolicy,
        *,
        usage: LLMContextUsage | dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.usage = LLMContextUsage.from_any(usage)

    def plan_call(
        self,
        *,
        messages: list[LLMMessage],
        tools: object,
        purpose: str,
        options: LLMCallOptions | None,
    ) -> tuple[list[LLMMessage], LLMCallOptions | None, LLMContextPlan]:
        purpose = purpose.strip() or "unspecified"
        output_reserve = (
            options.max_output_tokens
            if options is not None and options.max_output_tokens is not None
            else self.policy.reserved_output_tokens
        )
        output_reserve = max(1, output_reserve)
        max_input_tokens = self.policy.max_context_tokens - output_reserve - self.policy.safety_margin_tokens
        original_tokens = estimate_llm_input_tokens(messages=messages, tools=tools, options=options)

        if original_tokens <= max_input_tokens or self.policy.mode == "observe":
            plan = LLMContextPlan(
                purpose=purpose,
                original_input_tokens=original_tokens,
                planned_input_tokens=original_tokens,
                max_input_tokens=max_input_tokens,
                reserved_output_tokens=output_reserve,
                original_message_count=len(messages),
                planned_message_count=len(messages),
                removed_message_count=0,
                removed_group_count=0,
                fits=original_tokens <= max_input_tokens,
                compacted=False,
            )
            self._record(plan)
            return list(messages), self._effective_options(options, output_reserve), plan

        groups = _group_messages(messages)
        mandatory_indices = {group.index for group in groups if group.mandatory}
        selected_indices = set(mandatory_indices)
        mandatory_messages = _flatten_selected(groups, selected_indices)
        mandatory_tokens = estimate_llm_input_tokens(
            messages=mandatory_messages,
            tools=tools,
            options=options,
        )
        if mandatory_tokens > max_input_tokens:
            plan = LLMContextPlan(
                purpose=purpose,
                original_input_tokens=original_tokens,
                planned_input_tokens=mandatory_tokens,
                max_input_tokens=max_input_tokens,
                reserved_output_tokens=output_reserve,
                original_message_count=len(messages),
                planned_message_count=len(mandatory_messages),
                removed_message_count=len(messages) - len(mandatory_messages),
                removed_group_count=len(groups) - len(selected_indices),
                fits=False,
                compacted=len(mandatory_messages) != len(messages),
            )
            self._record(plan)
            raise LLMContextOverflowError(
                detail=(
                    "Mandatory context exceeds the input window: "
                    f"required={mandatory_tokens}, maximum={max_input_tokens}, purpose={purpose}"
                ),
                context_metadata=self.to_metadata(),
            )

        optional_groups = [group for group in groups if group.index not in mandatory_indices]
        for group in reversed(optional_groups):
            candidate_indices = {*selected_indices, group.index}
            candidate_messages = _flatten_selected(groups, candidate_indices)
            candidate_tokens = estimate_llm_input_tokens(
                messages=candidate_messages,
                tools=tools,
                options=options,
            )
            if candidate_tokens > max_input_tokens:
                break
            selected_indices.add(group.index)

        planned_messages = _flatten_selected(groups, selected_indices)
        planned_tokens = estimate_llm_input_tokens(messages=planned_messages, tools=tools, options=options)
        plan = LLMContextPlan(
            purpose=purpose,
            original_input_tokens=original_tokens,
            planned_input_tokens=planned_tokens,
            max_input_tokens=max_input_tokens,
            reserved_output_tokens=output_reserve,
            original_message_count=len(messages),
            planned_message_count=len(planned_messages),
            removed_message_count=len(messages) - len(planned_messages),
            removed_group_count=len(groups) - len(selected_indices),
            fits=planned_tokens <= max_input_tokens,
            compacted=len(planned_messages) != len(messages),
        )
        self._record(plan)
        return planned_messages, self._effective_options(options, output_reserve), plan

    def can_plan_call(
        self,
        *,
        messages: list[LLMMessage],
        tools: object,
        purpose: str,
        options: LLMCallOptions | None,
    ) -> bool:
        """Return whether a call can fit without mutating context-usage telemetry."""
        if self.policy.mode != "enforce":
            return True
        probe = LLMContextPlanner(self.policy)
        try:
            _, _, plan = probe.plan_call(
                messages=messages,
                tools=tools,
                purpose=purpose,
                options=options,
            )
        except LLMContextOverflowError:
            return False
        return plan.fits

    def to_metadata(self) -> dict[str, Any]:
        return {
            "llm_context_policy": self.policy.to_dict(),
            "llm_context_usage": self.usage.to_dict(),
        }

    def _effective_options(
        self,
        options: LLMCallOptions | None,
        output_reserve: int,
    ) -> LLMCallOptions | None:
        if self.policy.mode != "enforce":
            return options
        effective = replace(options) if options is not None else LLMCallOptions()
        effective.max_output_tokens = output_reserve
        return effective

    def _record(self, plan: LLMContextPlan) -> None:
        self.usage.plans_created += 1
        self.usage.plans_by_purpose[plan.purpose] = self.usage.plans_by_purpose.get(plan.purpose, 0) + 1
        self.usage.largest_original_input_tokens = max(
            self.usage.largest_original_input_tokens,
            plan.original_input_tokens,
        )
        self.usage.largest_planned_input_tokens = max(
            self.usage.largest_planned_input_tokens,
            plan.planned_input_tokens,
        )
        if plan.compacted:
            self.usage.calls_compacted += 1
            self.usage.messages_removed += plan.removed_message_count
            self.usage.groups_removed += plan.removed_group_count
            self.usage.estimated_input_tokens_removed += plan.removed_input_tokens
        if not plan.fits:
            self.usage.calls_overflowed += 1
        self.usage.last_plan = plan.to_dict()


def _group_messages(messages: list[LLMMessage]) -> list[_MessageGroup]:
    groups: list[_MessageGroup] = []
    last_user_index = max(
        (index for index, message in enumerate(messages) if message.role == "user"),
        default=-1,
    )
    index = 0
    while index < len(messages):
        message = messages[index]
        group_messages = [message]
        if (
            message.role == "user"
            and index + 1 < len(messages)
            and messages[index + 1].role == "assistant"
            and not messages[index + 1].tool_calls
        ):
            group_messages.append(messages[index + 1])
            index += 2
        elif message.role == "assistant" and message.tool_calls:
            expected_ids = {tool_call.id for tool_call in message.tool_calls}
            cursor = index + 1
            while cursor < len(messages):
                candidate = messages[cursor]
                if candidate.role != "tool" or candidate.tool_call_id not in expected_ids:
                    break
                group_messages.append(candidate)
                expected_ids.discard(candidate.tool_call_id)
                cursor += 1
            index = cursor
        else:
            index += 1
        group_index = len(groups)
        groups.append(
            _MessageGroup(
                index=group_index,
                messages=group_messages,
                mandatory=message.role == "system" or (last_user_index >= 0 and index > last_user_index),
            )
        )
    if last_user_index < 0:
        for group in reversed(groups):
            if any(message.role != "system" for message in group.messages):
                group.mandatory = True
                break
    return groups


def _flatten_selected(groups: list[_MessageGroup], selected_indices: set[int]) -> list[LLMMessage]:
    return [message for group in groups if group.index in selected_indices for message in group.messages]


def estimate_llm_input_tokens(
    *,
    messages: list[LLMMessage],
    tools: object = None,
    options: LLMCallOptions | None = None,
) -> int:
    payload: dict[str, Any] = {"messages": [message.to_history_dict() for message in messages]}
    if tools:
        payload["tools"] = tools
    if options is not None and options.response_format:
        payload["response_format"] = options.response_format
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return max(1, (len(raw) + 3) // 4)


_ACTIVE_LLM_CONTEXT_PLANNER: ContextVar[LLMContextPlanner | None] = ContextVar(
    "active_llm_context_planner",
    default=None,
)


@contextmanager
def llm_context_scope(planner: LLMContextPlanner | None) -> Iterator[LLMContextPlanner | None]:
    token = _ACTIVE_LLM_CONTEXT_PLANNER.set(planner)
    try:
        yield planner
    finally:
        _ACTIVE_LLM_CONTEXT_PLANNER.reset(token)


def active_llm_context_planner() -> LLMContextPlanner | None:
    return _ACTIVE_LLM_CONTEXT_PLANNER.get()
