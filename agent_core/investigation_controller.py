from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from agent_core.execution_context import ExecutionContext
from agent_core.investigation_models import FinalCritique, InvestigationDecision, StepReflection
from agent_core.investigation_prompts import (
    DEFAULT_INVESTIGATION_PROMPTS,
    InvestigationPromptSet,
)
from agent_core.investigation_state import InvestigationState
from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage
from agent_core.llm.errors import LLMProviderError
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.output_contracts import parse_json_object, render_json_object
from agent_core.run_options import RunOptions
from agent_core.settings import CoreSettings
from agent_core.structured_synthesizer import StructuredSynthesisRequest, StructuredSynthesizer
from agent_core.turn_steps import PendingResumeState, ToolExecutionStepResult
from agent_core.types import AgentTurnResult

logger = get_logger(__name__)


class ModelCaller(Protocol):
    def __call__(
        self,
        *,
        messages: list[LLMMessage],
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        ...


class FinalModelCaller(Protocol):
    def __call__(
        self,
        *,
        messages: list[LLMMessage],
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        ...


class ToolExecutor(Protocol):
    def __call__(
        self,
        *,
        user_input: str,
        session_id: str,
        context: ExecutionContext,
        messages: list[LLMMessage],
        turn_index: int,
        exchange_index: int,
        tool_calls_used: int,
        assistant_message: LLMMessage,
        max_tool_calls: int,
        pending_metadata_extra: dict[str, Any] | None = None,
    ) -> ToolExecutionStepResult:
        ...


class ConversationPersister(Protocol):
    def __call__(self, *, turn_index: int, user_input: str, assistant_content: str) -> None:
        ...


class MemoryRefresher(Protocol):
    def __call__(self, *, turn_index: int) -> None:
        ...


class ProviderFailureHandler(Protocol):
    def __call__(self, *, error: LLMProviderError, user_input: str, turn_index: int) -> AgentTurnResult:
        ...


class TraceRecorder(Protocol):
    def __call__(
        self,
        *,
        event_type: str,
        summary: str,
        iteration: int | None = None,
        payload: dict[str, Any] | None = None,
        related_tool_call_id: str | None = None,
    ) -> None:
        ...


INVESTIGATION_STATE_MESSAGE_PREFIX = "Investigation controller state:"


def with_investigation_guidance(
    messages: list[LLMMessage],
    *,
    options: RunOptions,
    prompt_set: InvestigationPromptSet | None = None,
) -> list[LLMMessage]:
    if any(
        message.role == "system" and message.content.startswith(f"Run mode: {options.mode}.")
        for message in messages
    ):
        return list(messages)

    prompts = prompt_set or DEFAULT_INVESTIGATION_PROMPTS
    guidance = LLMMessage(
        role="system",
        content=prompts.render_run_guidance(mode=options.mode),
    )
    if messages and messages[-1].role == "user":
        return [*messages[:-1], guidance, messages[-1]]
    return [*messages, guidance]

class InvestigationController:
    """Bounded, domain-agnostic investigation loop.

    The controller stores only structured artifacts and delegates all tool
    execution to orchestrator callbacks so policy checks and transcript
    persistence stay on the same path as direct mode.
    """

    def __init__(
        self,
        *,
        settings: CoreSettings,
        structured_synthesizer: StructuredSynthesizer,
        call_model_once: ModelCaller,
        call_final_model_once: FinalModelCaller,
        execute_tool_calls_once: ToolExecutor,
        persist_conversation_turn_once: ConversationPersister,
        refresh_memory_after_turn: MemoryRefresher,
        handle_provider_failure: ProviderFailureHandler,
        record_event: TraceRecorder | None = None,
        prompt_set: InvestigationPromptSet | None = None,
    ) -> None:
        self.settings = settings
        self.structured_synthesizer = structured_synthesizer
        self.call_model_once = call_model_once
        self.call_final_model_once = call_final_model_once
        self.execute_tool_calls_once = execute_tool_calls_once
        self.persist_conversation_turn_once = persist_conversation_turn_once
        self.refresh_memory_after_turn = refresh_memory_after_turn
        self.handle_provider_failure = handle_provider_failure
        self.record_event = record_event
        self.prompt_set = prompt_set or DEFAULT_INVESTIGATION_PROMPTS

    def _record_event(
        self,
        *,
        event_type: str,
        summary: str,
        iteration: int | None = None,
        payload: dict[str, Any] | None = None,
        related_tool_call_id: str | None = None,
    ) -> None:
        if self.record_event is None:
            return
        self.record_event(
            event_type=event_type,
            summary=summary,
            iteration=iteration,
            payload=payload,
            related_tool_call_id=related_tool_call_id,
        )

    def run(
        self,
        *,
        user_input: str,
        session_id: str,
        context: ExecutionContext,
        messages: list[LLMMessage],
        turn_index: int,
        options: RunOptions,
    ) -> AgentTurnResult:
        state = InvestigationState.create_template(objective=user_input)
        if options.require_initial_plan:
            self._record_event(
                event_type="initial_plan_started",
                summary="Initial investigation plan synthesis started",
                payload={"mode": options.mode},
            )
            try:
                state = self._synthesize_initial_plan(user_input=user_input, state=state, options=options)
            except LLMProviderError as exc:
                self._record_event(
                    event_type="llm_provider_failure",
                    summary="Initial plan provider failure handled",
                    payload={"kind": exc.kind},
                )
                failure_result = self.handle_provider_failure(error=exc, user_input=user_input, turn_index=turn_index)
                return self._attach_metadata(
                    failure_result,
                    options=options,
                    iterations_used=0,
                    tool_calls_used=0,
                    stop_reason="provider_failure",
                    state=state,
                )
            except ValueError as exc:
                if not options.recover_internal_synthesis_errors:
                    raise
                logger.warning(
                    "Initial investigation plan synthesis failed; continuing with template state",
                    extra={"error_preview": safe_preview(str(exc), limit=200)},
                )
                state.metadata["initial_plan_synthesis_error"] = safe_preview(str(exc), limit=200)
                self._record_event(
                    event_type="structured_synthesis_recovered",
                    summary="Initial plan synthesis failed; continuing with template investigation state",
                    payload={
                        "target": "investigation_initial_plan",
                        "error_preview": state.metadata["initial_plan_synthesis_error"],
                    },
                )
            else:
                self._record_event(
                    event_type="initial_plan_created",
                    summary="Initial investigation plan created",
                    payload={"investigation_state": state.compact_summary()},
                )

        return self._run_loop(
            user_input=user_input,
            session_id=session_id,
            context=context,
            messages=messages,
            turn_index=turn_index,
            options=options,
            state=state,
            iterations_used=0,
            tool_calls_used=0,
            exchange_index=0,
            no_progress_iterations=0,
        )

    def resume_after_pending(
        self,
        *,
        pending: PendingResumeState,
        session_id: str,
        context: ExecutionContext,
        options: RunOptions,
        state: InvestigationState,
        iterations_used: int,
        no_progress_iterations: int,
        tool_step: ToolExecutionStepResult | None = None,
    ) -> AgentTurnResult:
        completed_tool_step = tool_step or ToolExecutionStepResult(
            messages=pending.messages,
            tool_messages=pending.tool_messages,
            exchange_index=pending.exchange_index,
            tool_calls_used=pending.tool_calls_used,
            tool_statuses=pending.tool_statuses or [pending.tool_status],
            tool_names=pending.tool_names or [str(pending.pending_payload.get("tool_name") or "unknown")],
        )
        result, no_progress_iterations = self._reflect_and_decide_after_tools(
            user_input=pending.user_input,
            turn_index=pending.turn_index,
            options=options,
            state=state,
            messages=completed_tool_step.messages,
            tool_step=completed_tool_step,
            iterations_used=iterations_used,
            no_progress_iterations=no_progress_iterations,
        )
        if result is not None:
            return result

        return self._run_loop(
            user_input=pending.user_input,
            session_id=session_id,
            context=context,
            messages=completed_tool_step.messages,
            turn_index=pending.turn_index,
            options=options,
            state=state,
            iterations_used=iterations_used,
            tool_calls_used=completed_tool_step.tool_calls_used,
            exchange_index=completed_tool_step.exchange_index,
            no_progress_iterations=no_progress_iterations,
        )

    def _run_loop(
        self,
        *,
        user_input: str,
        session_id: str,
        context: ExecutionContext,
        messages: list[LLMMessage],
        turn_index: int,
        options: RunOptions,
        state: InvestigationState,
        iterations_used: int,
        tool_calls_used: int,
        exchange_index: int,
        no_progress_iterations: int,
    ) -> AgentTurnResult:
        while iterations_used < options.max_iterations:
            iterations_used += 1
            self._record_event(
                event_type="investigation_iteration_started",
                summary="Investigation iteration started",
                iteration=iterations_used,
                payload={
                    "tool_calls_used": tool_calls_used,
                    "max_iterations": options.max_iterations,
                    "max_tool_calls": options.max_tool_calls,
                },
            )
            try:
                assistant_messages = self._messages_with_iteration_state(
                    messages=messages,
                    state=state,
                    iteration=iterations_used,
                )
                llm_response = self.call_model_once(
                    messages=assistant_messages,
                    options=self._call_options(options=options, target="assistant_step"),
                )
            except LLMProviderError as exc:
                self._record_event(
                    event_type="llm_provider_failure",
                    summary="Assistant step provider failure handled",
                    iteration=iterations_used,
                    payload={"kind": exc.kind},
                )
                failure_result = self.handle_provider_failure(error=exc, user_input=user_input, turn_index=turn_index)
                return self._attach_metadata(
                    failure_result,
                    options=options,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_calls_used,
                    stop_reason="provider_failure",
                    state=state,
                )

            assistant_message = LLMMessage(
                role="assistant",
                content=llm_response.content,
                tool_calls=list(llm_response.tool_calls),
            )
            messages.append(assistant_message)
            self._record_event(
                event_type="assistant_step_completed",
                summary="Assistant investigation step completed",
                iteration=iterations_used,
                payload={
                    "content_length": len(llm_response.content),
                    "tool_call_count": len(llm_response.tool_calls),
                    "tool_calls": [
                        {"id": tool_call.id, "name": tool_call.name}
                        for tool_call in llm_response.tool_calls
                    ],
                },
            )

            if not llm_response.tool_calls:
                self._record_event(
                    event_type="final_draft_received",
                    summary="Assistant produced a final draft",
                    iteration=iterations_used,
                    payload={"content_length": len(llm_response.content)},
                )
                return self._handle_final_draft(
                    user_input=user_input,
                    session_id=session_id,
                    context=context,
                    messages=messages,
                    turn_index=turn_index,
                    options=options,
                    state=state,
                    final_draft=llm_response.content,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_calls_used,
                    exchange_index=exchange_index,
                    no_progress_iterations=no_progress_iterations,
                )

            if tool_calls_used >= options.max_tool_calls:
                return self._complete_with_budget_answer(
                    user_input=user_input,
                    turn_index=turn_index,
                    options=options,
                    state=state,
                    messages=messages,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_calls_used,
                    stop_reason="max_tool_calls",
                )

            tool_step = self.execute_tool_calls_once(
                user_input=user_input,
                session_id=session_id,
                context=context,
                messages=messages,
                turn_index=turn_index,
                exchange_index=exchange_index,
                tool_calls_used=tool_calls_used,
                assistant_message=assistant_message,
                max_tool_calls=options.max_tool_calls,
                pending_metadata_extra={
                    "mode": options.mode,
                    "run_options": asdict(options),
                    "investigation_state": state.to_dict(),
                    "iterations_used": iterations_used,
                    "no_progress_iterations": no_progress_iterations,
                },
            )
            messages = tool_step.messages
            exchange_index = tool_step.exchange_index
            tool_calls_used = tool_step.tool_calls_used
            self._record_event(
                event_type="tool_step_completed",
                summary="Investigation tool step completed",
                iteration=iterations_used,
                payload={
                    "tool_names": list(tool_step.tool_names),
                    "tool_statuses": list(tool_step.tool_statuses),
                    "tool_calls_used": tool_calls_used,
                    "budget_exhausted": tool_step.budget_exhausted,
                    "pending": tool_step.pending_result is not None,
                },
            )

            if tool_step.pending_result is not None:
                return self._attach_metadata(
                    tool_step.pending_result,
                    options=options,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_calls_used,
                    stop_reason="pending_tool_result",
                    state=state,
                )

            if tool_step.budget_exhausted:
                return self._complete_with_budget_answer(
                    user_input=user_input,
                    turn_index=turn_index,
                    options=options,
                    state=state,
                    messages=messages,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_calls_used,
                    stop_reason="max_tool_calls",
                )

            terminal_result, no_progress_iterations = self._reflect_and_decide_after_tools(
                user_input=user_input,
                turn_index=turn_index,
                options=options,
                state=state,
                messages=messages,
                tool_step=tool_step,
                iterations_used=iterations_used,
                no_progress_iterations=no_progress_iterations,
            )
            if terminal_result is not None:
                return terminal_result

        return self._complete_with_budget_answer(
            user_input=user_input,
            turn_index=turn_index,
            options=options,
            state=state,
            messages=messages,
            iterations_used=iterations_used,
            tool_calls_used=tool_calls_used,
            stop_reason="max_iterations",
        )

    def _reflect_and_decide_after_tools(
        self,
        *,
        user_input: str,
        turn_index: int,
        options: RunOptions,
        state: InvestigationState,
        messages: list[LLMMessage],
        tool_step: ToolExecutionStepResult,
        iterations_used: int,
        no_progress_iterations: int,
    ) -> tuple[AgentTurnResult | None, int]:
        previous_fingerprint = state.progress_fingerprint()
        try:
            reflection = self._synthesize_reflection(state=state, tool_step=tool_step, options=options)
        except ValueError as exc:
            if not options.recover_internal_synthesis_errors:
                raise
            logger.warning(
                "Investigation reflection synthesis failed; continuing without state update",
                extra={"error_preview": safe_preview(str(exc), limit=200)},
            )
            self._record_event(
                event_type="structured_synthesis_recovered",
                summary="Reflection synthesis failed; continuing the conversation without state update",
                iteration=iterations_used,
                payload={
                    "target": "investigation_step_reflection",
                    "error_preview": safe_preview(str(exc), limit=200),
                },
            )
            return None, no_progress_iterations + 1
        state.apply_reflection(reflection)
        if state.progress_fingerprint() == previous_fingerprint:
            no_progress_iterations += 1
        else:
            no_progress_iterations = 0
        self._record_event(
            event_type="reflection_completed",
            summary="Investigation reflection completed",
            iteration=iterations_used,
            payload={
                "reflection": reflection.to_dict(),
                "no_progress_iterations": no_progress_iterations,
                "investigation_state": state.compact_summary(),
            },
        )

        try:
            decision = self._synthesize_decision(
                state=state,
                reflection=reflection,
                options=options,
                iterations_used=iterations_used,
                tool_calls_used=tool_step.tool_calls_used,
            )
        except ValueError as exc:
            if not options.recover_internal_synthesis_errors:
                raise
            logger.warning(
                "Investigation decision synthesis failed; falling back to reflection state",
                extra={"error_preview": safe_preview(str(exc), limit=200)},
            )
            self._record_event(
                event_type="structured_synthesis_recovered",
                summary="Decision synthesis failed; using reflection state as fallback",
                iteration=iterations_used,
                payload={
                    "target": "investigation_decision",
                    "error_preview": safe_preview(str(exc), limit=200),
                },
            )
            if reflection.should_continue and tool_step.tool_calls_used < options.max_tool_calls:
                return None, no_progress_iterations
            return (
                self._complete_turn(
                    user_input=user_input,
                    turn_index=turn_index,
                    options=options,
                    state=state,
                    content=self._answer_from_state(state=state, final=not reflection.should_continue),
                    messages=messages,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_step.tool_calls_used,
                    stop_reason=reflection.stop_reason or "decision_synthesis_unavailable",
                ),
                no_progress_iterations,
            )
        self._record_event(
            event_type="decision_completed",
            summary="Investigation decision completed",
            iteration=iterations_used,
            payload={"decision": decision.to_dict()},
        )

        if decision.kind == "ask_user":
            question = decision.question or decision.reason_summary
            return (
                self._complete_turn(
                    user_input=user_input,
                    turn_index=turn_index,
                    options=options,
                    state=state,
                    content=question,
                    messages=messages,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_step.tool_calls_used,
                    stop_reason="ask_user",
                ),
                no_progress_iterations,
            )

        if decision.kind == "blocked":
            return (
                self._complete_turn(
                    user_input=user_input,
                    turn_index=turn_index,
                    options=options,
                    state=state,
                    content=f"Investigation blocked: {decision.reason_summary}",
                    messages=messages,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_step.tool_calls_used,
                    stop_reason="blocked",
                ),
                no_progress_iterations,
            )

        if decision.kind == "final" and reflection.should_continue:
            self._record_event(
                event_type="decision_overridden",
                summary="Final decision overridden because reflection still requires continuation",
                iteration=iterations_used,
                payload={
                    "decision": decision.to_dict(),
                    "reflection": reflection.to_dict(),
                },
            )
        elif decision.kind == "final" or (
            not reflection.should_continue and state.confidence >= options.min_confidence_to_answer
        ):
            return (
                self._complete_turn(
                    user_input=user_input,
                    turn_index=turn_index,
                    options=options,
                    state=state,
                    content=self._answer_from_state(state=state, final=True),
                    messages=messages,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_step.tool_calls_used,
                    stop_reason=decision.reason_summary or reflection.stop_reason or "final",
                ),
                no_progress_iterations,
            )

        if tool_step.tool_calls_used >= options.max_tool_calls:
            return (
                self._complete_with_budget_answer(
                    user_input=user_input,
                    turn_index=turn_index,
                    options=options,
                    state=state,
                    messages=messages,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_step.tool_calls_used,
                    stop_reason="max_tool_calls",
                ),
                no_progress_iterations,
            )

        if no_progress_iterations > 0 and no_progress_iterations >= options.max_no_progress_iterations:
            return (
                self._complete_with_budget_answer(
                    user_input=user_input,
                    turn_index=turn_index,
                    options=options,
                    state=state,
                    messages=messages,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_step.tool_calls_used,
                    stop_reason="no_progress",
                ),
                no_progress_iterations,
            )

        return None, no_progress_iterations

    def _handle_final_draft(
        self,
        *,
        user_input: str,
        session_id: str,
        context: ExecutionContext,
        messages: list[LLMMessage],
        turn_index: int,
        options: RunOptions,
        state: InvestigationState,
        final_draft: str,
        iterations_used: int,
        tool_calls_used: int,
        exchange_index: int,
        no_progress_iterations: int,
    ) -> AgentTurnResult:
        if not options.require_final_critique:
            return self._complete_turn(
                user_input=user_input,
                turn_index=turn_index,
                options=options,
                state=state,
                content=final_draft,
                messages=messages,
                iterations_used=iterations_used,
                tool_calls_used=tool_calls_used,
                stop_reason="final",
            )

        self._record_event(
            event_type="final_critique_started",
            summary="Final critique synthesis started",
            iteration=iterations_used,
            payload={"draft_length": len(final_draft)},
        )
        try:
            critique = self._synthesize_final_critique(state=state, final_draft=final_draft, options=options)
        except ValueError as exc:
            if not options.recover_internal_synthesis_errors:
                raise
            logger.warning(
                "Final critique synthesis failed; returning assistant draft",
                extra={"error_preview": safe_preview(str(exc), limit=200)},
            )
            self._record_event(
                event_type="structured_synthesis_recovered",
                summary="Final critique synthesis failed; returning assistant draft",
                iteration=iterations_used,
                payload={
                    "target": "investigation_final_critique",
                    "error_preview": safe_preview(str(exc), limit=200),
                },
            )
            return self._complete_turn(
                user_input=user_input,
                turn_index=turn_index,
                options=options,
                state=state,
                content=final_draft,
                messages=messages,
                iterations_used=iterations_used,
                tool_calls_used=tool_calls_used,
                stop_reason="final_critique_unavailable",
            )
        self._record_event(
            event_type="final_critique_completed",
            summary="Final critique completed",
            iteration=iterations_used,
            payload={"critique": critique.to_dict()},
        )
        if critique.approved:
            return self._complete_turn(
                user_input=user_input,
                turn_index=turn_index,
                options=options,
                state=state,
                content=final_draft,
                messages=messages,
                iterations_used=iterations_used,
                tool_calls_used=tool_calls_used,
                stop_reason="final_critique_approved",
            )

        state.apply_critique(critique)
        if iterations_used >= options.max_iterations:
            return self._complete_with_budget_answer(
                user_input=user_input,
                turn_index=turn_index,
                options=options,
                state=state,
                messages=messages,
                iterations_used=iterations_used,
                tool_calls_used=tool_calls_used,
                stop_reason="final_critique_rejected",
            )

        logger.info(
            "Final critique rejected the draft; continuing investigation",
            extra={"unsupported_claim_count": len(critique.unsupported_claims)},
        )
        return self._run_loop(
            user_input=user_input,
            session_id=session_id,
            context=context,
            messages=messages,
            turn_index=turn_index,
            options=options,
            state=state,
            iterations_used=iterations_used,
            tool_calls_used=tool_calls_used,
            exchange_index=exchange_index,
            no_progress_iterations=no_progress_iterations,
        )

    def _complete_turn(
        self,
        *,
        user_input: str,
        turn_index: int,
        options: RunOptions,
        state: InvestigationState,
        content: str,
        messages: list[LLMMessage],
        iterations_used: int,
        tool_calls_used: int,
        stop_reason: str,
    ) -> AgentTurnResult:
        state.stop_reason = stop_reason
        final_content = content
        final_metadata: dict[str, Any] = {}
        if options.final_output_mode == "json_schema":
            final_content, final_metadata = self._render_final_json_schema(
                content=content,
                messages=messages,
                options=options,
                state=state,
                stop_reason=stop_reason,
            )
        self._record_event(
            event_type="investigation_completed",
            summary="Investigation completed",
            payload={
                "stop_reason": stop_reason,
                "iterations_used": iterations_used,
                "tool_calls_used": tool_calls_used,
                "investigation_state": state.compact_summary(),
            },
        )
        self.persist_conversation_turn_once(
            turn_index=turn_index,
            user_input=user_input,
            assistant_content=final_content,
        )
        self.refresh_memory_after_turn(turn_index=turn_index)
        return AgentTurnResult(
            status="completed",
            content=final_content,
            metadata={
                **self._metadata(
                    options=options,
                    iterations_used=iterations_used,
                    tool_calls_used=tool_calls_used,
                    stop_reason=stop_reason,
                    state=state,
                ),
                **final_metadata,
            },
        )

    def _render_final_json_schema(
        self,
        *,
        content: str,
        messages: list[LLMMessage],
        options: RunOptions,
        state: InvestigationState,
        stop_reason: str,
    ) -> tuple[str, dict[str, Any]]:
        contract = options.final_output_contract
        if contract is None:
            raise ValueError("json_schema final output requires a final output contract")

        payload = {
            "candidate_final_answer": content,
            "stop_reason": stop_reason,
            "investigation_state": state.to_dict(),
            "contract_name": contract.name,
            "contract_instructions": list(contract.instructions),
        }
        instruction_lines = [
            "Formalize the final answer as one JSON object matching the provider-enforced JSON Schema.",
            "Use only the candidate answer and investigation state already present in this transcript.",
            "Do not request tools. Do not add prose, markdown fences, comments, or a second JSON object.",
            "Preserve uncertainty from the investigation state instead of inventing missing facts.",
        ]
        if contract.instructions:
            instruction_lines.append("Contract-specific instructions:")
            instruction_lines.extend(f"- {instruction}" for instruction in contract.instructions)

        final_messages = [
            *messages,
            LLMMessage(role="system", content="\n".join(instruction_lines)),
            LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ]
        llm_response = self.call_final_model_once(
            messages=final_messages,
            options=LLMCallOptions(
                reasoning_effort=options.reasoning_effort,
                reasoning_summary=options.reasoning_summary,
                response_format=contract.response_format(),
                max_output_tokens=self.settings.llm_max_output_tokens,
                metadata={
                    "mode": options.mode,
                    "target": "final_output_json_schema",
                    "final_output_contract": contract.name,
                    **options.metadata,
                },
            ),
        )
        if llm_response.tool_calls:
            raise LLMProviderError(
                kind="response_error",
                user_message="The final JSON Schema renderer unexpectedly requested tools.",
                detail=f"tool_call_count={len(llm_response.tool_calls)}",
            )
        try:
            rendered_payload = parse_json_object(llm_response.content, target_name="final_output_json_schema")
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMProviderError(
                kind="response_error",
                user_message="The final JSON Schema renderer returned invalid JSON.",
                detail=str(exc),
            ) from exc

        return render_json_object(rendered_payload), {
            "final_output_mode": "json_schema",
            "final_output_contract": contract.name,
        }

    def _complete_with_budget_answer(
        self,
        *,
        user_input: str,
        turn_index: int,
        options: RunOptions,
        state: InvestigationState,
        messages: list[LLMMessage],
        iterations_used: int,
        tool_calls_used: int,
        stop_reason: str,
    ) -> AgentTurnResult:
        state.stop_reason = stop_reason
        return self._complete_turn(
            user_input=user_input,
            turn_index=turn_index,
            options=options,
            state=state,
            content=self._answer_from_state(state=state, final=False),
            messages=messages,
            iterations_used=iterations_used,
            tool_calls_used=tool_calls_used,
            stop_reason=stop_reason,
        )

    def _synthesize_initial_plan(
        self,
        *,
        user_input: str,
        state: InvestigationState,
        options: RunOptions,
    ) -> InvestigationState:
        return self.structured_synthesizer.synthesize(
            request=StructuredSynthesisRequest(
                target_name="investigation_initial_plan",
                instructions=self.prompt_set.initial_plan,
                output_format=InvestigationState.create_template(objective=user_input).to_dict(),
                payload={"objective": user_input, "current_state": state.to_dict()},
                parser=InvestigationState.from_any,
                options=self._call_options(options=options, target="investigation_initial_plan"),
            )
        )

    def _synthesize_reflection(
        self,
        *,
        state: InvestigationState,
        tool_step: ToolExecutionStepResult,
        options: RunOptions,
    ) -> StepReflection:
        payload = {
            "current_state": state.to_dict(),
            "tool_results": [
                {
                    "role": message.role,
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
                for message in tool_step.tool_messages
            ],
            "tool_statuses": tool_step.tool_statuses,
            "tool_names": tool_step.tool_names,
        }
        return self.structured_synthesizer.synthesize(
            request=StructuredSynthesisRequest(
                target_name="investigation_step_reflection",
                instructions=self.prompt_set.step_reflection,
                output_format=StepReflection.create_template().to_dict(),
                payload=payload,
                parser=StepReflection.from_any,
                options=self._call_options(options=options, target="investigation_step_reflection"),
            )
        )

    def _synthesize_decision(
        self,
        *,
        state: InvestigationState,
        reflection: StepReflection,
        options: RunOptions,
        iterations_used: int,
        tool_calls_used: int,
    ) -> InvestigationDecision:
        payload = {
            "current_state": state.to_dict(),
            "latest_reflection": reflection.to_dict(),
            "budgets": {
                "iterations_used": iterations_used,
                "max_iterations": options.max_iterations,
                "tool_calls_used": tool_calls_used,
                "max_tool_calls": options.max_tool_calls,
                "min_confidence_to_answer": options.min_confidence_to_answer,
            },
        }
        return self.structured_synthesizer.synthesize(
            request=StructuredSynthesisRequest(
                target_name="investigation_decision",
                instructions=self.prompt_set.decision,
                output_format=InvestigationDecision.create_template().to_dict(),
                payload=payload,
                parser=InvestigationDecision.from_any,
                options=self._call_options(options=options, target="investigation_decision"),
            )
        )

    def _synthesize_final_critique(
        self,
        *,
        state: InvestigationState,
        final_draft: str,
        options: RunOptions,
    ) -> FinalCritique:
        return self.structured_synthesizer.synthesize(
            request=StructuredSynthesisRequest(
                target_name="investigation_final_critique",
                instructions=self.prompt_set.final_critique,
                output_format=FinalCritique.create_template().to_dict(),
                payload={"current_state": state.to_dict(), "final_draft": final_draft},
                parser=FinalCritique.from_any,
                options=self._call_options(options=options, target="investigation_final_critique"),
            )
        )

    def _answer_from_state(self, *, state: InvestigationState, final: bool) -> str:
        title = "Investigation complete." if final else "Investigation stopped before a complete answer."
        lines = [title]
        if state.stop_reason:
            lines.append(f"Stop reason: {state.stop_reason}.")
        lines.append("")
        lines.append("Established facts:")
        if state.facts:
            lines.extend(f"- {fact.summary}" for fact in state.facts)
        else:
            lines.append("- None established.")
        lines.append("")
        lines.append("Remaining uncertainty:")
        if state.evidence_gaps:
            lines.extend(f"- {gap}" for gap in state.evidence_gaps)
        else:
            lines.append("- None recorded.")
        lines.append("")
        lines.append("Recommended next steps:")
        if state.next_actions:
            lines.extend(f"- {action}" for action in state.next_actions)
        else:
            lines.append("- None recorded.")
        lines.append("")
        lines.append(f"Confidence: {state.confidence:.2f}")
        return "\n".join(lines)

    def _messages_with_iteration_state(
        self,
        *,
        messages: list[LLMMessage],
        state: InvestigationState,
        iteration: int,
    ) -> list[LLMMessage]:
        state_payload = {
            "iteration": iteration,
            "objective": state.objective,
            "plan": list(state.plan),
            "facts": [fact.to_dict() for fact in state.facts],
            "hypotheses": [hypothesis.to_dict() for hypothesis in state.hypotheses],
            "evidence_gaps": list(state.evidence_gaps),
            "completed_actions": list(state.completed_actions),
            "next_actions": list(state.next_actions),
            "risk_notes": list(state.risk_notes),
            "confidence": state.confidence,
            "stop_reason": state.stop_reason,
        }
        state_message = LLMMessage(
            role="system",
            content="\n".join(
                [
                    INVESTIGATION_STATE_MESSAGE_PREFIX,
                    "This is the controller's current auditable state, not new user evidence.",
                    "Use established facts, prioritize next actions, close evidence gaps, and respect risk notes.",
                    "Do not repeat completed actions unless verification requires it.",
                    json.dumps(state_payload, ensure_ascii=False, separators=(",", ":")),
                ]
            ),
        )
        if messages and messages[-1].role == "user":
            return [*messages[:-1], state_message, messages[-1]]
        return [*messages, state_message]

    def _call_options(self, *, options: RunOptions, target: str) -> LLMCallOptions:
        return LLMCallOptions(
            reasoning_effort=options.reasoning_effort,
            reasoning_summary=options.reasoning_summary,
            response_format={"type": "json_object"} if target != "assistant_step" else None,
            metadata={"mode": options.mode, "target": target, **options.metadata},
        )

    def _attach_metadata(
        self,
        result: AgentTurnResult,
        *,
        options: RunOptions,
        iterations_used: int,
        tool_calls_used: int,
        stop_reason: str,
        state: InvestigationState,
    ) -> AgentTurnResult:
        result.metadata = {
            **result.metadata,
            **self._metadata(
                options=options,
                iterations_used=iterations_used,
                tool_calls_used=tool_calls_used,
                stop_reason=stop_reason,
                state=state,
            ),
        }
        return result

    def _metadata(
        self,
        *,
        options: RunOptions,
        iterations_used: int,
        tool_calls_used: int,
        stop_reason: str,
        state: InvestigationState,
    ) -> dict[str, Any]:
        return {
            "mode": options.mode,
            "final_output_mode": options.final_output_mode,
            "iterations_used": iterations_used,
            "tool_calls_used": tool_calls_used,
            "stop_reason": stop_reason,
            "investigation_state": state.compact_summary(),
        }
