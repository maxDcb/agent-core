from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol

from agent_core.execution_context import ExecutionContext
from agent_core.investigation_models import FinalCritique, InvestigationDecision, StepReflection
from agent_core.investigation_prompts import (
    FINAL_CRITIQUE_PROMPT,
    INITIAL_PLAN_PROMPT,
    INVESTIGATION_DECISION_PROMPT,
    STEP_REFLECTION_PROMPT,
)
from agent_core.investigation_state import InvestigationState
from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage
from agent_core.llm.errors import LLMProviderError
from agent_core.logging_utils import get_logger
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


def with_investigation_guidance(messages: list[LLMMessage], *, options: RunOptions) -> list[LLMMessage]:
    if any(
        message.role == "system" and message.content.startswith(f"Run mode: {options.mode}.")
        for message in messages
    ):
        return list(messages)

    guidance = LLMMessage(
        role="system",
        content=(
            f"Run mode: {options.mode}. Work within the bounded investigation loop. "
            "Use tools only when useful and in scope. Do not expose chain-of-thought; "
            "final responses should summarize auditable findings and uncertainty."
        ),
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
        execute_tool_calls_once: ToolExecutor,
        persist_conversation_turn_once: ConversationPersister,
        refresh_memory_after_turn: MemoryRefresher,
        handle_provider_failure: ProviderFailureHandler,
        record_event: TraceRecorder | None = None,
    ) -> None:
        self.settings = settings
        self.structured_synthesizer = structured_synthesizer
        self.call_model_once = call_model_once
        self.execute_tool_calls_once = execute_tool_calls_once
        self.persist_conversation_turn_once = persist_conversation_turn_once
        self.refresh_memory_after_turn = refresh_memory_after_turn
        self.handle_provider_failure = handle_provider_failure
        self.record_event = record_event

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
            self._record_event(
                event_type="initial_plan_created",
                summary="Initial investigation plan created",
                payload={"investigation_state": state.compact_summary()},
            )

        return self._run_loop(
            user_input=user_input,
            session_id=session_id,
            context=context,
            messages=with_investigation_guidance(messages, options=options),
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
    ) -> AgentTurnResult:
        result, no_progress_iterations = self._reflect_and_decide_after_tools(
            user_input=pending.user_input,
            turn_index=pending.turn_index,
            options=options,
            state=state,
            messages=pending.messages,
            tool_step=ToolExecutionStepResult(
                messages=pending.messages,
                tool_messages=pending.tool_messages,
                exchange_index=pending.exchange_index,
                tool_calls_used=pending.tool_calls_used,
                tool_statuses=[pending.tool_status],
                tool_names=[str(pending.pending_payload.get("tool_name") or "unknown")],
            ),
            iterations_used=iterations_used,
            no_progress_iterations=no_progress_iterations,
        )
        if result is not None:
            return result

        return self._run_loop(
            user_input=pending.user_input,
            session_id=session_id,
            context=context,
            messages=pending.messages,
            turn_index=pending.turn_index,
            options=options,
            state=state,
            iterations_used=iterations_used,
            tool_calls_used=pending.tool_calls_used,
            exchange_index=pending.exchange_index,
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
                llm_response = self.call_model_once(
                    messages=messages,
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
        reflection = self._synthesize_reflection(state=state, tool_step=tool_step, options=options)
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

        decision = self._synthesize_decision(
            state=state,
            reflection=reflection,
            options=options,
            iterations_used=iterations_used,
            tool_calls_used=tool_step.tool_calls_used,
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
        critique = self._synthesize_final_critique(state=state, final_draft=final_draft, options=options)
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
        iterations_used: int,
        tool_calls_used: int,
        stop_reason: str,
    ) -> AgentTurnResult:
        state.stop_reason = stop_reason
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
            assistant_content=content,
        )
        self.refresh_memory_after_turn(turn_index=turn_index)
        return AgentTurnResult(
            status="completed",
            content=content,
            metadata=self._metadata(
                options=options,
                iterations_used=iterations_used,
                tool_calls_used=tool_calls_used,
                stop_reason=stop_reason,
                state=state,
            ),
        )

    def _complete_with_budget_answer(
        self,
        *,
        user_input: str,
        turn_index: int,
        options: RunOptions,
        state: InvestigationState,
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
                instructions=INITIAL_PLAN_PROMPT,
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
                instructions=STEP_REFLECTION_PROMPT,
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
                instructions=INVESTIGATION_DECISION_PROMPT,
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
                instructions=FINAL_CRITIQUE_PROMPT,
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

    def _with_investigation_guidance(self, messages: list[LLMMessage], *, options: RunOptions) -> list[LLMMessage]:
        return with_investigation_guidance(messages, options=options)

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
            "iterations_used": iterations_used,
            "tool_calls_used": tool_calls_used,
            "stop_reason": stop_reason,
            "investigation_state": state.compact_summary(),
        }
