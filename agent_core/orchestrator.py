from __future__ import annotations

from dataclasses import asdict
import json
from inspect import Parameter, signature
from typing import Any
from uuid import uuid4

from agent_core.domain_hooks import DomainHooks
from agent_core.execution_context import ExecutionContext
from agent_core.investigation_controller import InvestigationController, with_investigation_guidance
from agent_core.investigation_prompts import DEFAULT_INVESTIGATION_PROMPTS, InvestigationPromptSet
from agent_core.investigation_state import InvestigationState
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.memory.context_block import ContextBlock, estimate_token_count
from agent_core.memory.session_summary import SessionSummary
from agent_core.memory.task_state import TaskState
from agent_core.memory.thread_state import (
    ThreadState,
    create_conversation_turn_block,
    create_tool_exchange_block,
    render_context_blocks_to_history_dicts,
)
from agent_core.policy_engine import PolicyEngine
from agent_core.prompt_builder import PromptBuilder
from agent_core.run_trace import PromptSnapshot, RunTrace
from agent_core.run_options import RunOptions
from agent_core.session_manager import SessionManager
from agent_core.settings import CoreSettings
from agent_core.structured_synthesizer import StructuredSynthesisRequest, StructuredSynthesizer
from agent_core.tool_registry import ToolRegistry
from agent_core.turn_steps import PendingResumeState, ToolExecutionStepResult
from agent_core.types import AgentTurnResult, ToolExecutionStatus
from agent_core.llm.base import BaseLLMProvider, LLMCallOptions, LLMCompletionResult, LLMMessage
from agent_core.llm.errors import LLMProviderError

logger = get_logger("core.orchestrator")


class AgentOrchestrator:
    """Coordinate one full user turn from prompt build to persisted memory.

    The orchestrator is the runtime entry point. It builds the prompt stack,
    runs the assistant/tool loop, persists the resulting conversation as
    context blocks, then refreshes structured memory objects after the turn.
    """

    SUMMARY_MARKER_KEY = "session_summary_marker_block_id"
    PENDING_TURN_META_KEY = "pending_agent_turn"

    def __init__(
        self,
        *,
        settings: CoreSettings,
        provider: BaseLLMProvider,
        registry: ToolRegistry,
        session_manager: SessionManager,
        policy_engine: PolicyEngine,
        domain_hooks: DomainHooks | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.registry = registry
        self.session_manager = session_manager
        self.policy_engine = policy_engine
        self.domain_hooks = domain_hooks or DomainHooks()
        self.prompt_builder = PromptBuilder(
            settings=settings,
            session_manager=session_manager,
            domain_hooks=self.domain_hooks,
        )
        self.structured_synthesizer = StructuredSynthesizer(
            settings=settings,
            provider=provider,
        )

    def _build_tool_history_item(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        tool_content: str,
        status: ToolExecutionStatus,
    ) -> dict[str, Any]:
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "status": status,
            "content_preview": tool_content[:500],
        }

    def _build_messages(self, user_input: str) -> list[LLMMessage]:
        messages = self.prompt_builder.build_messages(user_input=user_input)
        logger.trace(
            "Built LLM message list",
            extra={
                "message_count": len(messages),
                "history_block_count": len(self.session_manager.get_context_blocks()),
            },
        )
        return messages

    def _estimate_prompt_tokens(self, *, messages: list[LLMMessage]) -> int:
        return estimate_token_count([message.to_history_dict() for message in messages])

    def _start_run_trace(
        self,
        *,
        session_id: str,
        run_options: RunOptions,
        messages: list[LLMMessage],
        turn_index: int,
    ) -> RunTrace:
        return RunTrace.start(
            run_id=f"run-{turn_index:04d}-{uuid4().hex[:8]}",
            session_id=session_id,
            mode=run_options.mode,
            turn_index=turn_index,
            options={
                "run_options": asdict(run_options),
                "model": self.settings.model,
                "max_active_context_tokens": self.settings.max_active_context_tokens,
                "max_tool_calls_per_turn": self.settings.max_tool_calls_per_turn,
            },
            prompt_snapshot=PromptSnapshot.from_messages(
                messages=messages,
                context_window_tokens=self.settings.max_active_context_tokens,
            ),
        )

    def _record_trace_event(
        self,
        trace: RunTrace | None,
        *,
        event_type: str,
        summary: str,
        iteration: int | None = None,
        payload: dict[str, Any] | None = None,
        related_tool_call_id: str | None = None,
    ) -> None:
        if trace is None:
            return
        trace.add_event(
            event_type=event_type,
            summary=summary,
            iteration=iteration,
            payload=payload,
            related_tool_call_id=related_tool_call_id,
        )

    def _save_run_trace_safely(self, trace: RunTrace) -> None:
        try:
            self.session_manager.save_run_trace(trace)
        except Exception as exc:
            logger.warning(
                "Failed to persist run trace",
                extra={"run_id": trace.run_id, "error_preview": safe_preview(str(exc), limit=200)},
            )

    def _finalize_run_trace_result(
        self,
        *,
        trace: RunTrace,
        result: AgentTurnResult,
        expose_trace_id: bool,
    ) -> AgentTurnResult:
        if expose_trace_id:
            result.metadata = {**result.metadata, "run_trace_id": trace.run_id}
        event_type = "run_completed" if result.status == "completed" else "run_pending"
        self._record_trace_event(
            trace,
            event_type=event_type,
            summary="Run completed" if result.status == "completed" else "Run is pending tool result",
            payload={
                "status": result.status,
                "content_length": len(result.content),
                "metadata_keys": sorted(result.metadata.keys()),
            },
        )
        trace.complete(status=result.status, final_metadata=result.metadata)
        self._save_run_trace_safely(trace)
        return result

    def _fail_run_trace(self, trace: RunTrace, exc: Exception) -> None:
        self._record_trace_event(
            trace,
            event_type="run_failed",
            summary="Run failed with an unhandled exception",
            payload={"exception_type": type(exc).__name__, "error_preview": safe_preview(str(exc), limit=300)},
        )
        trace.complete(
            status="failed",
            final_metadata={"exception_type": type(exc).__name__, "error_preview": safe_preview(str(exc), limit=300)},
        )
        self._save_run_trace_safely(trace)

    def _refresh_memory_after_turn(self, *, turn_index: int) -> None:
        # Memory synthesis belongs after the assistant has finished the turn, not
        # inside the tool-calling loop. Synthesis failures must not break the
        # user-visible turn result.
        thread_state = self._compact_history_after_turn()
        try:
            task_state = self._synthesize_task_state(thread_state=thread_state, turn_index=turn_index)
        except (LLMProviderError, ValueError) as exc:
            logger.warning(
                "TaskState synthesis failed; keeping the previous task state",
                extra={"error_preview": safe_preview(str(exc), limit=200)},
            )
        else:
            self.session_manager.set_task_state(task_state)
            thread_state.task_state = task_state

        thread_state = self.session_manager.get_thread_state()
        if not thread_state.overflow_blocks:
            self.session_manager.set_summary(None)
            self.session_manager.set_meta_value(self.SUMMARY_MARKER_KEY, None)
            self._run_post_turn_hooks(turn_index=turn_index)
            return

        try:
            summary = self._synthesize_session_summary(thread_state=thread_state)
        except (LLMProviderError, ValueError) as exc:
            logger.warning(
                "SessionSummary synthesis failed; keeping the previous summary",
                extra={"error_preview": safe_preview(str(exc), limit=200)},
            )
        else:
            self.session_manager.set_summary(summary)
            self.session_manager.set_meta_value(self.SUMMARY_MARKER_KEY, summary.covers_blocks_until)

        self._run_post_turn_hooks(turn_index=turn_index)

    def _run_post_turn_hooks(self, *, turn_index: int) -> None:
        try:
            self.domain_hooks.after_turn(
                session_manager=self.session_manager,
                thread_state=self.session_manager.get_thread_state(),
                turn_index=turn_index,
            )
        except Exception as exc:
            logger.warning(
                "Domain post-turn hook failed",
                extra={"error_preview": safe_preview(str(exc), limit=200)},
            )

    def _compact_history_after_turn(self) -> ThreadState:
        return self.session_manager.compact_history(
            max_active_tokens=self.settings.max_active_context_tokens,
        )

    def _application_context_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_manager.session_id or "default",
            "scope": list(self.settings.allowed_http_hosts),
            "source_code_locations": [str(path.resolve()) for path in self.settings.allowed_read_roots],
            "knowledge_base_dir": str(self.settings.knowledge_base_dir.resolve()),
            "allowed_http_methods": list(self.settings.allowed_http_methods),
        }

    def _synthesize_task_state(self, *, thread_state: ThreadState, turn_index: int) -> TaskState:
        # TaskState is synthesized from the active slice of the conversation.
        # It is meant to steer the next prompt, not to be a full audit log.
        runtime_context = self._application_context_payload()
        template = TaskState.create_template(
            run_id=f"run-{turn_index:04d}",
            objective=thread_state.task_state.objective if thread_state.task_state is not None else "Investigate the current in-scope application",
            scope=runtime_context["scope"],
            source_code_locations=runtime_context["source_code_locations"],
        )
        template.domain_extensions = self.domain_hooks.task_state_extensions_template(
            thread_state=thread_state,
            turn_index=turn_index,
        )
        payload = {
            "application_context": runtime_context,
            "previous_task_state": thread_state.task_state.to_dict() if thread_state.task_state is not None else None,
            "session_summary": thread_state.summary.to_dict() if thread_state.summary is not None else None,
            "recent_context_blocks": [block.to_dict() for block in thread_state.active_blocks],
            "recent_history": render_context_blocks_to_history_dicts(thread_state.active_blocks),
        }
        payload.update(
            self.domain_hooks.extend_task_state_payload(
                thread_state=thread_state,
                turn_index=turn_index,
            )
        )
        synthesized = self.structured_synthesizer.synthesize(
            request=StructuredSynthesisRequest(
                target_name="task_state",
                instructions=self.settings.task_state_synthesis_prompt,
                output_format=template.to_dict(),
                payload=payload,
                parser=TaskState.from_any,
            )
        )
        return synthesized.with_runtime_context(
            run_id=template.run_id,
            scope=runtime_context["scope"],
            source_code_locations=runtime_context["source_code_locations"],
        )

    def _synthesize_session_summary(self, *, thread_state: ThreadState) -> SessionSummary:
        # SessionSummary is built only from overflow blocks, then merged with
        # the previous summary so compaction can keep shrinking prompt history
        # without losing the long-running narrative of the engagement.
        new_overflow_blocks = self._unsummarized_overflow_blocks(thread_state=thread_state)
        if not new_overflow_blocks:
            raise ValueError("Cannot synthesize SessionSummary without overflow blocks")

        delta_template = SessionSummary.create_template(
            thread_id=thread_state.thread_id,
            covers_blocks_until=new_overflow_blocks[-1].block_id,
            source_block_count=len(new_overflow_blocks),
            previous_summary=None,
        )
        delta_template.domain_extensions = self.domain_hooks.session_summary_extensions_template(
            thread_state=thread_state,
        )
        delta_payload = {
            "thread_id": thread_state.thread_id,
            "summary_marker_block_id": self._summary_marker_block_id(thread_state=thread_state),
            "task_state": thread_state.task_state.to_dict() if thread_state.task_state is not None else None,
            "new_overflow_context_blocks": [block.to_dict() for block in new_overflow_blocks],
            "new_overflow_history": render_context_blocks_to_history_dicts(new_overflow_blocks),
        }
        delta_payload.update(
            self.domain_hooks.extend_session_summary_delta_payload(
                thread_state=thread_state,
                new_overflow_blocks=new_overflow_blocks,
            )
        )
        delta_summary = self.structured_synthesizer.synthesize(
            request=StructuredSynthesisRequest(
                target_name="session_summary",
                instructions=self.settings.session_summary_synthesis_prompt,
                output_format=delta_template.to_dict(),
                payload=delta_payload,
                parser=SessionSummary.from_any,
            )
        )
        merge_template = SessionSummary.create_template(
            thread_id=thread_state.thread_id,
            covers_blocks_until=new_overflow_blocks[-1].block_id,
            source_block_count=len(new_overflow_blocks),
            previous_summary=thread_state.summary,
        )
        merge_template.domain_extensions = self.domain_hooks.session_summary_extensions_template(
            thread_state=thread_state,
        )
        merge_payload = {
            "thread_id": thread_state.thread_id,
            "old_summary": thread_state.summary.to_dict() if thread_state.summary is not None else None,
            "new_summary_delta": delta_summary.to_dict(),
            "summary_marker_block_id": self._summary_marker_block_id(thread_state=thread_state),
        }
        merged_summary = self.structured_synthesizer.synthesize(
            request=StructuredSynthesisRequest(
                target_name="session_summary_merge",
                instructions=self.settings.session_summary_merge_prompt,
                output_format=merge_template.to_dict(),
                payload=merge_payload,
                parser=SessionSummary.from_any,
            )
        )
        return SessionSummary(
            summary_id=merge_template.summary_id,
            thread_id=thread_state.thread_id,
            covers_blocks_until=merge_template.covers_blocks_until,
            generated_at=merge_template.generated_at,
            source_block_count=len(new_overflow_blocks),
            facts_confirmed=list(merged_summary.facts_confirmed),
            hypotheses_open=list(merged_summary.hypotheses_open),
            decisions=list(merged_summary.decisions),
            completed_actions=list(merged_summary.completed_actions),
            pending_actions=list(merged_summary.pending_actions),
            relevant_artifacts=list(merged_summary.relevant_artifacts),
            domain_extensions=dict(merged_summary.domain_extensions),
            schema_version=merge_template.schema_version,
        )

    def _summary_marker_block_id(self, *, thread_state: ThreadState) -> str:
        marker = thread_state.meta.get(self.SUMMARY_MARKER_KEY)
        return marker if isinstance(marker, str) else ""

    def _unsummarized_overflow_blocks(self, *, thread_state: ThreadState) -> list[ContextBlock]:
        overflow_blocks = thread_state.overflow_blocks
        marker_block_id = self._summary_marker_block_id(thread_state=thread_state)
        if not marker_block_id:
            return overflow_blocks
        marker_index = next((index for index, block in enumerate(overflow_blocks) if block.block_id == marker_block_id), None)
        if marker_index is None:
            return []
        return overflow_blocks[marker_index + 1 :]

    def _persist_conversation_turn_once(self, *, turn_index: int, user_input: str, assistant_content: str) -> None:
        # The provider still receives flat messages, but persisted history now
        # stores one whole user/assistant turn as a single atomic block.
        conversation_block = create_conversation_turn_block(
            turn_index=turn_index,
            user_message=LLMMessage(role="user", content=user_input).to_history_dict(),
            assistant_message=LLMMessage(role="assistant", content=assistant_content).to_history_dict(),
        )
        self.session_manager.append_context_block(conversation_block)

    def _persist_tool_exchange_once(
        self,
        *,
        turn_index: int,
        exchange_index: int,
        assistant_message: LLMMessage,
        tool_messages: list[LLMMessage],
    ) -> None:
        tool_exchange_block = create_tool_exchange_block(
            turn_index=turn_index,
            exchange_index=exchange_index,
            assistant_message=assistant_message.to_history_dict(),
            tool_messages=[message.to_history_dict() for message in tool_messages],
        )
        self.session_manager.append_context_block(tool_exchange_block)

    def _persist_conversation_turn(self, *, turn_index: int, user_input: str, assistant_content: str) -> None:
        self._persist_conversation_turn_once(
            turn_index=turn_index,
            user_input=user_input,
            assistant_content=assistant_content,
        )

    def _persist_tool_exchange(
        self,
        *,
        turn_index: int,
        exchange_index: int,
        assistant_message: LLMMessage,
        tool_messages: list[LLMMessage],
    ) -> None:
        self._persist_tool_exchange_once(
            turn_index=turn_index,
            exchange_index=exchange_index,
            assistant_message=assistant_message,
            tool_messages=tool_messages,
        )

    def _append_budget_exhausted_tool_messages(
        self,
        *,
        messages: list[LLMMessage],
        tool_calls: list[Any],
        tool_messages: list[LLMMessage],
        tool_names: list[str],
        tool_statuses: list[ToolExecutionStatus],
        trace: RunTrace | None = None,
        iteration: int | None = None,
    ) -> None:
        for tool_call in tool_calls:
            tool_content = f"Tool call skipped: maximum tool-call budget reached before executing {tool_call.name}."
            tool_message = LLMMessage(role="tool", tool_call_id=tool_call.id, content=tool_content)
            messages.append(tool_message)
            tool_messages.append(tool_message)
            tool_names.append(tool_call.name)
            tool_statuses.append("budget_exhausted")
            try:
                arguments = json.loads(tool_call.arguments_json or "{}")
            except json.JSONDecodeError:
                arguments = {}
            self.session_manager.append_tool_history(
                self._build_tool_history_item(
                    tool_name=tool_call.name,
                    arguments=arguments,
                    tool_content=tool_content,
                    status="budget_exhausted",
                )
            )
            self._record_trace_event(
                trace,
                event_type="tool_call_skipped_budget_exhausted",
                summary=f"Tool call skipped due to budget: {tool_call.name}",
                iteration=iteration,
                payload={"tool_name": tool_call.name, "content_length": len(tool_content)},
                related_tool_call_id=tool_call.id,
            )

    def _handle_provider_failure(self, *, error: LLMProviderError, user_input: str, turn_index: int) -> AgentTurnResult:
        logger.error(
            "LLM provider failure",
            extra={"error_kind": error.kind, "detail_preview": safe_preview(error.detail or error.user_message)},
        )
        self._persist_conversation_turn(
            turn_index=turn_index,
            user_input=user_input,
            assistant_content=error.user_message,
        )
        self._refresh_memory_after_turn(turn_index=turn_index)
        return AgentTurnResult(status="completed", content=error.user_message)

    def _build_pending_id(
        self,
        *,
        session_id: str,
        turn_index: int,
        exchange_index: int,
        tool_call_id: str,
    ) -> str:
        return f"{session_id}:turn-{turn_index:04d}:exchange-{exchange_index:02d}:{tool_call_id}"

    def _persist_pending_turn(
        self,
        *,
        pending_id: str,
        user_input: str,
        messages: list[LLMMessage],
        turn_index: int,
        exchange_index: int,
        tool_calls_used: int,
        assistant_message: LLMMessage,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_metadata: dict[str, Any],
        tool_messages: list[LLMMessage] | None = None,
        pending_metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "pending_id": pending_id,
            "user_input": user_input,
            "messages": [message.to_history_dict() for message in messages],
            "turn_index": turn_index,
            "exchange_index": exchange_index,
            "tool_calls_used": tool_calls_used,
            "assistant_message": assistant_message.to_history_dict(),
            "tool_messages": [message.to_history_dict() for message in tool_messages or []],
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "metadata": result_metadata,
        }
        if pending_metadata_extra:
            payload.update(pending_metadata_extra)
        self.session_manager.set_meta_value(self.PENDING_TURN_META_KEY, payload)

    def run_turn(self, user_input: str, session_id: str = "default") -> str:
        return self.run_turn_result(user_input=user_input, session_id=session_id).content

    def run_turn_result(
        self,
        user_input: str,
        session_id: str = "default",
        options: RunOptions | None = None,
    ) -> AgentTurnResult:
        with self.session_manager.session_scope(session_id):
            return self._run_turn_result_active(user_input=user_input, session_id=session_id, options=options)

    def _run_turn_result_active(
        self,
        user_input: str,
        session_id: str,
        options: RunOptions | None,
    ) -> AgentTurnResult:
        run_options = options or RunOptions.direct()
        logger.info(
            "Starting run_turn",
            extra={"session_id": session_id, "user_input_length": len(user_input), "mode": run_options.mode},
        )
        state = self.session_manager.get_state()
        context = ExecutionContext(
            session_id=session_id,
            settings=self.settings,
            session_state=state,
        )

        messages = self._build_messages(user_input)
        turn_index = self.session_manager.get_next_turn_index()
        investigation_prompt_set: InvestigationPromptSet | None = None
        if run_options.mode in {"investigate", "deep_investigate"}:
            investigation_prompt_set = self._build_investigation_prompt_set(options=run_options)
            messages = with_investigation_guidance(
                messages,
                options=run_options,
                prompt_set=investigation_prompt_set,
            )
        trace = self._start_run_trace(
            session_id=session_id,
            run_options=run_options,
            messages=messages,
            turn_index=turn_index,
        )
        expose_trace_id = options is not None or run_options.mode in {"investigate", "deep_investigate"}
        try:
            if run_options.mode in {"investigate", "deep_investigate"}:
                result = self._build_investigation_controller(
                    trace=trace,
                    prompt_set=investigation_prompt_set,
                ).run(
                    user_input=user_input,
                    session_id=session_id,
                    context=context,
                    messages=messages,
                    turn_index=turn_index,
                    options=run_options,
                )
                return self._finalize_run_trace_result(
                    trace=trace,
                    result=result,
                    expose_trace_id=expose_trace_id,
                )

            tool_history_start_count = len(state.get("tool_history", [])) if isinstance(state.get("tool_history"), list) else 0
            result = self._continue_turn(
                user_input=user_input,
                session_id=session_id,
                context=context,
                messages=messages,
                turn_index=turn_index,
                tool_calls_used=0,
                exchange_index=0,
                trace=trace,
            )
            if options is not None:
                result.metadata = {
                    **result.metadata,
                    "mode": run_options.mode,
                    "iterations_used": 1,
                    "tool_calls_used": self._tool_history_delta(start_count=tool_history_start_count),
                    "stop_reason": result.status,
                }
            return self._finalize_run_trace_result(
                trace=trace,
                result=result,
                expose_trace_id=expose_trace_id,
            )
        except Exception as exc:
            self._fail_run_trace(trace, exc)
            raise

    def resume_turn(
        self,
        *,
        pending_id: str,
        tool_content: str,
        ok: bool = True,
        session_id: str = "default",
    ) -> AgentTurnResult:
        with self.session_manager.session_scope(session_id):
            return self._resume_turn_active(
                pending_id=pending_id,
                tool_content=tool_content,
                ok=ok,
                session_id=session_id,
            )

    def _resume_turn_active(
        self,
        *,
        pending_id: str,
        tool_content: str,
        ok: bool,
        session_id: str,
    ) -> AgentTurnResult:
        state = self.session_manager.get_state()
        pending = state.get("meta", {}).get(self.PENDING_TURN_META_KEY)
        if not isinstance(pending, dict) or pending.get("pending_id") != pending_id:
            return AgentTurnResult(
                status="completed",
                content=f"No pending agent turn found for pending_id={pending_id}",
            )

        trace = self._load_run_trace_from_pending(pending)
        self._record_trace_event(
            trace,
            event_type="pending_tool_result_received",
            summary="External pending tool result received",
            payload={"ok": ok, "content_length": len(tool_content)},
        )
        resumed = self._handle_pending_tool_result_once(pending=pending, tool_content=tool_content, ok=ok, trace=trace)
        if isinstance(resumed, AgentTurnResult):
            return (
                self._finalize_run_trace_result(trace=trace, result=resumed, expose_trace_id=bool(resumed.metadata))
                if trace is not None
                else resumed
            )

        context = ExecutionContext(
            session_id=session_id,
            settings=self.settings,
            session_state=state,
        )
        mode = resumed.pending_payload.get("mode")
        if mode in {"investigate", "deep_investigate"}:
            run_options = self._run_options_from_pending(resumed.pending_payload)
            investigation_prompt_set = self._build_investigation_prompt_set(options=run_options)
            investigation_state = InvestigationState.from_any(resumed.pending_payload.get("investigation_state"))
            if investigation_state is None:
                investigation_state = InvestigationState.create_template(objective=resumed.user_input)
            iterations_used = resumed.pending_payload.get("iterations_used")
            no_progress_iterations = resumed.pending_payload.get("no_progress_iterations")
            result = self._build_investigation_controller(
                trace=trace,
                prompt_set=investigation_prompt_set,
            ).resume_after_pending(
                pending=resumed,
                session_id=session_id,
                context=context,
                options=run_options,
                state=investigation_state,
                iterations_used=iterations_used if isinstance(iterations_used, int) else 1,
                no_progress_iterations=no_progress_iterations if isinstance(no_progress_iterations, int) else 0,
            )
            return (
                self._finalize_run_trace_result(trace=trace, result=result, expose_trace_id=True)
                if trace is not None
                else result
            )

        result = self._continue_turn(
            user_input=resumed.user_input,
            session_id=session_id,
            context=context,
            messages=resumed.messages,
            turn_index=resumed.turn_index,
            tool_calls_used=resumed.tool_calls_used,
            exchange_index=resumed.exchange_index,
            trace=trace,
        )
        return (
            self._finalize_run_trace_result(trace=trace, result=result, expose_trace_id=bool(result.metadata))
            if trace is not None
            else result
        )

    def _build_investigation_prompt_set(self, *, options: RunOptions) -> InvestigationPromptSet:
        return self.domain_hooks.customize_investigation_prompts(
            prompt_set=DEFAULT_INVESTIGATION_PROMPTS,
            settings=self.settings,
            options=options,
        )

    def _build_investigation_controller(
        self,
        *,
        trace: RunTrace | None = None,
        prompt_set: InvestigationPromptSet | None = None,
    ) -> InvestigationController:
        return InvestigationController(
            settings=self.settings,
            structured_synthesizer=self.structured_synthesizer,
            call_model_once=self._call_model_once,
            execute_tool_calls_once=lambda **kwargs: self._execute_tool_calls_once(**kwargs, trace=trace),
            persist_conversation_turn_once=self._persist_conversation_turn_once,
            refresh_memory_after_turn=self._refresh_memory_after_turn,
            handle_provider_failure=self._handle_provider_failure,
            record_event=lambda **kwargs: self._record_trace_event(trace, **kwargs),
            prompt_set=prompt_set,
        )

    def _load_run_trace_from_pending(self, pending: dict[str, Any]) -> RunTrace | None:
        run_trace_id = pending.get("run_trace_id")
        if not isinstance(run_trace_id, str) or not run_trace_id:
            return None
        payload = self.session_manager.load_run_trace(run_trace_id)
        trace = RunTrace.from_any(payload)
        if trace is None:
            logger.warning("Pending run trace could not be loaded", extra={"run_trace_id": run_trace_id})
        return trace

    def _call_model_once(
        self,
        *,
        messages: list[LLMMessage],
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        if options is not None and self._provider_accepts_options("complete_with_tools"):
            return self.provider.complete_with_tools(
                messages=messages,
                tools=self.registry.get_tool_specs(),
                model=self.settings.model,
                temperature=self.settings.temperature,
                options=options,
            )
        return self.provider.complete_with_tools(
            messages=messages,
            tools=self.registry.get_tool_specs(),
            model=self.settings.model,
            temperature=self.settings.temperature,
        )

    def _execute_tool_calls_once(
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
        trace: RunTrace | None = None,
    ) -> ToolExecutionStepResult:
        tool_messages: list[LLMMessage] = []
        tool_statuses: list[ToolExecutionStatus] = []
        tool_names: list[str] = []
        tool_budget_exhausted = False
        current_exchange_index = exchange_index + 1
        effective_pending_metadata_extra = dict(pending_metadata_extra or {})
        if trace is not None:
            effective_pending_metadata_extra["run_trace_id"] = trace.run_id

        for tool_call_offset, tc in enumerate(assistant_message.tool_calls):
            # The live provider transcript keeps growing inside the loop, but
            # persisted storage records each tool phase as an atomic
            # tool-exchange block once execution completes.
            if tool_calls_used >= max_tool_calls:
                logger.error("Tool-call budget exhausted before executing remaining tool calls")
                tool_budget_exhausted = True
                self._record_trace_event(
                    trace,
                    event_type="tool_budget_exhausted",
                    summary="Tool-call budget exhausted before executing remaining tool calls",
                    payload={"tool_calls_used": tool_calls_used, "max_tool_calls": max_tool_calls},
                )
                self._append_budget_exhausted_tool_messages(
                    messages=messages,
                    tool_calls=assistant_message.tool_calls[tool_call_offset:],
                    tool_messages=tool_messages,
                    tool_names=tool_names,
                    tool_statuses=tool_statuses,
                    trace=trace,
                )
                break

            tool_calls_used += 1
            tool_names.append(tc.name)
            logger.info(
                "Executing tool call",
                extra={"tool_name": tc.name, "tool_call_index": tool_calls_used},
            )
            self._record_trace_event(
                trace,
                event_type="tool_call_started",
                summary=f"Tool call started: {tc.name}",
                payload={
                    "tool_name": tc.name,
                    "tool_call_index": tool_calls_used,
                    "arguments_length": len(tc.arguments_json or ""),
                },
                related_tool_call_id=tc.id,
            )

            try:
                arguments = json.loads(tc.arguments_json or "{}")
            except json.JSONDecodeError:
                logger.exception("Failed to parse tool arguments JSON", extra={"tool_name": tc.name})
                arguments = {}
                tool_content = f"Invalid JSON arguments for tool {tc.name}"
                tool_status: ToolExecutionStatus = "invalid_arguments"
            else:
                authz = self.policy_engine.authorize(tc.name, arguments, context)
                if not authz.allowed:
                    tool_content = f"Tool denied by policy: {authz.reason}"
                    tool_status = "policy_denied"
                    logger.info(
                        "Tool denied by policy",
                        extra={"tool_name": tc.name, "reason": authz.reason},
                    )
                else:
                    try:
                        logger.debug(
                            "Dispatching tool execution",
                            extra={"tool_name": tc.name, "argument_keys": sorted(arguments.keys())},
                        )
                        result = self.registry.execute(tc.name, arguments, context)
                        tool_content = result.content
                        if result.pending:
                            pending_id = result.pending_id or self._build_pending_id(
                                session_id=session_id,
                                turn_index=turn_index,
                                exchange_index=current_exchange_index,
                                tool_call_id=tc.id,
                            )
                            self.session_manager.append_tool_history(
                                self._build_tool_history_item(
                                    tool_name=tc.name,
                                    arguments=arguments,
                                    tool_content=tool_content,
                                    status="pending",
                                )
                            )
                            self._persist_pending_turn(
                                pending_id=pending_id,
                                user_input=user_input,
                                messages=messages,
                                turn_index=turn_index,
                                exchange_index=current_exchange_index,
                                tool_calls_used=tool_calls_used,
                                assistant_message=assistant_message,
                                tool_call_id=tc.id,
                                tool_name=tc.name,
                                arguments=arguments,
                                result_metadata=result.metadata,
                                tool_messages=tool_messages,
                                pending_metadata_extra=effective_pending_metadata_extra or None,
                            )
                            logger.info(
                                "Tool execution is pending external result",
                                extra={"tool_name": tc.name, "pending_id": pending_id},
                            )
                            self._record_trace_event(
                                trace,
                                event_type="tool_call_pending",
                                summary=f"Tool call pending: {tc.name}",
                                payload={
                                    "tool_name": tc.name,
                                    "pending_id": pending_id,
                                    "content_length": len(tool_content),
                                    "metadata": result.metadata,
                                },
                                related_tool_call_id=tc.id,
                            )
                            return ToolExecutionStepResult(
                                messages=messages,
                                tool_messages=tool_messages,
                                exchange_index=current_exchange_index,
                                tool_calls_used=tool_calls_used,
                                pending_result=AgentTurnResult(
                                    status="pending_tool_result",
                                    content=tool_content,
                                    pending_id=pending_id,
                                    tool_name=tc.name,
                                    tool_arguments=arguments,
                                    metadata=result.metadata,
                                ),
                                tool_statuses=tool_statuses + ["pending"],
                                tool_names=tool_names,
                            )
                        tool_status = "ok" if result.ok else "tool_error"
                        logger.debug(
                            "Tool execution completed",
                            extra={
                                "tool_name": tc.name,
                                "ok": result.ok,
                                "content_preview": safe_preview(tool_content, limit=200),
                            },
                        )
                    except Exception as exc:
                        tool_content = f"Tool execution failed: {exc}"
                        tool_status = "execution_failed"
                        logger.exception("Tool execution crashed", extra={"tool_name": tc.name})

            tool_message = LLMMessage(role="tool", tool_call_id=tc.id, content=tool_content)
            messages.append(tool_message)
            tool_messages.append(tool_message)
            tool_statuses.append(tool_status)
            self.session_manager.append_tool_history(
                self._build_tool_history_item(
                    tool_name=tc.name,
                    arguments=arguments,
                    tool_content=tool_content,
                    status=tool_status,
                )
            )
            self._record_trace_event(
                trace,
                event_type="tool_call_completed",
                summary=f"Tool call completed: {tc.name}",
                payload={
                    "tool_name": tc.name,
                    "status": tool_status,
                    "content_length": len(tool_content),
                },
                related_tool_call_id=tc.id,
            )

        self._persist_tool_exchange_once(
            turn_index=turn_index,
            exchange_index=current_exchange_index,
            assistant_message=assistant_message,
            tool_messages=tool_messages,
        )
        self._record_trace_event(
            trace,
            event_type="tool_exchange_completed",
            summary="Tool exchange persisted",
            payload={
                "exchange_index": current_exchange_index,
                "tool_names": list(tool_names),
                "tool_statuses": list(tool_statuses),
                "budget_exhausted": tool_budget_exhausted,
            },
        )
        return ToolExecutionStepResult(
            messages=messages,
            tool_messages=tool_messages,
            exchange_index=current_exchange_index,
            tool_calls_used=tool_calls_used,
            budget_exhausted=tool_budget_exhausted,
            tool_statuses=tool_statuses,
            tool_names=tool_names,
        )

    def _handle_pending_tool_result_once(
        self,
        *,
        pending: dict[str, Any],
        tool_content: str,
        ok: bool,
        trace: RunTrace | None = None,
    ) -> PendingResumeState | AgentTurnResult:
        raw_messages = pending.get("messages")
        if not isinstance(raw_messages, list):
            return AgentTurnResult(status="completed", content="Pending agent turn is corrupt: missing messages.")

        messages = [
            LLMMessage.from_history_dict(item)
            for item in raw_messages
            if isinstance(item, dict)
        ]
        tool_call_id = pending.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return AgentTurnResult(status="completed", content="Pending agent turn is corrupt: missing tool call id.")

        tool_message = LLMMessage(role="tool", tool_call_id=tool_call_id, content=tool_content)
        messages.append(tool_message)

        turn_index = pending.get("turn_index")
        exchange_index = pending.get("exchange_index")
        tool_calls_used = pending.get("tool_calls_used")
        if not isinstance(turn_index, int) or not isinstance(exchange_index, int) or not isinstance(tool_calls_used, int):
            return AgentTurnResult(status="completed", content="Pending agent turn is corrupt: invalid turn counters.")

        assistant_payload = pending.get("assistant_message")
        assistant_message = (
            LLMMessage.from_history_dict(assistant_payload)
            if isinstance(assistant_payload, dict)
            else LLMMessage(role="assistant", content="")
        )
        raw_tool_messages = pending.get("tool_messages")
        previous_tool_messages = [
            LLMMessage.from_history_dict(item)
            for item in raw_tool_messages
            if isinstance(item, dict)
        ] if isinstance(raw_tool_messages, list) else []
        persisted_tool_messages = [*previous_tool_messages, tool_message]
        self._persist_tool_exchange_once(
            turn_index=turn_index,
            exchange_index=exchange_index,
            assistant_message=assistant_message,
            tool_messages=persisted_tool_messages,
        )
        pending_arguments = pending.get("arguments")
        self.session_manager.append_tool_history(
            self._build_tool_history_item(
                tool_name=str(pending.get("tool_name") or "unknown"),
                arguments=dict(pending_arguments) if isinstance(pending_arguments, dict) else {},
                tool_content=tool_content,
                status="ok" if ok else "tool_error",
            )
        )
        self._record_trace_event(
            trace,
            event_type="pending_tool_result_persisted",
            summary="Pending tool result persisted",
            payload={
                "tool_name": str(pending.get("tool_name") or "unknown"),
                "status": "ok" if ok else "tool_error",
                "content_length": len(tool_content),
            },
            related_tool_call_id=tool_call_id,
        )
        self.session_manager.set_meta_value(self.PENDING_TURN_META_KEY, None)
        return PendingResumeState(
            user_input=str(pending.get("user_input") or ""),
            messages=messages,
            turn_index=turn_index,
            exchange_index=exchange_index,
            tool_calls_used=tool_calls_used,
            tool_messages=persisted_tool_messages,
            tool_status="ok" if ok else "tool_error",
            pending_payload=dict(pending),
        )

    def _run_options_from_pending(self, pending: dict[str, Any]) -> RunOptions:
        raw_options = pending.get("run_options")
        if isinstance(raw_options, dict):
            allowed_keys = set(RunOptions.__dataclass_fields__.keys())
            try:
                return RunOptions(**{key: value for key, value in raw_options.items() if key in allowed_keys})
            except (TypeError, ValueError):
                logger.warning("Pending investigation run options were invalid; falling back to mode defaults")

        mode = pending.get("mode")
        if mode == "deep_investigate":
            return RunOptions.deep_investigate()
        return RunOptions.investigate(require_initial_plan=False)

    def _provider_accepts_options(self, method_name: str) -> bool:
        method = getattr(self.provider, method_name)
        try:
            parameters = signature(method).parameters.values()
        except (TypeError, ValueError):
            return True
        return any(
            parameter.kind == Parameter.VAR_KEYWORD or parameter.name == "options"
            for parameter in parameters
        )

    def _tool_history_delta(self, *, start_count: int) -> int:
        history = self.session_manager.get_state().get("tool_history", [])
        return max(0, len(history) - start_count) if isinstance(history, list) else 0

    def _continue_turn(
        self,
        *,
        user_input: str,
        session_id: str,
        context: ExecutionContext,
        messages: list[LLMMessage],
        turn_index: int,
        tool_calls_used: int,
        exchange_index: int,
        trace: RunTrace | None = None,
    ) -> AgentTurnResult:
        start_prompt_tokens = self._estimate_prompt_tokens(messages=messages)
        tool_loop_reserve_tokens = max(1, self.settings.max_active_context_tokens)
        prompt_reserve_warning_emitted = False
        model_call_index = 0

        while True:
            model_call_index += 1
            prompt_tokens = self._estimate_prompt_tokens(messages=messages)
            prompt_growth_tokens = max(0, prompt_tokens - start_prompt_tokens)
            logger.debug(
                "Calling LLM",
                extra={
                    "model": self.settings.model,
                    "message_count": len(messages),
                    "estimated_prompt_tokens": prompt_tokens,
                    "start_turn_prompt_tokens": start_prompt_tokens,
                    "tool_loop_reserve_tokens": tool_loop_reserve_tokens,
                    "prompt_growth_tokens": prompt_growth_tokens,
                },
            )
            if (
                tool_calls_used > 0
                and prompt_growth_tokens >= tool_loop_reserve_tokens
                and not prompt_reserve_warning_emitted
            ):
                logger.warning(
                    "Tool loop consumed the start-turn prompt reserve",
                    extra={
                        "estimated_prompt_tokens": prompt_tokens,
                        "start_turn_prompt_tokens": start_prompt_tokens,
                        "prompt_growth_tokens": prompt_growth_tokens,
                        "tool_loop_reserve_tokens": tool_loop_reserve_tokens,
                        "tool_calls_used": tool_calls_used,
                    },
                )
                prompt_reserve_warning_emitted = True
            self._record_trace_event(
                trace,
                event_type="llm_call_started",
                summary="LLM call started",
                iteration=model_call_index,
                payload={
                    "message_count": len(messages),
                    "estimated_prompt_tokens": prompt_tokens,
                    "tool_calls_used": tool_calls_used,
                    "exchange_index": exchange_index,
                },
            )
            try:
                llm_response = self._call_model_once(messages=messages)
            except LLMProviderError as exc:
                self._record_trace_event(
                    trace,
                    event_type="llm_provider_failure",
                    summary="LLM provider failure handled",
                    iteration=model_call_index,
                    payload={"kind": exc.kind, "detail_preview": safe_preview(exc.detail or exc.user_message, limit=200)},
                )
                return self._handle_provider_failure(
                    error=exc,
                    user_input=user_input,
                    turn_index=turn_index,
                )

            logger.debug(
                "Received LLM response",
                extra={
                    "content_length": len(llm_response.content),
                    "tool_call_count": len(llm_response.tool_calls),
                },
            )

            assistant_message = LLMMessage(
                role="assistant",
                content=llm_response.content,
                tool_calls=list(llm_response.tool_calls),
            )
            messages.append(assistant_message)
            self._record_trace_event(
                trace,
                event_type="assistant_response_received",
                summary="Assistant response received",
                iteration=model_call_index,
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
                self._persist_conversation_turn(
                    turn_index=turn_index,
                    user_input=user_input,
                    assistant_content=llm_response.content,
                )
                self._refresh_memory_after_turn(turn_index=turn_index)
                logger.info("Completing run_turn without additional tool calls")
                return AgentTurnResult(status="completed", content=llm_response.content)

            if tool_calls_used >= self.settings.max_tool_calls_per_turn:
                exchange_index += 1
                tool_messages: list[LLMMessage] = []
                tool_statuses: list[ToolExecutionStatus] = []
                tool_names: list[str] = []
                self._append_budget_exhausted_tool_messages(
                    messages=messages,
                    tool_calls=assistant_message.tool_calls,
                    tool_messages=tool_messages,
                    tool_names=tool_names,
                    tool_statuses=tool_statuses,
                    trace=trace,
                    iteration=model_call_index,
                )
                self._persist_tool_exchange_once(
                    turn_index=turn_index,
                    exchange_index=exchange_index,
                    assistant_message=assistant_message,
                    tool_messages=tool_messages,
                )
                msg = "Maximum number of tool calls reached for this turn."
                logger.error(msg)
                self._record_trace_event(
                    trace,
                    event_type="tool_budget_exhausted",
                    summary=msg,
                    iteration=model_call_index,
                    payload={"tool_calls_used": tool_calls_used},
                )
                self._persist_conversation_turn(
                    turn_index=turn_index,
                    user_input=user_input,
                    assistant_content=msg,
                )
                self._refresh_memory_after_turn(turn_index=turn_index)
                return AgentTurnResult(status="completed", content=msg)

            tool_step = self._execute_tool_calls_once(
                user_input=user_input,
                session_id=session_id,
                context=context,
                messages=messages,
                turn_index=turn_index,
                exchange_index=exchange_index,
                tool_calls_used=tool_calls_used,
                assistant_message=assistant_message,
                max_tool_calls=self.settings.max_tool_calls_per_turn,
                trace=trace,
            )
            messages = tool_step.messages
            exchange_index = tool_step.exchange_index
            tool_calls_used = tool_step.tool_calls_used

            if tool_step.pending_result is not None:
                return tool_step.pending_result

            if tool_step.budget_exhausted:
                msg = "Maximum number of tool calls reached for this turn."
                logger.error(msg)
                self._record_trace_event(
                    trace,
                    event_type="tool_budget_exhausted",
                    summary=msg,
                    iteration=model_call_index,
                    payload={"tool_calls_used": tool_calls_used},
                )
                self._persist_conversation_turn(
                    turn_index=turn_index,
                    user_input=user_input,
                    assistant_content=msg,
                )
                self._refresh_memory_after_turn(turn_index=turn_index)
                return AgentTurnResult(status="completed", content=msg)
