from __future__ import annotations

import json
from typing import Any

from agent_core.execution_context import ExecutionContext
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.policy_engine import PolicyEngine
from agent_core.specialists.registry import SpecialistRegistry
from agent_core.settings import CoreSettings
from agent_core.specialists.types import SpecialistOutput, SpecialistProfile, SpecialistRunRequest, SpecialistRunResult
from agent_core.tool_registry import ToolRegistry
from agent_core.types import ToolExecutionStatus, build_empty_session_state
from agent_core.llm.base import BaseLLMProvider, LLMMessage
from agent_core.llm.errors import LLMProviderError

logger = get_logger(__name__)


class SpecialistRunner:
    """Run one ephemeral specialist investigation with a filtered tool subset."""

    def __init__(
        self,
        *,
        settings: CoreSettings,
        provider: BaseLLMProvider,
        tool_registry: ToolRegistry,
        specialist_registry: SpecialistRegistry,
        policy_engine: PolicyEngine,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.tool_registry = tool_registry
        self.specialist_registry = specialist_registry
        self.policy_engine = policy_engine

    def run(self, *, request: SpecialistRunRequest, session_id: str = "default") -> SpecialistRunResult:
        profile = self.specialist_registry.require(request.profile_id)
        registry = self.tool_registry.build_subset(profile.allowed_tools)
        specialist_session_id = f"{session_id}:specialist:{profile.profile_id}"
        context = ExecutionContext(
            session_id=specialist_session_id,
            settings=self.settings,
            session_state=build_empty_session_state(session_id=specialist_session_id),
        )
        messages = self._build_messages(profile=profile, request=request, session_id=session_id)
        tool_history: list[dict[str, Any]] = []
        iterations = 0
        tool_calls_used = 0

        while iterations < profile.max_iterations:
            iterations += 1
            logger.debug(
                "Calling specialist LLM",
                extra={
                    "profile_id": profile.profile_id,
                    "iteration": iterations,
                    "tool_count": len(registry.list_tool_names()),
                },
            )
            try:
                llm_response = self.provider.complete_with_tools(
                    messages=messages,
                    tools=registry.get_tool_specs(),
                    model=profile.model or self.settings.model,
                    temperature=profile.temperature if profile.temperature is not None else self.settings.temperature,
                )
            except LLMProviderError as exc:
                logger.error(
                    "Specialist provider failure",
                    extra={"profile_id": profile.profile_id, "error_kind": exc.kind},
                )
                return SpecialistRunResult(
                    ok=False,
                    profile_id=profile.profile_id,
                    failure_reason=exc.user_message,
                    raw_content=exc.detail or exc.user_message,
                    tool_history=tool_history,
                    iterations=iterations,
                    tool_calls_used=tool_calls_used,
                )

            assistant_message = LLMMessage(
                role="assistant",
                content=llm_response.content,
                tool_calls=list(llm_response.tool_calls),
            )
            messages.append(assistant_message)

            if not llm_response.tool_calls:
                return self._finalize_result(
                    profile_id=profile.profile_id,
                    raw_content=llm_response.content,
                    tool_history=tool_history,
                    iterations=iterations,
                    tool_calls_used=tool_calls_used,
                )

            if tool_calls_used >= profile.max_tool_calls:
                return SpecialistRunResult(
                    ok=False,
                    profile_id=profile.profile_id,
                    raw_content=llm_response.content,
                    failure_reason="Maximum number of specialist tool calls reached.",
                    tool_history=tool_history,
                    iterations=iterations,
                    tool_calls_used=tool_calls_used,
                )

            for tool_call in llm_response.tool_calls:
                if tool_calls_used >= profile.max_tool_calls:
                    return SpecialistRunResult(
                        ok=False,
                        profile_id=profile.profile_id,
                        raw_content=llm_response.content,
                        failure_reason="Maximum number of specialist tool calls reached.",
                        tool_history=tool_history,
                        iterations=iterations,
                        tool_calls_used=tool_calls_used,
                    )

                tool_calls_used += 1
                tool_message, history_item = self._execute_tool_call(
                    registry=registry,
                    tool_name=tool_call.name,
                    arguments_json=tool_call.arguments_json,
                    tool_call_id=tool_call.id,
                    context=context,
                )
                messages.append(tool_message)
                tool_history.append(history_item)

        return SpecialistRunResult(
            ok=False,
            profile_id=profile.profile_id,
            failure_reason="Maximum number of specialist iterations reached.",
            tool_history=tool_history,
            iterations=iterations,
            tool_calls_used=tool_calls_used,
        )

    def _build_messages(
        self,
        *,
        profile: SpecialistProfile,
        request: SpecialistRunRequest,
        session_id: str,
    ) -> list[LLMMessage]:
        messages = [
            LLMMessage(role="system", content=self.settings.base_system_prompt),
            LLMMessage(role="system", content=self._build_specialist_system_prompt(profile=profile)),
            LLMMessage(role="system", content=self._build_scope_prompt_block(session_id=session_id)),
        ]
        messages.append(
            LLMMessage(
                role="user",
                content=json.dumps(request.to_payload(), ensure_ascii=False, indent=2),
            )
        )
        return messages

    def _build_specialist_system_prompt(self, *, profile: SpecialistProfile) -> str:
        lines = [
            f"Specialist profile: {profile.profile_id}",
            profile.system_prompt,
            "",
            "Specialist operating rules:",
            "- You are running as an ephemeral sub-agent invoked by a master agent.",
            "- Use only the tools exposed in this run. Do not assume hidden tools exist.",
            "- Drive the investigation to a bounded conclusion within this one run.",
            "- Base conclusions only on observed evidence and tool results.",
            "- Return exactly one JSON object matching the requested output format when you are done.",
            "- Do not wrap the final JSON in markdown fences.",
            "",
            "Output format:",
            json.dumps(SpecialistOutput.create_template(), ensure_ascii=False, indent=2),
        ]
        return "\n".join(lines)

    def _build_scope_prompt_block(self, *, session_id: str) -> str:
        allowed_roots = [str(path.resolve()) for path in self.settings.allowed_read_roots]
        knowledge_root = str(self.settings.knowledge_base_dir.resolve())
        allowed_hosts = self.settings.allowed_http_hosts or []
        allowed_methods = self.settings.allowed_http_methods or []
        lines = [
            "Execution scope:",
            f"- Parent session ID: {session_id}",
            "- Allowed local code roots:",
        ]
        if allowed_roots:
            lines.extend(f"  - {root}" for root in allowed_roots)
        else:
            lines.append("  - none")

        lines.extend(
            [
                "- Allowed knowledge base root:",
                f"  - {knowledge_root}",
                "- Allowed web hosts:",
            ]
        )
        if allowed_hosts:
            lines.extend(f"  - {host}" for host in allowed_hosts)
        else:
            lines.append("  - none")

        lines.append("- Allowed HTTP methods:")
        if allowed_methods:
            lines.extend(f"  - {method}" for method in allowed_methods)
        else:
            lines.append("  - none")
        return "\n".join(lines)

    def _execute_tool_call(
        self,
        *,
        registry: ToolRegistry,
        tool_name: str,
        arguments_json: str,
        tool_call_id: str,
        context: ExecutionContext,
    ) -> tuple[LLMMessage, dict[str, Any]]:
        arguments: dict[str, Any]
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            arguments = {}
            tool_content = f"Invalid JSON arguments for tool {tool_name}"
            tool_status: ToolExecutionStatus = "invalid_arguments"
        else:
            authz = self.policy_engine.authorize(tool_name, arguments, context)
            if not authz.allowed:
                tool_content = f"Tool denied by policy: {authz.reason}"
                tool_status = "policy_denied"
            else:
                try:
                    result = registry.execute(tool_name, arguments, context)
                    tool_content = result.content
                    tool_status = "ok" if result.ok else "tool_error"
                except Exception as exc:
                    logger.exception("Specialist tool execution crashed", extra={"tool_name": tool_name})
                    tool_content = f"Tool execution failed: {exc}"
                    tool_status = "execution_failed"

        tool_message = LLMMessage(role="tool", tool_call_id=tool_call_id, content=tool_content)
        history_item = {
            "tool_name": tool_name,
            "arguments": arguments,
            "status": tool_status,
            "content_preview": safe_preview(tool_content, limit=500),
        }
        return tool_message, history_item

    def _finalize_result(
        self,
        *,
        profile_id: str,
        raw_content: str,
        tool_history: list[dict[str, Any]],
        iterations: int,
        tool_calls_used: int,
    ) -> SpecialistRunResult:
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            return SpecialistRunResult(
                ok=False,
                profile_id=profile_id,
                raw_content=raw_content,
                failure_reason="Specialist returned invalid JSON output.",
                tool_history=tool_history,
                iterations=iterations,
                tool_calls_used=tool_calls_used,
            )

        output = SpecialistOutput.from_dict(payload)
        if output is None:
            return SpecialistRunResult(
                ok=False,
                profile_id=profile_id,
                raw_content=raw_content,
                failure_reason="Specialist returned an invalid output schema.",
                tool_history=tool_history,
                iterations=iterations,
                tool_calls_used=tool_calls_used,
            )

        return SpecialistRunResult(
            ok=True,
            profile_id=profile_id,
            output=output,
            raw_content=raw_content,
            tool_history=tool_history,
            iterations=iterations,
            tool_calls_used=tool_calls_used,
        )
