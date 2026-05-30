from __future__ import annotations

from agent_core.context_assembler import ContextAssembler
from agent_core.domain_hooks import DomainHooks
from agent_core.execution_context import (
    effective_allowed_http_hosts,
    effective_allowed_http_methods,
    effective_allowed_read_roots,
)
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.session_manager import SessionManager
from agent_core.settings import CoreSettings
from agent_core.llm.base import LLMMessage

logger = get_logger(__name__)


class PromptBuilder:
    """Build the provider-facing prompt stack for one user turn.

    The stack is assembled in layers:
    1. invariant system guidance from settings,
    2. explicit execution scope and domain guidance,
    3. memory blocks selected from the current session,
    4. the live user message for the turn.

    `ContextAssembler` owns history selection and budgeting. This class owns the
    stable prompt scaffolding that should always wrap that history.
    """

    def __init__(
        self,
        *,
        settings: CoreSettings,
        session_manager: SessionManager,
        domain_hooks: DomainHooks | None = None,
    ) -> None:
        self.settings = settings
        self.session_manager = session_manager
        self.domain_hooks = domain_hooks or DomainHooks()
        self.context_assembler = ContextAssembler(
            settings=settings,
            session_manager=session_manager,
        )

    def build_messages(self, *, user_input: str) -> list[LLMMessage]:
        base_messages: list[LLMMessage] = [LLMMessage(role="system", content=self.settings.base_system_prompt)]

        scope_block = self._build_scope_prompt_block()
        if scope_block:
            base_messages.append(LLMMessage(role="system", content=scope_block))

        for domain_block in self.domain_hooks.build_system_prompt_blocks(
            settings=self.settings,
            session_manager=self.session_manager,
        ):
            if domain_block:
                base_messages.append(LLMMessage(role="system", content=domain_block))

        # The assembler keeps history atomic as context blocks for as long as
        # possible, then flattens only the selected slice into provider-facing
        # chat messages.
        assembly = self.context_assembler.assemble(
            base_messages=base_messages,
            user_input=user_input,
        )
        messages = self._sanitize_messages(assembly.messages)

        logger.trace(
            "Built prompt messages",
            extra={
                "message_count": len(messages),
                "history_block_count": len(assembly.selected_blocks),
                "overflow_block_count": len(assembly.overflow_blocks),
            },
        )
        return messages

    def _build_scope_prompt_block(self) -> str:
        session_state = self.session_manager.get_state()
        allowed_roots = [str(path.resolve()) for path in effective_allowed_read_roots(self.settings, session_state)]
        knowledge_root = str(self.settings.knowledge_base_dir.resolve())
        allowed_hosts = effective_allowed_http_hosts(self.settings, session_state)
        allowed_methods = effective_allowed_http_methods(self.settings, session_state)
        session_id = self.session_manager.session_id or "default"

        lines = [
            "Execution scope:",
            f"- Session ID: {session_id}",
            "- Allowed local code roots:",
        ]
        if allowed_roots:
            lines.extend(f"  - {root}" for root in allowed_roots)
            lines.append("- For local code tools, use absolute paths inside these roots or paths relative to one of these roots.")
        else:
            lines.append("  - none")

        lines.extend(
            [
                "- Allowed knowledge base root:",
                f"  - {knowledge_root}",
                "- For knowledge tools, use this absolute root for broad searches or paths relative to it; pass exact returned paths to read_knowledge_chunk.",
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

        lines.extend(
            [
                "",
                "Scope rules:",
                "- Only read local files inside the allowed local code roots.",
                "- Only read knowledge files inside the allowed knowledge base root.",
                "- Only send HTTP requests to allowed web hosts using allowed methods.",
                "- Treat anything outside this scope as out of bounds unless the user changes the configuration.",
            ]
        )
        return "\n".join(lines)

    def _sanitize_messages(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        sanitized: list[LLMMessage] = []
        pending_assistant: LLMMessage | None = None
        pending_tool_messages: list[LLMMessage] = []
        pending_tool_calls: set[str] = set()

        def flush_pending_tool_exchange() -> None:
            nonlocal pending_assistant, pending_tool_messages, pending_tool_calls
            if pending_assistant is None:
                return
            if pending_tool_calls:
                logger.warning(
                    "Dropping assistant tool-call message with missing tool responses",
                    extra={"missing_tool_call_ids": sorted(pending_tool_calls)},
                )
            else:
                sanitized.append(pending_assistant)
                sanitized.extend(pending_tool_messages)
            pending_assistant = None
            pending_tool_messages = []
            pending_tool_calls = set()

        for msg in messages:
            role = msg.role

            if role == "assistant":
                # Once an assistant message with tool calls is emitted, only
                # matching tool responses should survive. This prevents stale or
                # orphan tool messages from breaking the chat transcript shape.
                flush_pending_tool_exchange()
                if msg.tool_calls:
                    pending_assistant = msg
                    pending_tool_calls = {tool_call.id for tool_call in msg.tool_calls}
                else:
                    sanitized.append(msg)
            elif role == "tool":
                tool_call_id = msg.tool_call_id
                if pending_assistant is not None and tool_call_id and tool_call_id in pending_tool_calls:
                    pending_tool_messages.append(msg)
                    pending_tool_calls.remove(tool_call_id)
                else:
                    logger.debug(
                        "Dropping orphan tool message",
                        extra={"tool_call_id": tool_call_id, "content_preview": safe_preview(msg.content)},
                    )
            else:
                flush_pending_tool_exchange()
                sanitized.append(msg)

        flush_pending_tool_exchange()

        return sanitized
