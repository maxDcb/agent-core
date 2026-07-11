from __future__ import annotations

import json

from agent_core.llm.base import LLMCompletionResult, LLMToolCall
from agent_core.orchestrator import AgentOrchestrator
from agent_core.policy_engine import PolicyEngine
from agent_core.session_manager import SessionManager
from agent_core.session_repo import SessionRepository
from agent_core.settings import CoreSettings
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult


class FakeProvider:
    def __init__(self) -> None:
        self.responses = [
            LLMCompletionResult(
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="call-1",
                        name="delayed_tool",
                        arguments_json=json.dumps({"value": "whoami"}),
                    )
                ],
            ),
            LLMCompletionResult(content="final after tool output"),
        ]

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        return self.responses.pop(0)

    def complete_text(self, *, messages, model, temperature, options=None):
        return json.dumps(
            {
                "run_id": "run-0000",
                "objective": "Use delayed tool",
                "scope": [],
                "source_code_locations": [],
                "domain_extensions": {},
                "open_questions": [],
                "next_action": None,
                "stop_conditions": [],
                "constraints": [],
                "relevant_artifacts": [],
                "status": "active",
            }
        )


class DelayedTool:
    name = "delayed_tool"
    description = "Returns a pending result."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

    def execute(self, arguments, context):
        return ToolResult.pending_result("waiting", metadata={"value": arguments["value"]})


class EchoTool:
    name = "echo"
    description = "Returns its input immediately."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

    def execute(self, arguments, context):
        return ToolResult(ok=True, content=arguments["value"])


class MultiToolProvider(FakeProvider):
    def __init__(self) -> None:
        self.seen_messages = []
        self.responses = [
            LLMCompletionResult(
                content="",
                tool_calls=[
                    LLMToolCall(id="call-1", name="echo", arguments_json=json.dumps({"value": "before"})),
                    LLMToolCall(id="call-2", name="delayed_tool", arguments_json=json.dumps({"value": "pending"})),
                    LLMToolCall(id="call-3", name="echo", arguments_json=json.dumps({"value": "after"})),
                ],
            ),
            LLMCompletionResult(content="final after all tool outputs"),
        ]

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        self.seen_messages.append(list(messages))
        return self.responses.pop(0)


def build_orchestrator(tmp_path) -> AgentOrchestrator:
    settings = CoreSettings(
        openai_api_key="test",
        model="test-model",
        memory_model="test-model",
        session_file=tmp_path / "session.json",
        base_system_prompt="system",
        task_state_synthesis_prompt="task",
        session_summary_synthesis_prompt="summary",
        session_summary_merge_prompt="merge",
    )
    registry = ToolRegistry()
    registry.register(DelayedTool())
    return AgentOrchestrator(
        settings=settings,
        provider=FakeProvider(),
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )


def test_agent_core_can_resume_pending_tool_result(tmp_path) -> None:
    orchestrator = build_orchestrator(tmp_path)

    pending = orchestrator.run_turn_result("call the delayed tool")

    assert pending.status == "pending_tool_result"
    assert pending.pending_id
    assert pending.tool_name == "delayed_tool"
    assert pending.metadata == {"value": "whoami"}

    completed = orchestrator.resume_turn(
        pending_id=pending.pending_id,
        tool_content="tool output",
    )

    assert completed.status == "completed"
    assert completed.content == "final after tool output"
    assert [block.kind for block in orchestrator.session_manager.get_context_blocks()] == [
        "tool_exchange",
        "conversation_turn",
    ]

    tool_history_count = len(orchestrator.session_manager.get_state()["tool_history"])
    block_count = len(orchestrator.session_manager.get_context_blocks())
    repeated = orchestrator.resume_turn(
        pending_id=pending.pending_id,
        tool_content="different duplicate output",
    )

    assert repeated == completed
    assert len(orchestrator.session_manager.get_state()["tool_history"]) == tool_history_count
    assert len(orchestrator.session_manager.get_context_blocks()) == block_count

    reloaded_orchestrator = build_orchestrator(tmp_path)
    repeated_after_restart = reloaded_orchestrator.resume_turn(
        pending_id=pending.pending_id,
        tool_content="another duplicate output",
    )
    assert repeated_after_restart == completed


def test_pending_multi_tool_exchange_resumes_remaining_calls_before_model_call(tmp_path) -> None:
    settings = CoreSettings(
        openai_api_key="test",
        model="test-model",
        memory_model="test-model",
        session_file=tmp_path / "session.json",
        base_system_prompt="system",
        task_state_synthesis_prompt="task",
        session_summary_synthesis_prompt="summary",
        session_summary_merge_prompt="merge",
    )
    provider = MultiToolProvider()
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(DelayedTool())
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )

    pending = orchestrator.run_turn_result("run all three tools")
    assert pending.status == "pending_tool_result"

    completed = orchestrator.resume_turn(
        pending_id=pending.pending_id or "",
        tool_content="resolved",
    )

    assert completed.content == "final after all tool outputs"
    resumed_messages = provider.seen_messages[1]
    tool_messages = [message for message in resumed_messages if message.role == "tool"]
    assert [(message.tool_call_id, message.content) for message in tool_messages] == [
        ("call-1", "before"),
        ("call-2", "resolved"),
        ("call-3", "after"),
    ]
    assert [item["status"] for item in orchestrator.session_manager.get_state()["tool_history"]] == [
        "ok",
        "pending",
        "ok",
        "ok",
    ]
    assert [block.kind for block in orchestrator.session_manager.get_context_blocks()] == [
        "tool_exchange",
        "conversation_turn",
    ]
