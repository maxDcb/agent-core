from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from agent_core.llm.base import LLMCompletionResult, LLMToolCall
from agent_core.llm.errors import LLMProviderError
from agent_core.orchestrator import AgentOrchestrator
from agent_core.policy_engine import PolicyEngine
from agent_core.session_manager import SessionManager
from agent_core.session_repo import SessionRepository
from agent_core.settings import CoreSettings
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult
from tests.run_helpers import resume_turn, run_turn, turn_memory_payload


class ScriptedProvider:
    def __init__(self, responses: list[LLMCompletionResult | Exception]) -> None:
        self.responses = list(responses)
        self.chat_calls = 0

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        self.chat_calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def complete_text(self, *, messages, model, temperature, options=None):
        return json.dumps(turn_memory_payload(objective="Kernel parity"))


class EchoTool:
    name = "echo"
    description = "Echo a value."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments, context):
        return ToolResult(ok=True, content=f"echo:{arguments['value']}")


class PendingTool(EchoTool):
    name = "pending"
    description = "Wait for an externally supplied result."

    def execute(self, arguments, context):
        return ToolResult.pending_result("waiting", metadata={"job_id": arguments["value"]})


def tool_call(name: str = "echo", *, value: str = "hello") -> LLMCompletionResult:
    return LLMCompletionResult(
        content="",
        tool_calls=[
            LLMToolCall(
                id="call-1",
                name=name,
                arguments_json=json.dumps({"value": value}),
            )
        ],
        provider="azure_openai",
        model_backend="langchain",
        model="deployment",
        provider_request_id="request-tool",
    )


def build_orchestrator(
    root: Path,
    *,
    backend: str,
    provider: ScriptedProvider,
    tool: EchoTool | None = None,
    max_tool_calls: int = 100,
) -> AgentOrchestrator:
    settings = CoreSettings(
        openai_api_key="test",
        model="fake",
        memory_model="fake",
        session_file=root / "session.json",
        base_system_prompt="system",
        turn_memory_synthesis_prompt="memory",
        max_active_context_tokens=100000,
        max_tool_calls_per_turn=max_tool_calls,
        agent_kernel_backend=backend,
    )
    registry = ToolRegistry()
    registry.register(tool or EchoTool())
    return AgentOrchestrator(
        settings=settings,
        provider=provider,
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )


def _trace_event_types(orchestrator: AgentOrchestrator, run_trace_id: str) -> list[str]:
    trace = orchestrator.session_manager.load_run_trace(run_trace_id)
    assert trace is not None
    events = trace["events"]
    assert isinstance(events, list)
    return [event["type"] for event in events if isinstance(event, dict)]


def _direct_tool_snapshot(root: Path, backend: str) -> dict[str, Any]:
    provider = ScriptedProvider([tool_call(), LLMCompletionResult(content="final")])
    orchestrator = build_orchestrator(root, backend=backend, provider=provider)

    result = run_turn(orchestrator, "echo once")
    trace_id = result.metadata["run_trace_id"]
    assert isinstance(trace_id, str)
    trace = orchestrator.session_manager.load_run_trace(trace_id)
    assert trace is not None
    assert trace["options"]["agent_kernel_backend"] == backend
    blocks = orchestrator.session_manager.get_context_blocks()
    return {
        "kernel_backend": orchestrator._direct_turn_kernel.backend,
        "status": result.status,
        "content": result.content,
        "provider_calls": provider.chat_calls,
        "block_kinds": [block.kind for block in blocks],
        "tool_statuses": [item["status"] for item in orchestrator.session_manager.get_state()["tool_history"]],
        "trace_events": _trace_event_types(orchestrator, trace_id),
    }


def test_langgraph_direct_tool_loop_matches_native_observable_behavior(tmp_path: Path) -> None:
    native = _direct_tool_snapshot(tmp_path / "native", "native")
    langgraph = _direct_tool_snapshot(tmp_path / "langgraph", "langgraph")

    assert native.pop("kernel_backend") == "native"
    assert langgraph.pop("kernel_backend") == "langgraph"
    assert langgraph == native


def test_langgraph_direct_kernel_is_a_multi_node_graph(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(
        tmp_path,
        backend="langgraph",
        provider=ScriptedProvider([LLMCompletionResult(content="done")]),
    )

    node_names = set(orchestrator._direct_turn_kernel.graph.get_graph().nodes)

    assert {"call_model", "execute_tools", "complete_response", "complete_budget"}.issubset(node_names)


def test_langgraph_tracing_does_not_inherit_process_opt_in(tmp_path: Path, monkeypatch) -> None:
    import langsmith as ls

    observed: list[bool | str | None] = []
    original_tracing_context = ls.tracing_context

    @contextmanager
    def recording_tracing_context(*args, **kwargs):
        observed.append(kwargs.get("enabled"))
        with original_tracing_context(*args, **kwargs):
            yield

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(ls, "tracing_context", recording_tracing_context)
    orchestrator = build_orchestrator(
        tmp_path,
        backend="langgraph",
        provider=ScriptedProvider([LLMCompletionResult(content="done")]),
    )

    result = run_turn(orchestrator, "answer")

    assert result.content == "done"
    assert observed[0] is False


def test_langgraph_direct_pending_resume_uses_existing_persistence_contract(tmp_path: Path) -> None:
    provider = ScriptedProvider([tool_call("pending", value="job-1"), LLMCompletionResult(content="resolved")])
    orchestrator = build_orchestrator(
        tmp_path,
        backend="langgraph",
        provider=provider,
        tool=PendingTool(),
    )

    pending = run_turn(orchestrator, "start external work")
    assert pending.status == "pending_tool_result"
    assert pending.pending_id
    persisted = orchestrator.session_manager.get_state()["meta"][AgentOrchestrator.PENDING_TURN_META_KEY]
    assert persisted["pending_id"] == pending.pending_id

    completed = resume_turn(orchestrator, pending_id=pending.pending_id, tool_content="external result")

    assert completed.status == "completed"
    assert completed.content == "resolved"
    assert provider.chat_calls == 2
    assert [block.kind for block in orchestrator.session_manager.get_context_blocks()] == [
        "tool_exchange",
        "conversation_turn",
    ]


def test_langgraph_direct_preserves_provider_failure_handling(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            LLMProviderError(
                kind="request_error",
                user_message="The model is temporarily unavailable.",
                detail="synthetic provider failure",
            )
        ]
    )
    orchestrator = build_orchestrator(tmp_path, backend="langgraph", provider=provider)

    result = run_turn(orchestrator, "answer")

    assert result.status == "completed"
    assert result.content == "The model is temporarily unavailable."
    assert result.metadata["stop_reason"] == "provider_failure"
    assert result.metadata["provider_error_kind"] == "request_error"


def test_langgraph_direct_preserves_tool_budget_completion(tmp_path: Path) -> None:
    provider = ScriptedProvider([tool_call()])
    orchestrator = build_orchestrator(
        tmp_path,
        backend="langgraph",
        provider=provider,
        max_tool_calls=0,
    )

    result = run_turn(orchestrator, "echo")

    assert result.content == "Maximum number of tool calls reached for this turn."
    assert orchestrator.session_manager.get_state()["tool_history"][0]["status"] == "budget_exhausted"


def test_unknown_agent_kernel_backend_fails_at_orchestrator_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported agent kernel backend"):
        build_orchestrator(
            tmp_path,
            backend="custom",
            provider=ScriptedProvider([LLMCompletionResult(content="unused")]),
        )
