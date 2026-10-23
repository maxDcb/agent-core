from __future__ import annotations

import json

import pytest

from agent_core import AgentRunService, CoreSettings, JsonFileRunStore, RunContext, StructuredTaskSpec
from agent_core.llm.base import LLMCompletionResult, LLMToolCall
from agent_core.policy_engine import PolicyEngine
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult


class DelayedTool:
    name = "delayed"
    description = "Wait for an external result."

    def __init__(self) -> None:
        self.calls = 0

    def schema(self):
        return build_tool_definition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, arguments, context):
        _ = (arguments, context)
        self.calls += 1
        return ToolResult.pending_result("waiting", pending_id="external-1", metadata={"job_id": "job-1"})


class Provider:
    def __init__(self) -> None:
        self.calls = 0
        self.messages = []

    def complete_with_tools(self, **kwargs):
        self.calls += 1
        self.messages = list(kwargs["messages"])
        if self.calls == 1:
            return LLMCompletionResult(
                content="", tool_calls=[LLMToolCall(id="call-1", name="delayed", arguments_json="{}")],
            )
        return LLMCompletionResult(content=json.dumps({"summary": "received"}))


def service(tmp_path, tool, provider, *, backend="native"):
    registry = ToolRegistry()
    registry.register(tool)
    return AgentRunService(
        settings=CoreSettings(session_file=tmp_path / "session.json", agent_kernel_backend=backend),
        provider=provider, tool_registry=registry, policy_engine=PolicyEngine(),
        run_store=JsonFileRunStore(tmp_path / "runs"),
    )


def spec():
    return StructuredTaskSpec(
        task_id="delayed-test", system_prompt="Use the delayed tool once.",
        objective="Report its result.", allowed_tools=["delayed"], max_tool_calls=1,
    )


@pytest.mark.parametrize("backend", ["native", "langgraph"])
def test_structured_run_waits_without_worker_then_resumes_after_reload(tmp_path, backend):
    tool = DelayedTool()
    provider = Provider()
    first = service(tmp_path, tool, provider, backend=backend)
    context = RunContext(namespace_id="test")

    pending = first.execute(spec=spec(), context=context, run_id="run-1")
    assert pending.status == "pending"
    assert pending.metadata["pending_id"] == "external-1"
    assert pending.metadata["job_id"] == "job-1"
    assert tool.calls == 1 and provider.calls == 1

    reloaded = service(tmp_path, tool, provider, backend=backend)
    assert reloaded.execute(spec=spec(), context=context, run_id="run-1").to_dict() == pending.to_dict()
    assert reloaded.resume(spec=spec(), context=context, run_id="run-1").status == "pending"
    assert tool.calls == 1 and provider.calls == 1

    completed = reloaded.resume_pending_tool(
        spec=spec(), context=context, run_id="run-1", pending_id="external-1",
        tool_content="remote output", ok=True,
    )
    assert completed.status == "completed"
    assert completed.tool_calls_used == 1
    assert completed.tool_history[0]["status"] == "ok"
    assert tool.calls == 1 and provider.calls == 2
    assert any(message.role == "tool" and "remote output" in message.content for message in provider.messages)
    assert reloaded.resume_pending_tool(
        spec=spec(), context=context, run_id="run-1", pending_id="external-1",
        tool_content="duplicate", ok=True,
    ).to_dict() == completed.to_dict()
    assert tool.calls == 1 and provider.calls == 2


def test_structured_pending_rejects_wrong_id_and_preserves_failed_result(tmp_path):
    tool = DelayedTool()
    provider = Provider()
    run = service(tmp_path, tool, provider)
    context = RunContext(namespace_id="test")
    run.execute(spec=spec(), context=context, run_id="run-2")
    with pytest.raises(ValueError, match="Pending id"):
        run.resume_pending_tool(
            spec=spec(), context=context, run_id="run-2", pending_id="wrong",
            tool_content="ignored",
        )
    assert run.get(namespace_id="test", run_id="run-2").status == "pending"

    completed = run.resume_pending_tool(
        spec=spec(), context=context, run_id="run-2", pending_id="external-1",
        tool_content="no response", ok=False,
    )
    assert completed.status == "completed"
    assert completed.tool_history[0]["status"] == "tool_error"
    assert tool.calls == 1
