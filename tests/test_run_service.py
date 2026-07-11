from __future__ import annotations

import json

import pytest

from agent_core.llm.base import LLMCompletionResult, LLMToolCall
from agent_core.policy_engine import PolicyEngine
from agent_core.run_context import RunContext
from agent_core.run_service import AgentRunService
from agent_core.run_store import JsonFileRunStore
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskSpec
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult


class ContextTool:
    name = "read_run_context"
    description = "Return the explicit run context."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, arguments, context):
        _ = arguments
        return ToolResult(
            True,
            json.dumps(
                {
                    "namespace_id": context.namespace_id,
                    "run_id": context.run_id,
                    "phase": context.correlation["phase"],
                    "target": context.application_context["target"],
                }
            ),
        )


class ScriptedProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_with_tools(self, **kwargs):
        _ = kwargs
        self.calls += 1
        if self.calls == 1:
            return LLMCompletionResult(
                content="",
                tool_calls=[LLMToolCall(id="context-1", name="read_run_context", arguments_json="{}")],
            )
        return LLMCompletionResult(content=json.dumps({"summary": "context observed"}))


def test_run_service_persists_context_result_and_is_idempotent(tmp_path) -> None:
    settings = CoreSettings(session_file=tmp_path / "threads.json")
    registry = ToolRegistry()
    registry.register(ContextTool())
    provider = ScriptedProvider()
    store = JsonFileRunStore(tmp_path / "runs")
    service = AgentRunService(
        settings=settings,
        provider=provider,
        tool_registry=registry,
        policy_engine=PolicyEngine(),
        run_store=store,
    )
    context = RunContext(
        namespace_id="assessment-1",
        parent_id="job-1",
        correlation={"phase": "recon"},
        application_context={"target": "https://example.test"},
    )
    spec = StructuredTaskSpec(
        task_id="recon",
        system_prompt="Inspect the explicit context.",
        objective="Return a summary.",
        allowed_tools=["read_run_context"],
    )

    first = service.execute(spec=spec, context=context, run_id="run-1")
    repeated = service.execute(spec=spec, context=context, run_id="run-1")

    assert first.ok is True
    assert repeated.to_dict() == first.to_dict()
    assert provider.calls == 2
    assert json.loads(first.tool_history[0]["content_preview"])["run_id"] == "run-1"
    persisted = service.get(namespace_id="assessment-1", run_id="run-1")
    assert persisted is not None
    assert persisted.status == "completed"
    assert [run.run_id for run in service.list(namespace_id="assessment-1", parent_id="job-1")] == ["run-1"]


def test_run_id_cannot_be_rebound_to_another_context(tmp_path) -> None:
    settings = CoreSettings(session_file=tmp_path / "threads.json")
    provider = ScriptedProvider()
    service = AgentRunService(
        settings=settings,
        provider=provider,
        tool_registry=ToolRegistry(),
        policy_engine=PolicyEngine(),
        run_store=JsonFileRunStore(tmp_path / "runs"),
    )
    spec = StructuredTaskSpec(task_id="task", system_prompt="Return text.", objective="Finish.")
    service.execute(spec=spec, context=RunContext(namespace_id="one"), run_id="fixed")

    with pytest.raises(ValueError, match="different request"):
        service.execute(spec=spec, context=RunContext(namespace_id="one", parent_id="other"), run_id="fixed")
