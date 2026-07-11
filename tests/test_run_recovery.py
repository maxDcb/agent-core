from __future__ import annotations

import json

import pytest

from agent_core import StructuredOutputContract
from agent_core.llm.base import LLMCompletionResult, LLMToolCall
from agent_core.policy_engine import PolicyEngine
from agent_core.run_context import RunContext
from agent_core.run_service import AgentRunService
from agent_core.run_store import JsonFileRunStore, RunExecutionBusyError
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskSpec
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult

pytestmark = pytest.mark.chaos


class CountingTool:
    name = "count"
    description = "Increment a deterministic test counter."

    def __init__(self, *, crash: bool = False) -> None:
        self.calls = 0
        self.crash = crash

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, arguments, context):
        _ = (arguments, context)
        self.calls += 1
        if self.crash:
            raise SystemExit("simulated process loss after external tool effect")
        return ToolResult(ok=True, content=f"count:{self.calls}")


class NamedCountingTool(CountingTool):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class TwoToolProvider:
    def complete_with_tools(self, **kwargs):
        _ = kwargs
        return LLMCompletionResult(
            content="",
            tool_calls=[
                LLMToolCall(id="first-1", name="first", arguments_json="{}"),
                LLMToolCall(id="second-1", name="second", arguments_json="{}"),
            ],
        )


class ToolThenCrashProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_with_tools(self, **kwargs):
        _ = kwargs
        self.calls += 1
        if self.calls == 1:
            return LLMCompletionResult(
                content="",
                tool_calls=[LLMToolCall(id="count-1", name="count", arguments_json="{}")],
            )
        raise SystemExit("simulated process loss during provider request")


class FinalProvider:
    def __init__(self) -> None:
        self.messages = []

    def complete_with_tools(self, **kwargs):
        self.messages = list(kwargs["messages"])
        return LLMCompletionResult(content=json.dumps({"summary": "resumed"}))


class ImmediateCrashProvider:
    def complete_with_tools(self, **kwargs):
        _ = kwargs
        raise SystemExit("simulated process loss during first provider request")


class ImmediateFinalProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_with_tools(self, **kwargs):
        _ = kwargs
        self.calls += 1
        return LLMCompletionResult(content="persisted final answer")


class NeverCallProvider:
    def complete_with_tools(self, **kwargs):
        _ = kwargs
        raise AssertionError("persisted provider response should be reused")


class DraftAndFinalProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_with_tools(self, **kwargs):
        _ = kwargs
        self.calls += 1
        if self.calls == 1:
            return LLMCompletionResult(content="investigation draft")
        return LLMCompletionResult(content=json.dumps({"summary": "validated final"}))


def _service(tmp_path, *, provider, tool: CountingTool | None = None):
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    return AgentRunService(
        settings=CoreSettings(session_file=tmp_path / "session.json"),
        provider=provider,
        tool_registry=registry,
        policy_engine=PolicyEngine(),
        run_store=JsonFileRunStore(tmp_path / "runs"),
    )


def _spec(*, objective: str = "Finish the task.") -> StructuredTaskSpec:
    return StructuredTaskSpec(
        task_id="recoverable",
        system_prompt="Use the counter once, then return JSON.",
        objective=objective,
        allowed_tools=["count"],
        max_iterations=3,
    )


def _contract_spec() -> StructuredTaskSpec:
    return StructuredTaskSpec(
        task_id="recoverable-contract",
        system_prompt="Investigate, then return the contract.",
        objective="Return a validated summary.",
        allowed_tools=["count"],
        output_contract=StructuredOutputContract(
            name="recoverable_contract",
            schema={
                "type": "object",
                "required": ["summary"],
                "additionalProperties": False,
                "properties": {"summary": {"type": "string"}},
            },
        ),
    )


def test_resume_continues_after_completed_tool_without_replaying_it(tmp_path) -> None:
    tool = CountingTool()
    first_service = _service(tmp_path, provider=ToolThenCrashProvider(), tool=tool)
    context = RunContext(namespace_id="assessment")

    try:
        first_service.execute(spec=_spec(), context=context, run_id="run-1")
    except SystemExit:
        pass

    interrupted = first_service.get(namespace_id="assessment", run_id="run-1")
    assert interrupted is not None
    assert interrupted.status == "running"
    assert interrupted.checkpoint is not None
    assert interrupted.checkpoint.payload["phase"] == "model_request"
    assert tool.calls == 1

    final_provider = FinalProvider()
    resumed = _service(tmp_path, provider=final_provider, tool=tool).resume(
        spec=_spec(),
        context=context,
        run_id="run-1",
    )

    assert resumed.ok is True
    assert json.loads(resumed.raw_content) == {"summary": "resumed"}
    assert resumed.tool_calls_used == 1
    assert tool.calls == 1
    assert any(message.role == "tool" and message.content == "count:1" for message in final_provider.messages)
    state = first_service.get(namespace_id="assessment", run_id="run-1")
    assert state is not None
    assert [attempt.status for attempt in state.attempts] == ["interrupted", "completed"]


def test_resume_blocks_when_tool_effect_is_ambiguous(tmp_path) -> None:
    tool = CountingTool(crash=True)
    provider = ToolThenCrashProvider()
    service = _service(tmp_path, provider=provider, tool=tool)
    context = RunContext(namespace_id="assessment")

    try:
        service.execute(spec=_spec(), context=context, run_id="run-ambiguous")
    except SystemExit:
        pass

    blocked = _service(tmp_path, provider=FinalProvider(), tool=tool).resume(
        spec=_spec(),
        context=context,
        run_id="run-ambiguous",
    )

    assert blocked.status == "blocked"
    assert blocked.error is not None
    assert blocked.error.kind == "ambiguous_tool_execution"
    assert blocked.metadata["tool_call_id"] == "count-1"
    assert tool.calls == 1

    tool.crash = False
    reconciled_provider = FinalProvider()
    completed = _service(tmp_path, provider=reconciled_provider, tool=tool).resolve_ambiguous_tool(
        spec=_spec(),
        context=context,
        run_id="run-ambiguous",
        tool_call_id="count-1",
        result=ToolResult(ok=True, content="count:1"),
    )

    assert completed.ok is True
    assert tool.calls == 1
    assert any(message.role == "tool" and message.content == "count:1" for message in reconciled_provider.messages)
    state = service.get(namespace_id="assessment", run_id="run-ambiguous")
    assert state is not None
    assert [attempt.status for attempt in state.attempts] == ["interrupted", "blocked", "completed"]


def test_resume_blocks_when_spec_fingerprint_changed(tmp_path) -> None:
    service = _service(tmp_path, provider=ImmediateCrashProvider(), tool=CountingTool())
    context = RunContext(namespace_id="assessment")
    try:
        service.execute(spec=_spec(), context=context, run_id="run-spec")
    except SystemExit:
        pass

    blocked = _service(tmp_path, provider=FinalProvider(), tool=CountingTool()).resume(
        spec=_spec(objective="A different objective."),
        context=context,
        run_id="run-spec",
    )

    assert blocked.status == "blocked"
    assert blocked.error is not None
    assert blocked.error.kind == "spec_mismatch"


def test_json_run_store_rejects_concurrent_execution_owners(tmp_path) -> None:
    first = JsonFileRunStore(tmp_path / "runs")
    second = JsonFileRunStore(tmp_path / "runs")

    with first.acquire_execution(namespace_id="assessment", run_id="run-locked"):
        with pytest.raises(RunExecutionBusyError):
            with second.acquire_execution(namespace_id="assessment", run_id="run-locked"):
                raise AssertionError("second execution owner should not acquire the lock")


def test_resume_reuses_persisted_final_provider_response(tmp_path) -> None:
    provider = ImmediateFinalProvider()
    first_service = _service(tmp_path, provider=provider, tool=CountingTool())
    context = RunContext(namespace_id="assessment")
    original_finalize = first_service.executor._finalize_result

    def crash_before_terminal_commit(**kwargs):
        _ = kwargs
        raise SystemExit("simulated process loss after final response checkpoint")

    first_service.executor._finalize_result = crash_before_terminal_commit  # type: ignore[method-assign]
    try:
        first_service.execute(spec=_spec(), context=context, run_id="run-final")
    except SystemExit:
        pass
    finally:
        first_service.executor._finalize_result = original_finalize  # type: ignore[method-assign]

    interrupted = first_service.get(namespace_id="assessment", run_id="run-final")
    assert interrupted is not None
    assert interrupted.checkpoint is not None
    assert interrupted.checkpoint.payload["phase"] == "result"
    assert provider.calls == 1

    completed = _service(tmp_path, provider=NeverCallProvider(), tool=CountingTool()).resume(
        spec=_spec(),
        context=context,
        run_id="run-final",
    )

    assert completed.ok is True
    assert completed.raw_content == "persisted final answer"
    assert provider.calls == 1


def test_resume_continues_remaining_multi_tool_calls_only(tmp_path) -> None:
    first_tool = NamedCountingTool("first")
    second_tool = NamedCountingTool("second")
    registry = ToolRegistry()
    registry.register(first_tool)
    registry.register(second_tool)
    settings = CoreSettings(session_file=tmp_path / "session.json")
    store = JsonFileRunStore(tmp_path / "runs")
    service = AgentRunService(
        settings=settings,
        provider=TwoToolProvider(),
        tool_registry=registry,
        policy_engine=PolicyEngine(),
        run_store=store,
    )
    original_save = store.save

    def crash_after_first_tool(state):
        original_save(state)
        checkpoint = state.checkpoint
        payload = checkpoint.payload if checkpoint is not None else {}
        pending = payload.get("pending_tool_calls")
        if (
            payload.get("phase") == "tools"
            and payload.get("next_tool_call_index") == 1
            and isinstance(pending, list)
            and [item.get("status") for item in pending if isinstance(item, dict)] == ["completed", "prepared"]
        ):
            raise SystemExit("simulated loss between tools")

    store.save = crash_after_first_tool  # type: ignore[method-assign]
    spec = StructuredTaskSpec(
        task_id="multi-recovery",
        system_prompt="Use both tools.",
        objective="Finish both calls.",
        allowed_tools=["first", "second"],
    )
    context = RunContext(namespace_id="assessment")
    try:
        service.execute(spec=spec, context=context, run_id="run-multi")
    except SystemExit:
        pass
    finally:
        store.save = original_save  # type: ignore[method-assign]

    assert first_tool.calls == 1
    assert second_tool.calls == 0

    resume_registry = ToolRegistry()
    resume_registry.register(first_tool)
    resume_registry.register(second_tool)
    resumed = AgentRunService(
        settings=settings,
        provider=FinalProvider(),
        tool_registry=resume_registry,
        policy_engine=PolicyEngine(),
        run_store=JsonFileRunStore(tmp_path / "runs"),
    ).resume(spec=spec, context=context, run_id="run-multi")

    assert resumed.ok is True
    assert first_tool.calls == 1
    assert second_tool.calls == 1
    assert [item["tool_name"] for item in resumed.tool_history] == ["first", "second"]


def test_resume_reuses_persisted_contract_finalization_response(tmp_path) -> None:
    provider = DraftAndFinalProvider()
    first_service = _service(tmp_path, provider=provider, tool=CountingTool())
    context = RunContext(namespace_id="assessment")
    original_finalize = first_service.executor._finalize_result

    def crash_before_contract_commit(**kwargs):
        _ = kwargs
        raise SystemExit("simulated loss after contract response checkpoint")

    first_service.executor._finalize_result = crash_before_contract_commit  # type: ignore[method-assign]
    try:
        first_service.execute(spec=_contract_spec(), context=context, run_id="run-contract")
    except SystemExit:
        pass
    finally:
        first_service.executor._finalize_result = original_finalize  # type: ignore[method-assign]

    interrupted = first_service.get(namespace_id="assessment", run_id="run-contract")
    assert interrupted is not None
    assert interrupted.checkpoint is not None
    assert interrupted.checkpoint.payload["phase"] == "result"
    assert interrupted.checkpoint.payload["result_kind"] == "contract"
    assert provider.calls == 2

    completed = _service(tmp_path, provider=NeverCallProvider(), tool=CountingTool()).resume(
        spec=_contract_spec(),
        context=context,
        run_id="run-contract",
    )

    assert completed.ok is True
    assert completed.output == {"summary": "validated final"}
    assert provider.calls == 2
