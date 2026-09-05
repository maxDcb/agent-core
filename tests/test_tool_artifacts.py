from __future__ import annotations

import json
from typing import Any, Literal

import pytest

from agent_core import LLMContextPolicy, StructuredOutputContract, ToolArtifactPolicy
from agent_core.context_planner import LLMContextPlanner, estimate_llm_input_tokens, llm_context_scope
from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import LLMCompletionResult, LLMMessage, LLMToolCall
from agent_core.llm_budget import run_budgeted_llm_call
from agent_core.orchestrator import AgentOrchestrator
from agent_core.policy_engine import PolicyEngine
from agent_core.run_options import RunOptions
from agent_core.session_manager import SessionManager
from agent_core.session_repo import SessionRepository
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskCheckpoint, StructuredTaskRunner, StructuredTaskSpec
from agent_core.tool_artifacts import (
    READ_ARTIFACT_TOOL_NAME,
    JsonFileArtifactStore,
    ToolArtifactRuntime,
    artifact_descriptor_from_message,
    message_to_persistence_dict,
    tool_artifact_scope,
)
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult
from tests.run_helpers import execution_context, resume_turn, run_turn


class _LargeResultTool:
    name = "large_result"
    description = "Return a large deterministic result."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        return ToolResult(ok=True, content="PLAYWRIGHT-EVIDENCE:" + "x" * 2_000)


class _HugeResultTool:
    name = "huge_result"
    description = "Return a huge deterministic result."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        return ToolResult(ok=True, content="XML-EVIDENCE:<item>\"quoted\" & useful</item>\n" * 500)


class _ReadBackProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.artifact_id = ""
        self.next_offset = 0

    def complete_with_tools(self, **kwargs: Any) -> LLMCompletionResult:
        self.calls.append(kwargs)
        call_index = len(self.calls)
        if call_index == 1:
            assert all(tool.name != READ_ARTIFACT_TOOL_NAME for tool in kwargs["tools"])
            return LLMCompletionResult(
                content="",
                tool_calls=[LLMToolCall(id="call-large", name="large_result", arguments_json="{}")],
            )
        if call_index == 2:
            assert any(tool.name == READ_ARTIFACT_TOOL_NAME for tool in kwargs["tools"])
            result_message = next(message for message in kwargs["messages"] if message.tool_call_id == "call-large")
            envelope = json.loads(result_message.content)
            assert envelope["kind"] == "artifact_result"
            assert envelope["materialization"] == "preview"
            assert envelope["content"].startswith("PLAYWRIGHT-EVIDENCE")
            assert envelope["complete"] is False
            assert len(envelope["content"].encode("utf-8")) == envelope["returned_bytes"]
            assert envelope["returned_bytes"] <= 64
            assert envelope["returned_bytes"] < envelope["artifact"]["size_bytes"]
            self.artifact_id = envelope["artifact"]["artifact_id"]
            self.next_offset = envelope["next_offset"]
            return LLMCompletionResult(
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="call-read",
                        name=READ_ARTIFACT_TOOL_NAME,
                        arguments_json=json.dumps(
                            {
                                "artifact_id": self.artifact_id,
                                "offset": self.next_offset,
                                "limit": 80,
                            }
                        ),
                    )
                ],
            )
        read_message = next(message for message in kwargs["messages"] if message.tool_call_id == "call-read")
        payload = json.loads(read_message.content)
        assert payload["kind"] == "artifact_chunk"
        assert payload["artifact_id"] == self.artifact_id
        assert payload["offset"] == self.next_offset
        assert payload["content"].startswith("x")
        return LLMCompletionResult(content="done")

    def complete_text(self, **kwargs: Any) -> LLMCompletionResult:
        return LLMCompletionResult(content="{}")


class _PendingTool:
    name = "pending_result"
    description = "Return a result asynchronously."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        return ToolResult.pending_result("waiting")


class _PendingReadProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.artifact_id = ""

    def complete_with_tools(self, **kwargs: Any) -> LLMCompletionResult:
        self.calls += 1
        if self.calls == 1:
            return LLMCompletionResult(
                content="",
                tool_calls=[LLMToolCall(id="call-pending", name="pending_result", arguments_json="{}")],
            )
        if self.calls == 2:
            message = next(item for item in kwargs["messages"] if item.tool_call_id == "call-pending")
            envelope = json.loads(message.content)
            assert envelope["kind"] == "artifact_result"
            assert envelope["materialization"] == "preview"
            assert envelope["content"].startswith("ASYNC-EVIDENCE")
            self.artifact_id = envelope["artifact"]["artifact_id"]
            return LLMCompletionResult(
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="call-pending-read",
                        name=READ_ARTIFACT_TOOL_NAME,
                        arguments_json=json.dumps({"artifact_id": self.artifact_id, "limit": 32}),
                    )
                ],
            )
        return LLMCompletionResult(content="resumed")

    def complete_text(self, **kwargs: Any) -> LLMCompletionResult:
        return LLMCompletionResult(content="{}")


class _ReadUntilContextBoundaryProvider:
    def __init__(self, *, max_context_tokens: int, reserved_output_tokens: int) -> None:
        self.max_context_tokens = max_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.calls: list[dict[str, Any]] = []
        self.artifact_id = ""
        self.next_offset = 0
        self.returned_bytes: list[int] = []

    def complete_with_tools(self, **kwargs: Any) -> LLMCompletionResult:
        self.calls.append(kwargs)
        options = kwargs.get("options")
        estimated = estimate_llm_input_tokens(
            messages=kwargs["messages"],
            tools=kwargs["tools"],
            options=options,
        )
        output_reserve = (
            options.max_output_tokens
            if options is not None and options.max_output_tokens is not None
            else self.reserved_output_tokens
        )
        assert estimated <= self.max_context_tokens - output_reserve

        if not kwargs["tools"]:
            return LLMCompletionResult(content='{"answer":"bounded final answer"}')
        if len(self.calls) == 1:
            return LLMCompletionResult(
                content="",
                tool_calls=[LLMToolCall(id="call-large", name="huge_result", arguments_json="{}")],
            )

        last_tool_message = next(
            (message for message in reversed(kwargs["messages"]) if message.role == "tool"),
            None,
        )
        assert last_tool_message is not None
        payload = json.loads(last_tool_message.content)
        if payload["kind"] == "artifact_result":
            self.artifact_id = payload["artifact"]["artifact_id"]
            self.next_offset = payload["next_offset"]
        else:
            assert payload["kind"] == "artifact_chunk"
            self.returned_bytes.append(payload["returned_bytes"])
            self.next_offset = payload["next_offset"]
        return LLMCompletionResult(
            content="",
            tool_calls=[
                LLMToolCall(
                    id=f"call-read-{len(self.calls)}",
                    name=READ_ARTIFACT_TOOL_NAME,
                    arguments_json=json.dumps(
                        {"artifact_id": self.artifact_id, "offset": self.next_offset, "limit": 900}
                    ),
                )
            ],
        )

    def complete_text(self, **kwargs: Any) -> LLMCompletionResult:
        return LLMCompletionResult(content="{}")


def test_file_artifact_store_is_namespace_scoped_and_reads_bounded_chunks(tmp_path) -> None:
    store = JsonFileArtifactStore(tmp_path / "artifacts")
    descriptor = store.put_text(
        namespace_id="tenant-a",
        run_id="run-1",
        tool_name="browser",
        content="héllo world",
    )

    first = store.read_text(
        namespace_id="tenant-a",
        artifact_id=descriptor.artifact_id,
        offset=0,
        limit=2,
    )
    second = store.read_text(
        namespace_id="tenant-a",
        artifact_id=descriptor.artifact_id,
        offset=first.next_offset,
        limit=100,
    )

    assert first.content == "h"
    assert first.eof is False
    assert first.content + second.content == "héllo world"
    assert second.eof is True
    with pytest.raises(FileNotFoundError):
        store.read_all_text(namespace_id="tenant-b", artifact_id=descriptor.artifact_id)


def test_core_settings_require_artifact_storage_policy(tmp_path) -> None:
    settings = CoreSettings(
        base_system_prompt="system",
        artifacts_directory=tmp_path / "artifacts",
    )

    assert settings.tool_artifact_policy == ToolArtifactPolicy()
    with pytest.raises(ValueError, match="Tool artifact policy"):
        CoreSettings(
            base_system_prompt="system",
            artifacts_directory=tmp_path / "disabled",
            tool_artifact_policy=None,  # type: ignore[arg-type]
        )


def test_runtime_keeps_only_newest_artifact_results_hot_and_persists_references(tmp_path) -> None:
    store = JsonFileArtifactStore(tmp_path / "artifacts")
    runtime = ToolArtifactRuntime(
        policy=ToolArtifactPolicy(
            hot_context_bytes=10,
            max_complete_result_bytes=10,
            preview_bytes=4,
            max_read_bytes=16,
            max_reads_per_run=2,
            max_total_read_bytes=32,
        ),
        store=store,
        namespace_id="tenant",
        run_id="run",
    )
    older = runtime.externalize(
        tool_name="one",
        content="12345678",
        tool_call_id="call-1",
        status="ok",
    )
    newest = runtime.externalize(
        tool_name="two",
        content="abcdefgh",
        tool_call_id="call-2",
        status="tool_error",
    )

    projected = runtime.project_messages([older, newest])

    older_envelope = json.loads(projected[0].content)
    newest_envelope = json.loads(projected[1].content)
    assert older_envelope["materialization"] == "reference"
    assert older_envelope["content"] is None
    assert newest_envelope["materialization"] == "complete"
    assert newest_envelope["content"] == "abcdefgh"
    assert newest_envelope["status"] == "tool_error"
    persisted_envelope = json.loads(message_to_persistence_dict(newest)["content"])
    assert persisted_envelope["materialization"] == "reference"
    assert persisted_envelope["artifact"]["artifact_id"] == (
        artifact_descriptor_from_message(newest).artifact_id  # type: ignore[union-attr]
    )


def test_runtime_reduces_artifact_read_to_largest_context_safe_utf8_chunk(tmp_path) -> None:
    settings = CoreSettings(base_system_prompt="system", artifacts_directory=tmp_path / "artifacts")
    context = execution_context(settings, namespace_id="tenant")
    runtime = ToolArtifactRuntime(
        policy=ToolArtifactPolicy(
            hot_context_bytes=32,
            max_complete_result_bytes=32,
            preview_bytes=16,
            max_read_bytes=80,
            max_reads_per_run=3,
            max_total_read_bytes=240,
        ),
        store=JsonFileArtifactStore(settings.artifacts_directory),
        namespace_id=context.namespace_id,
        run_id=context.run_id,
    )
    message = runtime.externalize(
        tool_name="unicode_result",
        content="🧪" * 100,
        tool_call_id="call-unicode",
        status="ok",
    )
    descriptor = artifact_descriptor_from_message(message)
    assert descriptor is not None

    result = runtime.execute(
        tool_name=READ_ARTIFACT_TOOL_NAME,
        arguments={"artifact_id": descriptor.artifact_id, "offset": 0, "limit": 80},
        context=context,
        content_fits=lambda content: json.loads(content)["returned_bytes"] <= 17,
    )

    payload = json.loads(result.content)
    assert result.ok is True
    assert payload["content"] == "🧪" * 4
    assert payload["returned_bytes"] == 16
    assert payload["eof"] is False
    assert result.metadata["artifact_read_context_limited"] is True
    assert runtime.usage.internal_tool_calls == 1
    assert runtime.usage.artifact_bytes_read == 16
    assert runtime.usage.reads_rejected == 0


def test_runtime_rejects_read_when_even_smallest_chunk_cannot_fit_context(tmp_path) -> None:
    settings = CoreSettings(base_system_prompt="system", artifacts_directory=tmp_path / "artifacts")
    context = execution_context(settings, namespace_id="tenant")
    runtime = ToolArtifactRuntime(
        policy=ToolArtifactPolicy(max_read_bytes=32, max_reads_per_run=2, max_total_read_bytes=64),
        store=JsonFileArtifactStore(settings.artifacts_directory),
        namespace_id=context.namespace_id,
        run_id=context.run_id,
    )
    message = runtime.externalize(
        tool_name="large",
        content="evidence" * 100,
        tool_call_id="call-large",
        status="ok",
    )
    descriptor = artifact_descriptor_from_message(message)
    assert descriptor is not None

    result = runtime.execute(
        tool_name=READ_ARTIFACT_TOOL_NAME,
        arguments={"artifact_id": descriptor.artifact_id, "limit": 32},
        context=context,
        content_fits=lambda content: False,
    )

    assert result.ok is False
    assert result.metadata["artifact_read_context_exhausted"] is True
    assert "remaining model context" in result.content
    assert runtime.usage.internal_tool_calls == 1
    assert runtime.usage.artifact_bytes_read == 0
    assert runtime.usage.reads_rejected == 1


def test_context_planning_demotes_hot_artifact_preview_before_provider_call(tmp_path) -> None:
    settings = CoreSettings(base_system_prompt="system", artifacts_directory=tmp_path / "artifacts")
    runtime = ToolArtifactRuntime(
        policy=ToolArtifactPolicy(
            hot_context_bytes=128,
            max_complete_result_bytes=32,
            preview_bytes=128,
            max_read_bytes=32,
            max_reads_per_run=2,
            max_total_read_bytes=64,
        ),
        store=JsonFileArtifactStore(settings.artifacts_directory),
        namespace_id="tenant",
        run_id="run",
    )
    artifact_message = runtime.externalize(
        tool_name="large",
        content="x" * 1_000,
        tool_call_id="call-large",
        status="ok",
    )
    persisted = message_to_persistence_dict(artifact_message)
    reference_message = LLMMessage.from_history_dict(persisted)
    messages = [
        LLMMessage(role="system", content="system"),
        LLMMessage(role="user", content="inspect"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(id="call-large", name="large", arguments_json="{}")],
        ),
        artifact_message,
    ]
    reserve = 32
    reference_tokens = estimate_llm_input_tokens(messages=[*messages[:-1], reference_message])
    planner = LLMContextPlanner(
        LLMContextPolicy(
            max_context_tokens=reference_tokens + reserve,
            reserved_output_tokens=reserve,
            safety_margin_tokens=0,
        )
    )
    seen_materializations: list[str] = []

    with tool_artifact_scope(runtime), llm_context_scope(planner):
        run_budgeted_llm_call(
            messages=messages,
            purpose="artifact_projection",
            options=None,
            invoke=lambda options: (
                seen_materializations.append(json.loads(messages[-1].content)["materialization"]),
                LLMCompletionResult(content="ok"),
            )[-1],
        )

    assert seen_materializations == ["reference"]
    assert planner.usage.calls_overflowed == 0
    assert planner.usage.plans_created == 1


def test_structured_loop_externalizes_results_and_dispatches_internal_reads_outside_app_budget(tmp_path) -> None:
    settings = CoreSettings(
        base_system_prompt="system",
        artifacts_directory=tmp_path / "artifacts",
    )
    registry = ToolRegistry()
    registry.register(_LargeResultTool())
    provider = _ReadBackProvider()
    runner = StructuredTaskRunner(
        settings=settings,
        provider=provider,
        tool_registry=registry,
        policy_engine=PolicyEngine(),
    )
    checkpoints: list[dict[str, Any]] = []
    policy = ToolArtifactPolicy(
        hot_context_bytes=128,
        max_complete_result_bytes=128,
        preview_bytes=64,
        max_read_bytes=80,
        max_reads_per_run=3,
        max_total_read_bytes=240,
    )

    result = runner.run(
        spec=StructuredTaskSpec(
            task_id="artifact-loop",
            system_prompt="Inspect the evidence.",
            objective="Read the stored result.",
            allowed_tools=["large_result"],
            max_tool_calls=1,
            max_iterations=4,
            tool_artifact_policy=policy,
        ),
        context=execution_context(settings, namespace_id="tenant"),
        on_checkpoint=lambda checkpoint: checkpoints.append(checkpoint.to_dict()),
    )

    assert result.ok is True
    assert result.raw_content == "done"
    assert result.tool_calls_used == 1
    assert [item["tool_kind"] for item in result.tool_history] == ["application", "runtime"]
    assert {key: result.metadata["tool_artifact_usage"][key] for key in (
        "artifacts_written", "artifact_bytes_written", "internal_tool_calls", "artifact_bytes_read", "reads_rejected"
    )} == {
        "artifacts_written": 1,
        "artifact_bytes_written": 2_020,
        "internal_tool_calls": 1,
        "artifact_bytes_read": 80,
        "reads_rejected": 0,
    }
    assert provider.artifact_id
    serialized = json.dumps(checkpoints)
    assert "x" * 1_000 not in serialized
    assert all(
        "PLAYWRIGHT-EVIDENCE" not in message["content"]
        for checkpoint in checkpoints
        for message in checkpoint["messages"]
        if message.get("tool_call_id") == "call-large"
    )
    assert provider.artifact_id in serialized
    restored = StructuredTaskCheckpoint.from_dict(checkpoints[-1])
    assert restored is not None
    assert restored.tool_artifact_policy == policy
    assert restored.tool_artifact_usage.internal_tool_calls == 1
    previous_contract = dict(checkpoints[-1])
    previous_contract["schema_version"] = 5
    assert StructuredTaskCheckpoint.from_dict(previous_contract) is None


@pytest.mark.parametrize("kernel_backend", ["native", "langgraph"])
def test_structured_reads_stop_at_context_boundary_and_preserve_finalization_capacity(
    tmp_path,
    kernel_backend: Literal["native", "langgraph"],
) -> None:
    max_context_tokens = 2_400
    reserved_output_tokens = 160
    settings = CoreSettings(
        base_system_prompt="system",
        artifacts_directory=tmp_path / "artifacts",
        agent_kernel_backend=kernel_backend,
    )
    registry = ToolRegistry()
    registry.register(_HugeResultTool())
    provider = _ReadUntilContextBoundaryProvider(
        max_context_tokens=max_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    runner = StructuredTaskRunner(
        settings=settings,
        provider=provider,
        tool_registry=registry,
        policy_engine=PolicyEngine(),
    )
    checkpoints: list[dict[str, Any]] = []

    result = runner.run(
        spec=StructuredTaskSpec(
            task_id="bounded-artifact-loop",
            system_prompt="Inspect as much stored evidence as the context permits.",
            objective="Read the artifact until no safe chunk remains.",
            allowed_tools=["huge_result"],
            max_tool_calls=1,
            max_iterations=30,
            output_contract=StructuredOutputContract(
                name="bounded_answer",
                schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            ),
            llm_context_policy=LLMContextPolicy(
                max_context_tokens=max_context_tokens,
                reserved_output_tokens=reserved_output_tokens,
                safety_margin_tokens=0,
            ),
            tool_artifact_policy=ToolArtifactPolicy(
                hot_context_bytes=128,
                max_complete_result_bytes=128,
                preview_bytes=64,
                max_read_bytes=900,
                max_reads_per_run=30,
                max_total_read_bytes=20_000,
            ),
        ),
        context=execution_context(settings, namespace_id="tenant"),
        on_checkpoint=lambda checkpoint: checkpoints.append(checkpoint.to_dict()),
    )

    assert result.ok is True
    assert result.raw_content == '{"answer":"bounded final answer"}'
    assert result.output == {"answer": "bounded final answer"}
    assert result.metadata["forced_finalization"] is True
    assert "remaining model context" in result.metadata["budget_failure_reason"]
    assert result.metadata["tool_artifact_usage"]["reads_rejected"] == 1
    assert result.metadata["tool_artifact_usage"]["artifact_bytes_read"] < 20_000
    assert len(provider.returned_bytes) >= 1
    assert result.tool_history[-1]["artifact_read_context_exhausted"] is True
    assert any(checkpoint["phase"] == "finalization" for checkpoint in checkpoints)


def test_conversation_loop_exposes_internal_reader_without_consuming_application_budget(tmp_path) -> None:
    settings = CoreSettings(
        base_system_prompt="system",
        memory_model="fake",
        session_file=tmp_path / "session.json",
        artifacts_directory=tmp_path / "artifacts",
        max_tool_calls_per_turn=1,
    )
    registry = ToolRegistry()
    registry.register(_LargeResultTool())
    provider = _ReadBackProvider()
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        memory_provider=provider,
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )
    policy = ToolArtifactPolicy(
        hot_context_bytes=128,
        max_complete_result_bytes=128,
        preview_bytes=64,
        max_read_bytes=80,
        max_reads_per_run=3,
        max_total_read_bytes=240,
    )

    result = run_turn(
        orchestrator,
        "inspect",
        options=RunOptions.direct(tool_artifact_policy=policy),
    )

    assert result.status == "completed"
    assert result.content == "done"
    assert result.metadata["tool_calls_used"] == 1
    assert result.metadata["tool_artifact_usage"]["internal_tool_calls"] == 1
    state = orchestrator.session_manager.get_state()
    exchange_blocks = [block for block in state["context_blocks"] if block["kind"] == "tool_exchange"]
    serialized = json.dumps(exchange_blocks)
    assert provider.artifact_id in serialized
    assert "x" * 1_000 not in serialized


def test_pending_conversation_resume_restores_artifact_policy_and_usage(tmp_path) -> None:
    settings = CoreSettings(
        base_system_prompt="system",
        memory_model="fake",
        session_file=tmp_path / "session.json",
        artifacts_directory=tmp_path / "artifacts",
        max_tool_calls_per_turn=1,
    )
    registry = ToolRegistry()
    registry.register(_PendingTool())
    provider = _PendingReadProvider()
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        memory_provider=provider,
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )
    policy = ToolArtifactPolicy(
        hot_context_bytes=64,
        max_complete_result_bytes=64,
        preview_bytes=32,
        max_read_bytes=32,
        max_reads_per_run=2,
        max_total_read_bytes=64,
    )

    pending = run_turn(
        orchestrator,
        "start",
        options=RunOptions.direct(tool_artifact_policy=policy),
    )
    assert pending.status == "pending_tool_result"
    completed = resume_turn(
        orchestrator,
        pending_id=pending.pending_id or "",
        tool_content="ASYNC-EVIDENCE:" + "z" * 1_000,
    )

    assert completed.status == "completed"
    assert completed.content == "resumed"
    assert completed.metadata["tool_artifact_policy"] == policy.to_dict()
    assert completed.metadata["tool_artifact_usage"]["artifacts_written"] == 1
    assert completed.metadata["tool_artifact_usage"]["internal_tool_calls"] == 1
    assert provider.artifact_id
