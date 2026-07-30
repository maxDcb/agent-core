from __future__ import annotations

import json
from typing import Any

import pytest

from agent_core import ToolArtifactPolicy
from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import LLMCompletionResult, LLMToolCall
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
    assert result.metadata["tool_artifact_usage"] == {
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
