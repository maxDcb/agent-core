"""Exercise real artifact reads through both execution loops and kernel backends."""

from __future__ import annotations

import json

import pytest

from agent_core.llm.base import LLMCompletionResult, LLMToolCall
from agent_core.llm_budget import LLMBudget
from agent_core.policy_engine import PolicyEngine
from agent_core.run_options import RunOptions
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskCheckpoint, StructuredTaskRunner, StructuredTaskSpec
from agent_core.tool_artifacts import ToolArtifactPolicy
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult
from tests.run_helpers import execution_context, resume_turn, run_turn, turn_memory_payload
from tests.test_investigation_modes import (
    EchoTool,
    ScriptedProvider,
    build_orchestrator,
    decision_payload,
    reflection_payload,
    tool_call,
)


class SyntheticDocument:
    name = "read_document"
    description = "Read a synthetic document containing numbered observations."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, arguments, context):
        return ToolResult(True, "".join(f"observation {i}: value={i * 7}\n" for i in range(1, 101)))


class RecoveryProvider:
    """Attempt a premature final answer unless the runtime offers a correction."""

    def __init__(self, *, repeat_error=False, scenario="small_page"):
        self.repeat_error = repeat_error
        self.scenario = scenario
        self.calls = []
        self.stage = "initial"
        self.recovery_seen = False

    def complete_text(self, *, messages, model, temperature, options=None):
        target = (options.metadata or {}).get("target") if options else None
        if target == "investigation_step_reflection":
            for message in messages:
                if "observation 40: value=280" in message.content and "artifact_page" in message.content:
                    self.stage = "read"
            return json.dumps(
                reflection_payload(
                    new_facts=["Observation 40 has value 280"] if self.stage == "read" else [],
                    remaining_gaps=[] if self.stage == "read" else ["The selected observation has not been read"],
                    should_continue=self.stage == "document",
                )
            )
        if target == "investigation_decision":
            return json.dumps(decision_payload("continue" if self.stage == "document" else "final"))
        return json.dumps(turn_memory_payload())

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        target = (options.metadata or {}).get("target") if options else None
        if target == "investigation_final_response" or not tools:
            return LLMCompletionResult(
                content="Observation 40 has value 280" if self.stage == "read" else "Document unread"
            )
        self.calls.append(list(messages))
        latest = next((message for message in reversed(messages) if message.role == "tool"), None)
        if latest is None:
            self.stage = "document"
            return self.call("read_document", {})
        payload = json.loads(latest.content)
        if payload["kind"] == "artifact_result":
            args = {
                "artifact_id": payload["artifact"]["artifact_id"],
                "operation": "read",
                "start_line": 40,
                "line_count": 1,
                "max_bytes": 120,
            }
            if self.scenario == "denied":
                args["artifact_id"] = "art_" + "0" * 32
            self.bad_args = args
            self.stage = "error"
            return self.call("agent_core_read_artifact", args)
        if payload["kind"] == "artifact_read_error":
            recovery = next(
                (
                    message
                    for message in reversed(messages)
                    if message.role == "system" and "suggested_action=" in message.content
                ),
                None,
            )
            if recovery is not None and messages.index(recovery) > messages.index(latest):
                self.recovery_seen = True
                action = json.loads(recovery.content.split("suggested_action=", 1)[1])
                return self.call(
                    "agent_core_read_artifact", self.bad_args if self.repeat_error else action["arguments"]
                )
            return LLMCompletionResult(content="Document unread")
        assert payload["kind"] == "artifact_page"
        assert payload["selection_complete"]
        assert payload["content"] == [{"line": 40, "text": "observation 40: value=280\n"}]
        self.stage = "read"
        return LLMCompletionResult(content="Observation 40 has value 280")

    def call(self, name, arguments):
        return LLMCompletionResult(
            content="",
            tool_calls=[
                LLMToolCall(
                    id=f"call-{len(self.calls)}",
                    name=name,
                    arguments_json=json.dumps(arguments),
                )
            ],
        )


def make_structured(tmp_path, provider, backend, **spec_options):
    settings = CoreSettings(
        base_system_prompt="Read synthetic documents.",
        artifacts_directory=tmp_path / "artifacts",
        agent_kernel_backend=backend,
    )
    registry = ToolRegistry()
    registry.register(SyntheticDocument())
    runner = StructuredTaskRunner(
        settings=settings, provider=provider, tool_registry=registry, policy_engine=PolicyEngine()
    )
    spec = StructuredTaskSpec(
        task_id="artifact-recovery",
        system_prompt="Read the requested observation.",
        objective="Read observation 40.",
        allowed_tools=["read_document"],
        max_iterations=8,
        max_tool_calls=1,
        tool_artifact_policy=ToolArtifactPolicy(max_complete_result_bytes=64, preview_bytes=32),
        **spec_options,
    )
    return runner, spec, execution_context(settings)


@pytest.mark.parametrize("backend", ["native", "langgraph"])
@pytest.mark.parametrize("mode", ["conversation", "direct", "structured"])
@pytest.mark.parametrize("scenario", ["small_page", "denied"])
def test_read_correction_in_execution_modes(tmp_path, backend, mode, scenario):
    provider = RecoveryProvider(scenario=scenario)
    if mode == "structured":
        runner, spec, context = make_structured(tmp_path, provider, backend)
        result = runner.run(spec=spec, context=context)
        answer = result.raw_content
    else:
        orchestrator = build_orchestrator(tmp_path, provider, agent_kernel_backend=backend)
        orchestrator.registry.register(SyntheticDocument())
        result = run_turn(
            orchestrator,
            "Read observation 40",
            options=RunOptions(
                mode="direct" if mode == "direct" else "investigate",
                max_iterations=8,
                max_tool_calls=2,
                require_initial_plan=False,
                tool_artifact_policy=ToolArtifactPolicy(max_complete_result_bytes=64, preview_bytes=32),
            ),
        )
        answer = result.content
    assert result.metadata["tool_artifact_usage"]["recovery_attempts"] == (1 if scenario == "small_page" else 0)
    if scenario == "small_page":
        assert provider.recovery_seen
        assert "280" in answer
    else:
        assert not provider.recovery_seen


@pytest.mark.parametrize("backend", ["native", "langgraph"])
@pytest.mark.parametrize("repeat_error", [False, True])
def test_structured_recovery_checkpoint_resumes_and_cannot_reset_counter(tmp_path, backend, repeat_error):
    provider = RecoveryProvider(repeat_error=repeat_error)
    runner, spec, context = make_structured(tmp_path, provider, backend)
    saved = None

    class Interrupted(Exception):
        pass

    def stop_after_recovery(checkpoint):
        nonlocal saved
        if checkpoint.tool_artifact_usage.recovery_attempts == 1:
            saved = checkpoint.to_dict()
            raise Interrupted()

    with pytest.raises(Interrupted):
        runner.run(spec=spec, context=context, on_checkpoint=stop_after_recovery)
    checkpoint = StructuredTaskCheckpoint.from_dict(saved)
    assert checkpoint is not None
    assert checkpoint.tool_artifact_usage.recovery_fingerprints
    result = runner.resume(spec=spec, context=context, checkpoint=checkpoint)
    assert result.metadata["tool_artifact_usage"]["recovery_attempts"] == 1
    assert len(provider.calls) < 8
    assert result.raw_content == ("Document unread" if repeat_error else "Observation 40 has value 280")


@pytest.mark.parametrize("backend", ["native", "langgraph"])
@pytest.mark.parametrize("constraint", ["iterations", "reads", "bytes", "llm_calls"])
def test_structured_read_correction_preserves_hard_budgets(tmp_path, backend, constraint):
    provider = RecoveryProvider()
    runner, spec, context = make_structured(tmp_path, provider, backend)
    if constraint == "iterations":
        spec.max_iterations = 2
    elif constraint == "reads":
        spec.tool_artifact_policy = ToolArtifactPolicy(
            max_complete_result_bytes=64, preview_bytes=32, max_reads_per_run=1
        )
    elif constraint == "bytes":
        spec.tool_artifact_policy = ToolArtifactPolicy(
            max_complete_result_bytes=64, preview_bytes=32, max_total_read_bytes=50
        )
    else:
        spec.llm_budget = LLMBudget(max_calls=3)
    result = runner.run(spec=spec, context=context)
    assert "280" not in result.raw_content
    assert result.metadata["tool_artifact_usage"]["internal_tool_calls"] == 1
    assert result.metadata["tool_artifact_usage"]["artifact_bytes_read"] == 0
    if constraint == "llm_calls":
        assert len(provider.calls) == 3
        assert not result.ok
    else:
        assert result.metadata["tool_artifact_usage"]["recovery_attempts"] == 0


@pytest.mark.parametrize("backend", ["native", "langgraph"])
@pytest.mark.parametrize("mode", ["direct", "investigate"])
def test_conversation_pending_resume_preserves_claimed_read_recovery(tmp_path, backend, mode):
    class PendingDocument(EchoTool):
        name = "read_document"

        def execute(self, arguments, context):
            if arguments["value"] == "wrong":
                return ToolResult(
                    False,
                    json.dumps(
                        {
                            "kind": "artifact_read_error",
                            "code": "invalid_selection",
                            "recoverable": True,
                            "suggested_action": {"tool": self.name, "arguments": {"value": "root"}},
                        }
                    ),
                )
            return ToolResult.pending_result("Waiting for document storage.")

    chat = [tool_call(name="read_document", value="wrong")]
    if mode == "direct":
        chat.append(LLMCompletionResult(content="Document unread"))
    chat.append(tool_call(name="read_document", value="root", call_id="corrected-read"))
    if mode == "direct":
        chat.append(LLMCompletionResult(content="Document total is 42"))
    provider = ScriptedProvider(
        chat=chat,
        reflections=[
            reflection_payload(remaining_gaps=["Unread"], should_continue=False),
            reflection_payload(new_facts=["Document total is 42"], should_continue=False),
        ],
        decisions=[decision_payload("blocked"), decision_payload("final")],
    )
    orchestrator = build_orchestrator(tmp_path, provider, agent_kernel_backend=backend)
    orchestrator.registry.register(PendingDocument())
    result = run_turn(
        orchestrator,
        "Read the document total",
        options=RunOptions(
            mode=mode,
            max_iterations=5,
            max_tool_calls=3,
            require_initial_plan=False,
        ),
    )
    assert result.status == "pending_tool_result"
    assert result.metadata["tool_artifact_usage"]["recovery_attempts"] == 1
    result = resume_turn(orchestrator, pending_id=result.pending_id, tool_content='{"document_total":42}')
    assert "42" in result.content
    assert result.metadata["tool_artifact_usage"]["recovery_attempts"] == 1
    assert len(result.metadata["tool_artifact_usage"]["recovery_fingerprints"]) == 1
