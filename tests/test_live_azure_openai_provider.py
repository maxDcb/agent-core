from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage, LLMToolDefinition
from agent_core.llm.provider_factory import LLMProviderConfig, build_provider_from_config
from agent_core.orchestrator import AgentOrchestrator
from agent_core.output_contracts import StructuredOutputContract
from agent_core.policy_engine import PolicyEngine
from agent_core.run_context import RunContext
from agent_core.run_options import RunOptions
from agent_core.session_manager import SessionManager
from agent_core.session_repo import SessionRepository
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskRunner, StructuredTaskSpec
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult
from tests.run_helpers import resume_turn, run_turn, turn_memory_payload

pytestmark = pytest.mark.live_llm


@dataclass(frozen=True, slots=True)
class LiveAzureConfig:
    endpoint: str
    api_key: str = field(repr=False)
    api_version: str
    model: str
    enabled_backends: frozenset[str]


class DeterministicMemoryProvider:
    """Keep live kernel checks focused on the agent model and graph flow."""

    def complete_text(self, *, messages, model, temperature, options=None):
        target = (options.metadata or {}).get("target") if options is not None else None
        if target == "investigation_step_reflection":
            return json.dumps(
                {
                    "observation_summary": "The live tool returned the requested marker.",
                    "new_facts": ["live-investigation-result"],
                    "updated_hypotheses": [],
                    "rejected_hypotheses": [],
                    "remaining_gaps": [],
                    "resolved_gaps": [],
                    "recommended_next_actions": [],
                    "risk_notes": [],
                    "confidence": 1.0,
                    "should_continue": False,
                    "stop_reason": "live integration evidence collected",
                }
            )
        if target == "investigation_decision":
            return json.dumps(
                {
                    "kind": "final",
                    "reason_summary": "The live integration evidence is sufficient.",
                    "next_action": None,
                    "question": None,
                    "required_approval": False,
                }
            )
        return json.dumps(turn_memory_payload(objective="Live LangGraph kernel validation"))

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        raise AssertionError("The deterministic memory provider must not run the agent tool loop")


class RecordingProvider:
    """Record real provider latency and usage without changing its contract."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def complete_text(self, *, messages, model, temperature, options=None):
        return self._record(
            method="complete_text",
            target=(options.metadata or {}).get("target") if options is not None else None,
            call=lambda: self.delegate.complete_text(
                messages=messages,
                model=model,
                temperature=temperature,
                options=options,
            ),
        )

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        return self._record(
            method="complete_with_tools",
            target=(options.metadata or {}).get("target") if options is not None else None,
            call=lambda: self.delegate.complete_with_tools(
                messages=messages,
                tools=tools,
                model=model,
                temperature=temperature,
                options=options,
            ),
        )

    def _record(self, *, method: str, target: str | None, call: Any) -> LLMCompletionResult:
        started_at = time.perf_counter()
        result = call()
        wall_seconds = time.perf_counter() - started_at
        usage = result.usage
        self.calls.append(
            {
                "method": method,
                "target": target,
                "wall_seconds": wall_seconds,
                "provider_seconds": result.duration_seconds,
                "input_tokens": usage.input_tokens if usage is not None else 0,
                "output_tokens": usage.output_tokens if usage is not None else 0,
                "reasoning_tokens": usage.reasoning_output_tokens if usage is not None else 0,
                "total_tokens": usage.total_tokens if usage is not None else 0,
                "tool_call_count": len(result.tool_calls),
                "provider_attempts": result.provider_attempts,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class KernelEvalObservation:
    scenario: str
    backend: str
    status: str
    wall_seconds: float
    provider_seconds: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    call_targets: tuple[str, ...]
    tool_names: tuple[str, ...]
    tool_statuses: tuple[str, ...]
    context_block_kinds: tuple[str, ...]
    trace_event_types: tuple[str, ...]
    stop_reason: str | None
    iterations_used: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "backend": self.backend,
            "status": self.status,
            "wall_seconds": round(self.wall_seconds, 4),
            "provider_seconds": round(self.provider_seconds, 4),
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "call_targets": list(self.call_targets),
            "tool_names": list(self.tool_names),
            "tool_statuses": list(self.tool_statuses),
            "context_block_kinds": list(self.context_block_kinds),
            "trace_event_types": list(self.trace_event_types),
            "stop_reason": self.stop_reason,
            "iterations_used": self.iterations_used,
        }


class LiveEchoTool:
    name = "live_echo"
    description = "Return the supplied text unchanged. Always use this when the user asks for live_echo."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments, context):
        return ToolResult(ok=True, content=str(arguments["text"]))


class LivePendingTool(LiveEchoTool):
    name = "live_pending"
    description = "Start external work and wait for its result. Always use this when the user asks for live_pending."

    def execute(self, arguments, context):
        return ToolResult.pending_result("live-pending-wait", metadata={"requested_text": arguments["text"]})


class LiveReverseTool(LiveEchoTool):
    name = "live_reverse"
    description = "Return the supplied text reversed. Use only when the user explicitly asks for live_reverse."

    def execute(self, arguments, context):
        return ToolResult(ok=True, content=str(arguments["text"])[::-1])


@pytest.fixture(scope="session")
def live_azure_config() -> LiveAzureConfig:
    if not _env_flag("AGENT_CORE_RUN_LIVE_LLM_TESTS"):
        pytest.skip("Set AGENT_CORE_RUN_LIVE_LLM_TESTS=1 to run paid Azure OpenAI integration tests")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if not endpoint or not api_key:
        pytest.fail("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required for live_llm tests")

    raw_backends = os.getenv("AGENT_CORE_LIVE_LLM_BACKENDS", "native,langchain")
    enabled_backends = frozenset(value.strip() for value in raw_backends.split(",") if value.strip())
    if not enabled_backends:
        pytest.fail("AGENT_CORE_LIVE_LLM_BACKENDS must select native, langchain, or both")
    unsupported = enabled_backends - {"native", "langchain"}
    if unsupported:
        pytest.fail(f"Unsupported AGENT_CORE_LIVE_LLM_BACKENDS values: {sorted(unsupported)}")

    return LiveAzureConfig(
        endpoint=endpoint,
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip(),
        model=os.getenv("AGENT_CORE_LIVE_LLM_MODEL", "gpt-5.4-mini").strip(),
        enabled_backends=enabled_backends,
    )


@pytest.fixture(params=["native", "langchain"])
def live_provider(request, live_azure_config: LiveAzureConfig):
    backend = str(request.param)
    if backend not in live_azure_config.enabled_backends:
        pytest.skip(f"Backend {backend} not selected by AGENT_CORE_LIVE_LLM_BACKENDS")
    provider = build_provider_from_config(
        LLMProviderConfig(
            provider="azure_openai",
            model_backend=backend,
            azure_openai_endpoint=live_azure_config.endpoint,
            azure_openai_api_key=live_azure_config.api_key,
            azure_openai_api_version=live_azure_config.api_version,
            timeout_seconds=120.0,
            langchain_tracing_enabled=False,
        )
    )
    return backend, provider


def test_live_text_reasoning_and_usage(live_provider, live_azure_config: LiveAzureConfig) -> None:
    backend, provider = live_provider

    result = provider.complete_text(
        messages=[
            LLMMessage(role="system", content="Solve carefully and return only the integer."),
            LLMMessage(role="user", content="What is 17 multiplied by 19?"),
        ],
        model=live_azure_config.model,
        temperature=0.0,
        options=LLMCallOptions(reasoning_effort="low", max_output_tokens=256),
    )

    assert result.content.strip() == "323"
    assert result.provider == "azure_openai"
    assert result.model_backend == backend
    assert result.provider_request_id
    assert result.provider_attempts >= 1
    assert result.usage is not None
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


def test_live_strict_json_schema(live_provider, live_azure_config: LiveAzureConfig) -> None:
    backend, provider = live_provider
    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "backend_contract": {"type": "string", "enum": ["stable"]},
        },
        "required": ["ok", "backend_contract"],
        "additionalProperties": False,
    }

    result = provider.complete_text(
        messages=[
            LLMMessage(
                role="user",
                content='Return the object with "ok" true and "backend_contract" set to "stable".',
            )
        ],
        model=live_azure_config.model,
        temperature=0.0,
        options=LLMCallOptions(
            max_output_tokens=256,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "live_contract", "strict": True, "schema": schema},
            },
        ),
    )

    assert json.loads(result.content) == {"ok": True, "backend_contract": "stable"}
    assert result.model_backend == backend


def test_live_tool_call_and_result_roundtrip(live_provider, live_azure_config: LiveAzureConfig) -> None:
    backend, provider = live_provider
    user_message = LLMMessage(
        role="user",
        content=(
            "Call echo exactly once with text live-tool-result. "
            "After receiving its result, answer exactly with the returned text."
        ),
    )
    tool = LLMToolDefinition(
        name="echo",
        description="Return the provided text unchanged.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )

    first = provider.complete_with_tools(
        messages=[user_message],
        tools=[tool],
        model=live_azure_config.model,
        temperature=0.0,
        options=LLMCallOptions(max_output_tokens=256),
    )

    assert len(first.tool_calls) == 1
    tool_call = first.tool_calls[0]
    assert tool_call.name == "echo"
    assert json.loads(tool_call.arguments_json) == {"text": "live-tool-result"}

    final = provider.complete_with_tools(
        messages=[
            user_message,
            LLMMessage(role="assistant", content=first.content, tool_calls=first.tool_calls),
            LLMMessage(role="tool", content="live-tool-result", tool_call_id=tool_call.id),
        ],
        tools=[tool],
        model=live_azure_config.model,
        temperature=0.0,
        options=LLMCallOptions(max_output_tokens=256),
    )

    assert final.tool_calls == []
    assert final.content.strip().strip("\"'").rstrip(".") == "live-tool-result"
    assert first.model_backend == final.model_backend == backend


def test_live_structured_task_runner(live_provider, live_azure_config: LiveAzureConfig, tmp_path) -> None:
    backend, provider = live_provider
    settings = CoreSettings(
        llm_provider="azure_openai",
        llm_model_backend=backend,
        agent_kernel_backend="langgraph",
        azure_openai_endpoint=live_azure_config.endpoint,
        azure_openai_api_key=live_azure_config.api_key,
        azure_openai_api_version=live_azure_config.api_version,
        model=live_azure_config.model,
        memory_model=live_azure_config.model,
        llm_max_output_tokens=256,
        session_file=tmp_path / "session.json",
        base_system_prompt="live test",
        turn_memory_synthesis_prompt="live test",
    )
    runner = StructuredTaskRunner(
        settings=settings,
        provider=provider,
        tool_registry=ToolRegistry(),
        policy_engine=PolicyEngine(),
    )

    result = runner.run(
        spec=StructuredTaskSpec(
            task_id=f"live-structured-{backend}",
            system_prompt="Return the requested contract without prose.",
            objective='Return {"ok": true, "component": "structured_task"}.',
            output_contract=StructuredOutputContract(
                name="live_structured_task",
                strict=True,
                schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "component": {"type": "string", "enum": ["structured_task"]},
                    },
                    "required": ["ok", "component"],
                    "additionalProperties": False,
                },
            ),
            allowed_tools=[],
            max_iterations=1,
        ),
        context=ExecutionContext.from_run_context(
            context=RunContext(namespace_id="live-tests", run_id=f"live-{backend}"),
            settings=settings,
        ),
    )

    assert result.ok
    assert result.output == {"ok": True, "component": "structured_task"}
    assert result.llm_calls
    assert all(call.model_backend == backend for call in result.llm_calls)
    assert all(call.model == live_azure_config.model for call in result.llm_calls)
    assert runner._kernel.backend == "langgraph"


def test_live_langgraph_structured_task_tool_loop(
    live_provider,
    live_azure_config: LiveAzureConfig,
    tmp_path,
) -> None:
    backend, provider = live_provider
    settings = CoreSettings(
        llm_provider="azure_openai",
        llm_model_backend=backend,
        agent_kernel_backend="langgraph",
        azure_openai_endpoint=live_azure_config.endpoint,
        azure_openai_api_key=live_azure_config.api_key,
        azure_openai_api_version=live_azure_config.api_version,
        model=live_azure_config.model,
        memory_model=live_azure_config.model,
        llm_max_output_tokens=512,
        session_file=tmp_path / "session.json",
        base_system_prompt="live test",
        turn_memory_synthesis_prompt="live test",
    )
    registry = ToolRegistry()
    registry.register(LiveEchoTool())
    runner = StructuredTaskRunner(
        settings=settings,
        provider=provider,
        tool_registry=registry,
        policy_engine=PolicyEngine(),
    )
    checkpoint_phases: list[str] = []

    result = runner.run(
        spec=StructuredTaskSpec(
            task_id=f"live-langgraph-tool-{backend}",
            system_prompt="Follow the requested tool workflow and return the strict contract without prose.",
            objective=(
                "Call live_echo exactly once with text structured-langgraph-live. "
                'After receiving the tool result, return {"ok": true, "marker": "structured-langgraph-live"}.'
            ),
            constraints=["The live_echo tool call is mandatory."],
            allowed_tools=["live_echo"],
            output_contract=StructuredOutputContract(
                name="live_langgraph_structured_tool",
                strict=True,
                schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "marker": {
                            "type": "string",
                            "enum": ["structured-langgraph-live"],
                        },
                    },
                    "required": ["ok", "marker"],
                    "additionalProperties": False,
                },
            ),
            max_tool_calls=1,
            max_iterations=3,
        ),
        context=ExecutionContext.from_run_context(
            context=RunContext(namespace_id="live-tests", run_id=f"live-langgraph-tool-{backend}"),
            settings=settings,
        ),
        on_checkpoint=lambda checkpoint: checkpoint_phases.append(checkpoint.phase),
    )

    assert result.ok
    assert result.output == {"ok": True, "marker": "structured-langgraph-live"}
    assert result.tool_calls_used == 1
    assert [item["tool_name"] for item in result.tool_history] == ["live_echo"]
    assert {"tools", "finalization", "result"}.issubset(checkpoint_phases)
    assert all(call.model_backend == backend for call in result.llm_calls)
    assert all(call.model == live_azure_config.model for call in result.llm_calls)
    assert runner._kernel.backend == "langgraph"


def _build_live_conversation_orchestrator(
    *,
    live_azure_config: LiveAzureConfig,
    tmp_path,
    agent_kernel_backend: str,
    tool: LiveEchoTool,
    extra_tools: tuple[LiveEchoTool, ...] = (),
    real_internal_synthesis: bool = False,
) -> AgentOrchestrator:
    provider = RecordingProvider(
        build_provider_from_config(
            LLMProviderConfig(
                provider="azure_openai",
                model_backend="langchain",
                azure_openai_endpoint=live_azure_config.endpoint,
                azure_openai_api_key=live_azure_config.api_key,
                azure_openai_api_version=live_azure_config.api_version,
                timeout_seconds=120.0,
                langchain_tracing_enabled=False,
            )
        )
    )
    settings = CoreSettings(
        llm_provider="azure_openai",
        llm_model_backend="langchain",
        agent_kernel_backend=agent_kernel_backend,
        azure_openai_endpoint=live_azure_config.endpoint,
        azure_openai_api_key=live_azure_config.api_key,
        azure_openai_api_version=live_azure_config.api_version,
        model=live_azure_config.model,
        memory_model=live_azure_config.model,
        llm_max_output_tokens=512,
        session_file=tmp_path / "session.json",
        base_system_prompt=(
            "You are a deterministic integration-test assistant. Follow the user's explicit tool instruction exactly, "
            "then return the exact requested marker without commentary."
        ),
        turn_memory_synthesis_prompt="unused by the deterministic memory provider",
    )
    registry = ToolRegistry()
    registry.register(tool)
    for extra_tool in extra_tools:
        registry.register(extra_tool)
    return AgentOrchestrator(
        settings=settings,
        provider=provider,
        memory_provider=provider if real_internal_synthesis else DeterministicMemoryProvider(),
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )


def _observe_live_kernel_run(
    *,
    scenario: str,
    backend: str,
    orchestrator: AgentOrchestrator,
    result,
    wall_seconds: float,
    trace_id: str | None = None,
) -> KernelEvalObservation:
    provider = orchestrator.provider
    assert isinstance(provider, RecordingProvider)
    resolved_trace_id = trace_id or str(result.metadata["run_trace_id"])
    trace = orchestrator.session_manager.load_run_trace(resolved_trace_id)
    assert trace is not None
    calls = provider.calls
    return KernelEvalObservation(
        scenario=scenario,
        backend=backend,
        status=result.status,
        wall_seconds=wall_seconds,
        provider_seconds=sum(float(call["provider_seconds"] or 0.0) for call in calls),
        llm_calls=len(calls),
        input_tokens=sum(int(call["input_tokens"]) for call in calls),
        output_tokens=sum(int(call["output_tokens"]) for call in calls),
        reasoning_tokens=sum(int(call["reasoning_tokens"] or 0) for call in calls),
        total_tokens=sum(int(call["total_tokens"]) for call in calls),
        call_targets=tuple(str(call["target"] or call["method"]) for call in calls),
        tool_names=tuple(
            str(item["tool_name"]) for item in orchestrator.session_manager.get_state()["tool_history"]
        ),
        tool_statuses=tuple(
            str(item["status"]) for item in orchestrator.session_manager.get_state()["tool_history"]
        ),
        context_block_kinds=tuple(
            block.kind for block in orchestrator.session_manager.get_context_blocks()
        ),
        trace_event_types=tuple(str(event["type"]) for event in trace["events"]),
        stop_reason=(
            str(result.metadata["stop_reason"])
            if result.metadata.get("stop_reason") is not None
            else None
        ),
        iterations_used=(
            int(result.metadata["iterations_used"])
            if isinstance(result.metadata.get("iterations_used"), int)
            else None
        ),
    )


def _percent_delta(current: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return round(((current - baseline) / baseline) * 100, 2)


def _report_kernel_pair(native: KernelEvalObservation, langgraph: KernelEvalObservation) -> None:
    assert native.scenario == langgraph.scenario
    assert native.status == langgraph.status
    assert native.call_targets == langgraph.call_targets
    assert native.tool_names == langgraph.tool_names
    assert native.tool_statuses == langgraph.tool_statuses
    assert native.context_block_kinds == langgraph.context_block_kinds
    assert native.trace_event_types == langgraph.trace_event_types
    report = {
        "scenario": native.scenario,
        "native": native.to_dict(),
        "langgraph": langgraph.to_dict(),
        "langgraph_delta_percent": {
            "wall_seconds": _percent_delta(langgraph.wall_seconds, native.wall_seconds),
            "provider_seconds": _percent_delta(langgraph.provider_seconds, native.provider_seconds),
            "llm_calls": _percent_delta(float(langgraph.llm_calls), float(native.llm_calls)),
            "input_tokens": _percent_delta(float(langgraph.input_tokens), float(native.input_tokens)),
            "output_tokens": _percent_delta(float(langgraph.output_tokens), float(native.output_tokens)),
            "total_tokens": _percent_delta(float(langgraph.total_tokens), float(native.total_tokens)),
        },
    }
    print(f"LIVE_KERNEL_EVAL {json.dumps(report, sort_keys=True)}")


def test_live_conversation_tool_loop_kernel_parity(
    live_azure_config: LiveAzureConfig,
    tmp_path,
) -> None:
    if "langchain" not in live_azure_config.enabled_backends:
        pytest.skip("The live agent-kernel matrix requires the LangChain model backend")
    observations: dict[str, KernelEvalObservation] = {}
    for backend in ("native", "langgraph"):
        orchestrator = _build_live_conversation_orchestrator(
            live_azure_config=live_azure_config,
            tmp_path=tmp_path / backend,
            agent_kernel_backend=backend,
            tool=LiveEchoTool(),
            extra_tools=(LiveReverseTool(),),
        )

        started_at = time.perf_counter()
        result = run_turn(
            orchestrator,
            "Call live_echo exactly once with text live-kernel-result. Then answer exactly: live-kernel-result",
        )
        elapsed = time.perf_counter() - started_at

        assert result.status == "completed"
        assert result.content.strip().strip("\"'").rstrip(".") == "live-kernel-result"
        trace_id = str(result.metadata["run_trace_id"])
        trace = orchestrator.session_manager.load_run_trace(trace_id)
        assert trace is not None
        assert trace["options"]["agent_kernel_backend"] == backend
        assistant_events = [
            event for event in trace["events"] if event["type"] == "assistant_response_received"
        ]
        assert len(assistant_events) == 2
        assert all(event["payload"]["model_backend"] == "langchain" for event in assistant_events)
        observations[backend] = _observe_live_kernel_run(
            scenario="direct_tool_roundtrip",
            backend=backend,
            orchestrator=orchestrator,
            result=result,
            wall_seconds=elapsed,
        )

    native = observations["native"]
    langgraph = observations["langgraph"]
    assert native.llm_calls == langgraph.llm_calls == 2
    assert native.tool_names == langgraph.tool_names == ("live_echo",)
    assert native.tool_statuses == langgraph.tool_statuses == ("ok",)
    assert native.context_block_kinds == langgraph.context_block_kinds == (
        "tool_exchange",
        "conversation_turn",
    )
    _report_kernel_pair(native, langgraph)


def test_live_pending_resume_kernel_parity(live_azure_config: LiveAzureConfig, tmp_path) -> None:
    if "langchain" not in live_azure_config.enabled_backends:
        pytest.skip("The live agent-kernel matrix requires the LangChain model backend")
    observations: dict[str, KernelEvalObservation] = {}
    for backend in ("native", "langgraph"):
        orchestrator = _build_live_conversation_orchestrator(
            live_azure_config=live_azure_config,
            tmp_path=tmp_path / backend,
            agent_kernel_backend=backend,
            tool=LivePendingTool(),
            extra_tools=(LiveReverseTool(),),
        )

        started_at = time.perf_counter()
        pending = run_turn(
            orchestrator,
            "Call live_pending exactly once with text live-resume-result. Wait for its result before answering.",
        )

        assert pending.status == "pending_tool_result"
        assert pending.pending_id
        assert pending.tool_name == "live_pending"
        payload = orchestrator.session_manager.get_state()["meta"][AgentOrchestrator.PENDING_TURN_META_KEY]
        assert payload["agent_graph_checkpoint"] == {
            "schema_version": "1",
            "graph": "direct",
            "backend": backend,
            "resume_node": "resume_tool_exchange",
        }
        trace_id = str(payload["run_trace_id"])

        completed = resume_turn(
            orchestrator,
            pending_id=pending.pending_id,
            tool_content="live-resume-result",
        )
        elapsed = time.perf_counter() - started_at

        assert completed.status == "completed"
        assert "live-resume-result" in completed.content
        observations[backend] = _observe_live_kernel_run(
            scenario="direct_pending_resume",
            backend=backend,
            orchestrator=orchestrator,
            result=completed,
            wall_seconds=elapsed,
            trace_id=trace_id,
        )

    native = observations["native"]
    langgraph = observations["langgraph"]
    assert native.llm_calls == langgraph.llm_calls == 2
    assert native.tool_names == langgraph.tool_names == ("live_pending", "live_pending")
    assert native.tool_statuses == langgraph.tool_statuses == ("pending", "ok")
    assert native.context_block_kinds == langgraph.context_block_kinds == (
        "tool_exchange",
        "conversation_turn",
    )
    assert "agent_graph_checkpoint_restored" in native.trace_event_types
    assert "agent_graph_checkpoint_restored" in langgraph.trace_event_types
    _report_kernel_pair(native, langgraph)


def test_live_investigation_tool_flow_kernel_parity(live_azure_config: LiveAzureConfig, tmp_path) -> None:
    if "langchain" not in live_azure_config.enabled_backends:
        pytest.skip("The live agent-kernel matrix requires the LangChain model backend")
    observations: dict[str, KernelEvalObservation] = {}
    for backend in ("native", "langgraph"):
        orchestrator = _build_live_conversation_orchestrator(
            live_azure_config=live_azure_config,
            tmp_path=tmp_path / backend,
            agent_kernel_backend=backend,
            tool=LiveEchoTool(),
            extra_tools=(LiveReverseTool(),),
        )

        started_at = time.perf_counter()
        result = run_turn(
            orchestrator,
            (
                "Investigate by calling live_echo exactly once with text live-investigation-result. "
                "Use the returned evidence in the final answer."
            ),
            options=RunOptions.investigate(max_iterations=2, require_initial_plan=False),
        )
        elapsed = time.perf_counter() - started_at

        assert result.status == "completed"
        assert "live-investigation-result" in result.content
        assert result.metadata["mode"] == "investigate"
        assert result.metadata["investigation_state"]["facts"] == ["live-investigation-result"]
        trace = orchestrator.session_manager.load_run_trace(str(result.metadata["run_trace_id"]))
        assert trace is not None
        assert trace["options"]["agent_kernel_backend"] == backend
        assert "decision_completed" in [event["type"] for event in trace["events"]]
        observations[backend] = _observe_live_kernel_run(
            scenario="investigation_tool_reflect_decide",
            backend=backend,
            orchestrator=orchestrator,
            result=result,
            wall_seconds=elapsed,
        )

    native = observations["native"]
    langgraph = observations["langgraph"]
    assert native.llm_calls == langgraph.llm_calls == 2
    assert native.tool_names == langgraph.tool_names == ("live_echo",)
    assert native.tool_statuses == langgraph.tool_statuses == ("ok",)
    assert native.stop_reason == langgraph.stop_reason
    assert native.iterations_used == langgraph.iterations_used == 1
    _report_kernel_pair(native, langgraph)


def test_live_full_real_investigation_kernel_parity(live_azure_config: LiveAzureConfig, tmp_path) -> None:
    """Exercise plan, tool use, reflection, decision, and finalization with the real LLM."""

    if "langchain" not in live_azure_config.enabled_backends:
        pytest.skip("The live agent-kernel matrix requires the LangChain model backend")
    observations: dict[str, KernelEvalObservation] = {}
    for backend in ("native", "langgraph"):
        orchestrator = _build_live_conversation_orchestrator(
            live_azure_config=live_azure_config,
            tmp_path=tmp_path / backend,
            agent_kernel_backend=backend,
            tool=LiveEchoTool(),
            extra_tools=(LiveReverseTool(),),
            real_internal_synthesis=True,
        )

        started_at = time.perf_counter()
        result = run_turn(
            orchestrator,
            (
                "Investigate the marker by calling live_echo exactly once with text full-real-evidence. "
                "The returned marker is sufficient evidence; finish after observing it and include it verbatim."
            ),
            options=RunOptions.investigate(max_iterations=2, require_initial_plan=True),
        )
        elapsed = time.perf_counter() - started_at

        assert result.status == "completed"
        assert "full-real-evidence" in result.content
        assert result.metadata["mode"] == "investigate"
        trace = orchestrator.session_manager.load_run_trace(str(result.metadata["run_trace_id"]))
        assert trace is not None
        event_types = [event["type"] for event in trace["events"]]
        assert {
            "initial_plan_created",
            "tool_step_completed",
            "reflection_completed",
            "decision_completed",
        }.issubset(event_types)
        observations[backend] = _observe_live_kernel_run(
            scenario="full_real_investigation",
            backend=backend,
            orchestrator=orchestrator,
            result=result,
            wall_seconds=elapsed,
        )

    native = observations["native"]
    langgraph = observations["langgraph"]
    assert native.status == langgraph.status == "completed"
    assert native.tool_names == langgraph.tool_names == ("live_echo",)
    assert native.tool_statuses == langgraph.tool_statuses == ("ok",)
    assert native.context_block_kinds == langgraph.context_block_kinds
    _report_kernel_pair(native, langgraph)


def test_live_structured_investigation_output_kernel_parity(
    live_azure_config: LiveAzureConfig,
    tmp_path,
) -> None:
    if "langchain" not in live_azure_config.enabled_backends:
        pytest.skip("The live agent-kernel matrix requires the LangChain model backend")
    contract = StructuredOutputContract(
        name="live_kernel_structured_output",
        strict=True,
        schema={
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "marker": {"type": "string", "enum": ["structured-kernel-result"]},
            },
            "required": ["ok", "marker"],
            "additionalProperties": False,
        },
    )
    observations: dict[str, KernelEvalObservation] = {}
    for backend in ("native", "langgraph"):
        orchestrator = _build_live_conversation_orchestrator(
            live_azure_config=live_azure_config,
            tmp_path=tmp_path / backend,
            agent_kernel_backend=backend,
            tool=LiveEchoTool(),
            extra_tools=(LiveReverseTool(),),
        )

        started_at = time.perf_counter()
        result = run_turn(
            orchestrator,
            'Return an answer that establishes ok=true and the marker "structured-kernel-result".',
            options=RunOptions.investigate(
                max_iterations=1,
                require_initial_plan=False,
                final_output_mode="json_schema",
                final_output_contract=contract,
            ),
        )
        elapsed = time.perf_counter() - started_at

        assert result.status == "completed"
        assert json.loads(result.content) == {"ok": True, "marker": "structured-kernel-result"}
        assert result.metadata["final_output_mode"] == "json_schema"
        observations[backend] = _observe_live_kernel_run(
            scenario="investigation_structured_output",
            backend=backend,
            orchestrator=orchestrator,
            result=result,
            wall_seconds=elapsed,
        )

    native = observations["native"]
    langgraph = observations["langgraph"]
    assert native.llm_calls == langgraph.llm_calls == 2
    assert native.tool_names == langgraph.tool_names == ()
    assert native.tool_statuses == langgraph.tool_statuses == ()
    assert native.stop_reason == langgraph.stop_reason
    _report_kernel_pair(native, langgraph)


def test_live_full_real_deep_critique_kernel_parity(live_azure_config: LiveAzureConfig, tmp_path) -> None:
    if "langchain" not in live_azure_config.enabled_backends:
        pytest.skip("The live agent-kernel matrix requires the LangChain model backend")
    observations: dict[str, KernelEvalObservation] = {}
    for backend in ("native", "langgraph"):
        orchestrator = _build_live_conversation_orchestrator(
            live_azure_config=live_azure_config,
            tmp_path=tmp_path / backend,
            agent_kernel_backend=backend,
            tool=LiveEchoTool(),
            extra_tools=(LiveReverseTool(),),
            real_internal_synthesis=True,
        )

        started_at = time.perf_counter()
        result = run_turn(
            orchestrator,
            "Answer concisely with the exact factual marker deep-critique-result and no unsupported claims.",
            options=RunOptions.deep_investigate(max_iterations=1, require_initial_plan=False),
        )
        elapsed = time.perf_counter() - started_at

        assert result.status == "completed"
        assert "deep-critique-result" in result.content
        assert result.metadata["mode"] == "deep_investigate"
        trace = orchestrator.session_manager.load_run_trace(str(result.metadata["run_trace_id"]))
        assert trace is not None
        event_types = [event["type"] for event in trace["events"]]
        assert "final_critique_completed" in event_types
        observations[backend] = _observe_live_kernel_run(
            scenario="full_real_deep_critique",
            backend=backend,
            orchestrator=orchestrator,
            result=result,
            wall_seconds=elapsed,
        )

    native = observations["native"]
    langgraph = observations["langgraph"]
    assert native.status == langgraph.status == "completed"
    assert native.tool_names == langgraph.tool_names == ()
    assert native.tool_statuses == langgraph.tool_statuses == ()
    assert native.context_block_kinds == langgraph.context_block_kinds
    _report_kernel_pair(native, langgraph)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
