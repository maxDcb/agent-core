from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from agent_core.execution_context import ExecutionContext
from agent_core.llm.base import LLMCallOptions, LLMMessage, LLMToolDefinition
from agent_core.llm.provider_factory import LLMProviderConfig, build_provider_from_config
from agent_core.orchestrator import AgentOrchestrator
from agent_core.output_contracts import StructuredOutputContract
from agent_core.policy_engine import PolicyEngine
from agent_core.run_context import RunContext
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
    api_key: str
    api_version: str
    model: str
    enabled_backends: frozenset[str]


class DeterministicMemoryProvider:
    """Keep live kernel checks focused on the agent model and graph flow."""

    def complete_text(self, *, messages, model, temperature, options=None):
        return json.dumps(turn_memory_payload(objective="Live LangGraph kernel validation"))

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        raise AssertionError("The deterministic memory provider must not run the agent tool loop")


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


def _build_live_conversation_orchestrator(
    *,
    live_azure_config: LiveAzureConfig,
    tmp_path,
    agent_kernel_backend: str,
    tool: LiveEchoTool,
) -> AgentOrchestrator:
    provider = build_provider_from_config(
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
    return AgentOrchestrator(
        settings=settings,
        provider=provider,
        memory_provider=DeterministicMemoryProvider(),
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )


@pytest.mark.parametrize("agent_kernel_backend", ["native", "langgraph"])
def test_live_conversation_tool_loop_kernel_parity(
    live_azure_config: LiveAzureConfig,
    tmp_path,
    agent_kernel_backend: str,
) -> None:
    if "langchain" not in live_azure_config.enabled_backends:
        pytest.skip("The live agent-kernel matrix requires the LangChain model backend")
    orchestrator = _build_live_conversation_orchestrator(
        live_azure_config=live_azure_config,
        tmp_path=tmp_path,
        agent_kernel_backend=agent_kernel_backend,
        tool=LiveEchoTool(),
    )

    result = run_turn(
        orchestrator,
        "Call live_echo exactly once with text live-kernel-result. Then answer exactly: live-kernel-result",
    )

    assert result.status == "completed"
    assert result.content.strip().strip("\"'").rstrip(".") == "live-kernel-result"
    assert [item["status"] for item in orchestrator.session_manager.get_state()["tool_history"]] == ["ok"]
    assert [block.kind for block in orchestrator.session_manager.get_context_blocks()] == [
        "tool_exchange",
        "conversation_turn",
    ]
    trace_id = result.metadata["run_trace_id"]
    trace = orchestrator.session_manager.load_run_trace(str(trace_id))
    assert trace is not None
    assert trace["options"]["agent_kernel_backend"] == agent_kernel_backend
    assistant_events = [event for event in trace["events"] if event["type"] == "assistant_response_received"]
    assert len(assistant_events) == 2
    assert all(event["payload"]["model_backend"] == "langchain" for event in assistant_events)


def test_live_langgraph_pending_resume(live_azure_config: LiveAzureConfig, tmp_path) -> None:
    if "langchain" not in live_azure_config.enabled_backends:
        pytest.skip("The live agent-kernel matrix requires the LangChain model backend")
    orchestrator = _build_live_conversation_orchestrator(
        live_azure_config=live_azure_config,
        tmp_path=tmp_path,
        agent_kernel_backend="langgraph",
        tool=LivePendingTool(),
    )

    pending = run_turn(
        orchestrator,
        "Call live_pending exactly once with text live-resume-result. Wait for its result before answering.",
    )

    assert pending.status == "pending_tool_result"
    assert pending.pending_id
    assert pending.tool_name == "live_pending"

    completed = resume_turn(
        orchestrator,
        pending_id=pending.pending_id,
        tool_content="live-resume-result",
    )

    assert completed.status == "completed"
    assert "live-resume-result" in completed.content
    assert [item["status"] for item in orchestrator.session_manager.get_state()["tool_history"]] == [
        "pending",
        "ok",
    ]
    assert [block.kind for block in orchestrator.session_manager.get_context_blocks()] == [
        "tool_exchange",
        "conversation_turn",
    ]


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
