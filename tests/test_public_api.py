from __future__ import annotations

from contextlib import contextmanager

import agent_core
import agent_core.conversation as conversation
import agent_core.observability as observability
import agent_core.spi as spi
from agent_core import (
    AgentRunResult,
    AgentRunService,
    CoreSettings,
    JsonFileRunStore,
    RunContext,
    RunOptions,
    StructuredOutputContract,
    StructuredTaskSpec,
)


def test_run_engine_public_api_is_small_and_explicit() -> None:
    assert agent_core.__version__ == "0.3.0"
    assert set(agent_core.__all__) == {
        "AgentRunError",
        "AgentRunAttempt",
        "AgentRunMode",
        "AgentRunResult",
        "AgentRunService",
        "AgentRunState",
        "CoreSettings",
        "ExecutionContext",
        "ExecutionScope",
        "FinalOutputMode",
        "JSON_SCHEMA_DRAFT",
        "JsonFileRunStore",
        "RunContext",
        "RunCheckpoint",
        "RunExecutionBusyError",
        "RunOptions",
        "RunStatus",
        "RunStore",
        "RunStrategy",
        "StructuredOutputContract",
        "StructuredOutputValidationError",
        "StructuredOutputValidationIssue",
        "StructuredTaskResult",
        "StructuredTaskRunner",
        "StructuredTaskSpec",
        "__version__",
    }
    assert all(
        item is not None
        for item in (
            AgentRunResult,
            AgentRunService,
            CoreSettings,
            JsonFileRunStore,
            RunContext,
            RunOptions,
            StructuredOutputContract,
            StructuredTaskSpec,
        )
    )
    assert not hasattr(agent_core, "AgentOrchestrator")
    assert not hasattr(agent_core, "SessionManager")
    assert not hasattr(agent_core, "ToolRegistry")


def test_extension_conversation_and_observability_facades_are_explicit() -> None:
    assert set(spi.__all__) == {
        "AuthorizationResult",
        "BaseLLMProvider",
        "BaseTool",
        "DomainHooks",
        "InvestigationPromptSet",
        "LLMCallOptions",
        "LLMCompletionResult",
        "LLMMessage",
        "LLMProviderConfig",
        "LLMProviderError",
        "LLMToolCall",
        "LLMToolDefinition",
        "PolicyEngine",
        "ToolExecutionStatus",
        "ToolRegistry",
        "ToolResult",
        "build_memory_provider",
        "build_provider",
        "build_provider_from_config",
        "build_tool_definition",
        "load_prompt",
        "normalize_provider_name",
    }
    assert set(conversation.__all__) == {
        "AgentOrchestrator",
        "AgentTurnResult",
        "ConversationAgent",
        "ConversationStateView",
        "JsonFileSessionStore",
        "SessionManager",
        "SessionRepository",
        "SessionState",
        "SessionStore",
    }
    assert set(observability.__all__) == {
        "ContextBudget",
        "ExtraAwareFormatter",
        "PromptBlock",
        "PromptSnapshot",
        "RunTrace",
        "TraceEvent",
        "configure_logging",
        "get_logger",
        "safe_preview",
    }


def test_external_extensions_can_be_declared_using_only_public_facades() -> None:
    class ExternalProvider:
        def complete_text(self, *, messages, model, temperature, options=None):
            return "ok"

        def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
            return spi.LLMCompletionResult(content="ok")

    class ExternalRunStore:
        def __init__(self) -> None:
            self.states: dict[tuple[str, str], agent_core.AgentRunState] = {}

        def create(self, state: agent_core.AgentRunState) -> None:
            self.states[(state.context.namespace_id, state.run_id)] = state

        def save(self, state: agent_core.AgentRunState) -> None:
            self.create(state)

        def load(self, *, namespace_id: str, run_id: str) -> agent_core.AgentRunState | None:
            return self.states.get((namespace_id, run_id))

        def list(self, *, namespace_id: str, parent_id: str | None = None) -> list[agent_core.AgentRunState]:
            return [
                state
                for (stored_namespace, _), state in self.states.items()
                if stored_namespace == namespace_id and (parent_id is None or state.context.parent_id == parent_id)
            ]

        @contextmanager
        def acquire_execution(self, *, namespace_id: str, run_id: str):
            _ = (namespace_id, run_id)
            yield

    class ExternalDomain(spi.DomainHooks):
        pass

    class ExternalPolicy(spi.PolicyEngine):
        pass

    provider: spi.BaseLLMProvider = ExternalProvider()
    store: agent_core.RunStore = ExternalRunStore()

    assert provider.complete_text(messages=[], model="test", temperature=0) == "ok"
    assert store.list(namespace_id="test") == []
    assert ExternalDomain() is not None
    assert ExternalPolicy() is not None
    assert spi.ToolRegistry() is not None
