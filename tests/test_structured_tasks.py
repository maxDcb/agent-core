from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage, LLMToolCall
from agent_core.policy_engine import PolicyEngine
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import (
    StructuredOutputContract,
    StructuredTaskRunner,
    StructuredTaskSpec,
    _safe_tool_argument_summary,
)
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult


class FakeProvider:
    def __init__(self, responses: list[LLMCompletionResult]) -> None:
        self.responses = list(responses)
        self.last_messages: list[LLMMessage] = []
        self.last_tools: list[object] = []
        self.last_options: LLMCallOptions | None = None
        self.options_history: list[LLMCallOptions | None] = []
        self.tools_history: list[list[object]] = []

    def complete_text(
        self,
        *,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> str:
        raise AssertionError("complete_text should not be used by StructuredTaskRunner")

    def complete_with_tools(
        self,
        *,
        messages: list[LLMMessage],
        tools: list,
        model: str,
        temperature: float,
        options: LLMCallOptions | None = None,
    ) -> LLMCompletionResult:
        self.last_messages = list(messages)
        self.last_tools = list(tools)
        self.last_options = options
        self.options_history.append(options)
        self.tools_history.append(list(tools))
        return self.responses.pop(0)


class EchoTool:
    name = "echo_tool"
    description = "Echo one value."

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

    def execute(self, arguments: dict, context) -> ToolResult:
        return ToolResult(ok=True, content=f"echo:{arguments['value']}")


class SessionIdTool:
    name = "session_id_tool"
    description = "Return the execution context session id."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        )

    def execute(self, arguments: dict, context) -> ToolResult:
        return ToolResult(ok=True, content=context.session_id)


def _settings(root: Path) -> CoreSettings:
    return CoreSettings(
        allowed_read_roots=[root],
        knowledge_base_dir=root / "knowledge",
        base_system_prompt="Base system prompt.",
    )


def test_safe_tool_argument_summary_keeps_useful_context_without_sensitive_values() -> None:
    summary = _safe_tool_argument_summary(
        {
            "path": "workspace/server.ts",
            "query": "router.get",
            "password": "secret",
            "fields": {"email": "user@example.test", "password": "secret"},
        }
    )

    assert summary["path"] == "workspace/server.ts"
    assert summary["query"] == "router.get"
    assert "password" not in summary
    assert "fields" not in summary
    assert summary["redacted_argument_keys"] == ["fields", "password"]


def test_structured_task_runner_without_contract_returns_raw_text_after_tool_loop() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry = ToolRegistry()
        registry.register(EchoTool())
        provider = FakeProvider(
            [
                LLMCompletionResult(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call-1",
                            name="echo_tool",
                            arguments_json=json.dumps({"value": "pre-inventory"}),
                        )
                    ],
                ),
                LLMCompletionResult(content="done after pre-inventory"),
            ]
        )
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=provider,
            tool_registry=registry,
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="pre_inventory",
                system_prompt="Map the initial workspace state.",
                objective="Build a first workspace summary.",
                allowed_tools=["echo_tool"],
            ),
            session_id="session-1",
        )

        assert result.ok is True
        assert result.output is None
        assert result.raw_content == "done after pre-inventory"
        assert result.metadata["final_output_mode"] == "text"
        assert result.tool_calls_used == 1
        assert result.tool_history[0]["status"] == "ok"
        assert result.tool_history[0]["content_preview"] == "echo:pre-inventory"
        assert provider.last_options is not None
        assert provider.last_options.response_format is None
        assert provider.last_options.metadata["structured_task_id"] == "pre_inventory"
        assert provider.last_tools[0].name == "echo_tool"


def test_structured_task_runner_uses_structured_output_contract_when_requested() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        contract_schema = {
            "type": "object",
            "required": ["observations", "confidence"],
            "additionalProperties": False,
            "properties": {
                "observations": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
        }
        provider = FakeProvider(
            [
                LLMCompletionResult(
                    content=json.dumps(
                        {
                            "observations": ["ok"],
                            "confidence": 0.9,
                        }
                    )
                )
            ]
        )
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=provider,
            tool_registry=ToolRegistry(),
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="analysis",
                system_prompt="Analyze the input.",
                objective="Return structured analysis.",
                output_contract=StructuredOutputContract(
                    name="Analysis Contract!",
                    schema=contract_schema,
                    strict=True,
                    instructions=["Use confidence between 0 and 1."],
                ),
            ),
            session_id="session-1",
        )

        assert result.ok is True
        assert provider.last_options is not None
        assert provider.last_options.response_format == {
            "type": "json_schema",
            "json_schema": {
                "name": "Analysis_Contract",
                "schema": contract_schema,
                "strict": True,
            },
        }
        assert provider.last_options.response_format_fallback is None
        task_prompt = provider.last_messages[1].content
        assert "Provider-enforced structured output contract" in task_prompt
        assert "Analysis_Contract" in task_prompt
        assert "Use confidence between 0 and 1." in task_prompt


def test_structured_task_runner_enforces_contract_only_on_final_no_tool_output() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry = ToolRegistry()
        registry.register(EchoTool())
        contract_schema = {
            "type": "object",
            "required": ["observations", "confidence"],
            "additionalProperties": False,
            "properties": {
                "observations": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
        }
        provider = FakeProvider(
            [
                LLMCompletionResult(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call-1",
                            name="echo_tool",
                            arguments_json=json.dumps({"value": "route"}),
                        )
                    ],
                ),
                LLMCompletionResult(content="Draft: route evidence observed."),
                LLMCompletionResult(content=json.dumps({"observations": ["route evidence observed"], "confidence": 0.8})),
            ]
        )
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=provider,
            tool_registry=registry,
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="analysis_with_tools",
                system_prompt="Analyze the input with tools.",
                objective="Return structured analysis.",
                allowed_tools=["echo_tool"],
                output_contract=StructuredOutputContract(
                    name="analysis_contract",
                    schema=contract_schema,
                    strict=True,
                ),
            ),
            session_id="session-1",
        )

        assert result.ok is True
        assert result.output == {"observations": ["route evidence observed"], "confidence": 0.8}
        assert result.metadata["contract_finalization"] is True
        assert result.metadata["contract_name"] == "analysis_contract"
        assert result.tool_calls_used == 1
        assert result.iterations == 3
        assert len(provider.options_history) == 3
        assert provider.options_history[0] is not None
        assert provider.options_history[0].response_format is None
        assert provider.options_history[0].response_format_fallback is None
        assert provider.options_history[1] is not None
        assert provider.options_history[1].response_format is None
        assert provider.options_history[1].response_format_fallback is None
        assert provider.options_history[2] is not None
        assert provider.options_history[2].response_format == {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis_contract",
                "schema": contract_schema,
                "strict": True,
            },
        }
        assert provider.options_history[2].response_format_fallback is None
        assert provider.tools_history[0][0].name == "echo_tool"
        assert provider.tools_history[1][0].name == "echo_tool"
        assert provider.tools_history[2] == []
        assert provider.last_messages[-1].role == "system"
        assert "Investigation is complete" in provider.last_messages[-1].content


def test_structured_task_runner_passes_configured_max_output_tokens_to_final_output() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        provider = FakeProvider([LLMCompletionResult(content="done")])
        settings = _settings(root)
        settings.llm_max_output_tokens = 12345
        runner = StructuredTaskRunner(
            settings=settings,
            provider=provider,
            tool_registry=ToolRegistry(),
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="max_output_tokens",
                system_prompt="Return JSON.",
                objective="Validate max output token forwarding.",
            )
        )

        assert result.ok is True
        assert provider.last_options is not None
        assert provider.last_options.max_output_tokens == 12345


def test_structured_task_runner_does_not_pass_configured_max_output_tokens_to_tool_loop() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry = ToolRegistry()
        registry.register(EchoTool())
        contract_schema = {
            "type": "object",
            "required": ["summary"],
            "additionalProperties": False,
            "properties": {"summary": {"type": "string"}},
        }
        provider = FakeProvider(
            [
                LLMCompletionResult(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call-1",
                            name="echo_tool",
                            arguments_json=json.dumps({"value": "route"}),
                        )
                    ],
                ),
                LLMCompletionResult(content="Draft after tool evidence."),
                LLMCompletionResult(content=json.dumps({"summary": "done"})),
            ]
        )
        settings = _settings(root)
        settings.llm_max_output_tokens = 12345
        runner = StructuredTaskRunner(
            settings=settings,
            provider=provider,
            tool_registry=registry,
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="max_output_tokens_tool_loop",
                system_prompt="Return JSON after checking the tool.",
                objective="Validate max output token forwarding.",
                allowed_tools=["echo_tool"],
                output_contract=StructuredOutputContract(name="summary_contract", schema=contract_schema),
            )
        )

        assert result.ok is True
        assert len(provider.options_history) == 3
        assert provider.options_history[0] is not None
        assert provider.options_history[0].max_output_tokens is None
        assert provider.options_history[1] is not None
        assert provider.options_history[1].max_output_tokens is None
        assert provider.options_history[2] is not None
        assert provider.options_history[2].max_output_tokens == 12345


def test_structured_task_runner_rejects_invalid_json_output() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=FakeProvider([LLMCompletionResult(content="not-json")]),
            tool_registry=ToolRegistry(),
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="text_output",
                system_prompt="Return a concise answer.",
                objective="Validate text output.",
            )
        )

        assert result.ok is True
        assert result.output is None
        assert result.raw_content == "not-json"


def test_structured_task_runner_rejects_invalid_json_schema_output() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        first = {"summary": "first object"}
        second = {"summary": "duplicate object should be rejected"}
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=FakeProvider([LLMCompletionResult(content=json.dumps(first) + "\n" + json.dumps(second))]),
            tool_registry=ToolRegistry(),
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="json_schema_strict_parse",
                system_prompt="Return JSON.",
                objective="Validate strict JSON Schema parsing.",
                output_contract=StructuredOutputContract(
                    name="strict_output",
                    schema={
                        "type": "object",
                        "required": ["summary"],
                        "additionalProperties": False,
                        "properties": {"summary": {"type": "string"}},
                    },
                ),
            )
        )

        assert result.ok is False
        assert result.failure_reason == "Structured task returned invalid JSON Schema output."


def test_structured_task_prompt_forbids_appended_second_json_object_for_schema_contract() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        provider = FakeProvider([LLMCompletionResult(content=json.dumps({"summary": "done"}))])
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=provider,
            tool_registry=ToolRegistry(),
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="json_prompt_guard",
                system_prompt="Return JSON.",
                objective="Validate prompt guard.",
                output_contract=StructuredOutputContract(
                    name="guarded_output",
                    schema={
                        "type": "object",
                        "required": ["summary"],
                        "additionalProperties": False,
                        "properties": {"summary": {"type": "string"}},
                    },
                ),
            )
        )

        assert result.ok is True
        task_prompt = provider.last_messages[1].content
        assert "no second JSON object after it" in task_prompt
        assert "Provider-enforced structured output contract" in task_prompt


def test_structured_task_runner_uses_parent_session_id_for_tool_context() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry = ToolRegistry()
        registry.register(SessionIdTool())
        provider = FakeProvider(
            [
                LLMCompletionResult(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call-1",
                            name="session_id_tool",
                            arguments_json="{}",
                        )
                    ],
                ),
                LLMCompletionResult(content=json.dumps({"summary": "done"})),
            ]
        )
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=provider,
            tool_registry=registry,
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="session_context",
                system_prompt="Return JSON after checking session id.",
                objective="Check session id.",
                allowed_tools=["session_id_tool"],
            ),
            session_id="workspace-session",
        )

        assert result.ok is True
        assert result.tool_history[0]["content_preview"] == "workspace-session"


def test_structured_task_runner_fails_fast_on_unknown_allowed_tool() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        provider = FakeProvider([])
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=provider,
            tool_registry=ToolRegistry(),
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="unknown_tool",
                system_prompt="Use scoped tools only.",
                objective="Should not start.",
                allowed_tools=["missing_tool"],
            )
        )

        assert result.ok is False
        assert "Unknown tool: missing_tool" in result.failure_reason
        assert provider.responses == []


def test_structured_task_runner_forces_schema_finalization_after_iteration_budget() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry = ToolRegistry()
        registry.register(EchoTool())
        provider = FakeProvider(
            [
                LLMCompletionResult(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call-1",
                            name="echo_tool",
                            arguments_json=json.dumps({"value": "budget"}),
                        )
                    ],
                ),
                LLMCompletionResult(content=json.dumps({"summary": "finalized from observed evidence"})),
            ]
        )
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=provider,
            tool_registry=registry,
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="budgeted",
                system_prompt="Return JSON after using the echo tool.",
                objective="Exercise forced finalization.",
                allowed_tools=["echo_tool"],
                output_contract=StructuredOutputContract(
                    name="budgeted_output",
                    schema={
                        "type": "object",
                        "required": ["summary"],
                        "additionalProperties": False,
                        "properties": {"summary": {"type": "string"}},
                    },
                ),
                max_iterations=1,
                max_tool_calls=3,
            )
        )

        assert result.ok is True
        assert result.output == {"summary": "finalized from observed evidence"}
        assert result.metadata["forced_finalization"] is True
        assert result.metadata["budget_failure_reason"] == "Maximum number of structured task iterations reached."
        assert provider.last_tools == []


def test_structured_task_budget_answers_every_tool_call_before_finalization() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry = ToolRegistry()
        registry.register(EchoTool())
        provider = FakeProvider(
            [
                LLMCompletionResult(
                    content="",
                    tool_calls=[
                        LLMToolCall(id="call-1", name="echo_tool", arguments_json=json.dumps({"value": "run"})),
                        LLMToolCall(id="call-2", name="echo_tool", arguments_json=json.dumps({"value": "skip"})),
                    ],
                ),
                LLMCompletionResult(content="finalized"),
            ]
        )
        runner = StructuredTaskRunner(
            settings=_settings(root),
            provider=provider,
            tool_registry=registry,
            policy_engine=PolicyEngine(),
        )

        result = runner.run(
            spec=StructuredTaskSpec(
                task_id="partial_tool_budget",
                system_prompt="Use tools when useful.",
                objective="Exercise a partial tool-call budget.",
                allowed_tools=["echo_tool"],
                max_iterations=2,
                max_tool_calls=1,
            )
        )

        assert result.ok is True
        tool_messages = [message for message in provider.last_messages if message.role == "tool"]
        assert [message.tool_call_id for message in tool_messages] == ["call-1", "call-2"]
        assert "maximum tool-call budget" in tool_messages[1].content
        assert [item["status"] for item in result.tool_history] == ["ok", "budget_exhausted"]
