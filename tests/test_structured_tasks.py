from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent_core.llm.base import LLMCallOptions, LLMCompletionResult, LLMMessage, LLMToolCall
from agent_core.policy_engine import PolicyEngine
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskRunner, StructuredTaskSpec
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult


class FakeProvider:
    def __init__(self, responses: list[LLMCompletionResult]) -> None:
        self.responses = list(responses)
        self.last_messages: list[LLMMessage] = []
        self.last_tools: list[object] = []
        self.last_options: LLMCallOptions | None = None

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


def _settings(root: Path) -> CoreSettings:
    return CoreSettings(
        allowed_read_roots=[root],
        knowledge_base_dir=root / "knowledge",
        base_system_prompt="Base system prompt.",
    )


def test_structured_task_runner_executes_tool_loop_and_parses_json_output() -> None:
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
                            arguments_json=json.dumps({"value": "pre-recon"}),
                        )
                    ],
                ),
                LLMCompletionResult(
                    content=json.dumps(
                        {
                            "summary": "done",
                            "evidence": [{"tool": "echo_tool", "value": "pre-recon"}],
                        }
                    )
                ),
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
                task_id="pre_recon",
                system_prompt="Map the initial target state.",
                objective="Build a first target summary.",
                allowed_tools=["echo_tool"],
                output_schema={
                    "type": "object",
                    "required": ["summary", "evidence"],
                    "properties": {
                        "summary": {"type": "string"},
                        "evidence": {"type": "array"},
                    },
                },
            ),
            session_id="session-1",
        )

        assert result.ok is True
        assert result.output is not None
        assert result.output["summary"] == "done"
        assert result.tool_calls_used == 1
        assert result.tool_history[0]["status"] == "ok"
        assert result.tool_history[0]["content_preview"] == "echo:pre-recon"
        assert provider.last_options is not None
        assert provider.last_options.response_format == {"type": "json_object"}
        assert provider.last_options.metadata["structured_task_id"] == "pre_recon"
        assert provider.last_tools[0].name == "echo_tool"


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
                task_id="bad_output",
                system_prompt="Return JSON.",
                objective="Validate strict output parsing.",
            )
        )

        assert result.ok is False
        assert result.failure_reason == "Structured task returned invalid JSON output."


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
