from __future__ import annotations

from agent_core.execution_context import ExecutionContext
from agent_core.settings import CoreSettings
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult


class EchoTool:
    name = "echo"
    description = "Echo a value."

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

    def execute(self, arguments, context):
        return ToolResult(ok=True, content=f"echo:{arguments['value']}")


def test_tool_registry_executes_registered_tool(tmp_path) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    context = ExecutionContext(
        session_id="default",
        settings=CoreSettings(session_file=tmp_path / "session.json"),
        session_state={},
    )

    result = registry.execute("echo", {"value": "hello"}, context)

    assert result.ok is True
    assert result.content == "echo:hello"
    assert registry.list_tool_names() == ["echo"]


def test_tool_registry_reports_unknown_tool(tmp_path) -> None:
    registry = ToolRegistry()
    context = ExecutionContext(
        session_id="default",
        settings=CoreSettings(session_file=tmp_path / "session.json"),
        session_state={},
    )

    result = registry.execute("missing", {}, context)

    assert result.ok is False
    assert result.content == "Unknown tool: missing"
