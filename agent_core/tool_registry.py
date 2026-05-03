from __future__ import annotations

from typing import Any

from agent_core.execution_context import ExecutionContext
from agent_core.logging_utils import get_logger
from agent_core.types import ToolResult
from agent_core.llm.base import LLMToolDefinition
from agent_core.tools import BaseTool

logger = get_logger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        logger.debug("Initialized empty tool registry")

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info("Registered tool", extra={"tool_name": tool.name})

    def get_tool_specs(self) -> list[LLMToolDefinition]:
        specs = [tool.schema() for tool in self._tools.values()]
        logger.trace("Built tool schemas", extra={"tool_count": len(specs)})
        return specs

    def execute(self, tool_name: str, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            logger.error("Attempted to execute unknown tool", extra={"tool_name": tool_name})
            return ToolResult(ok=False, content=f"Unknown tool: {tool_name}")
        logger.debug("Executing registry dispatch", extra={"tool_name": tool_name})
        return tool.execute(arguments, context)

    def get_tool(self, tool_name: str) -> BaseTool | None:
        return self._tools.get(tool_name)

    def build_subset(self, tool_names: list[str]) -> "ToolRegistry":
        subset = ToolRegistry()
        for tool_name in tool_names:
            tool = self.get_tool(tool_name)
            if tool is None:
                raise KeyError(f"Unknown tool: {tool_name}")
            subset.register(tool)
        return subset

    def list_tool_names(self) -> list[str]:
        return sorted(self._tools.keys())
