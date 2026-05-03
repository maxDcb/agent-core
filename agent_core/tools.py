"""Tool abstraction layer for agent_core.

This module defines the protocol that tools must implement to be compatible
with the ToolRegistry. It has no external dependencies beyond agent_core types.
"""

from typing import Protocol

from agent_core.execution_context import ExecutionContext
from agent_core.types import ToolResult
from agent_core.llm.base import LLMToolDefinition


class BaseTool(Protocol):
    """Protocol that defines the interface for tools.
    
    Any object implementing these methods and attributes can be used with
    ToolRegistry, without needing to inherit from a base class.
    """
    name: str
    description: str

    def schema(self) -> LLMToolDefinition:
        """Return the JSON schema for this tool."""
        ...

    def execute(self, arguments: dict, context: ExecutionContext) -> ToolResult:
        """Execute the tool with the given arguments and context."""
        ...


def build_tool_definition(*, name: str, description: str, parameters: dict) -> LLMToolDefinition:
    """Helper to construct a tool definition."""
    return LLMToolDefinition(name=name, description=description, parameters=parameters)
