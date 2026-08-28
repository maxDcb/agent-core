"""Stable extension contracts for providers, domains, policies, and tools."""

from agent_core.domain_hooks import DomainHooks
from agent_core.investigation_prompts import InvestigationPromptSet
from agent_core.llm.base import (
    BaseLLMProvider,
    LLMCallOptions,
    LLMCallRecord,
    LLMCompletionResult,
    LLMMessage,
    LLMTokenUsage,
    LLMToolCall,
    LLMToolDefinition,
    LLMUsageSummary,
)
from agent_core.llm.errors import LLMProviderError
from agent_core.llm.provider_factory import (
    LLMProviderConfig,
    build_memory_provider,
    build_provider,
    build_provider_from_config,
    normalize_model_backend,
    normalize_provider_name,
)
from agent_core.policy_engine import PolicyEngine
from agent_core.prompt_repository import load_prompt
from agent_core.tool_artifacts import ArtifactStore
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import BaseTool, build_tool_definition
from agent_core.types import AuthorizationResult, ToolExecutionStatus, ToolResult

__all__ = [
    "AuthorizationResult",
    "ArtifactStore",
    "BaseLLMProvider",
    "BaseTool",
    "DomainHooks",
    "InvestigationPromptSet",
    "LLMCallOptions",
    "LLMCallRecord",
    "LLMCompletionResult",
    "LLMMessage",
    "LLMProviderConfig",
    "LLMProviderError",
    "LLMToolCall",
    "LLMToolDefinition",
    "LLMTokenUsage",
    "LLMUsageSummary",
    "PolicyEngine",
    "ToolExecutionStatus",
    "ToolRegistry",
    "ToolResult",
    "build_memory_provider",
    "build_provider",
    "build_provider_from_config",
    "build_tool_definition",
    "load_prompt",
    "normalize_model_backend",
    "normalize_provider_name",
]
