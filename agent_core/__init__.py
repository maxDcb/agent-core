"""Public API for the reusable agent_core runtime package."""

from agent_core.conversation import ConversationAgent
from agent_core.domain_hooks import DomainHooks
from agent_core.execution_context import ExecutionContext
from agent_core.investigation_models import FinalCritique, InvestigationDecision, StepReflection
from agent_core.investigation_prompts import InvestigationPromptSet
from agent_core.investigation_state import EvidenceItem, Hypothesis, InvestigationState
from agent_core.orchestrator import AgentOrchestrator
from agent_core.output_contracts import (
    JSON_SCHEMA_DRAFT,
    FinalOutputMode,
    StructuredOutputContract,
    StructuredOutputValidationError,
    StructuredOutputValidationIssue,
)
from agent_core.policy_engine import PolicyEngine
from agent_core.run_context import ExecutionScope, RunContext
from agent_core.run_models import AgentRunError, AgentRunResult, AgentRunState, RunStatus, RunStrategy
from agent_core.run_options import AgentRunMode, RunOptions
from agent_core.run_service import AgentRunService
from agent_core.run_store import JsonFileRunStore, RunStore
from agent_core.run_trace import ContextBudget, PromptBlock, PromptSnapshot, RunTrace, TraceEvent
from agent_core.session_manager import SessionManager
from agent_core.session_repo import JsonFileSessionStore, SessionRepository, SessionStore
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import (
    StructuredTaskResult,
    StructuredTaskRunner,
    StructuredTaskSpec,
)
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import BaseTool, build_tool_definition
from agent_core.types import AgentTurnResult, ToolResult

__version__ = "0.3.0"

__all__ = [
    "AgentOrchestrator",
    "AgentRunError",
    "AgentRunMode",
    "AgentRunResult",
    "AgentRunService",
    "AgentRunState",
    "BaseTool",
    "CoreSettings",
    "ConversationAgent",
    "DomainHooks",
    "EvidenceItem",
    "ExecutionContext",
    "ExecutionScope",
    "FinalOutputMode",
    "FinalCritique",
    "Hypothesis",
    "InvestigationDecision",
    "InvestigationPromptSet",
    "InvestigationState",
    "JSON_SCHEMA_DRAFT",
    "JsonFileRunStore",
    "PolicyEngine",
    "ContextBudget",
    "PromptBlock",
    "PromptSnapshot",
    "RunTrace",
    "RunOptions",
    "RunContext",
    "RunStatus",
    "RunStore",
    "RunStrategy",
    "SessionManager",
    "SessionRepository",
    "SessionStore",
    "StructuredTaskResult",
    "StructuredTaskRunner",
    "StructuredTaskSpec",
    "StructuredOutputContract",
    "StructuredOutputValidationError",
    "StructuredOutputValidationIssue",
    "JsonFileSessionStore",
    "StepReflection",
    "TraceEvent",
    "ToolRegistry",
    "AgentTurnResult",
    "ToolResult",
    "__version__",
    "build_tool_definition",
]
