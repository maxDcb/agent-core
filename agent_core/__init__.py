"""Public API for the reusable agent_core runtime package."""

from agent_core.domain_hooks import DomainHooks
from agent_core.execution_context import ExecutionContext
from agent_core.investigation_models import FinalCritique, InvestigationDecision, StepReflection
from agent_core.investigation_prompts import InvestigationPromptSet
from agent_core.investigation_state import EvidenceItem, Hypothesis, InvestigationState
from agent_core.orchestrator import AgentOrchestrator
from agent_core.policy_engine import PolicyEngine
from agent_core.run_trace import ContextBudget, PromptBlock, PromptSnapshot, RunTrace, TraceEvent
from agent_core.run_options import AgentRunMode, RunOptions
from agent_core.session_manager import SessionManager
from agent_core.session_repo import JsonFileSessionStore, SessionRepository, SessionStore
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskResult, StructuredTaskRunner, StructuredTaskSpec
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import BaseTool, build_tool_definition
from agent_core.types import AgentTurnResult, ToolResult

__version__ = "0.1.0"

__all__ = [
    "AgentOrchestrator",
    "AgentRunMode",
    "BaseTool",
    "CoreSettings",
    "DomainHooks",
    "EvidenceItem",
    "ExecutionContext",
    "FinalCritique",
    "Hypothesis",
    "InvestigationDecision",
    "InvestigationPromptSet",
    "InvestigationState",
    "PolicyEngine",
    "ContextBudget",
    "PromptBlock",
    "PromptSnapshot",
    "RunTrace",
    "RunOptions",
    "SessionManager",
    "SessionRepository",
    "SessionStore",
    "StructuredTaskResult",
    "StructuredTaskRunner",
    "StructuredTaskSpec",
    "JsonFileSessionStore",
    "StepReflection",
    "TraceEvent",
    "ToolRegistry",
    "AgentTurnResult",
    "ToolResult",
    "__version__",
    "build_tool_definition",
]
