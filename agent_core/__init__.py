"""Stable public API for the autonomous agent-core run engine.

Extension contracts live in :mod:`agent_core.spi`. Optional conversation
adapters live in :mod:`agent_core.conversation`, and telemetry helpers live in
:mod:`agent_core.observability`. All other modules are implementation details.
"""

from agent_core.execution_context import ExecutionContext
from agent_core.output_contracts import (
    JSON_SCHEMA_DRAFT,
    FinalOutputMode,
    StructuredOutputContract,
    StructuredOutputValidationError,
    StructuredOutputValidationIssue,
)
from agent_core.run_context import ExecutionScope, RunContext
from agent_core.run_models import (
    AgentRunAttempt,
    AgentRunError,
    AgentRunResult,
    AgentRunState,
    RunCheckpoint,
    RunStatus,
    RunStrategy,
)
from agent_core.run_options import AgentRunMode, RunOptions
from agent_core.run_service import AgentRunService
from agent_core.run_store import JsonFileRunStore, RunExecutionBusyError, RunStore
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import StructuredTaskResult, StructuredTaskRunner, StructuredTaskSpec

__version__ = "0.3.0"

__all__ = [
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
]
