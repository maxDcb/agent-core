from __future__ import annotations

import agent_core
from agent_core import (
    AgentOrchestrator,
    CoreSettings,
    InvestigationState,
    JsonFileSessionStore,
    RunOptions,
    RunTrace,
    SessionStore,
    ToolRegistry,
)


def test_public_api_exports_runtime_entrypoints() -> None:
    assert agent_core.__version__ == "0.1.0"
    assert AgentOrchestrator is not None
    assert CoreSettings is not None
    assert InvestigationState is not None
    assert JsonFileSessionStore is not None
    assert RunOptions is not None
    assert RunTrace is not None
    assert SessionStore is not None
    assert ToolRegistry is not None
