from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.llm.base import LLMCompletionResult, LLMToolCall
from agent_core.policy_engine import PolicyEngine
from agent_core.settings import CoreSettings
from agent_core.structured_tasks import (
    StructuredOutputContract,
    StructuredTaskRunner,
    StructuredTaskSpec,
)
from agent_core.tool_registry import ToolRegistry
from agent_core.tools import build_tool_definition
from agent_core.types import ToolResult
from tests.run_helpers import execution_context


class ScriptedProvider:
    def __init__(self) -> None:
        self.responses = [
            LLMCompletionResult(
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="echo-1",
                        name="echo",
                        arguments_json=json.dumps({"value": "inventory"}),
                    )
                ],
            ),
            LLMCompletionResult(content="Investigation complete."),
            LLMCompletionResult(content=json.dumps({"summary": "echo:inventory"})),
        ]

    def complete_with_tools(self, **kwargs) -> LLMCompletionResult:
        _ = kwargs
        return self.responses.pop(0)


class EchoTool:
    name = "echo"
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
        _ = context
        return ToolResult(ok=True, content=f"echo:{arguments['value']}")


def _runner(root: Path, backend: str) -> StructuredTaskRunner:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return StructuredTaskRunner(
        settings=CoreSettings(
            allowed_read_roots=[root],
            knowledge_base_dir=root / "knowledge",
            agent_kernel_backend=backend,
        ),
        provider=ScriptedProvider(),
        tool_registry=registry,
        policy_engine=PolicyEngine(),
    )


def _spec() -> StructuredTaskSpec:
    return StructuredTaskSpec(
        task_id="pre_recon_inventory",
        system_prompt="Inventory the target.",
        objective="Return the inventory summary.",
        allowed_tools=["echo"],
        output_contract=StructuredOutputContract(
            name="pre_recon_inventory",
            schema={
                "type": "object",
                "required": ["summary"],
                "additionalProperties": False,
                "properties": {"summary": {"type": "string"}},
            },
        ),
    )


def test_native_and_langgraph_structured_kernels_have_the_same_result_and_checkpoints(
    tmp_path: Path,
) -> None:
    outcomes: dict[str, dict] = {}
    checkpoint_phases: dict[str, list[str]] = {}

    for backend in ("native", "langgraph"):
        runner = _runner(tmp_path / backend, backend)
        phases: list[str] = []
        result = runner.run(
            spec=_spec(),
            context=execution_context(runner.settings, namespace_id="assessment"),
            on_checkpoint=lambda checkpoint, phases=phases: phases.append(checkpoint.phase),
        )
        outcomes[backend] = result.to_dict()
        for history_item in outcomes[backend]["tool_history"]:
            history_item.pop("artifact_id", None)
        checkpoint_phases[backend] = phases
        assert runner._kernel.backend == backend

    assert outcomes["langgraph"] == outcomes["native"]
    assert checkpoint_phases["langgraph"] == checkpoint_phases["native"]
    assert outcomes["langgraph"]["output"] == {"summary": "echo:inventory"}
    assert outcomes["langgraph"]["tool_calls_used"] == 1


def test_unknown_backend_is_rejected_for_structured_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected 'native' or 'langgraph'"):
        _runner(tmp_path, "other")
