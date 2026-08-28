from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_core.agent_graph.investigation import LangGraphInvestigationKernel
from agent_core.llm.base import LLMCompletionResult
from agent_core.orchestrator import AgentOrchestrator
from agent_core.run_options import RunOptions
from tests.run_helpers import resume_turn, run_turn
from tests.test_investigation_modes import (
    ScriptedProvider,
    build_orchestrator,
    critique_payload,
    decision_payload,
    reflection_payload,
    tool_call,
)


def _trace_events(orchestrator: AgentOrchestrator, trace_id: str) -> list[str]:
    trace = orchestrator.session_manager.load_run_trace(trace_id)
    assert trace is not None
    return [event["type"] for event in trace["events"]]


def _initial_plan() -> dict[str, Any]:
    return {
        "objective": "investigate",
        "plan": ["collect evidence"],
        "facts": [],
        "hypotheses": [],
        "evidence_gaps": ["evidence not collected"],
        "completed_actions": [],
        "next_actions": ["collect evidence"],
        "risk_notes": [],
        "confidence": 0.1,
        "stop_reason": None,
        "metadata": {},
    }


def _no_tool_snapshot(root: Path, backend: str) -> dict[str, Any]:
    provider = ScriptedProvider(
        chat=[LLMCompletionResult(content="draft")],
        plans=[_initial_plan()],
        finals=[LLMCompletionResult(content="final answer")],
    )
    orchestrator = build_orchestrator(root, provider, agent_kernel_backend=backend)

    result = run_turn(orchestrator, "investigate", options=RunOptions.investigate(max_iterations=1))

    trace_id = str(result.metadata["run_trace_id"])
    trace = orchestrator.session_manager.load_run_trace(trace_id)
    assert trace is not None
    return {
        "content": result.content,
        "mode": result.metadata["mode"],
        "iterations_used": result.metadata["iterations_used"],
        "tool_calls_used": result.metadata["tool_calls_used"],
        "stop_reason": result.metadata["stop_reason"],
        "final_response_origin": result.metadata["final_response_origin"],
        "block_kinds": [block.kind for block in orchestrator.session_manager.get_context_blocks()],
        "trace_events": _trace_events(orchestrator, trace_id),
        "trace_backend": trace["options"]["agent_kernel_backend"],
    }


def test_langgraph_investigation_initial_plan_and_finalization_match_native(tmp_path: Path) -> None:
    native = _no_tool_snapshot(tmp_path / "native", "native")
    langgraph = _no_tool_snapshot(tmp_path / "langgraph", "langgraph")

    assert native.pop("trace_backend") == "native"
    assert langgraph.pop("trace_backend") == "langgraph"
    assert langgraph == native


def test_langgraph_investigation_kernel_exposes_explicit_control_nodes(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(
        tmp_path,
        ScriptedProvider(chat=[]),
        agent_kernel_backend="langgraph",
    )
    kernel = LangGraphInvestigationKernel(orchestrator._build_investigation_controller())

    assert {
        "initialize_plan",
        "assistant_step",
        "execute_tools",
        "reflect_decide",
        "handle_final_draft",
        "complete_max_tools",
        "complete_max_iterations",
    }.issubset(kernel.graph.get_graph().nodes)


def _tool_snapshot(root: Path, backend: str) -> dict[str, Any]:
    provider = ScriptedProvider(
        chat=[tool_call(value="fact")],
        reflections=[
            reflection_payload(
                new_facts=["echo returned fact"],
                remaining_gaps=["need second source"],
                recommended_next_actions=["verify independently"],
                should_continue=False,
            )
        ],
        decisions=[decision_payload("final", reason_summary="enough evidence")],
    )
    orchestrator = build_orchestrator(root, provider, agent_kernel_backend=backend)

    result = run_turn(
        orchestrator,
        "investigate",
        options=RunOptions.investigate(max_iterations=2, require_initial_plan=False),
    )

    trace_id = str(result.metadata["run_trace_id"])
    return {
        "content": result.content,
        "stop_reason": result.metadata["stop_reason"],
        "iterations_used": result.metadata["iterations_used"],
        "tool_calls_used": result.metadata["tool_calls_used"],
        "facts": result.metadata["investigation_state"]["facts"],
        "journal_kinds": [item.kind for item in orchestrator.session_manager.get_memory_journal().exchanges],
        "tool_statuses": [item["status"] for item in orchestrator.session_manager.get_state()["tool_history"]],
        "trace_events": _trace_events(orchestrator, trace_id),
    }


def test_langgraph_investigation_tool_reflection_decision_matches_native(tmp_path: Path) -> None:
    assert _tool_snapshot(tmp_path / "langgraph", "langgraph") == _tool_snapshot(tmp_path / "native", "native")


def _deep_snapshot(root: Path, backend: str) -> dict[str, Any]:
    provider = ScriptedProvider(
        chat=[LLMCompletionResult(content="unsupported draft"), LLMCompletionResult(content="revised draft")],
        critiques=[
            critique_payload(approved=False, unsupported_claims=["unsupported claim"]),
            critique_payload(approved=True),
        ],
    )
    orchestrator = build_orchestrator(root, provider, agent_kernel_backend=backend)

    result = run_turn(
        orchestrator,
        "answer",
        options=RunOptions.deep_investigate(
            max_iterations=2,
            require_initial_plan=False,
        ),
    )

    return {
        "content": result.content,
        "mode": result.metadata["mode"],
        "stop_reason": result.metadata["stop_reason"],
        "iterations_used": result.metadata["iterations_used"],
        "evidence_gaps": result.metadata["investigation_state"]["evidence_gaps"],
        "next_actions": result.metadata["investigation_state"]["next_actions"],
    }


def test_langgraph_deep_investigation_critique_loop_matches_native(tmp_path: Path) -> None:
    assert _deep_snapshot(tmp_path / "langgraph", "langgraph") == _deep_snapshot(tmp_path / "native", "native")


@pytest.mark.parametrize("backend", ["native", "langgraph"])
def test_investigation_pending_checkpoint_and_resume_contract(tmp_path: Path, backend: str) -> None:
    provider = ScriptedProvider(
        chat=[tool_call(name="pending", value="job-1")],
        reflections=[reflection_payload(new_facts=["external result arrived"], should_continue=False)],
        decisions=[decision_payload("final", reason_summary="pending result resolved")],
    )
    orchestrator = build_orchestrator(
        tmp_path,
        provider,
        pending=True,
        agent_kernel_backend=backend,
    )

    pending = run_turn(
        orchestrator,
        "start pending",
        options=RunOptions.investigate(max_iterations=2, require_initial_plan=False),
    )

    assert pending.status == "pending_tool_result"
    assert pending.pending_id
    payload = orchestrator.session_manager.get_state()["meta"][AgentOrchestrator.PENDING_TURN_META_KEY]
    assert payload["agent_graph_checkpoint"] == {
        "schema_version": "1",
        "graph": "investigation",
        "backend": backend,
        "resume_node": "resume_tool_exchange",
    }

    completed = resume_turn(orchestrator, pending_id=pending.pending_id, tool_content="done")

    assert completed.status == "completed"
    assert "external result arrived" in completed.content
    assert completed.metadata["mode"] == "investigate"
    assert [block.kind for block in orchestrator.session_manager.get_context_blocks()] == [
        "tool_exchange",
        "conversation_turn",
    ]
