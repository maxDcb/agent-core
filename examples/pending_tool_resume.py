from __future__ import annotations

import json
from pathlib import Path

from agent_core import (
    AgentOrchestrator,
    CoreSettings,
    ExecutionContext,
    PolicyEngine,
    SessionManager,
    SessionRepository,
    ToolResult,
    build_tool_definition,
)
from agent_core.llm.base import LLMCompletionResult, LLMToolCall
from agent_core.tool_registry import ToolRegistry


class FakeProvider:
    """Deterministic provider used to demonstrate pending/resume without an API key."""

    def __init__(self) -> None:
        self._responses = [
            LLMCompletionResult(
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="call-1",
                        name="start_external_job",
                        arguments_json=json.dumps({"command": "whoami"}),
                    )
                ],
            ),
            LLMCompletionResult(content="The external job completed successfully: demo-user"),
        ]

    def complete_with_tools(self, *, messages, tools, model, temperature):
        return self._responses.pop(0)

    def complete_text(self, *, messages, model, temperature):
        return json.dumps(
            {
                "run_id": "run-0000",
                "objective": "Demonstrate pending tool resume",
                "scope": [],
                "source_code_locations": [],
                "open_questions": [],
                "next_action": None,
                "stop_conditions": [],
                "constraints": [],
                "relevant_artifacts": [],
                "status": "active",
                "domain_extensions": {},
            }
        )


class ExternalJobTool:
    name = "start_external_job"
    description = "Start an external job and return pending while the host application waits for the result."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "External command to schedule.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict, context: ExecutionContext) -> ToolResult:
        return ToolResult.pending_result(
            "External job scheduled. Waiting for host application result.",
            metadata={"job_id": "job-123", "command": arguments.get("command")},
        )


def build_orchestrator(session_file: Path) -> AgentOrchestrator:
    settings = CoreSettings(
        openai_api_key="not-used",
        model="fake-model",
        memory_model="fake-model",
        session_file=session_file,
        base_system_prompt="You demonstrate pending tool result resume.",
        task_state_synthesis_prompt="Return JSON.",
        session_summary_synthesis_prompt="Return JSON.",
        session_summary_merge_prompt="Return JSON.",
    )
    registry = ToolRegistry()
    registry.register(ExternalJobTool())
    return AgentOrchestrator(
        settings=settings,
        provider=FakeProvider(),
        registry=registry,
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )


def run_demo(session_file: Path) -> tuple[str, str]:
    orchestrator = build_orchestrator(session_file)
    pending = orchestrator.run_turn_result("Run the external job.")
    if not pending.pending_id:
        raise RuntimeError("Expected a pending tool result.")
    completed = orchestrator.resume_turn(
        pending_id=pending.pending_id,
        tool_content="demo-user",
    )
    return pending.status, completed.content


def main() -> int:
    pending_status, final_content = run_demo(Path(".agent-core-demo/pending-session.json"))
    print(f"First turn status: {pending_status}")
    print(final_content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
