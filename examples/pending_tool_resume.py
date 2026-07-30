from __future__ import annotations

import json
from pathlib import Path

from agent_core import (
    CoreSettings,
    ExecutionContext,
    RunContext,
)
from agent_core.conversation import AgentOrchestrator, SessionManager, SessionRepository
from agent_core.spi import (
    LLMCompletionResult,
    LLMToolCall,
    PolicyEngine,
    ToolRegistry,
    ToolResult,
    build_tool_definition,
)


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

    def complete_with_tools(self, *, messages, tools, model, temperature, options=None):
        return self._responses.pop(0)

    def complete_text(self, *, messages, model, temperature, options=None):
        return json.dumps(
            {
                "turn_summary": "The external job completed successfully.",
                "next_handoff": (
                    "Current objective:\n"
                    "Demonstrate pending tool resume.\n\n"
                    "Latest outcome:\n"
                    "The external job completed successfully; no follow-up action remains."
                ),
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
        turn_memory_synthesis_prompt="Return JSON.",
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
    context = RunContext(namespace_id="demo", run_id="demo-run", thread_id="demo")
    pending = orchestrator.run_turn_result(
        user_input="Run the external job.",
        thread_id="demo",
        context=context,
    )
    if not pending.pending_id:
        raise RuntimeError("Expected a pending tool result.")
    completed = orchestrator.resume_turn(
        pending_id=pending.pending_id,
        tool_content="demo-user",
        thread_id="demo",
        context=context,
    )
    return pending.status, completed.content


def main() -> int:
    pending_status, final_content = run_demo(Path(".agent-core-demo/pending-session.json"))
    print(f"First turn status: {pending_status}")
    print(final_content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
