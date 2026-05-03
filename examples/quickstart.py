from __future__ import annotations

import argparse
import os
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_core import (
    AgentOrchestrator,
    CoreSettings,
    ExecutionContext,
    PolicyEngine,
    SessionManager,
    SessionRepository,
    ToolRegistry,
    ToolResult,
    build_tool_definition,
)
from agent_core.llm.openai_provider import OpenAIProvider


DEFAULT_PROMPT = "What time is it? Use get_current_time, then answer in one sentence."
DEMO_STATE_DIR = Path(".agent-core-demo")

BASE_SYSTEM_PROMPT = """You are a concise assistant running inside an agent-core quickstart.
Use tools when they are relevant. Answer directly and do not mention internal implementation details."""

TASK_STATE_PROMPT = """Update the task state from the conversation.
Return only valid JSON matching the provided output format."""

SESSION_SUMMARY_PROMPT = """Summarize the supplied overflow conversation blocks.
Return only valid JSON matching the provided output format."""

SESSION_SUMMARY_MERGE_PROMPT = """Merge the old session summary with the new summary delta.
Return only valid JSON matching the provided output format."""


class CurrentTimeTool:
    name = "get_current_time"
    description = "Return the current UTC timestamp."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        return ToolResult(ok=True, content=datetime.now(UTC).isoformat())


class EchoTool:
    name = "echo"
    description = "Return the provided text. Useful for checking tool wiring."

    def schema(self):
        return build_tool_definition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to return.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        text = str(arguments.get("text", ""))
        return ToolResult(ok=True, content=text)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            try:
                parsed_value = shlex.split(value, comments=True)[0] if value else ""
            except (IndexError, ValueError):
                parsed_value = value.strip("\"'")
            os.environ[key] = parsed_value


def build_settings(
    *,
    api_key: str | None,
    model: str,
    memory_model: str,
    session_file: Path,
) -> CoreSettings:
    return CoreSettings(
        openai_api_key=api_key,
        model=model,
        memory_model=memory_model,
        session_file=session_file,
        allowed_read_roots=[Path.cwd()],
        allowed_http_hosts=[],
        base_system_prompt=BASE_SYSTEM_PROMPT,
        task_state_synthesis_prompt=TASK_STATE_PROMPT,
        session_summary_synthesis_prompt=SESSION_SUMMARY_PROMPT,
        session_summary_merge_prompt=SESSION_SUMMARY_MERGE_PROMPT,
    )


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CurrentTimeTool())
    registry.register(EchoTool())
    return registry


def build_orchestrator(settings: CoreSettings) -> AgentOrchestrator:
    return AgentOrchestrator(
        settings=settings,
        provider=OpenAIProvider(api_key=settings.openai_api_key),
        registry=build_registry(),
        session_manager=SessionManager(SessionRepository(settings.session_file)),
        policy_engine=PolicyEngine(),
    )


def run_prompt(orchestrator: AgentOrchestrator, prompt: str, *, session_id: str) -> None:
    result = orchestrator.run_turn_result(prompt, session_id=session_id)
    print(result.content)
    if result.is_pending:
        print(f"\nPending tool result: {result.pending_id}")
        print("Call orchestrator.resume_turn(...) from your application when the external result is ready.")


def run_repl(orchestrator: AgentOrchestrator, *, session_id: str) -> None:
    print("agent-core quickstart. Type /exit to quit.")
    while True:
        prompt = input("> ").strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return
        run_prompt(orchestrator, prompt, session_id=session_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal agent-core assistant.")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Prompt to send for a one-shot run.")
    parser.add_argument("--interactive", action="store_true", help="Start a small REPL instead of one one-shot prompt.")
    parser.add_argument("--session-id", default="quickstart", help="Session id used for persisted memory.")
    parser.add_argument("--session-file", type=Path, default=DEMO_STATE_DIR / "session.json", help="Session JSON path.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional .env file to load.")
    parser.add_argument("--model", default=None, help="Chat model.")
    parser.add_argument(
        "--memory-model",
        default=None,
        help="Model used for structured memory synthesis.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY. Set it in the environment or create a .env file from .env.example.")
        return 2

    settings = build_settings(
        api_key=api_key,
        model=args.model or os.getenv("AGENT_CORE_MODEL", "gpt-4.1-mini"),
        memory_model=args.memory_model or os.getenv("AGENT_CORE_MEMORY_MODEL", "gpt-4.1-mini"),
        session_file=args.session_file,
    )
    orchestrator = build_orchestrator(settings)

    if args.interactive:
        run_repl(orchestrator, session_id=args.session_id)
    else:
        run_prompt(orchestrator, args.prompt, session_id=args.session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
