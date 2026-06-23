from __future__ import annotations

import argparse
import json
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
    StructuredOutputContract,
    StructuredTaskRunner,
    StructuredTaskSpec,
    ToolRegistry,
    ToolResult,
    build_tool_definition,
)
from agent_core.llm.base import BaseLLMProvider, LLMCallOptions, LLMMessage, LLMToolDefinition
from agent_core.llm.errors import LLMProviderError
from agent_core.llm.provider_factory import build_provider, normalize_provider_name


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


def build_settings(*, model: str, memory_model: str, session_file: Path) -> CoreSettings:
    return CoreSettings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_anthropic_endpoint=os.getenv("AZURE_ANTHROPIC_ENDPOINT"),
        azure_anthropic_api_key=os.getenv("AZURE_ANTHROPIC_API_KEY"),
        azure_anthropic_api_version=os.getenv("AZURE_ANTHROPIC_API_VERSION"),
        azure_anthropic_version=os.getenv("AZURE_ANTHROPIC_VERSION"),
        llm_timeout_seconds=float(os.getenv("AGENT_CORE_LLM_TIMEOUT_SECONDS", "120")),
        llm_max_output_tokens=_optional_positive_int(os.getenv("AGENT_CORE_LLM_MAX_OUTPUT_TOKENS")),
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
        provider=build_provider(settings),
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


def _print_check(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def _json_object_matches(content: str, expected: dict[str, Any]) -> tuple[bool, str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return False, f"expected object, got {type(payload).__name__}"

    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        return False, f"mismatches={mismatches}"
    return True, f"keys={sorted(payload.keys())}"


def _run_plain_chat_check(provider: BaseLLMProvider, *, model: str) -> bool:
    try:
        content = provider.complete_text(
            messages=[
                LLMMessage(role="system", content="You are a compatibility checker. Answer exactly as requested."),
                LLMMessage(role="user", content="Return exactly: OK"),
            ],
            model=model,
            temperature=0.0,
        )
    except LLMProviderError as exc:
        return _print_check("plain chat", False, f"{exc.kind}: {exc.user_message}")
    normalized = content.strip().strip("\"'").rstrip(".")
    if normalized == "OK":
        return _print_check("plain chat", True, f"content={content!r}")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and payload.get("answer") == "OK":
        return _print_check("plain chat", True, f"content={content!r}; note=json wrapper")

    return _print_check("plain chat", False, f"content={content!r}")


def _run_tool_call_check(provider: BaseLLMProvider, *, model: str) -> bool:
    try:
        result = provider.complete_with_tools(
            messages=[
                LLMMessage(
                    role="user",
                    content="Call the echo tool once with text set to compatibility-check. Do not answer directly.",
                )
            ],
            tools=[
                LLMToolDefinition(
                    name="echo",
                    description="Return the provided text.",
                    parameters={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                )
            ],
            model=model,
            temperature=0.0,
        )
    except LLMProviderError as exc:
        return _print_check("tool calls", False, f"{exc.kind}: {exc.user_message}")

    if not result.tool_calls:
        return _print_check("tool calls", False, f"no tool call returned; content={result.content!r}")
    call = result.tool_calls[0]
    try:
        arguments = json.loads(call.arguments_json or "{}")
    except json.JSONDecodeError:
        arguments = {}
    ok = call.name == "echo" and isinstance(arguments, dict) and arguments.get("text") == "compatibility-check"
    return _print_check("tool calls", ok, f"name={call.name!r}, arguments={call.arguments_json!r}")


def _run_json_object_check(provider: BaseLLMProvider, *, model: str) -> bool:
    try:
        content = provider.complete_text(
            messages=[
                LLMMessage(
                    role="user",
                    content='Return one JSON object with fields "ok": true and "mode": "json_object". No prose.',
                )
            ],
            model=model,
            temperature=0.0,
            options=LLMCallOptions(response_format={"type": "json_object"}),
        )
    except LLMProviderError as exc:
        return _print_check("response_format json_object", False, f"{exc.kind}: {exc.user_message}")

    ok, detail = _json_object_matches(content, {"ok": True, "mode": "json_object"})
    return _print_check("response_format json_object", ok, detail)


def _run_json_schema_check(provider: BaseLLMProvider, *, model: str) -> bool:
    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["json_schema"]},
        },
        "required": ["ok", "mode"],
        "additionalProperties": False,
    }
    try:
        content = provider.complete_text(
            messages=[
                LLMMessage(
                    role="user",
                    content='Return one JSON object with fields "ok": true and "mode": "json_schema". No prose.',
                )
            ],
            model=model,
            temperature=0.0,
            options=LLMCallOptions(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "compatibility_check",
                        "schema": schema,
                        "strict": True,
                    },
                },
                response_format_fallback={"type": "json_object"},
            ),
        )
    except LLMProviderError as exc:
        return _print_check("response_format json_schema", False, f"{exc.kind}: {exc.user_message}")

    ok, detail = _json_object_matches(content, {"ok": True, "mode": "json_schema"})
    return _print_check("response_format json_schema", ok, detail)


def _run_structured_task_check(settings: CoreSettings, provider: BaseLLMProvider) -> bool:
    runner = StructuredTaskRunner(
        settings=settings,
        provider=provider,
        tool_registry=ToolRegistry(),
        policy_engine=PolicyEngine(),
    )


def _optional_positive_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None
    result = runner.run(
        spec=StructuredTaskSpec(
            task_id="compatibility_check",
            system_prompt="You are validating provider structured output support.",
            objective='Return {"ok": true, "component": "structured_task"} as the final output.',
            output_contract=StructuredOutputContract(
                name="compatibility_check",
                strict=True,
                schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "component": {"type": "string", "enum": ["structured_task"]},
                    },
                    "required": ["ok", "component"],
                    "additionalProperties": False,
                },
            ),
            allowed_tools=[],
            max_iterations=1,
        )
    )
    output = result.output or {}
    output_matches = result.ok and output.get("ok") is True and output.get("component") == "structured_task"
    detail = result.failure_reason or f"output={result.output}"
    return _print_check("StructuredTaskRunner final output", output_matches, detail)


def run_compatibility_checks(settings: CoreSettings) -> int:
    provider = build_provider(settings)
    print(f"Running provider compatibility checks with provider={settings.llm_provider!r}, model={settings.model!r}")
    print("Checks cover plain chat, OpenAI-style tool calls, response_format, and StructuredTaskRunner.")

    checks = [
        _run_plain_chat_check(provider, model=settings.model),
        _run_tool_call_check(provider, model=settings.model),
        _run_json_object_check(provider, model=settings.model),
        _run_json_schema_check(provider, model=settings.model),
        _run_structured_task_check(settings, provider),
    ]

    passed = sum(1 for ok in checks if ok)
    print(f"\nCompatibility summary: {passed}/{len(checks)} checks passed.")
    return 0 if all(checks) else 1


def missing_provider_config(settings: CoreSettings) -> list[str]:
    provider_name = normalize_provider_name(settings.llm_provider)
    if provider_name == "openai":
        return [] if settings.openai_api_key else ["OPENAI_API_KEY"]
    if provider_name == "azure_openai":
        missing = []
        if not settings.azure_openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not settings.azure_openai_api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        return missing
    if provider_name == "azure_anthropic":
        missing = []
        if not settings.azure_anthropic_endpoint:
            missing.append("AZURE_ANTHROPIC_ENDPOINT")
        if not settings.azure_anthropic_api_key:
            missing.append("AZURE_ANTHROPIC_API_KEY")
        return missing
    return [f"unsupported LLM_PROVIDER={settings.llm_provider!r}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal agent-core assistant.")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Prompt to send for a one-shot run.")
    parser.add_argument(
        "--compat-check",
        action="store_true",
        help="Run compatibility checks for the configured LLM provider.",
    )
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

    settings = build_settings(
        model=args.model or os.getenv("AGENT_CORE_MODEL", "gpt-4.1-mini"),
        memory_model=args.memory_model or os.getenv("AGENT_CORE_MEMORY_MODEL", "gpt-4.1-mini"),
        session_file=args.session_file,
    )
    missing_config = missing_provider_config(settings)
    if missing_config:
        print(
            f"Missing provider configuration for LLM_PROVIDER={settings.llm_provider!r}: "
            f"{', '.join(missing_config)}"
        )
        return 2

    if args.compat_check:
        return run_compatibility_checks(settings)

    orchestrator = build_orchestrator(settings)

    if args.interactive:
        run_repl(orchestrator, session_id=args.session_id)
    else:
        run_prompt(orchestrator, args.prompt, session_id=args.session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
