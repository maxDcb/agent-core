from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent_core.prompt_repository import load_prompt


@dataclass(slots=True)
class CoreSettings:
    llm_provider: str = "openai"

    openai_api_key: str | None = None

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2025-01-01-preview"

    azure_anthropic_endpoint: str | None = None
    azure_anthropic_api_key: str | None = None
    azure_anthropic_api_version: str | None = None

    model: str = "gpt-4.1-mini"
    memory_model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    memory_temperature: float = 0.0

    max_active_context_tokens: int = 16000
    max_tool_calls_per_turn: int = 100
    log_synthesis_payloads: bool = False

    debug: bool = False
    log_level: str | None = None

    session_file: Path = field(default_factory=lambda: Path("./sessions/session.json"))
    reports_directory: Path = field(default_factory=lambda: Path("./reports"))
    prompts_dir: Path = field(default_factory=lambda: Path("./prompts").resolve())
    knowledge_base_dir: Path = field(default_factory=lambda: Path("./knowledge").resolve())

    allowed_read_roots: list[Path] = field(default_factory=lambda: [Path("workspace").resolve()])
    allowed_http_hosts: list[str] = field(default_factory=lambda: ["example.com"])
    allowed_http_methods: list[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    )
    http_proxy: str | None = None

    base_system_prompt: str = ""
    task_state_synthesis_prompt: str = ""
    session_summary_synthesis_prompt: str = ""
    session_summary_merge_prompt: str = ""

    def __post_init__(self) -> None:
        self.prompts_dir = self.prompts_dir.resolve()
        self.knowledge_base_dir = self.knowledge_base_dir.resolve()
        self.allowed_read_roots = [path.resolve() for path in self.allowed_read_roots]
        self.allowed_http_methods = [method.strip().upper() for method in self.allowed_http_methods if method.strip()]

        # Load system prompts gracefully. If the prompts directory doesn't exist or files are missing,
        # leave the prompts empty. Domain layers (like AppSettings) can override prompts_dir and reload.
        if not self.base_system_prompt:
            try:
                self.base_system_prompt = load_prompt(self.prompts_dir, "system/main_agent.md")
            except (FileNotFoundError, ValueError):
                pass
        if not self.task_state_synthesis_prompt:
            try:
                self.task_state_synthesis_prompt = load_prompt(self.prompts_dir, "memory/task_state.md")
            except (FileNotFoundError, ValueError):
                pass
        if not self.session_summary_synthesis_prompt:
            try:
                self.session_summary_synthesis_prompt = load_prompt(self.prompts_dir, "memory/session_summary.md")
            except (FileNotFoundError, ValueError):
                pass
        if not self.session_summary_merge_prompt:
            try:
                self.session_summary_merge_prompt = load_prompt(self.prompts_dir, "memory/session_summary_merge.md")
            except (FileNotFoundError, ValueError):
                pass
