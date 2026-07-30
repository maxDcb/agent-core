from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent_core.context_planner import LLMContextPolicy
from agent_core.llm_budget import LLMBudget
from agent_core.prompt_repository import load_prompt
from agent_core.tool_artifacts import ToolArtifactPolicy


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
    azure_anthropic_version: str | None = None

    memory_llm_provider: str | None = None
    memory_openai_api_key: str | None = None
    memory_azure_openai_endpoint: str | None = None
    memory_azure_openai_api_key: str | None = None
    memory_azure_openai_api_version: str | None = None
    memory_azure_anthropic_endpoint: str | None = None
    memory_azure_anthropic_api_key: str | None = None
    memory_azure_anthropic_api_version: str | None = None
    memory_azure_anthropic_version: str | None = None

    model: str = "gpt-4.1-mini"
    memory_model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    memory_temperature: float = 0.0

    max_active_context_tokens: int = 16000
    max_tool_calls_per_turn: int = 100
    llm_timeout_seconds: float = 120.0
    llm_max_output_tokens: int | None = None
    llm_budget: LLMBudget | None = None
    llm_context_policy: LLMContextPolicy | None = None
    tool_artifact_policy: ToolArtifactPolicy = field(default_factory=ToolArtifactPolicy)
    log_synthesis_payloads: bool = False
    memory_max_turn_input_chars: int = 64_000
    memory_max_handoff_chars: int = 6_000
    memory_max_turn_summary_chars: int = 4_000

    debug: bool = False
    log_level: str | None = None

    session_file: Path = field(default_factory=lambda: Path("./sessions/session.json"))
    reports_directory: Path = field(default_factory=lambda: Path("./reports"))
    artifacts_directory: Path = field(default_factory=lambda: Path("./artifacts"))
    prompts_dir: Path = field(default_factory=lambda: Path("./prompts").resolve())
    knowledge_base_dir: Path = field(default_factory=lambda: Path("./knowledge").resolve())

    allowed_read_roots: list[Path] = field(default_factory=lambda: [Path("workspace").resolve()])
    allowed_http_hosts: list[str] = field(default_factory=lambda: ["example.com"])
    allowed_http_methods: list[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    )
    http_proxy: str | None = None

    base_system_prompt: str = ""
    turn_memory_synthesis_prompt: str = ""

    def __post_init__(self) -> None:
        self.llm_budget = LLMBudget.from_any(self.llm_budget)
        self.llm_context_policy = LLMContextPolicy.from_any(self.llm_context_policy)
        self.tool_artifact_policy = ToolArtifactPolicy.from_any(self.tool_artifact_policy)
        self.artifacts_directory = self.artifacts_directory.resolve()
        self.prompts_dir = self.prompts_dir.resolve()
        self.knowledge_base_dir = self.knowledge_base_dir.resolve()
        self.allowed_read_roots = [path.resolve() for path in self.allowed_read_roots]
        self.allowed_http_methods = [method.strip().upper() for method in self.allowed_http_methods if method.strip()]
        for field_name in (
            "memory_max_turn_input_chars",
            "memory_max_handoff_chars",
            "memory_max_turn_summary_chars",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        # Load system prompts gracefully. If the prompts directory doesn't exist or files are missing,
        # leave the prompts empty. Domain layers (like AppSettings) can override prompts_dir and reload.
        if not self.base_system_prompt:
            try:
                self.base_system_prompt = load_prompt(self.prompts_dir, "system/main_agent.md")
            except (FileNotFoundError, ValueError):
                pass
        if not self.turn_memory_synthesis_prompt:
            try:
                self.turn_memory_synthesis_prompt = load_prompt(self.prompts_dir, "memory/turn_memory.md")
            except (FileNotFoundError, ValueError):
                pass
