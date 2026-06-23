from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_example(name: str):
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quickstart_registry_registers_demo_tools() -> None:
    quickstart = load_example("quickstart")

    registry = quickstart.build_registry()

    assert registry.list_tool_names() == ["echo", "get_current_time"]


def test_quickstart_loads_dotenv_without_overwriting_existing_values(tmp_path, monkeypatch) -> None:
    quickstart = load_example("quickstart")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=from-file",
                "AGENT_CORE_MODEL=demo-model",
                "AGENT_CORE_MEMORY_MODEL='demo-memory-model'",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.delenv("AGENT_CORE_MODEL", raising=False)
    monkeypatch.delenv("AGENT_CORE_MEMORY_MODEL", raising=False)

    quickstart.load_dotenv(env_file)

    assert quickstart.os.environ["OPENAI_API_KEY"] == "from-env"
    assert quickstart.os.environ["AGENT_CORE_MODEL"] == "demo-model"
    assert quickstart.os.environ["AGENT_CORE_MEMORY_MODEL"] == "demo-memory-model"


def test_quickstart_build_settings_reads_azure_anthropic_env(tmp_path, monkeypatch) -> None:
    quickstart = load_example("quickstart")
    monkeypatch.setenv("LLM_PROVIDER", "azure_anthropic")
    monkeypatch.setenv("AZURE_ANTHROPIC_ENDPOINT", "https://example.services.ai.azure.com/anthropic")
    monkeypatch.setenv("AZURE_ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("AZURE_ANTHROPIC_API_VERSION", raising=False)
    monkeypatch.setenv("AZURE_ANTHROPIC_VERSION", "2023-06-01")
    monkeypatch.setenv("AGENT_CORE_LLM_TIMEOUT_SECONDS", "321")

    settings = quickstart.build_settings(
        model="claude-opus-4-6",
        memory_model="claude-opus-4-6",
        session_file=tmp_path / "session.json",
    )

    assert settings.llm_provider == "azure_anthropic"
    assert settings.azure_anthropic_endpoint == "https://example.services.ai.azure.com/anthropic"
    assert settings.azure_anthropic_api_key == "test-key"
    assert settings.azure_anthropic_api_version is None
    assert settings.azure_anthropic_version == "2023-06-01"
    assert settings.llm_timeout_seconds == 321
    assert quickstart.missing_provider_config(settings) == []


def test_pending_tool_resume_example_runs(tmp_path) -> None:
    pending_demo = load_example("pending_tool_resume")

    pending_status, final_content = pending_demo.run_demo(tmp_path / "session.json")

    assert pending_status == "pending_tool_result"
    assert final_content == "The external job completed successfully: demo-user"
