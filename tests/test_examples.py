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


def test_pending_tool_resume_example_runs(tmp_path) -> None:
    pending_demo = load_example("pending_tool_resume")

    pending_status, final_content = pending_demo.run_demo(tmp_path / "session.json")

    assert pending_status == "pending_tool_result"
    assert final_content == "The external job completed successfully: demo-user"
