from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.execution_context import ExecutionContext
from agent_core.policy_engine import PolicyEngine
from agent_core.settings import CoreSettings
from agent_core.types import AuthorizationResult, build_empty_session_state


def _context(tmp_path: Path) -> ExecutionContext:
    workspace = tmp_path / "workspace"
    knowledge = tmp_path / "knowledge"
    workspace.mkdir()
    knowledge.mkdir()
    settings = CoreSettings(
        allowed_read_roots=[workspace],
        knowledge_base_dir=knowledge,
        allowed_http_hosts=["example.test"],
        allowed_http_methods=["GET", "POST"],
    )
    return ExecutionContext(
        session_id="policy-test",
        settings=settings,
        session_state=build_empty_session_state(session_id="policy-test"),
    )


def test_unknown_tool_is_allowed_for_compatibility(tmp_path: Path) -> None:
    result = PolicyEngine().authorize("application_tool", {}, _context(tmp_path))
    assert result == AuthorizationResult(True, "allowed")


def test_custom_validator_can_allow_and_deny(tmp_path: Path) -> None:
    context = _context(tmp_path)
    engine = PolicyEngine(validators={"guarded": lambda arguments, _: AuthorizationResult(arguments.get("ok") is True)})

    assert engine.authorize("guarded", {"ok": True}, context).allowed is True
    assert engine.authorize("guarded", {"ok": False}, context).allowed is False


@pytest.mark.parametrize("tool_name", ["list_directory", "read_file_chunk", "search_code", "tree_directory"])
def test_filesystem_policy_normalizes_allowed_relative_paths(tmp_path: Path, tool_name: str) -> None:
    context = _context(tmp_path)
    target = context.settings.allowed_read_roots[0] / "target.txt"
    target.write_text("evidence", encoding="utf-8")
    arguments = {"path": "target.txt"}

    result = PolicyEngine().authorize(tool_name, arguments, context)

    assert result.allowed is True
    assert arguments["path"] == str(target.resolve())


def test_filesystem_policy_denies_missing_and_out_of_scope_paths(tmp_path: Path) -> None:
    context = _context(tmp_path)
    engine = PolicyEngine()

    assert engine.authorize("read_file_chunk", {}, context).allowed is False
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    result = engine.authorize("read_file_chunk", {"path": str(outside)}, context)
    assert result.allowed is False
    assert "Path not allowed" in result.reason


def test_knowledge_policy_is_limited_to_knowledge_root(tmp_path: Path) -> None:
    context = _context(tmp_path)
    article = context.settings.knowledge_base_dir / "article.md"
    article.write_text("reference", encoding="utf-8")
    arguments = {"path": "article.md"}

    assert PolicyEngine().authorize("read_knowledge_chunk", arguments, context).allowed is True
    assert arguments["path"] == str(article.resolve())
    assert PolicyEngine().authorize("search_knowledge", {"path": str(tmp_path)}, context).allowed is False


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({"url": "https://example.test"}, "Missing HTTP method"),
        ({"method": "DELETE", "url": "https://example.test"}, "HTTP method not allowed"),
        ({"method": "GET"}, "Missing URL"),
        ({"method": "GET", "url": "file:///tmp/x"}, "Invalid URL scheme"),
        ({"method": "GET", "url": "https:///missing-host"}, "Invalid URL"),
        ({"method": "GET", "url": "https://other.test"}, "Host not allowed"),
        (
            {"method": "POST", "url": "https://example.test", "json_body": {}, "raw_body": "x"},
            "Provide only one HTTP body type",
        ),
        (
            {"method": "GET", "url": "https://example.test", "proxy_url": "socks5://proxy.test"},
            "Invalid proxy URL",
        ),
    ],
)
def test_http_policy_denies_invalid_requests(tmp_path: Path, arguments: dict, reason: str) -> None:
    result = PolicyEngine().authorize("http_request", arguments, _context(tmp_path))
    assert result.allowed is False
    assert reason in result.reason


def test_http_policy_allows_configured_request_and_proxy(tmp_path: Path) -> None:
    arguments = {
        "method": "post",
        "url": "https://example.test/api",
        "json_body": {"ok": True},
        "proxy_url": "http://proxy.test:8080",
    }
    assert PolicyEngine().authorize("http_request", arguments, _context(tmp_path)).allowed is True


def test_session_execution_scope_overrides_global_scope(tmp_path: Path) -> None:
    context = _context(tmp_path)
    scoped_root = tmp_path / "scoped"
    scoped_root.mkdir()
    scoped_file = scoped_root / "allowed.txt"
    scoped_file.write_text("ok", encoding="utf-8")
    context.session_state["execution_scope"] = {
        "allowed_read_roots": [str(scoped_root)],
        "allowed_http_hosts": ["scoped.test"],
        "allowed_http_methods": ["PATCH"],
    }

    assert PolicyEngine().authorize("read_file_chunk", {"path": str(scoped_file)}, context).allowed is True
    assert PolicyEngine().authorize(
        "http_request",
        {"method": "PATCH", "url": "https://scoped.test/resource"},
        context,
    ).allowed is True
