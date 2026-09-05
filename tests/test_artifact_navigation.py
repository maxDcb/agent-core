from __future__ import annotations

import json

import pytest

from agent_core.artifact_navigation import NavigationError, decode_cursor, next_read
from agent_core.llm.base import LLMMessage
from agent_core.settings import CoreSettings
from agent_core.tool_artifacts import JsonFileArtifactStore, ToolArtifactPolicy, ToolArtifactRuntime, ToolArtifactUsage
from tests.run_helpers import execution_context


def setup_runtime(tmp_path, content, **policy_options):
    settings = CoreSettings(base_system_prompt="system", artifacts_directory=tmp_path / "artifacts")
    context = execution_context(settings, namespace_id="reader")
    runtime = ToolArtifactRuntime(
        policy=ToolArtifactPolicy(**policy_options),
        store=JsonFileArtifactStore(settings.artifacts_directory),
        namespace_id="reader",
        run_id=context.run_id,
    )
    message = runtime.externalize(tool_name="document", content=content, tool_call_id="doc", status="ok")
    artifact_id = json.loads(message.content)["artifact"]["artifact_id"]
    return runtime, context, artifact_id, message


def read(runtime, context, artifact_id, **query):
    return runtime.execute(
        tool_name="agent_core_read_artifact", arguments={"artifact_id": artifact_id, **query}, context=context
    )


def pages(runtime, context, artifact_id, **query):
    result = read(runtime, context, artifact_id, **query)
    for _ in range(100):
        assert result.ok, result.content
        page = json.loads(result.content)
        yield page
        if page["next_read"] is None:
            return
        result = runtime.execute(
            tool_name="agent_core_read_artifact", arguments=page["next_read"]["arguments"], context=context
        )
    pytest.fail("Pagination did not terminate")


def test_raw_preview_continuation_is_lossless_and_survives_runtime_recreation(tmp_path):
    source = "alpha🧪é" * 80
    runtime, context, artifact_id, message = setup_runtime(
        tmp_path, source, preview_bytes=16, max_complete_result_bytes=32, max_read_bytes=37, max_reads_per_run=100
    )
    envelope = json.loads(message.content)
    content = envelope["content"]
    arguments = envelope["next_read"]["arguments"]
    while arguments:
        result = runtime.execute(tool_name="agent_core_read_artifact", arguments=arguments, context=context)
        assert result.ok, result.content
        page = json.loads(result.content)
        content += page["content"]
        arguments = page["next_read"]["arguments"] if page["next_read"] else None
        runtime = ToolArtifactRuntime(
            policy=runtime.policy,
            store=runtime.store,
            namespace_id="reader",
            run_id=context.run_id,
            usage=runtime.usage.to_dict(),
        )
    assert content == source
    assert artifact_id in runtime.usage.read_progress
    assert runtime.usage.read_progress[artifact_id]["last_page_complete"]


def test_json_projection_pages_are_complete_and_do_not_repeat_items(tmp_path):
    rows = [{"id": i, "name": "é" * 30, "large": "x" * 5000} for i in range(40)]
    runtime, context, artifact_id, _ = setup_runtime(
        tmp_path, json.dumps({"nested": {"items": rows}}), max_read_bytes=1600, max_reads_per_run=100
    )
    output = list(
        pages(runtime, context, artifact_id, operation="read", json_pointer="/nested/items", fields=["id", "name"])
    )
    assert len(output) > 1
    assert [item for page in output for item in page["content"]] == [
        {"id": row["id"], "name": row["name"]} for row in rows
    ]
    assert all(len(json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode()) <= 1600 for page in output)
    assert output[-1]["selection_complete"]


def test_inspection_returns_escaped_pointers_without_leaf_content(tmp_path):
    runtime, context, artifact_id, _ = setup_runtime(
        tmp_path, json.dumps({"a/b~c": [1, 2], "secret": "not-in-inspection"})
    )
    payload = json.loads(read(runtime, context, artifact_id, operation="inspect").content)
    assert payload["content"][0] == {"json_pointer": "/a~1b~0c", "type": "array", "length": 2}
    assert "not-in-inspection" not in json.dumps(payload)
    page = json.loads(read(runtime, context, artifact_id, json_pointer="/a~1b~0c").content)
    assert page["content"] == [1, 2]


def test_search_finds_tail_and_reports_scan_completeness(tmp_path):
    source = "prefix " * 9000 + "TARGET final" + " TARGET" * 120
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, source, max_read_bytes=1600, max_reads_per_run=100)
    output = list(pages(runtime, context, artifact_id, operation="search", query="TARGET"))
    matches = [item for page in output for item in page["content"]]
    assert len(matches) == 121
    assert len({item["char_offset"] for item in matches}) == 121
    assert matches[0]["char_offset"] == 63000
    assert output[-1]["selection_complete"]


def test_text_line_selection(tmp_path):
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, "a\nb\nc\n")
    payload = json.loads(read(runtime, context, artifact_id, operation="read", start_line=2).content)
    assert payload["content"] == [{"line": 2, "text": "b\n"}, {"line": 3, "text": "c\n"}]


def test_oversized_item_and_invalid_pointer_are_recoverable(tmp_path):
    runtime, context, artifact_id, _ = setup_runtime(
        tmp_path, json.dumps({"items": [{"id": 1, "huge": "x" * 10000}]}), max_read_bytes=1000
    )
    for pointer in ("/items", "/absent"):
        result = read(runtime, context, artifact_id, json_pointer=pointer)
        assert not result.ok
        assert json.loads(result.content)["recoverable"] is True
        assert runtime.claim_read_recovery([LLMMessage(role="tool", content=result.content)], ["tool_error"])
    assert not runtime.claim_read_recovery([LLMMessage(role="tool", content=result.content)], ["tool_error"])
    corrected = read(runtime, context, artifact_id, json_pointer="/items", fields=["id"])
    assert corrected.ok
    assert json.loads(corrected.content)["content"] == [{"id": 1}]


def test_namespace_denial_and_changed_artifact_do_not_recover(tmp_path):
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, "example")
    other = ToolArtifactRuntime(policy=runtime.policy, store=runtime.store, namespace_id="other", run_id=context.run_id)
    other_context = execution_context(
        CoreSettings(base_system_prompt="system", artifacts_directory=tmp_path / "artifacts"), namespace_id="other"
    )
    result = read(other, other_context, artifact_id, operation="inspect")
    assert not result.ok
    assert not json.loads(result.content)["recoverable"]
    token = next_read(artifact_id, "old-version", {"offset": 0})["arguments"]["continuation"]
    with pytest.raises(NavigationError, match="version changed"):
        decode_cursor(token, artifact_id, "new-version")


@pytest.mark.parametrize(
    "query",
    [
        {"continuation": "garbage"},
        {"operation": "search", "query": ""},
        {"json_pointer": "/x~2"},
        {"fields": ["id"]},
        {"operation": "inspect", "offset": -1},
    ],
)
def test_invalid_queries_return_errors_instead_of_crashing(tmp_path, query):
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, "{}")
    result = read(runtime, context, artifact_id, **query)
    assert not result.ok
    assert json.loads(result.content)["recoverable"]


def test_navigation_respects_context_source_and_total_budgets(tmp_path):
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, '{"a": 1}', max_total_read_bytes=50)
    result = read(runtime, context, artifact_id, operation="inspect")
    assert not result.ok
    assert runtime.usage.artifact_bytes_read == 0
    assert not json.loads(result.content)["recoverable"]
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, "x" * 1000, max_navigation_source_bytes=32)
    result = read(runtime, context, artifact_id, operation="inspect")
    assert json.loads(result.content)["code"] == "source_too_large"
    assert read(runtime, context, artifact_id, offset=500, limit=20).ok


def test_read_progress_roundtrip_is_bounded_and_contains_no_document_content(tmp_path):
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, '{"private": "value"}')
    assert read(runtime, context, artifact_id, operation="inspect").ok
    restored = ToolArtifactUsage.from_any(runtime.usage.to_dict())
    assert restored.read_progress == runtime.usage.read_progress
    assert "private" not in json.dumps(restored.to_dict())


def test_large_string_selection_remains_readable_without_loading_other_fields(tmp_path):
    value = "line🧪\n" * 400
    runtime, context, artifact_id, _ = setup_runtime(
        tmp_path, json.dumps({"data": value, "irrelevant": "x" * 10000}), max_read_bytes=1100, max_reads_per_run=100
    )
    output = list(pages(runtime, context, artifact_id, json_pointer="/data"))
    assert len(output) > 1
    assert "".join(page["content"] for page in output) == value
    assert output[-1]["selection_complete"]


def test_cache_reuses_parsed_document_and_is_bounded(tmp_path, monkeypatch):
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, '{"items":[1,2]}', max_navigation_cache_bytes=25)
    original_read = runtime.store.read_all_text
    calls = []

    def tracked(**kwargs):
        calls.append(kwargs["artifact_id"])
        return original_read(**kwargs)

    monkeypatch.setattr(runtime.store, "read_all_text", tracked)
    assert read(runtime, context, artifact_id, operation="inspect").ok
    assert read(runtime, context, artifact_id, json_pointer="/items").ok
    assert calls == [artifact_id]
    message = runtime.externalize(tool_name="document", content='{"items":[3,4]}', tool_call_id="doc2", status="ok")
    second = json.loads(message.content)["artifact"]["artifact_id"]
    assert read(runtime, context, second, operation="inspect").ok
    assert artifact_id not in runtime._navigation_cache


def test_inspect_after_checkpoint_exposes_previous_continuation(tmp_path):
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, json.dumps({"items": list(range(1000))}))
    page = read(runtime, context, artifact_id, json_pointer="/items", limit=1200)
    assert page.ok
    saved = runtime.usage.to_dict()
    restored = ToolArtifactRuntime(
        policy=runtime.policy, store=runtime.store, namespace_id="reader", run_id=context.run_id, usage=saved
    )
    inspection = json.loads(read(restored, context, artifact_id, operation="inspect").content)
    assert inspection["previous_read"]["next_read"] == json.loads(page.content)["next_read"]


def test_continuation_cannot_be_rebound_to_another_artifact(tmp_path):
    runtime, context, artifact_id, envelope = setup_runtime(
        tmp_path, "x" * 100, max_complete_result_bytes=32, preview_bytes=16
    )
    other = runtime.externalize(tool_name="document", content="x" * 100, tool_call_id="other", status="ok")
    other_id = json.loads(other.content)["artifact"]["artifact_id"]
    token = json.loads(envelope.content)["next_read"]["arguments"]["continuation"]
    result = read(runtime, context, other_id, continuation=token)
    assert not result.ok
    assert json.loads(result.content)["code"] == "invalid_cursor"


def test_navigation_context_callback_is_honored(tmp_path):
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, json.dumps({"rows": [{"n": i} for i in range(20)]}))
    result = runtime.execute(
        tool_name="agent_core_read_artifact",
        arguments={"artifact_id": artifact_id, "json_pointer": "/rows"},
        context=context,
        content_fits=lambda content: len(json.loads(content).get("content", [])) <= 2,
    )
    assert result.ok
    payload = json.loads(result.content)
    assert len(payload["content"]) == 2
    assert not payload["selection_complete"]


def test_raw_utf8_read_does_not_exceed_remaining_byte_budget(tmp_path):
    runtime, context, artifact_id, _ = setup_runtime(tmp_path, "🧪", max_total_read_bytes=1)
    result = read(runtime, context, artifact_id, limit=1)
    assert not result.ok
    assert runtime.usage.artifact_bytes_read == 0
    assert json.loads(result.content)["recoverable"] is False
