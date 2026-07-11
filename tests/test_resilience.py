from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_core.memory.context_block import ContextBlock
from agent_core.memory.history_compactor import CompactionPolicy, HistoryCompactor
from agent_core.memory.thread_state import ThreadState
from agent_core.session_repo import JsonFileSessionStore
from agent_core.types import build_empty_session_state


@pytest.mark.chaos
def test_atomic_write_failure_preserves_previous_session_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileSessionStore(tmp_path / "session.json")
    original = build_empty_session_state(session_id="stable")
    store.save("stable", original)
    session_file = store._resolve_session_file("stable")
    original_bytes = session_file.read_bytes()
    files_before = set(session_file.parent.iterdir())

    def fail_dump(*args: object, **kwargs: object) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(json, "dump", fail_dump)
    with pytest.raises(OSError, match="simulated disk failure"):
        store.save("stable", {**original, "domain_state": {"new": {"value": 1}}})

    assert session_file.read_bytes() == original_bytes
    assert set(session_file.parent.iterdir()) == files_before


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(session_id=st.text())
def test_session_ids_always_resolve_inside_store_directory(tmp_path: Path, session_id: str) -> None:
    store = JsonFileSessionStore(tmp_path / "session.json")
    resolved = store._resolve_session_file(session_id).resolve()
    assert resolved.is_relative_to(store.session_directory.resolve())
    assert resolved.suffix == ".json"


@given(
    block_id=st.text(min_size=1, max_size=30),
    content=st.dictionaries(
        keys=st.text(min_size=1, max_size=15),
        values=st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=30)),
        max_size=8,
    ),
    pinned=st.booleans(),
    priority=st.integers(min_value=-10, max_value=100),
)
def test_context_block_round_trip_is_stable(
    block_id: str,
    content: dict,
    pinned: bool,
    priority: int,
) -> None:
    block = ContextBlock(
        block_id=block_id,
        kind="retrieved_memory",
        content=content,
        token_estimate=1,
        pinned=pinned,
        priority=priority,
    )
    restored = ContextBlock.from_dict(block.to_dict())
    assert restored == block


@given(
    token_estimates=st.lists(st.integers(min_value=1, max_value=200), min_size=1, max_size=30),
    budget=st.integers(min_value=1, max_value=1000),
)
def test_history_compaction_preserves_every_atomic_turn_and_keeps_latest_active(
    token_estimates: list[int],
    budget: int,
) -> None:
    blocks = [
        ContextBlock(
            block_id=f"turn-{index}",
            kind="conversation_turn",
            content={},
            token_estimate=token_estimate,
            metadata={"turn_index": index},
        )
        for index, token_estimate in enumerate(token_estimates)
    ]
    compacted = HistoryCompactor(CompactionPolicy(max_active_tokens=budget)).compact(
        ThreadState(thread_id="property-test", context_blocks=blocks)
    )

    active_ids = {block.block_id for block in compacted.active_blocks}
    overflow_ids = {block.block_id for block in compacted.overflow_blocks}
    assert active_ids.isdisjoint(overflow_ids)
    assert active_ids | overflow_ids == {block.block_id for block in blocks}
    assert blocks[-1].block_id in active_ids
    active_tokens = sum(block.token_estimate for block in compacted.active_blocks)
    assert active_tokens <= budget or compacted.active_blocks == [blocks[-1]]
