"""Bounded, deterministic navigation of immutable text and JSON artifacts.

Cursors describe a view, never grant access. The caller must authorize and load
the artifact in its namespace before interpreting a continuation.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from typing import Any

READ_TOOL = "agent_core_read_artifact"


class NavigationError(ValueError):
    def __init__(
        self, code: str, message: str, *, recoverable: bool = True, suggested_query: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable
        self.suggested_query = suggested_query


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def next_read(artifact_id: str, sha256: str, query: dict[str, Any]) -> dict[str, Any]:
    token = base64.urlsafe_b64encode(
        dumps({"v": 1, "id": artifact_id, "sha256": sha256, "query": query}).encode()
    ).decode()
    return {"tool": READ_TOOL, "arguments": {"artifact_id": artifact_id, "continuation": token}}


def decode_cursor(token: Any, artifact_id: str, sha256: str) -> dict[str, Any]:
    try:
        if not isinstance(token, str) or len(token) > 8192:
            raise ValueError("invalid cursor")
        payload = json.loads(base64.b64decode(token.encode(), altchars=b"-_", validate=True))
        if not isinstance(payload, dict) or payload.get("v") != 1 or payload.get("id") != artifact_id:
            raise ValueError("cursor does not belong to this artifact")
        if payload.get("sha256") != sha256:
            raise NavigationError("stale_cursor", "Artifact version changed; inspect it again.")
        query = payload.get("query")
        if not isinstance(query, dict):
            raise ValueError("invalid cursor query")
        return query
    except (ValueError, UnicodeError, TypeError) as exc:
        if isinstance(exc, NavigationError):
            raise
        raise NavigationError("invalid_cursor", "Invalid continuation; inspect the artifact again.") from exc


def normalize_query(query: dict[str, Any], artifact_id: str, sha256: str, default_max_bytes: int) -> dict[str, Any]:
    """Keep a cursor's view authoritative while tolerating redundant LLM arguments."""
    if "continuation" not in query:
        validate_query(query)
        return dict(query)
    token = query["continuation"]
    selected = decode_cursor(token, artifact_id, sha256)
    validate_query(selected)
    recovery = {"continuation": token}
    extras = {key: value for key, value in query.items() if key != "continuation"}
    for key, value in extras.items():
        if key in {"limit", "max_bytes"}:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise NavigationError(
                    "invalid_arguments", "Page budgets must be positive byte counts.", suggested_query=recovery
                )
            ceiling = min(selected.get("max_bytes", selected.get("limit", default_max_bytes)), default_max_bytes)
            if value > ceiling:
                raise NavigationError(
                    "continuation_conflict", "A continuation may only reduce its page budget.", suggested_query=recovery
                )
        elif (key not in selected and key != "operation") or dumps(value) != dumps(selected.get(key, "read")):
            raise NavigationError(
                "continuation_conflict",
                "Arguments conflict with the continuation's view or position.",
                suggested_query=recovery,
            )
    if "limit" in extras and "max_bytes" in extras and extras["limit"] != extras["max_bytes"]:
        raise NavigationError("continuation_conflict", "limit and max_bytes disagree.", suggested_query=recovery)
    if "limit" in extras or "max_bytes" in extras:
        # Preserve legacy raw-read accounting when a caller repeats `limit`.
        budget_key = "max_bytes" if "max_bytes" in selected or "max_bytes" in extras else "limit"
        selected.pop("limit", None)
        selected.pop("max_bytes", None)
        selected[budget_key] = extras.get("max_bytes", extras.get("limit"))
    validate_query(selected)
    return selected


def resolve_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise NavigationError("invalid_selection", "json_pointer must be empty or start with '/'.")
    for part in pointer[1:].split("/"):
        # RFC 6901 only defines these two escape sequences.
        if "~" in part.replace("~1", "").replace("~0", ""):
            raise NavigationError("invalid_selection", "Invalid JSON pointer escape.")
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif (
            isinstance(value, list)
            and key.isascii()
            and key.isdigit()
            and (key == "0" or not key.startswith("0"))
            and int(key) < len(value)
        ):
            value = value[int(key)]
        else:
            raise NavigationError(
                "selection_not_found", "The requested JSON pointer does not exist; inspect its parent."
            )
    return value


def child_pointer(pointer: str, key: str) -> str:
    return pointer + "/" + key.replace("~", "~0").replace("/", "~1")


def shape(value: Any) -> dict[str, Any]:
    if isinstance(value, (dict, list, str)):
        return {"type": {dict: "object", list: "array", str: "string"}[type(value)], "length": len(value)}
    return {"type": "null" if value is None else "boolean" if isinstance(value, bool) else "number"}


def validate_query(query: dict[str, Any]) -> None:
    if len(dumps(query).encode("utf-8")) > 5500:
        raise NavigationError("invalid_arguments", "Selection parameters exceed the continuation size budget.")
    allowed = {
        "operation",
        "json_pointer",
        "fields",
        "position",
        "query",
        "offset",
        "limit",
        "max_bytes",
        "start_line",
        "line_count",
    }
    if set(query) - allowed or query.get("operation", "read") not in {"read", "inspect", "search"}:
        raise NavigationError("invalid_arguments", "Unsupported navigation operation or parameters.")
    for name in ("offset", "position", "start_line", "line_count", "limit", "max_bytes"):
        if name in query and (
            isinstance(query[name], bool)
            or not isinstance(query[name], int)
            or query[name] < (1 if name in {"limit", "max_bytes", "start_line", "line_count"} else 0)
        ):
            raise NavigationError("invalid_arguments", f"{name} is outside its valid integer range.")
    if "limit" in query and "max_bytes" in query and query["limit"] != query["max_bytes"]:
        raise NavigationError(
            "invalid_arguments", "limit and max_bytes are byte budgets and must agree; use line_count for lines."
        )
    if "line_count" in query and "start_line" not in query:
        raise NavigationError("invalid_arguments", "line_count requires start_line.")
    if "start_line" in query and query.get("operation", "read") != "read":
        raise NavigationError("invalid_arguments", "Line ranges are only supported by read.")
    if "json_pointer" in query and (not isinstance(query["json_pointer"], str) or len(query["json_pointer"]) > 2048):
        raise NavigationError("invalid_arguments", "json_pointer must be a string of at most 2048 characters.")
    if "fields" in query and (
        not isinstance(query["fields"], list)
        or not 1 <= len(query["fields"]) <= 32
        or any(not isinstance(f, str) or len(f) > 128 for f in query["fields"])
    ):
        raise NavigationError("invalid_arguments", "fields must contain 1 to 32 short field names.")
    if "fields" in query and sum(len(f) for f in query["fields"]) > 2048:
        raise NavigationError("invalid_arguments", "Combined field names exceed the continuation size budget.")
    if query.get("operation") == "search" and (
        not isinstance(query.get("query"), str) or not 1 <= len(query["query"]) <= 256
    ):
        raise NavigationError("invalid_arguments", "search requires a literal query of 1 to 256 characters.")
    if "start_line" in query and ("json_pointer" in query or "offset" in query):
        raise NavigationError("invalid_arguments", "start_line cannot be combined with json_pointer or offset.")
    if "fields" in query and "json_pointer" not in query:
        raise NavigationError("invalid_selection", "fields requires a json_pointer; use an empty pointer for the root.")
    if "offset" in query and (query.get("operation", "read") != "read" or "json_pointer" in query):
        raise NavigationError(
            "invalid_arguments", "offset is only for raw byte reading; follow next_read for selected views."
        )
    if "query" in query and query.get("operation") != "search":
        raise NavigationError("invalid_arguments", "query is only supported by search.")
    if "fields" in query and query.get("operation", "read") != "read":
        raise NavigationError("invalid_arguments", "fields is only supported by read.")


def render_page(
    *,
    artifact_id: str,
    text: str,
    parsed: Any,
    query: dict[str, Any],
    max_bytes: int,
    content_fits: Callable[[str], bool] | None = None,
    previous_read: dict[str, Any] | None = None,
    sha256: str | None = None,
) -> str:
    """Build a page without splitting structured items or hiding omitted fields."""
    validate_query(query)
    digest = sha256 or hashlib.sha256(text.encode("utf-8")).hexdigest()
    operation = query.get("operation", "read")
    pointer = query.get("json_pointer", "")
    value = resolve_pointer(parsed, pointer) if "json_pointer" in query else parsed
    fields = query.get("fields")
    position = query.get("position", 0)
    base: dict[str, Any] = {
        "schema_version": "2",
        "kind": "artifact_page",
        "artifact_id": artifact_id,
        "operation": operation,
        "selection": {"json_pointer": pointer} if "json_pointer" in query else {},
        "projection_fields": fields,
        "selection_complete": False,
        "next_read": None,
    }
    if operation == "inspect":
        base["structure"] = shape(value)
        if previous_read is not None:
            candidate = {**base, "previous_read": previous_read}
            if len(dumps(candidate).encode("utf-8")) < max_bytes // 2:
                base = candidate
        if isinstance(value, dict):
            entries = [(child_pointer(pointer, key), child) for key, child in value.items()]
        elif isinstance(value, list):
            entries = [(child_pointer(pointer, str(i)), child) for i, child in enumerate(value)]
        else:
            entries = []
        items = ({"json_pointer": path, **shape(child)} for path, child in entries)
        total = len(entries)
    elif operation == "search":
        source = dumps(value) if "json_pointer" in query else text
        needle = query["query"]
        if position > len(source):
            raise NavigationError("invalid_arguments", "Search position is beyond the selected content.")
        # Search windows overlap implicitly: the next start is immediately after
        # a match, not at the end of its displayed context.
        found: list[dict[str, Any]] = []
        cursor = position
        while len(found) < 100:
            index = source.find(needle, cursor)
            if index < 0:
                cursor = len(source)
                break
            found.append({"char_offset": index, "excerpt": source[max(0, index - 80) : index + len(needle) + 80]})
            cursor = index + len(needle)
        base["offset_unit"] = "characters_in_selection"
        return _search_page(base, found, source, needle, query, digest, max_bytes, content_fits)
    elif "json_pointer" in query:
        if fields is not None and not isinstance(value, (dict, list)):
            raise NavigationError("invalid_selection", "fields requires an object or array of objects.")
        if isinstance(value, str):
            return _string_page(base, value, query, digest, max_bytes, content_fits)
        values = value if isinstance(value, list) else [value]

        def project(item: Any) -> Any:
            if fields is None:
                return item
            if not isinstance(item, dict):
                raise NavigationError("invalid_selection", "fields requires object elements.")
            return {key: item[key] for key in fields if key in item}

        items = (project(item) for item in values)
        total = len(values)
        base["selected_type"] = shape(value)["type"]
    else:
        lines = text.splitlines(keepends=True)
        start = query.get("start_line", 1) - 1
        position = query.get("position", start)
        items = ({"line": i + 1, "text": line} for i, line in enumerate(lines))
        total = min(len(lines), start + query.get("line_count", len(lines)))
        if position < start:
            raise NavigationError("invalid_arguments", "Position is before the selected line range.")
        base["selection"] = {"start_line": start + 1}
        if "line_count" in query:
            base["selection"]["line_count"] = query["line_count"]
    if position > total:
        raise NavigationError("invalid_arguments", "Position is beyond the selected content.")
    base["total_items"] = total
    base["content"] = []
    base["position"] = position
    _require_capacity(base, max_bytes, content_fits)
    for i, item in enumerate(items):
        if i >= total:
            break
        if i < position:
            continue
        candidate = {**base, "content": [*base["content"], item]}
        candidate["selection_complete"] = i + 1 == total
        candidate["next_read"] = (
            None if candidate["selection_complete"] else next_read(artifact_id, digest, {**query, "position": i + 1})
        )
        if not _fits(candidate, max_bytes, content_fits):
            if not base["content"]:
                if _fits(candidate, max_bytes, None):
                    _require_capacity(candidate, max_bytes, content_fits)
                suggested = {
                    "operation": "inspect",
                    "json_pointer": child_pointer(pointer, str(i)) if isinstance(value, list) else pointer,
                }
                if operation == "inspect":
                    suggested = {"operation": "read", "offset": 0}
                elif "json_pointer" not in query:
                    suggested = {"operation": "read", "offset": len("".join(lines[:i]).encode("utf-8"))}
                raise NavigationError(
                    "item_too_large",
                    "One item exceeds the page budget. Inspect its structure, select a child or fewer fields, or use raw offset reading.",
                    suggested_query=suggested,
                )
            break
        base = candidate
    if not base["content"]:
        base["selection_complete"] = position == total
    _require_capacity(base, max_bytes, content_fits)
    return dumps(base)


def _fits(payload: dict[str, Any], max_bytes: int, predicate: Callable[[str], bool] | None) -> bool:
    content = dumps(payload)
    return len(content.encode("utf-8")) <= max_bytes and (predicate is None or predicate(content))


def _require_capacity(payload: dict[str, Any], max_bytes: int, predicate: Callable[[str], bool] | None) -> None:
    if not _fits(payload, max_bytes, None):
        raise NavigationError(
            "page_too_small", "The byte budget cannot hold a page with reading progress.", recoverable=False
        )
    if not _fits(payload, max_bytes, predicate):
        raise NavigationError(
            "context_exhausted", "The remaining model context cannot hold the next page.", recoverable=False
        )


def _string_page(
    base: dict[str, Any],
    value: str,
    query: dict[str, Any],
    digest: str,
    max_bytes: int,
    predicate: Callable[[str], bool] | None,
) -> str:
    position = query.get("position", 0)
    if position > len(value):
        raise NavigationError("invalid_arguments", "Position is beyond the selected string.")

    def candidate(end: int) -> dict[str, Any]:
        return {
            **base,
            "content": value[position:end],
            "position": position,
            "offset_unit": "characters_in_selection",
            "selection_complete": end == len(value),
            "next_read": None
            if end == len(value)
            else next_read(base["artifact_id"], digest, {**query, "position": end}),
        }

    low, high = position, min(len(value), position + max_bytes)
    best = None
    while low <= high:
        middle = (low + high) // 2
        page = candidate(middle)
        if _fits(page, max_bytes, predicate):
            best = page
            low = middle + 1
        else:
            high = middle - 1
    # The final page has no cursor, so its envelope can be smaller.
    if len(value) - position <= max_bytes and _fits(candidate(len(value)), max_bytes, predicate):
        best = candidate(len(value))
    if best is None or (not best["content"] and position < len(value)):
        _require_capacity(candidate(min(position + 1, len(value))), max_bytes, predicate)
        raise NavigationError("page_too_small", "The page cannot hold the next character.", recoverable=False)
    return dumps(best)


def _search_page(
    base: dict[str, Any],
    found: list[dict[str, Any]],
    source: str,
    needle: str,
    query: dict[str, Any],
    digest: str,
    max_bytes: int,
    predicate: Callable[[str], bool] | None,
) -> str:
    base["content"] = []
    base["position"] = query.get("position", 0)
    for item in found:
        end = item["char_offset"] + len(needle)
        candidate = {
            **base,
            "content": [*base["content"], item],
            "selection_complete": False,
            "next_read": next_read(base["artifact_id"], digest, {**query, "position": end}),
        }
        if not _fits(candidate, max_bytes, predicate):
            if not base["content"]:
                _require_capacity(candidate, max_bytes, predicate)
            break
        base = candidate
    if len(base["content"]) == len(found) and len(found) < 100:
        base["selection_complete"] = True
        base["next_read"] = None
    _require_capacity(base, max_bytes, predicate)
    return dumps(base)
