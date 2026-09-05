from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, cast
from uuid import uuid4

from agent_core.artifact_navigation import NavigationError, decode_cursor, next_read, render_page, validate_query
from agent_core.llm.base import LLMMessage, LLMToolDefinition
from agent_core.types import ToolExecutionStatus, ToolResult

if TYPE_CHECKING:
    from agent_core.execution_context import ExecutionContext

READ_ARTIFACT_TOOL_NAME = "agent_core_read_artifact"
_ARTIFACT_ID_PATTERN = re.compile(r"^art_[0-9a-f]{32}$")
_ARTIFACT_METADATA_KEY = "artifact_result"
logger = logging.getLogger(__name__)
_TOOL_EXECUTION_STATUSES = {
    "ok",
    "pending",
    "tool_error",
    "policy_denied",
    "invalid_arguments",
    "execution_failed",
    "budget_exhausted",
}

ArtifactMaterialization: TypeAlias = Literal["complete", "preview", "reference"]


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ToolArtifactPolicy:
    """Lossless storage and bounded context projection for tool results."""

    hot_context_bytes: int = 64 * 1024
    max_complete_result_bytes: int = 32 * 1024
    preview_bytes: int = 4 * 1024
    max_read_bytes: int = 16 * 1024
    max_reads_per_run: int = 20
    max_total_read_bytes: int = 256 * 1024
    max_navigation_source_bytes: int = 8 * 1024 * 1024
    max_navigation_cache_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "hot_context_bytes",
            "max_complete_result_bytes",
            "preview_bytes",
            "max_read_bytes",
            "max_reads_per_run",
            "max_total_read_bytes",
            "max_navigation_source_bytes",
            "max_navigation_cache_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), field_name=name))
        if self.preview_bytes > self.hot_context_bytes:
            raise ValueError("preview_bytes must not exceed hot_context_bytes")

    def to_dict(self) -> dict[str, int]:
        return {
            "hot_context_bytes": self.hot_context_bytes,
            "max_complete_result_bytes": self.max_complete_result_bytes,
            "preview_bytes": self.preview_bytes,
            "max_read_bytes": self.max_read_bytes,
            "max_reads_per_run": self.max_reads_per_run,
            "max_total_read_bytes": self.max_total_read_bytes,
            "max_navigation_source_bytes": self.max_navigation_source_bytes,
            "max_navigation_cache_bytes": self.max_navigation_cache_bytes,
        }

    @classmethod
    def from_any(cls, payload: object) -> ToolArtifactPolicy:
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("Tool artifact policy must be a ToolArtifactPolicy or dictionary")
        return cls(
            hot_context_bytes=cast(int, payload.get("hot_context_bytes", 64 * 1024)),
            max_complete_result_bytes=cast(
                int,
                payload.get("max_complete_result_bytes", 32 * 1024),
            ),
            preview_bytes=cast(int, payload.get("preview_bytes", 4 * 1024)),
            max_read_bytes=cast(int, payload.get("max_read_bytes", 16 * 1024)),
            max_reads_per_run=cast(int, payload.get("max_reads_per_run", 20)),
            max_total_read_bytes=cast(int, payload.get("max_total_read_bytes", 256 * 1024)),
            max_navigation_source_bytes=cast(int, payload.get("max_navigation_source_bytes", 8 * 1024 * 1024)),
            max_navigation_cache_bytes=cast(int, payload.get("max_navigation_cache_bytes", 16 * 1024 * 1024)),
        )


@dataclass(frozen=True, slots=True)
class ToolArtifactDescriptor:
    artifact_id: str
    tool_name: str
    size_bytes: int
    sha256: str
    content_type: str = "text/plain; charset=utf-8"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "tool_name": self.tool_name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "content_type": self.content_type,
        }

    @classmethod
    def from_any(cls, payload: object) -> ToolArtifactDescriptor | None:
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return None
        artifact_id = payload.get("artifact_id")
        tool_name = payload.get("tool_name")
        size_bytes = payload.get("size_bytes")
        sha256 = payload.get("sha256")
        content_type = payload.get("content_type", "text/plain; charset=utf-8")
        if (
            not isinstance(artifact_id, str)
            or not _ARTIFACT_ID_PATTERN.fullmatch(artifact_id)
            or not isinstance(tool_name, str)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or not isinstance(content_type, str)
        ):
            return None
        return cls(
            artifact_id=artifact_id,
            tool_name=tool_name,
            size_bytes=size_bytes,
            sha256=sha256,
            content_type=content_type,
        )


@dataclass(frozen=True, slots=True)
class ArtifactResultEnvelope:
    """Stable provider-facing contract for one stored application-tool result."""

    status: ToolExecutionStatus
    artifact: ToolArtifactDescriptor
    materialization: ArtifactMaterialization
    content: str | None
    returned_bytes: int
    complete: bool
    next_offset: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "kind": "artifact_result",
            "status": self.status,
            "artifact": self.artifact.to_dict(),
            "materialization": self.materialization,
            "content": self.content,
            "returned_bytes": self.returned_bytes,
            "complete": self.complete,
            "next_offset": self.next_offset,
            "read_tool": READ_ARTIFACT_TOOL_NAME,
            "next_read": None
            if self.complete
            else next_read(self.artifact.artifact_id, self.artifact.sha256, {"offset": self.next_offset or 0}),
        }

    def to_content(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def as_reference(self) -> ArtifactResultEnvelope:
        return ArtifactResultEnvelope.reference(status=self.status, artifact=self.artifact)

    @classmethod
    def reference(
        cls,
        *,
        status: ToolExecutionStatus,
        artifact: ToolArtifactDescriptor,
    ) -> ArtifactResultEnvelope:
        return cls(
            status=status,
            artifact=artifact,
            materialization="reference",
            content=None,
            returned_bytes=0,
            complete=False,
            next_offset=0,
        )

    @classmethod
    def from_any(cls, payload: object) -> ArtifactResultEnvelope | None:
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != "1" or payload.get("kind") != "artifact_result":
            return None
        status = payload.get("status")
        artifact = ToolArtifactDescriptor.from_any(payload.get("artifact"))
        materialization = payload.get("materialization")
        content = payload.get("content")
        returned_bytes = payload.get("returned_bytes")
        complete = payload.get("complete")
        next_offset = payload.get("next_offset")
        if (
            status not in _TOOL_EXECUTION_STATUSES
            or artifact is None
            or materialization not in {"complete", "preview", "reference"}
            or (content is not None and not isinstance(content, str))
            or not isinstance(returned_bytes, int)
            or isinstance(returned_bytes, bool)
            or returned_bytes < 0
            or not isinstance(complete, bool)
            or (
                next_offset is not None
                and (not isinstance(next_offset, int) or isinstance(next_offset, bool) or next_offset < 0)
            )
        ):
            return None
        if materialization == "complete" and (
            content is None
            or not complete
            or next_offset is not None
            or returned_bytes != artifact.size_bytes
            or len(content.encode("utf-8")) != returned_bytes
        ):
            return None
        if materialization == "preview" and (
            content is None
            or complete
            or next_offset != returned_bytes
            or returned_bytes >= artifact.size_bytes
            or len(content.encode("utf-8")) != returned_bytes
        ):
            return None
        if materialization == "reference" and (
            content is not None or complete or returned_bytes != 0 or next_offset != 0
        ):
            return None
        return cls(
            status=cast(ToolExecutionStatus, status),
            artifact=artifact,
            materialization=cast(ArtifactMaterialization, materialization),
            content=content,
            returned_bytes=returned_bytes,
            complete=complete,
            next_offset=next_offset,
        )


@dataclass(frozen=True, slots=True)
class ArtifactChunk:
    artifact_id: str
    offset: int
    next_offset: int
    size_bytes: int
    content: str
    eof: bool
    sha256: str = ""

    def to_content(self) -> str:
        return json.dumps(
            {
                "schema_version": "1",
                "kind": "artifact_chunk",
                "artifact_id": self.artifact_id,
                "offset": self.offset,
                "next_offset": self.next_offset,
                "returned_bytes": self.next_offset - self.offset,
                "size_bytes": self.size_bytes,
                "eof": self.eof,
                "content": self.content,
                "next_read": None
                if self.eof
                else next_read(self.artifact_id, self.sha256, {"offset": self.next_offset}),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class ArtifactStore(Protocol):
    def put_text(
        self,
        *,
        namespace_id: str,
        run_id: str,
        tool_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolArtifactDescriptor: ...

    def read_text(
        self,
        *,
        namespace_id: str,
        artifact_id: str,
        offset: int,
        limit: int,
    ) -> ArtifactChunk: ...

    def read_all_text(self, *, namespace_id: str, artifact_id: str) -> str: ...


class JsonFileArtifactStore:
    """Simple namespace-isolated artifact store with atomic file publication."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()

    def put_text(
        self,
        *,
        namespace_id: str,
        run_id: str,
        tool_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolArtifactDescriptor:
        raw = content.encode("utf-8")
        artifact_id = f"art_{uuid4().hex}"
        descriptor = ToolArtifactDescriptor(
            artifact_id=artifact_id,
            tool_name=tool_name,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        namespace_directory = self._namespace_directory(namespace_id)
        namespace_directory.mkdir(parents=True, exist_ok=True)
        content_path, metadata_path = self._paths(namespace_directory, artifact_id)
        self._atomic_write_bytes(content_path, raw)
        self._atomic_write_bytes(
            metadata_path,
            json.dumps(
                {
                    "schema_version": 1,
                    "namespace_id": namespace_id,
                    "run_id": run_id,
                    "descriptor": descriptor.to_dict(),
                    "metadata": dict(metadata or {}),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8"),
        )
        return descriptor

    def read_text(
        self,
        *,
        namespace_id: str,
        artifact_id: str,
        offset: int,
        limit: int,
    ) -> ArtifactChunk:
        descriptor, content_path = self._resolve_owned(namespace_id=namespace_id, artifact_id=artifact_id)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        limit = _positive_int(limit, field_name="limit")
        if offset > descriptor.size_bytes:
            raise ValueError("offset is beyond the end of the artifact")
        if content_path.stat().st_size != descriptor.size_bytes:
            raise ValueError("artifact content size does not match its descriptor")
        with content_path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(min(limit, descriptor.size_bytes - offset))
        decoded_raw = raw
        while decoded_raw:
            try:
                content = decoded_raw.decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                if exc.reason != "unexpected end of data" or len(decoded_raw) - exc.start > 4:
                    raise ValueError("artifact content is not valid UTF-8") from exc
                decoded_raw = decoded_raw[: exc.start]
        else:
            content = ""
        if not decoded_raw and raw and offset < descriptor.size_bytes:
            with content_path.open("rb") as handle:
                handle.seek(offset)
                prefix = handle.read(min(4, descriptor.size_bytes - offset))
            for end in range(1, len(prefix) + 1):
                try:
                    content = prefix[:end].decode("utf-8")
                except UnicodeDecodeError:
                    continue
                decoded_raw = prefix[:end]
                break
            if not decoded_raw:
                raise ValueError("artifact content is not valid UTF-8")
        next_offset = offset + len(decoded_raw)
        return ArtifactChunk(
            artifact_id=artifact_id,
            offset=offset,
            next_offset=next_offset,
            size_bytes=descriptor.size_bytes,
            content=content,
            eof=next_offset >= descriptor.size_bytes,
            sha256=descriptor.sha256,
        )

    def read_all_text(self, *, namespace_id: str, artifact_id: str) -> str:
        descriptor, content_path = self._resolve_owned(namespace_id=namespace_id, artifact_id=artifact_id)
        raw = content_path.read_bytes()
        if len(raw) != descriptor.size_bytes or hashlib.sha256(raw).hexdigest() != descriptor.sha256:
            raise ValueError("artifact content does not match its descriptor")
        return raw.decode("utf-8")

    def _resolve_owned(self, *, namespace_id: str, artifact_id: str) -> tuple[ToolArtifactDescriptor, Path]:
        if not _ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
            raise ValueError("invalid artifact id")
        namespace_directory = self._namespace_directory(namespace_id)
        content_path, metadata_path = self._paths(namespace_directory, artifact_id)
        if not metadata_path.exists() or not content_path.exists():
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"artifact metadata is invalid: {artifact_id}") from exc
        descriptor = ToolArtifactDescriptor.from_any(payload.get("descriptor") if isinstance(payload, dict) else None)
        if descriptor is None or payload.get("namespace_id") != namespace_id:
            raise PermissionError("artifact is not available in this namespace")
        return descriptor, content_path

    def _namespace_directory(self, namespace_id: str) -> Path:
        digest = hashlib.sha256(namespace_id.encode("utf-8")).hexdigest()[:24]
        return self.directory / digest

    @staticmethod
    def _paths(namespace_directory: Path, artifact_id: str) -> tuple[Path, Path]:
        return namespace_directory / f"{artifact_id}.txt", namespace_directory / f"{artifact_id}.json"

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


@dataclass(slots=True)
class ToolArtifactUsage:
    artifacts_written: int = 0
    artifact_bytes_written: int = 0
    internal_tool_calls: int = 0
    artifact_bytes_read: int = 0
    reads_rejected: int = 0
    recovery_attempts: int = 0
    read_progress: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts_written": self.artifacts_written,
            "artifact_bytes_written": self.artifact_bytes_written,
            "internal_tool_calls": self.internal_tool_calls,
            "artifact_bytes_read": self.artifact_bytes_read,
            "reads_rejected": self.reads_rejected,
            "recovery_attempts": self.recovery_attempts,
            "read_progress": {key: dict(value) for key, value in self.read_progress.items()},
        }

    @classmethod
    def from_any(cls, payload: object) -> ToolArtifactUsage:
        if isinstance(payload, cls):
            return cls(**payload.to_dict())
        if not isinstance(payload, dict):
            return cls()

        def count(name: str) -> int:
            value = payload.get(name)
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

        return cls(
            artifacts_written=count("artifacts_written"),
            artifact_bytes_written=count("artifact_bytes_written"),
            internal_tool_calls=count("internal_tool_calls"),
            artifact_bytes_read=count("artifact_bytes_read"),
            reads_rejected=count("reads_rejected"),
            recovery_attempts=count("recovery_attempts"),
            read_progress={
                key: dict(value)
                for key, value in list(payload.get("read_progress", {}).items())[-16:]
                if isinstance(key, str) and isinstance(value, dict)
            }
            if isinstance(payload.get("read_progress"), dict)
            else {},
        )


class ToolArtifactRuntime:
    def __init__(
        self,
        *,
        policy: ToolArtifactPolicy,
        store: ArtifactStore,
        namespace_id: str,
        run_id: str,
        usage: ToolArtifactUsage | dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.store = store
        self.namespace_id = namespace_id
        self.run_id = run_id
        self.usage = ToolArtifactUsage.from_any(usage)
        self._navigation_cache: OrderedDict[str, tuple[str, str, Any, int]] = OrderedDict()

    def externalize(
        self,
        *,
        tool_name: str,
        content: str,
        tool_call_id: str,
        status: ToolExecutionStatus,
        metadata: dict[str, Any] | None = None,
    ) -> LLMMessage:
        descriptor = self.store.put_text(
            namespace_id=self.namespace_id,
            run_id=self.run_id,
            tool_name=tool_name,
            content=content,
            metadata=metadata,
        )
        self.usage.artifacts_written += 1
        self.usage.artifact_bytes_written += descriptor.size_bytes
        envelope = self._materialize_new(
            descriptor=descriptor,
            status=status,
            content=content,
        )
        return LLMMessage(
            role="tool",
            tool_call_id=tool_call_id,
            content=envelope.to_content(),
            metadata={_ARTIFACT_METADATA_KEY: envelope.as_reference().to_dict()},
        )

    def project_messages(
        self,
        messages: list[LLMMessage],
        *,
        messages_fit: Callable[[list[LLMMessage]], bool] | None = None,
    ) -> list[LLMMessage]:
        projected = [
            replace(message, tool_calls=list(message.tool_calls), metadata=dict(message.metadata))
            for message in messages
        ]
        stored_envelopes: list[tuple[LLMMessage, ArtifactResultEnvelope]] = []
        for message in projected:
            stored_envelope = artifact_envelope_from_message(message)
            if stored_envelope is None:
                continue
            stored_envelopes.append((message, stored_envelope))
            message.content = stored_envelope.as_reference().to_content()

        remaining = self.policy.hot_context_bytes
        for message, stored_envelope in reversed(stored_envelopes):
            materialization_bytes = self._materialization_bytes(stored_envelope.artifact)
            if materialization_bytes <= remaining:
                envelope = self._materialize_stored(stored_envelope)
                if envelope.materialization == "reference":
                    continue
                message.content = envelope.to_content()
                if messages_fit is not None and not messages_fit(projected):
                    message.content = stored_envelope.as_reference().to_content()
                    continue
                remaining -= envelope.returned_bytes
        return projected

    def prepare_messages(
        self,
        messages: list[LLMMessage],
        *,
        messages_fit: Callable[[list[LLMMessage]], bool] | None = None,
    ) -> None:
        messages[:] = self.project_messages(messages, messages_fit=messages_fit)

    def tool_specs(self) -> list[LLMToolDefinition]:
        return [
            LLMToolDefinition(
                name=READ_ARTIFACT_TOOL_NAME,
                description=(
                    "Application tool results are artifact_result envelopes. If complete is true, content contains "
                    "the full result and no read is needed. If materialization is preview or reference, use this "
                    "tool only when missing details are needed. Start at the envelope next_offset after a preview, "
                    "then follow next_read.arguments exactly. operation='inspect' shows JSON structure; "
                    "operation='read' with json_pointer selects JSON and fields projects object fields; "
                    "operation='search' finds literal text. start_line reads text lines. "
                    "A continuation preserves the selection; do not combine it with selection parameters. "
                    "selection_complete concerns the selected view only, not the whole document. "
                    "Recoverable read errors contain a suggested_action. Try that action before treating available data as blocked."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": "string", "pattern": r"^art_[0-9a-f]{32}$"},
                        "operation": {"type": "string", "enum": ["read", "inspect", "search"]},
                        "continuation": {"type": "string", "maxLength": 8192},
                        "json_pointer": {"type": "string", "maxLength": 2048},
                        "fields": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 128},
                            "minItems": 1,
                            "maxItems": 32,
                        },
                        "query": {"type": "string", "minLength": 1, "maxLength": 256},
                        "start_line": {"type": "integer", "minimum": 1},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": self.policy.max_read_bytes,
                            "default": self.policy.max_read_bytes,
                        },
                    },
                    "required": ["artifact_id"],
                    "additionalProperties": False,
                },
            )
        ]

    def has_readable_artifacts(self, messages: list[LLMMessage]) -> bool:
        return self.usage.artifacts_written > 0 or any(
            artifact_descriptor_from_message(message) is not None for message in messages
        )

    def is_internal_tool(self, tool_name: str) -> bool:
        return tool_name == READ_ARTIFACT_TOOL_NAME

    def execute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
        content_fits: Callable[[str], bool] | None = None,
    ) -> ToolResult:
        if not self.is_internal_tool(tool_name):
            raise KeyError(tool_name)
        if context.namespace_id != self.namespace_id:
            return ToolResult(ok=False, content="Artifact read denied: namespace mismatch.")
        if self.usage.internal_tool_calls >= self.policy.max_reads_per_run:
            self.usage.reads_rejected += 1
            return ToolResult(ok=False, content="Artifact read denied: maximum internal read calls reached.")
        artifact_id = arguments.get("artifact_id")
        if isinstance(artifact_id, str) and any(
            key in arguments for key in ("continuation", "operation", "json_pointer", "fields", "query", "start_line")
        ):
            return self._navigate(arguments=arguments, context=context, content_fits=content_fits)
        offset = arguments.get("offset", 0)
        requested_limit = arguments.get("limit", self.policy.max_read_bytes)
        if not isinstance(artifact_id, str):
            return ToolResult(ok=False, content="Artifact read failed: artifact_id must be a string.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            return ToolResult(ok=False, content="Artifact read failed: offset must be a non-negative integer.")
        if isinstance(requested_limit, bool) or not isinstance(requested_limit, int) or requested_limit <= 0:
            return ToolResult(ok=False, content="Artifact read failed: limit must be a positive integer.")
        remaining = self.policy.max_total_read_bytes - self.usage.artifact_bytes_read
        if remaining <= 0:
            self.usage.reads_rejected += 1
            return ToolResult(ok=False, content="Artifact read denied: total artifact read budget reached.")
        limit = min(requested_limit, self.policy.max_read_bytes, remaining)
        self.usage.internal_tool_calls += 1
        try:
            chunk, context_limited = self._largest_fitting_chunk(
                artifact_id=artifact_id,
                offset=offset,
                limit=limit,
                content_fits=content_fits,
            )
        except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
            return ToolResult(ok=False, content=f"Artifact read failed: {exc}")
        if chunk is None:
            self.usage.reads_rejected += 1
            return ToolResult(
                ok=False,
                content=(
                    "Artifact read stopped: the remaining model context cannot safely hold another chunk. "
                    "Use the evidence already read and return the best supported final answer."
                ),
                metadata={
                    "tool_kind": "runtime",
                    "artifact_read": True,
                    "artifact_read_context_exhausted": True,
                    "externalize": False,
                },
            )
        if chunk.next_offset - chunk.offset > limit:
            recoverable = min(self.policy.max_read_bytes, remaining) >= 4
            return self._navigation_error(
                artifact_id,
                NavigationError(
                    "byte_budget_too_small",
                    "The byte limit cannot hold the next complete UTF-8 character.",
                    recoverable=recoverable,
                    suggested_query={"offset": offset, "limit": 4} if recoverable else None,
                ),
            )
        self.usage.artifact_bytes_read += chunk.next_offset - chunk.offset
        self._remember_read(artifact_id, json.loads(chunk.to_content()))
        metadata: dict[str, Any] = {
            "tool_kind": "runtime",
            "artifact_read": True,
            "externalize": False,
        }
        if context_limited:
            metadata["artifact_read_context_limited"] = True
        return ToolResult(
            ok=True,
            content=chunk.to_content(),
            metadata=metadata,
        )

    def _remember_read(self, artifact_id: str, payload: dict[str, Any]) -> None:
        self.usage.read_progress.pop(artifact_id, None)
        self.usage.read_progress[artifact_id] = {
            "selection": payload.get("selection", {}),
            "projection_fields": payload.get("projection_fields"),
            "last_page_complete": payload.get("selection_complete", payload.get("eof", False)),
            "next_read": payload.get("next_read"),
            "note": "Previously delivered; content may no longer be in the model context.",
        }
        while len(self.usage.read_progress) > 16:
            del self.usage.read_progress[next(iter(self.usage.read_progress))]
        logger.debug(
            "Artifact page delivered",
            extra={
                "artifact_id": artifact_id,
                "selection_complete": self.usage.read_progress[artifact_id]["last_page_complete"],
            },
        )

    def _navigation_error(self, artifact_id: str, error: NavigationError) -> ToolResult:
        self.usage.reads_rejected += 1
        logger.debug(
            "Artifact navigation rejected",
            extra={"artifact_id": artifact_id, "error_code": error.code, "recoverable": error.recoverable},
        )
        action = {"tool": READ_ARTIFACT_TOOL_NAME, "arguments": {"artifact_id": artifact_id, "operation": "inspect"}}
        if error.code in {"source_too_large", "item_too_large"}:
            action["arguments"] = {"artifact_id": artifact_id, "offset": 0}
        if error.suggested_query is not None:
            action["arguments"] = {"artifact_id": artifact_id, **error.suggested_query}
        return ToolResult(
            ok=False,
            content=json.dumps(
                {
                    "kind": "artifact_read_error",
                    "code": error.code,
                    "message": str(error),
                    "recoverable": error.recoverable,
                    "suggested_action": action if error.recoverable else None,
                }
            ),
            metadata={
                "tool_kind": "runtime",
                "artifact_read": True,
                "externalize": False,
                "artifact_read_recoverable": error.recoverable,
                "artifact_read_context_exhausted": error.code == "context_exhausted",
            },
        )

    def _navigate(
        self,
        *,
        arguments: dict[str, Any],
        context: ExecutionContext,
        content_fits: Callable[[str], bool] | None,
    ) -> ToolResult:
        artifact_id = arguments["artifact_id"]
        self.usage.internal_tool_calls += 1
        try:
            # Ownership is checked even on a cache hit and before cursor decoding.
            probe = self.store.read_text(namespace_id=self.namespace_id, artifact_id=artifact_id, offset=0, limit=1)
            query = {key: value for key, value in arguments.items() if key != "artifact_id"}
            if "continuation" in query:
                if set(query) != {"continuation"}:
                    raise NavigationError("invalid_arguments", "Pass continuation alone with artifact_id.")
                query = decode_cursor(query["continuation"], artifact_id, probe.sha256)
            validate_query(query)
            if query.get("operation", "read") == "read" and not any(
                key in query for key in ("json_pointer", "fields", "query", "start_line", "position")
            ):
                self.usage.internal_tool_calls -= 1
                return self.execute(
                    tool_name=READ_ARTIFACT_TOOL_NAME,
                    arguments={
                        "artifact_id": artifact_id,
                        "offset": query.get("offset", 0),
                        "limit": query.get("limit", self.policy.max_read_bytes),
                    },
                    context=context,
                    content_fits=content_fits,
                )
            remaining = self.policy.max_total_read_bytes - self.usage.artifact_bytes_read
            if remaining <= 0:
                raise NavigationError("budget_exhausted", "Total artifact read budget reached.", recoverable=False)
            if probe.size_bytes > self.policy.max_navigation_source_bytes:
                raise NavigationError(
                    "source_too_large", "Artifact exceeds the structured navigation size limit; use bounded raw reads."
                )
            cached = self._navigation_cache.get(artifact_id)
            if cached is not None and cached[0] == probe.sha256:
                _, text, parsed, _ = cached
                self._navigation_cache.move_to_end(artifact_id)
            else:
                text = self.store.read_all_text(namespace_id=self.namespace_id, artifact_id=artifact_id)
                try:
                    parsed = json.loads(text)
                except (ValueError, RecursionError):
                    parsed = text
                # Cache is bounded by source bytes and entry count. Parsed JSON
                # memory can exceed source size; source admission bounds each item.
                if probe.size_bytes <= self.policy.max_navigation_cache_bytes:
                    self._navigation_cache[artifact_id] = (probe.sha256, text, parsed, probe.size_bytes)
                    while (
                        len(self._navigation_cache) > 16
                        or sum(entry[3] for entry in self._navigation_cache.values())
                        > self.policy.max_navigation_cache_bytes
                    ):
                        self._navigation_cache.popitem(last=False)
            content = render_page(
                artifact_id=artifact_id,
                text=text,
                parsed=parsed,
                query=query,
                max_bytes=min(remaining, self.policy.max_read_bytes, query.get("limit", self.policy.max_read_bytes)),
                previous_read=self.usage.read_progress.get(artifact_id),
                sha256=probe.sha256,
                content_fits=content_fits,
            )
            self.usage.artifact_bytes_read += len(content.encode("utf-8"))
            self._remember_read(artifact_id, json.loads(content))
            return ToolResult(
                ok=True, content=content, metadata={"tool_kind": "runtime", "artifact_read": True, "externalize": False}
            )
        except NavigationError as exc:
            return self._navigation_error(artifact_id, exc)
        except (FileNotFoundError, PermissionError, OSError, ValueError, RecursionError):
            return self._navigation_error(
                artifact_id,
                NavigationError(
                    "artifact_unavailable", "Artifact cannot be read in this namespace.", recoverable=False
                ),
            )

    def claim_read_recovery(self, messages: Sequence[LLMMessage], statuses: Sequence[str]) -> bool:
        """Allow at most two reconsiderations of recoverable artifact-read errors.

        This does not execute actions or override authorization, context, or tool
        budgets. Only errors from the latest tool step are considered.
        """
        if (
            self.usage.recovery_attempts >= 2
            or self.usage.internal_tool_calls >= self.policy.max_reads_per_run
            or self.usage.artifact_bytes_read >= self.policy.max_total_read_bytes
        ):
            return False
        for message, status in zip(messages, statuses, strict=False):
            if status != "tool_error":
                continue
            try:
                payload = json.loads(message.content)
                if isinstance(payload, dict) and payload.get("kind") == "artifact_result":
                    payload = json.loads(payload.get("content") or "null")
            except (ValueError, TypeError):
                continue
            if (
                isinstance(payload, dict)
                and payload.get("kind") == "artifact_read_error"
                and payload.get("recoverable") is True
                and isinstance(payload.get("suggested_action"), dict)
            ):
                self.usage.recovery_attempts += 1
                return True
        return False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "tool_artifact_policy": self.policy.to_dict(),
            "tool_artifact_usage": self.usage.to_dict(),
        }

    def _materialize_new(
        self,
        *,
        descriptor: ToolArtifactDescriptor,
        status: ToolExecutionStatus,
        content: str,
    ) -> ArtifactResultEnvelope:
        if descriptor.size_bytes <= self.policy.max_complete_result_bytes:
            return ArtifactResultEnvelope(
                status=status,
                artifact=descriptor,
                materialization="complete",
                content=content,
                returned_bytes=descriptor.size_bytes,
                complete=True,
                next_offset=None,
            )
        try:
            return self._read_preview(descriptor=descriptor, status=status)
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            return ArtifactResultEnvelope.reference(status=status, artifact=descriptor)

    def _materialize_stored(
        self,
        envelope: ArtifactResultEnvelope,
    ) -> ArtifactResultEnvelope:
        descriptor = envelope.artifact
        try:
            if descriptor.size_bytes <= self.policy.max_complete_result_bytes:
                content = self.store.read_all_text(
                    namespace_id=self.namespace_id,
                    artifact_id=descriptor.artifact_id,
                )
                return ArtifactResultEnvelope(
                    status=envelope.status,
                    artifact=descriptor,
                    materialization="complete",
                    content=content,
                    returned_bytes=descriptor.size_bytes,
                    complete=True,
                    next_offset=None,
                )
            return self._read_preview(descriptor=descriptor, status=envelope.status)
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            return envelope.as_reference()

    def _read_preview(
        self,
        *,
        descriptor: ToolArtifactDescriptor,
        status: ToolExecutionStatus,
    ) -> ArtifactResultEnvelope:
        limit = min(self.policy.preview_bytes, descriptor.size_bytes - 1)
        chunk = self.store.read_text(
            namespace_id=self.namespace_id,
            artifact_id=descriptor.artifact_id,
            offset=0,
            limit=limit,
        )
        if chunk.eof:
            return ArtifactResultEnvelope.reference(status=status, artifact=descriptor)
        return ArtifactResultEnvelope(
            status=status,
            artifact=descriptor,
            materialization="preview",
            content=chunk.content,
            returned_bytes=chunk.next_offset,
            complete=False,
            next_offset=chunk.next_offset,
        )

    def _materialization_bytes(self, descriptor: ToolArtifactDescriptor) -> int:
        if descriptor.size_bytes <= self.policy.max_complete_result_bytes:
            return descriptor.size_bytes
        return min(self.policy.preview_bytes, descriptor.size_bytes - 1)

    def _largest_fitting_chunk(
        self,
        *,
        artifact_id: str,
        offset: int,
        limit: int,
        content_fits: Callable[[str], bool] | None,
    ) -> tuple[ArtifactChunk | None, bool]:
        chunk = self.store.read_text(
            namespace_id=self.namespace_id,
            artifact_id=artifact_id,
            offset=offset,
            limit=limit,
        )
        if content_fits is None or content_fits(chunk.to_content()):
            return chunk, False

        low = 1
        high = limit - 1
        best: ArtifactChunk | None = None
        while low <= high:
            candidate_limit = (low + high) // 2
            candidate = self.store.read_text(
                namespace_id=self.namespace_id,
                artifact_id=artifact_id,
                offset=offset,
                limit=candidate_limit,
            )
            if content_fits(candidate.to_content()):
                if best is None or candidate.next_offset > best.next_offset:
                    best = candidate
                low = candidate_limit + 1
            else:
                high = candidate_limit - 1
        return best, best is not None


def artifact_descriptor_from_message(message: LLMMessage) -> ToolArtifactDescriptor | None:
    envelope = artifact_envelope_from_message(message)
    return envelope.artifact if envelope is not None else None


def artifact_envelope_from_message(message: LLMMessage) -> ArtifactResultEnvelope | None:
    return ArtifactResultEnvelope.from_any(message.metadata.get(_ARTIFACT_METADATA_KEY))


def message_to_persistence_dict(message: LLMMessage) -> dict[str, Any]:
    payload = message.to_history_dict()
    envelope = artifact_envelope_from_message(message)
    if envelope is not None:
        payload["content"] = envelope.as_reference().to_content()
    if message.metadata:
        payload["_agent_core"] = dict(message.metadata)
    return payload


_ACTIVE_TOOL_ARTIFACT_RUNTIME: ContextVar[ToolArtifactRuntime | None] = ContextVar(
    "active_tool_artifact_runtime",
    default=None,
)


@contextmanager
def tool_artifact_scope(runtime: ToolArtifactRuntime | None) -> Iterator[ToolArtifactRuntime | None]:
    token = _ACTIVE_TOOL_ARTIFACT_RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _ACTIVE_TOOL_ARTIFACT_RUNTIME.reset(token)


def active_tool_artifact_runtime() -> ToolArtifactRuntime | None:
    return _ACTIVE_TOOL_ARTIFACT_RUNTIME.get()
