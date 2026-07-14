from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

from agent_core.llm.base import LLMMessage, LLMToolDefinition
from agent_core.types import ToolResult

if TYPE_CHECKING:
    from agent_core.execution_context import ExecutionContext

READ_ARTIFACT_TOOL_NAME = "agent_core_read_artifact"
_ARTIFACT_ID_PATTERN = re.compile(r"^art_[0-9a-f]{32}$")
_ARTIFACT_METADATA_KEY = "tool_artifact"


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ToolArtifactPolicy:
    """Lossless storage and bounded context projection for tool results."""

    hot_context_bytes: int = 64 * 1024
    max_inline_result_bytes: int = 32 * 1024
    max_read_bytes: int = 16 * 1024
    max_reads_per_run: int = 20
    max_total_read_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        for name in (
            "hot_context_bytes",
            "max_inline_result_bytes",
            "max_read_bytes",
            "max_reads_per_run",
            "max_total_read_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), field_name=name))

    def to_dict(self) -> dict[str, int]:
        return {
            "hot_context_bytes": self.hot_context_bytes,
            "max_inline_result_bytes": self.max_inline_result_bytes,
            "max_read_bytes": self.max_read_bytes,
            "max_reads_per_run": self.max_reads_per_run,
            "max_total_read_bytes": self.max_total_read_bytes,
        }

    @classmethod
    def from_any(cls, payload: object) -> ToolArtifactPolicy | None:
        if payload is None:
            return None
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("Tool artifact policy must be a ToolArtifactPolicy, dictionary, or None")
        return cls(
            hot_context_bytes=cast(int, payload.get("hot_context_bytes", 64 * 1024)),
            max_inline_result_bytes=cast(int, payload.get("max_inline_result_bytes", 32 * 1024)),
            max_read_bytes=cast(int, payload.get("max_read_bytes", 16 * 1024)),
            max_reads_per_run=cast(int, payload.get("max_reads_per_run", 20)),
            max_total_read_bytes=cast(int, payload.get("max_total_read_bytes", 256 * 1024)),
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

    def reference_content(self) -> str:
        return json.dumps(
            {
                **self.to_dict(),
                "externalized": True,
                "fully_loaded": False,
                "read_tool": READ_ARTIFACT_TOOL_NAME,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactChunk:
    artifact_id: str
    offset: int
    next_offset: int
    size_bytes: int
    content: str
    eof: bool

    def to_content(self) -> str:
        return json.dumps(
            {
                "artifact_id": self.artifact_id,
                "offset": self.offset,
                "next_offset": self.next_offset,
                "returned_bytes": self.next_offset - self.offset,
                "size_bytes": self.size_bytes,
                "eof": self.eof,
                "content": self.content,
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
    ) -> ToolArtifactDescriptor:
        ...

    def read_text(
        self,
        *,
        namespace_id: str,
        artifact_id: str,
        offset: int,
        limit: int,
    ) -> ArtifactChunk:
        ...

    def read_all_text(self, *, namespace_id: str, artifact_id: str) -> str:
        ...


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

    def to_dict(self) -> dict[str, int]:
        return {
            "artifacts_written": self.artifacts_written,
            "artifact_bytes_written": self.artifact_bytes_written,
            "internal_tool_calls": self.internal_tool_calls,
            "artifact_bytes_read": self.artifact_bytes_read,
            "reads_rejected": self.reads_rejected,
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

    def externalize(
        self,
        *,
        tool_name: str,
        content: str,
        tool_call_id: str,
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
        visible_content = content if descriptor.size_bytes <= self.policy.max_inline_result_bytes else descriptor.reference_content()
        return LLMMessage(
            role="tool",
            tool_call_id=tool_call_id,
            content=visible_content,
            metadata={_ARTIFACT_METADATA_KEY: descriptor.to_dict()},
        )

    def project_messages(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        projected = [replace(message, tool_calls=list(message.tool_calls), metadata=dict(message.metadata)) for message in messages]
        remaining = self.policy.hot_context_bytes
        for message in reversed(projected):
            descriptor = artifact_descriptor_from_message(message)
            if descriptor is None:
                continue
            if descriptor.size_bytes <= remaining and descriptor.size_bytes <= self.policy.max_inline_result_bytes:
                try:
                    message.content = self.store.read_all_text(
                        namespace_id=self.namespace_id,
                        artifact_id=descriptor.artifact_id,
                    )
                except (FileNotFoundError, PermissionError, ValueError, OSError):
                    message.content = descriptor.reference_content()
                else:
                    remaining -= descriptor.size_bytes
            else:
                message.content = descriptor.reference_content()
        return projected

    def prepare_messages(self, messages: list[LLMMessage]) -> None:
        messages[:] = self.project_messages(messages)

    def tool_specs(self) -> list[LLMToolDefinition]:
        return [
            LLMToolDefinition(
                name=READ_ARTIFACT_TOOL_NAME,
                description=(
                    "Read a bounded UTF-8 chunk from a tool-result artifact previously externalized by agent-core. "
                    "Use next_offset to continue; the artifact is not fully reviewed until eof is true."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": "string", "pattern": r"^art_[0-9a-f]{32}$"},
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

    def execute(self, *, tool_name: str, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        if not self.is_internal_tool(tool_name):
            raise KeyError(tool_name)
        if context.namespace_id != self.namespace_id:
            return ToolResult(ok=False, content="Artifact read denied: namespace mismatch.")
        if self.usage.internal_tool_calls >= self.policy.max_reads_per_run:
            self.usage.reads_rejected += 1
            return ToolResult(ok=False, content="Artifact read denied: maximum internal read calls reached.")
        artifact_id = arguments.get("artifact_id")
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
            chunk = self.store.read_text(
                namespace_id=self.namespace_id,
                artifact_id=artifact_id,
                offset=offset,
                limit=limit,
            )
        except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
            return ToolResult(ok=False, content=f"Artifact read failed: {exc}")
        self.usage.artifact_bytes_read += chunk.next_offset - chunk.offset
        return ToolResult(
            ok=True,
            content=chunk.to_content(),
            metadata={"tool_kind": "runtime", "artifact_read": True, "externalize": False},
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "tool_artifact_policy": self.policy.to_dict(),
            "tool_artifact_usage": self.usage.to_dict(),
        }


def artifact_descriptor_from_message(message: LLMMessage) -> ToolArtifactDescriptor | None:
    return ToolArtifactDescriptor.from_any(message.metadata.get(_ARTIFACT_METADATA_KEY))


def message_to_persistence_dict(message: LLMMessage) -> dict[str, Any]:
    payload = message.to_history_dict()
    descriptor = artifact_descriptor_from_message(message)
    if descriptor is not None:
        payload["content"] = descriptor.reference_content()
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
