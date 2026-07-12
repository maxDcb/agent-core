from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Generic, TypeVar

from agent_core.llm.base import BaseLLMProvider, LLMCallOptions, LLMMessage, completion_content
from agent_core.logging_utils import get_logger, safe_preview
from agent_core.settings import CoreSettings

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(slots=True)
class StructuredSynthesisRequest(Generic[T]):
    """Describe one structured-memory synthesis job for the LLM."""

    target_name: str
    instructions: str
    output_format: dict[str, Any]
    payload: dict[str, Any]
    parser: Callable[[object], T | None]
    options: LLMCallOptions | None = None


class StructuredSynthesizer:
    """Generic JSON synthesizer for structured memory objects.

    The synthesizer does not know about TaskState or SessionSummary directly. It
    only knows how to ask the LLM for a single JSON object that matches a caller
    supplied output format, then validate that object through the caller's
    parser.
    """

    def __init__(
        self,
        *,
        settings: CoreSettings,
        provider: BaseLLMProvider,
    ) -> None:
        self.settings = settings
        self.provider = provider

    def synthesize(self, *, request: StructuredSynthesisRequest[T]) -> T:
        system_prompt = self._build_system_prompt(
            instructions=request.instructions,
            output_format=request.output_format,
        )
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=json.dumps(request.payload, ensure_ascii=False, indent=2)),
        ]

        if self.settings.log_synthesis_payloads:
            logger.info(
                "Structured synthesis request for %s\n%s",
                request.target_name,
                json.dumps(
                    {
                        "target": request.target_name,
                        "model": self.settings.memory_model,
                        "temperature": self.settings.memory_temperature,
                        "messages": [message.to_history_dict() for message in messages],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

        options = request.options or LLMCallOptions(response_format={"type": "json_object"})
        raw_content = self._complete_text(messages=messages, options=options)

        if self.settings.log_synthesis_payloads:
            logger.info(
                "Structured synthesis response for %s\n%s",
                request.target_name,
                raw_content,
            )

        candidate = _load_json_object(raw_content, target_name=request.target_name)
        if not isinstance(candidate, dict):
            raise ValueError(f"{request.target_name} synthesis returned a non-object payload")

        parsed = request.parser(candidate)
        if parsed is None:
            raise ValueError(f"{request.target_name} synthesis returned an invalid structured payload")
        return parsed

    def _complete_text(self, *, messages: list[LLMMessage], options: LLMCallOptions | None) -> str:
        if options is not None and self._provider_accepts_options("complete_text"):
            return completion_content(self.provider.complete_text(
                messages=messages,
                model=self.settings.memory_model,
                temperature=self.settings.memory_temperature,
                options=options,
            ))
        return completion_content(self.provider.complete_text(
            messages=messages,
            model=self.settings.memory_model,
            temperature=self.settings.memory_temperature,
        ))

    def _provider_accepts_options(self, method_name: str) -> bool:
        method = getattr(self.provider, method_name)
        try:
            parameters = signature(method).parameters.values()
        except (TypeError, ValueError):
            return True
        return any(
            parameter.kind == Parameter.VAR_KEYWORD or parameter.name == "options"
            for parameter in parameters
        )

    def _build_system_prompt(self, *, instructions: str, output_format: dict[str, Any]) -> str:
        return "\n\n".join(
            [
                instructions.strip(),
                "Output format:",
                json.dumps(output_format, ensure_ascii=False, indent=2),
            ]
        )


def _load_json_object(raw_content: str, *, target_name: str) -> object:
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as exc:
        original_exc = exc

    for candidate in _json_recovery_candidates(raw_content):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            stripped = candidate.lstrip()
            if not stripped.startswith("{"):
                continue
            try:
                payload, end_index = json.JSONDecoder().raw_decode(stripped)
            except json.JSONDecodeError:
                continue
            trailing = stripped[end_index:].strip()
            if trailing:
                logger.warning(
                    "Structured synthesis response for %s had trailing content after JSON; using first object",
                    target_name,
                    extra={"trailing_preview": safe_preview(trailing, limit=300)},
                )
            return payload

    raise ValueError(f"{target_name} synthesis returned invalid JSON") from original_exc


def _json_recovery_candidates(raw_content: str) -> list[str]:
    stripped = raw_content.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)

    fenced = _extract_markdown_json_fence(stripped)
    if fenced is not None:
        candidates.append(fenced)

    first_brace = stripped.find("{")
    if first_brace > 0:
        candidates.append(stripped[first_brace:])
    return candidates


def _extract_markdown_json_fence(value: str) -> str | None:
    if not value.startswith("```"):
        return None
    first_newline = value.find("\n")
    if first_newline < 0:
        return None
    closing_fence = value.rfind("```")
    if closing_fence <= first_newline:
        return None
    return value[first_newline + 1 : closing_fence].strip()
