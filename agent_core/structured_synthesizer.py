from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from agent_core.logging_utils import get_logger
from agent_core.llm.base import BaseLLMProvider, LLMMessage
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

        raw_content = self.provider.complete_text(
            messages=messages,
            model=self.settings.memory_model,
            temperature=self.settings.memory_temperature,
        )

        if self.settings.log_synthesis_payloads:
            logger.info(
                "Structured synthesis response for %s\n%s",
                request.target_name,
                raw_content,
            )

        try:
            candidate = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{request.target_name} synthesis returned invalid JSON") from exc

        if not isinstance(candidate, dict):
            raise ValueError(f"{request.target_name} synthesis returned a non-object payload")

        parsed = request.parser(candidate)
        if parsed is None:
            raise ValueError(f"{request.target_name} synthesis returned an invalid structured payload")
        return parsed

    def _build_system_prompt(self, *, instructions: str, output_format: dict[str, Any]) -> str:
        return "\n\n".join(
            [
                instructions.strip(),
                "Output format:",
                json.dumps(output_format, ensure_ascii=False, indent=2),
            ]
        )
