from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProviderErrorKind = Literal[
    "configuration_error",
    "request_error",
    "response_error",
    "rate_limit_error",
    "unexpected_error",
]


@dataclass(slots=True)
class LLMProviderError(Exception):
    kind: ProviderErrorKind
    user_message: str
    detail: str = ""

    def __str__(self) -> str:
        return self.detail or self.user_message
