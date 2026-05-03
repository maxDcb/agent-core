from __future__ import annotations

import logging
import os
from typing import Any

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")
_STANDARD_LOG_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__.keys())


class AgentCoreLogger(logging.Logger):
    # TRACE is used for high-frequency state transitions that would be too noisy at DEBUG.
    def trace(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(TRACE_LEVEL):
            self._log(TRACE_LEVEL, msg, args, **kwargs)


logging.setLoggerClass(AgentCoreLogger)


class ExtraAwareFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        for key in sorted(record.__dict__.keys()):
            if key in _STANDARD_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            extras.append(f"{key}={safe_preview(record.__dict__[key], limit=80)}")

        if not extras:
            return base
        return f"{base} | {' '.join(extras)}"


def _coerce_level(raw_level: str | None, *, debug: bool) -> int:
    if raw_level:
        normalized = raw_level.strip().upper()
        if normalized == "TRACE":
            return TRACE_LEVEL
        return getattr(logging, normalized, logging.INFO)
    return logging.DEBUG if debug else logging.INFO


def configure_logging(*, debug: bool, level_name: str | None = None) -> None:
    level = _coerce_level(level_name or os.getenv("AGENT_CORE_LOG_LEVEL"), debug=debug)
    handler = logging.StreamHandler()
    handler.setFormatter(ExtraAwareFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # Third-party SDK transport logs are too noisy for normal interactive use. Keep the assistant's
    # own logs visible while suppressing low-signal HTTP request chatter from provider libraries.
    for logger_name in ("httpx", "httpcore", "openai"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    get_logger(__name__).info("Logging configured", extra={"configured_level": logging.getLevelName(level)})


def get_logger(name: str) -> AgentCoreLogger:
    return logging.getLogger(name)  # type: ignore[return-value]


def safe_preview(value: Any, limit: int = 120) -> str:
    # Log previews stay single-line so tool output and provider details do not flood the console.
    text = str(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."
