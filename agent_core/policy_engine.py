from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from agent_core.execution_context import ExecutionContext
from agent_core.logging_utils import get_logger
from agent_core.types import AuthorizationResult

logger = get_logger(__name__)

PolicyValidator = Callable[[dict, ExecutionContext], AuthorizationResult]


class PolicyEngine:
    def __init__(self, *, validators: dict[str, PolicyValidator] | None = None) -> None:
        self._validators: dict[str, PolicyValidator] = {
            "list_directory": self._validate_filesystem_tool,
            "read_file_chunk": self._validate_filesystem_tool,
            "search_code": self._validate_filesystem_tool,
            "tree_directory": self._validate_filesystem_tool,
            "search_knowledge": self._validate_knowledge_read_tool,
            "read_knowledge_chunk": self._validate_knowledge_read_tool,
            "http_request": self._validate_http_tool,
        }
        if validators:
            self.register_validators(validators)

    def register_validator(self, tool_name: str, validator: PolicyValidator) -> None:
        self._validators[tool_name] = validator

    def register_validators(self, validators: dict[str, PolicyValidator]) -> None:
        self._validators.update(validators)

    def authorize(self, tool_name: str, arguments: dict, context: ExecutionContext) -> AuthorizationResult:
        logger.trace("Evaluating policy", extra={"tool_name": tool_name, "argument_keys": sorted(arguments.keys())})
        validator = self._validators.get(tool_name)
        if validator is None:
            logger.trace("No dedicated validator registered for tool; allowing execution", extra={"tool_name": tool_name})
            return AuthorizationResult(True, "allowed")

        result = validator(arguments, context)
        if result.allowed:
            logger.trace("Policy allowed tool execution", extra={"tool_name": tool_name})
        return result

    def _validate_filesystem_tool(self, arguments: dict, context: ExecutionContext) -> AuthorizationResult:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            logger.info("Policy denied filesystem access because path is missing", extra={"path": raw_path})
            return AuthorizationResult(False, "Missing path")

        candidate = Path(raw_path).resolve()
        if not context.is_path_allowed(candidate):
            logger.info("Policy denied filesystem access", extra={"path": str(candidate)})
            return AuthorizationResult(False, f"Path not allowed: {candidate}")

        return AuthorizationResult(True, "allowed")

    def _validate_http_tool(self, arguments: dict, context: ExecutionContext) -> AuthorizationResult:
        raw_method = arguments.get("method")
        if not isinstance(raw_method, str) or not raw_method.strip():
            logger.info("Policy denied HTTP request with missing method")
            return AuthorizationResult(False, "Missing HTTP method")

        method = raw_method.upper()
        if method not in context.settings.allowed_http_methods:
            logger.info("Policy denied HTTP method", extra={"method": method})
            return AuthorizationResult(False, f"HTTP method not allowed: {method}")

        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            logger.info("Policy denied HTTP request with missing URL")
            return AuthorizationResult(False, "Missing URL")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            logger.info("Policy denied HTTP request with invalid scheme", extra={"url": url, "scheme": parsed.scheme})
            return AuthorizationResult(False, "Invalid URL scheme")

        host = parsed.hostname
        if not host:
            logger.info("Policy denied HTTP request with invalid URL", extra={"url": url})
            return AuthorizationResult(False, "Invalid URL")

        if host not in context.settings.allowed_http_hosts:
            logger.info("Policy denied HTTP host", extra={"host": host})
            return AuthorizationResult(False, f"Host not allowed: {host}")

        body_keys = [
            key
            for key in ("json_body", "form_body", "raw_body", "body")
            if arguments.get(key) is not None
        ]
        if len(body_keys) > 1:
            logger.info("Policy denied HTTP request with multiple body types", extra={"body_keys": body_keys})
            return AuthorizationResult(False, "Provide only one HTTP body type")

        proxy_url = arguments.get("proxy_url")
        if proxy_url is not None:
            if not isinstance(proxy_url, str) or not proxy_url.strip():
                logger.info("Policy denied HTTP request with invalid proxy URL", extra={"proxy_url": proxy_url})
                return AuthorizationResult(False, "Invalid proxy URL")

            parsed_proxy = urlparse(proxy_url)
            if parsed_proxy.scheme not in {"http", "https"} or not parsed_proxy.hostname:
                logger.info(
                    "Policy denied HTTP request with invalid proxy URL scheme",
                    extra={"proxy_url": proxy_url, "scheme": parsed_proxy.scheme},
                )
                return AuthorizationResult(False, "Invalid proxy URL")

        return AuthorizationResult(True, "allowed")

    def _validate_knowledge_read_tool(self, arguments: dict, context: ExecutionContext) -> AuthorizationResult:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            logger.info("Policy denied knowledge access because path is missing", extra={"path": raw_path})
            return AuthorizationResult(False, "Missing path")

        candidate = Path(raw_path).resolve()
        knowledge_root = context.settings.knowledge_base_dir.resolve()
        try:
            candidate.relative_to(knowledge_root)
        except ValueError:
            logger.info(
                "Policy denied knowledge access outside knowledge base",
                extra={"path": str(candidate), "knowledge_root": str(knowledge_root)},
            )
            return AuthorizationResult(False, f"Knowledge path not allowed: {candidate}")

        return AuthorizationResult(True, "allowed")

