from __future__ import annotations

from dataclasses import dataclass

from agent_core.llm.azure_anthropic_provider import AzureAnthropicProvider
from agent_core.llm.azure_openai_provider import AzureOpenAIProvider
from agent_core.llm.base import BaseLLMProvider
from agent_core.llm.langchain_azure_openai_provider import LangChainAzureOpenAIProvider
from agent_core.llm.openai_provider import OpenAIProvider
from agent_core.settings import CoreSettings


@dataclass(frozen=True, slots=True)
class LLMProviderConfig:
    provider: str
    openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str | None = None
    azure_anthropic_endpoint: str | None = None
    azure_anthropic_api_key: str | None = None
    azure_anthropic_api_version: str | None = None
    azure_anthropic_version: str | None = None
    timeout_seconds: float = 120.0
    model_backend: str = "native"


_MEMORY_PROVIDER_FIELDS = (
    "memory_llm_provider",
    "memory_llm_model_backend",
    "memory_openai_api_key",
    "memory_azure_openai_endpoint",
    "memory_azure_openai_api_key",
    "memory_azure_openai_api_version",
    "memory_azure_anthropic_endpoint",
    "memory_azure_anthropic_api_key",
    "memory_azure_anthropic_api_version",
    "memory_azure_anthropic_version",
)


def normalize_provider_name(provider_name: str | None) -> str:
    return (provider_name or "openai").strip().lower().replace("-", "_")


def normalize_model_backend(model_backend: str | None) -> str:
    return (model_backend or "native").strip().lower().replace("-", "_")


def build_provider(settings: CoreSettings) -> BaseLLMProvider:
    return build_provider_from_config(_primary_provider_config(settings))


def build_memory_provider(settings: CoreSettings) -> BaseLLMProvider | None:
    config = _memory_provider_config(settings)
    if config is None:
        return None
    return build_provider_from_config(config)


def build_provider_from_config(config: LLMProviderConfig) -> BaseLLMProvider:
    provider_name = normalize_provider_name(config.provider)
    model_backend = normalize_model_backend(config.model_backend)

    if model_backend not in {"native", "langchain"}:
        raise ValueError(
            f"Unsupported LLM model backend: {config.model_backend}. "
            "Supported values are native and langchain."
        )

    if model_backend == "langchain":
        if provider_name != "azure_openai":
            raise ValueError(
                "The LangChain model backend currently supports only provider=azure_openai. "
                f"Received provider={config.provider}."
            )
        return LangChainAzureOpenAIProvider(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            timeout_seconds=config.timeout_seconds,
        )

    if provider_name == "openai":
        return OpenAIProvider(
            api_key=config.openai_api_key,
            timeout_seconds=config.timeout_seconds,
        )

    if provider_name == "azure_openai":
        return AzureOpenAIProvider(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            timeout_seconds=config.timeout_seconds,
        )

    if provider_name == "azure_anthropic":
        return AzureAnthropicProvider(
            endpoint=config.azure_anthropic_endpoint,
            api_key=config.azure_anthropic_api_key,
            api_version=config.azure_anthropic_api_version,
            anthropic_version=config.azure_anthropic_version,
            timeout_seconds=config.timeout_seconds,
        )

    raise ValueError(
        f"Unsupported LLM provider: {config.provider}. "
        "Supported values are openai, azure_openai, and azure_anthropic."
    )


def _primary_provider_config(settings: CoreSettings) -> LLMProviderConfig:
    return LLMProviderConfig(
        provider=settings.llm_provider,
        model_backend=settings.llm_model_backend,
        openai_api_key=settings.openai_api_key,
        azure_openai_endpoint=settings.azure_openai_endpoint,
        azure_openai_api_key=settings.azure_openai_api_key,
        azure_openai_api_version=settings.azure_openai_api_version,
        azure_anthropic_endpoint=settings.azure_anthropic_endpoint,
        azure_anthropic_api_key=settings.azure_anthropic_api_key,
        azure_anthropic_api_version=settings.azure_anthropic_api_version,
        azure_anthropic_version=settings.azure_anthropic_version,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def _memory_provider_config(settings: CoreSettings) -> LLMProviderConfig | None:
    if not _has_memory_provider_overrides(settings):
        return None

    provider = _prefer_override(settings.memory_llm_provider, settings.llm_provider) or "openai"
    model_backend = _prefer_override(settings.memory_llm_model_backend, settings.llm_model_backend) or "native"
    return LLMProviderConfig(
        provider=provider,
        model_backend=model_backend,
        openai_api_key=_prefer_override(settings.memory_openai_api_key, settings.openai_api_key),
        azure_openai_endpoint=_prefer_override(settings.memory_azure_openai_endpoint, settings.azure_openai_endpoint),
        azure_openai_api_key=_prefer_override(settings.memory_azure_openai_api_key, settings.azure_openai_api_key),
        azure_openai_api_version=_prefer_override(settings.memory_azure_openai_api_version, settings.azure_openai_api_version),
        azure_anthropic_endpoint=_prefer_override(settings.memory_azure_anthropic_endpoint, settings.azure_anthropic_endpoint),
        azure_anthropic_api_key=_prefer_override(settings.memory_azure_anthropic_api_key, settings.azure_anthropic_api_key),
        azure_anthropic_api_version=_prefer_override(settings.memory_azure_anthropic_api_version, settings.azure_anthropic_api_version),
        azure_anthropic_version=_prefer_override(settings.memory_azure_anthropic_version, settings.azure_anthropic_version),
        timeout_seconds=settings.llm_timeout_seconds,
    )


def _has_memory_provider_overrides(settings: CoreSettings) -> bool:
    return any(_has_value(getattr(settings, field_name, None)) for field_name in _MEMORY_PROVIDER_FIELDS)


def _has_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _prefer_override(override: str | None, fallback: str | None) -> str | None:
    if isinstance(override, str) and not override.strip():
        return fallback
    return override if override is not None else fallback
