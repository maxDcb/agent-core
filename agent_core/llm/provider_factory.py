from __future__ import annotations

from agent_core.llm.azure_anthropic_provider import AzureAnthropicProvider
from agent_core.llm.azure_openai_provider import AzureOpenAIProvider
from agent_core.llm.openai_provider import OpenAIProvider
from agent_core.settings import CoreSettings


def normalize_provider_name(provider_name: str | None) -> str:
    return (provider_name or "openai").strip().lower().replace("-", "_")


def build_provider(settings: CoreSettings):
    provider_name = normalize_provider_name(settings.llm_provider)

    if provider_name == "openai":
        return OpenAIProvider(api_key=settings.openai_api_key)

    if provider_name == "azure_openai":
        return AzureOpenAIProvider(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    if provider_name == "azure_anthropic":
        return AzureAnthropicProvider(
            endpoint=settings.azure_anthropic_endpoint,
            api_key=settings.azure_anthropic_api_key,
            api_version=settings.azure_anthropic_api_version,
        )

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider}. "
        "Supported values are openai, azure_openai, and azure_anthropic."
    )
