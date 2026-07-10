from __future__ import annotations

from agent_core.llm.azure_openai_provider import AzureOpenAIProvider
from agent_core.llm.openai_provider import OpenAIProvider
from agent_core.llm.provider_factory import build_memory_provider, build_provider
from agent_core.settings import CoreSettings


def test_build_memory_provider_returns_none_without_memory_overrides() -> None:
    settings = CoreSettings(llm_provider="openai", openai_api_key="primary-key")

    assert build_memory_provider(settings) is None


def test_build_memory_provider_uses_memory_openai_api_key() -> None:
    settings = CoreSettings(
        llm_provider="openai",
        openai_api_key="primary-key",
        memory_llm_provider="openai",
        memory_openai_api_key="memory-key",
    )

    primary_provider = build_provider(settings)
    memory_provider = build_memory_provider(settings)

    assert isinstance(primary_provider, OpenAIProvider)
    assert isinstance(memory_provider, OpenAIProvider)
    assert primary_provider.api_key == "primary-key"
    assert memory_provider.api_key == "memory-key"


def test_build_memory_provider_can_override_azure_openai_endpoint_and_inherit_key() -> None:
    settings = CoreSettings(
        llm_provider="azure_openai",
        azure_openai_endpoint="https://primary.openai.azure.com",
        azure_openai_api_key="shared-key",
        azure_openai_api_version="2025-01-01-preview",
        memory_azure_openai_endpoint="https://memory.openai.azure.com",
    )

    memory_provider = build_memory_provider(settings)

    assert isinstance(memory_provider, AzureOpenAIProvider)
    assert memory_provider.azure_endpoint == "https://memory.openai.azure.com"
    assert memory_provider.api_key == "shared-key"
    assert memory_provider.api_version == "2025-01-01-preview"
