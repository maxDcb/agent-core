from __future__ import annotations

from agent_core.llm.azure_openai_provider import AzureOpenAIProvider
from agent_core.llm.langchain_azure_openai_provider import LangChainAzureOpenAIProvider
from agent_core.llm.openai_provider import OpenAIProvider
from agent_core.llm.provider_factory import (
    LLMProviderConfig,
    build_memory_provider,
    build_provider,
    build_provider_from_config,
)
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


def test_build_provider_selects_langchain_azure_openai_backend() -> None:
    settings = CoreSettings(
        llm_provider="azure_openai",
        llm_model_backend="langchain",
        azure_openai_endpoint="https://primary.openai.azure.com",
        azure_openai_api_key="shared-key",
    )

    provider = build_provider(settings)

    assert isinstance(provider, LangChainAzureOpenAIProvider)
    assert provider.azure_endpoint == "https://primary.openai.azure.com"
    assert provider.api_key == "shared-key"


def test_build_memory_provider_inherits_langchain_backend_from_primary() -> None:
    settings = CoreSettings(
        llm_provider="azure_openai",
        llm_model_backend="langchain",
        azure_openai_endpoint="https://primary.openai.azure.com",
        azure_openai_api_key="shared-key",
        memory_azure_openai_endpoint="https://memory.openai.azure.com",
    )

    provider = build_memory_provider(settings)

    assert isinstance(provider, LangChainAzureOpenAIProvider)
    assert provider.azure_endpoint == "https://memory.openai.azure.com"


def test_build_memory_provider_can_override_langchain_backend_with_native() -> None:
    settings = CoreSettings(
        llm_provider="azure_openai",
        llm_model_backend="langchain",
        azure_openai_endpoint="https://primary.openai.azure.com",
        azure_openai_api_key="shared-key",
        memory_llm_model_backend="native",
    )

    provider = build_memory_provider(settings)

    assert isinstance(provider, AzureOpenAIProvider)


def test_build_provider_rejects_langchain_backend_for_unsupported_provider() -> None:
    config = LLMProviderConfig(provider="openai", model_backend="langchain", openai_api_key="test-key")

    try:
        build_provider_from_config(config)
    except ValueError as exc:
        assert "currently supports only provider=azure_openai" in str(exc)
    else:
        raise AssertionError("Expected unsupported backend/provider combination to fail")


def test_build_provider_rejects_unknown_model_backend() -> None:
    config = LLMProviderConfig(provider="azure_openai", model_backend="unknown")

    try:
        build_provider_from_config(config)
    except ValueError as exc:
        assert "Unsupported LLM model backend" in str(exc)
    else:
        raise AssertionError("Expected unknown model backend to fail")
