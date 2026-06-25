# Changelog

## 0.3.0

- Added provider-enforced JSON Schema contracts for structured task final outputs.
- Made structured output schema requests fail instead of silently downgrading to JSON-object mode when a provider cannot enforce the schema.
- Added Azure Anthropic provider support for Claude deployments on Azure Foundry.
- Expanded provider compatibility checks in the quickstart example.
- Added provider configuration examples for OpenAI, Azure OpenAI and Azure Anthropic.

## 0.2.0

- Added bounded investigation modes with auditable state and optional final critique.
- Added run trace persistence, prompt snapshots and trace summaries.
- Replaced specialist profiles with generic structured task execution.
- Added OpenAI/Azure request normalization and adaptive retry handling.
- Added quickstart and pending tool resume examples.
- Kept core prompts and extension points domain-agnostic.

## 0.1.0

- Initial standalone extraction of `agent_core`.
- Packaging metadata for the `agent-core` Python distribution.
- Basic tests for public API, tool registry and pending tool result resume flow.
