# Changelog

## Unreleased

- Added optional run-level `LLMBudget` enforcement and observation across
  conversation, investigation, structured-task, finalization and memory calls.
- Persisted LLM budget usage in pending conversation state and structured-task
  checkpoint schema version 3 so resumed runs retain prior consumption.
- Added budget metadata with accounted and provider-reported token usage,
  elapsed time, violations and the exhausted dimension.
- Added an optional provider-window context planner covering messages, tools,
  response schemas, output reservation and tokenizer safety margin.
- Preserved system prompts and complete current tool turns while compacting
  only atomic historical groups, with observe/enforce modes and overflow errors.
- Persisted context-planning telemetry in pending turns and structured-task
  checkpoint schema version 4 so resumed runs retain planner state.

## 0.4.0

- Added exact provider token usage, per-call LLM telemetry, retry counts, cache
  details, checkpoint persistence, run-level summaries, and conversation capture.
- Added lossless structured-run checkpoints, explicit interrupted-run resume,
  run-attempt audit history, local execution locks, and fail-closed ambiguous
  tool reconciliation.
- Split and reduced the supported API into explicit run-engine, extension SPI,
  optional conversation, and observability facades.
- Added centralized local JSON Schema Draft 2020-12 validation for structured
  task and investigation outputs, including format checks and schema validation
  at contract construction.
- Added structured, payload-safe validation diagnostics and explicit failure for
  provider outputs that violate their declared contract.
- Added separate provider configuration for the main agent and memory synthesis.
- Made investigation answers conversational by default, with explicit strict or
  recoverable handling for malformed internal synthesis.
- Hardened multi-tool execution and pending-result resume behavior.

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
