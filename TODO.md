# Architecture TODO

## Context capacity and large artifacts

- Keep the current accuracy-first behavior: do not semantically compact or discard active-run evidence solely because a configurable context target was exceeded.
- Keep normal conversation compaction after the run completes.
- In a future change, distinguish the provider's hard context capacity from the configurable active-history target and expose accurate context-usage telemetry.
- If the provider's hard capacity cannot be respected, fail explicitly instead of silently truncating evidence or reporting a complete investigation.
- Consider lossless artifact references and evidence promotion as optional extension points. Storage, pagination, search, and externalization of large tool outputs remain the responsibility of the application/tool project using `agent-core`.
- Preserve the invariant that data removed from an active prompt is never implicitly treated as reviewed evidence.

## Security policy and sensitive persistence

- Add explicit `PolicyEngine` modes:
  - `permissive`: preserve the current compatibility behavior and allow tools without a registered policy;
  - `audit`: allow tools without a registered policy but emit structured warnings and trace events;
  - `strict`: deny tools without an explicit validator or explicit allow policy.
- Keep `permissive` as the `agent-core` compatibility default until a future major version. Let security-sensitive applications such as PentestAssistant opt into `audit`, then `strict`.
- Add startup registry validation, for example `policy_engine.validate_registry(registry)`, so strict-mode configuration errors list every unclassified tool before a run starts.
- Support an explicit allow policy for low-risk tools. Fail-closed should mean that every tool received a deliberate policy decision, not that every tool requires a complex validator.
- Suggested rollout for PentestAssistant:
  1. enable audit mode;
  2. inventory every unclassified tool;
  3. add validators or explicit allow policies in the application/tool project;
  4. enable strict mode once the inventory is complete.
- Keep responsibility for tool-specific security rules in the application or tool project. `agent-core` should provide enforcement, validation, audit events, and safe defaults without pretending to understand domain-specific tool semantics.
- Treat LLM visibility and persistence visibility as separate concerns: JWTs, cookies, credentials, and other sensitive values may be required in the live model context and for exact pending-run resume.
- Do not add irreversible redaction to the live LLM context or required session state without a complete resume design.
- Consider a separate optional trace policy such as `full` versus `metadata_only`; audit traces and operational logs can have stricter disclosure rules than resumable session state.
- Before adding disk protection, design encryption at rest, key ownership, file permissions, retention, trace policy, resume behavior, and application-owned stores as one coherent feature. Prefer encryption over destructive redaction when exact values must remain reloadable.

## Multi-worker session storage

- Keep `JsonFileSessionStore` as the simple default for local and single-worker deployments.
- Document that atomic file replacement prevents partial/corrupt writes but does not prevent lost updates when separate processes modify the same session concurrently.
- The current per-session locks are process-local and therefore do not coordinate Uvicorn/Gunicorn workers, containers, machines, or asynchronous workers resuming the same pending run.
- Add an optional transactional `SessionStore` implementation when multi-worker deployment becomes necessary:
  - SQLite for local multi-process deployments;
  - PostgreSQL or an application-provided transactional store for distributed deployments.
- Require optimistic versioning or transactional row locking so a stale worker cannot silently overwrite a newer session state.
- Make pending-result consumption idempotent and atomic across workers so the same external result cannot resume a run twice.
- Treat a conflict as an explicit reload/retry condition, never as last-write-wins.
- This work becomes necessary when the same session may be handled by multiple API workers, background jobs, containers, or machines. It is not required for the current single-process usage model.

## Provider and model capabilities

- Add an optional capability API that is evaluated per model or deployment rather than globally per provider.
- Introduce a compact `ProviderCapabilities` model covering at least:
  - tool calling;
  - parallel tool calls;
  - JSON-object output;
  - provider-enforced strict JSON Schema output;
  - reasoning effort and reasoning summary options;
  - maximum-output-token support;
  - known context-window capacity when reliable.
- Represent capability state as supported, unsupported, or unknown. Unknown capabilities must preserve the current compatibility behavior for third-party providers and custom Azure deployments.
- Add an optional capability-aware provider protocol without immediately breaking existing `BaseLLMProvider` implementations.
- Derive explicit provider requirements from each run or structured task and perform a preflight before expensive investigation/tool execution begins.
- Fail early with a structured compatibility error when a required capability is known to be unsupported. Allow a guarded attempt when support is unknown.
- Support provider-specific schema compatibility checks because a provider may support JSON Schema while rejecting particular keywords or object shapes.
- Reuse and centralize the existing OpenAI/Azure adaptive capability learning where possible, including cached parameter rejections for opaque deployment names.
- Expose declared and learned capabilities through the quickstart compatibility diagnostics and run traces.
- Add contract tests shared by all bundled providers for supported, unsupported, and unknown capability states.
