# Investigation Modes

`agent_core` supports three generic run modes through `RunOptions`.

- `direct`: the existing assistant/tool loop. This remains the default when no options are passed.
- `investigate`: a bounded loop that can plan, call tools, observe results, update structured state, and decide whether to continue, ask the user, stop, or answer.
- `deep_investigate`: a larger bounded investigation preset with final critique enabled by default.

Investigation state is domain-agnostic. It stores concise, auditable artifacts only: facts, hypotheses, evidence gaps, completed actions, next actions, risk notes, confidence, and stop reason. Raw chain-of-thought is not stored, exposed, or included in returned metadata.

Domain-specific behavior still belongs in `DomainHooks`, tools, or an external domain package. Core investigation prompts intentionally avoid domain-specific assumptions.

The final answer from `investigate` and `deep_investigate` is text by default.
Internal planning, reflection, decision and critique phases synthesize JSON
state, but this does not make JSON a public output mode. Use
`final_output_mode="json_schema"` with a `StructuredOutputContract` when a
caller needs a tool-backed investigation whose final phase is forced to one
provider-enforced JSON Schema object. The schema rendering happens in a final
no-tool model call and does not alter the configured reasoning options.

Set `RunOptions.recover_internal_synthesis_errors=True` for interactive
conversation surfaces that should continue when internal JSON state synthesis is
malformed. Leave it `False` for batch or pipeline flows where malformed
structured state should fail loudly.

```python
from agent_core import RunOptions

result = orchestrator.run_turn_result(
    "Investigate this issue using available tools.",
    options=RunOptions.investigate(),
)
```

Each completed investigation result includes compact metadata such as mode, iterations used, tool calls used, stop reason, and a compact investigation state summary.
