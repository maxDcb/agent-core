# Investigation Modes

`agent_core` supports three generic run modes through `RunOptions`.

- `direct`: the existing assistant/tool loop. This remains the default when no options are passed.
- `investigate`: a bounded loop that can plan, call tools, observe results, update structured state, and decide whether to continue, ask the user, stop, or answer.
- `deep_investigate`: a larger bounded investigation preset with final critique enabled by default.

Investigation state is domain-agnostic. It stores concise, auditable artifacts only: facts, hypotheses, evidence gaps, completed actions, next actions, risk notes, confidence, and stop reason. Raw chain-of-thought is not stored, exposed, or included in returned metadata.

Domain-specific behavior still belongs in `DomainHooks`, tools, or an external domain package. Core investigation prompts intentionally avoid domain-specific assumptions.

The final answer from `investigate` and `deep_investigate` is text by default.
Once the loop stops, a dedicated no-tool model call receives the original
request, candidate draft, completion status, stop reason, and auditable state.
It renders the conversational answer in the user's language without exposing
the controller format. A tool call, empty response, raw JSON document, or model
failure activates the bounded deterministic state renderer as a fail-safe.
Completed text runs expose `final_response_origin=model|fallback`; fallback
results also expose the exception class in `final_response_error_type`.

Internal planning, reflection, decision and critique phases synthesize JSON
state, but this does not make JSON a public output mode. Use
`final_output_mode="json_schema"` with a `StructuredOutputContract` when a
caller needs a tool-backed investigation whose final phase is forced to one
provider-enforced JSON Schema object. Rendering happens in a final no-tool model
call and does not alter the configured reasoning options. The parsed object is
then validated locally against the same Draft 2020-12 contract before it is
returned. Provider acceptance alone is not treated as proof that the output is
valid.

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
