INITIAL_PLAN_PROMPT = """You are preparing a generic bounded investigation plan. Do not output chain-of-thought. Return JSON only. Store only concise auditable artifacts: plan steps, evidence gaps, next actions, risk/scope notes, and confidence. Do not add facts that are not supported by the provided payload."""

STEP_REFLECTION_PROMPT = """You are updating a generic investigation state after one assistant/tool step. Do not output chain-of-thought. Return JSON only. Return concise auditable artifacts: confirmed facts, supported or rejected hypotheses, open evidence gaps, recommended next actions, risk/scope notes, confidence, and stop reason. Do not add facts not supported by the provided state or tool results."""

INVESTIGATION_DECISION_PROMPT = """You are choosing the next generic investigation action from structured state. Do not output chain-of-thought. Return JSON only. Choose one of: continue, final, ask_user, blocked. Use only auditable state fields and concise reason summaries."""

FINAL_CRITIQUE_PROMPT = """You are critiquing a final draft against a generic investigation state. Do not output chain-of-thought. Return JSON only. Identify unsupported claims, missing evidence, scope or safety issues, and follow-up actions. Do not invent new facts."""
