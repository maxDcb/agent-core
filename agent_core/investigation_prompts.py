from __future__ import annotations

from dataclasses import dataclass, replace


INITIAL_PLAN_PROMPT = """You are preparing a generic bounded investigation plan. Do not output chain-of-thought. Return JSON only. Store only concise auditable artifacts: plan steps, evidence gaps, next actions, risk/scope notes, and confidence. Do not add facts that are not supported by the provided payload."""

STEP_REFLECTION_PROMPT = """You are updating a generic investigation state after one assistant/tool step. Do not output chain-of-thought. Return JSON only. Return concise auditable artifacts: confirmed facts, supported or rejected hypotheses, open evidence gaps, resolved evidence gaps, recommended next actions, risk/scope notes, confidence, and stop reason. Do not add facts not supported by the provided state or tool results. Do not repeat previous evidence gaps that are now resolved by the latest tool results; list them in resolved_gaps instead. Do not treat output-format or response-style instructions as evidence gaps."""

INVESTIGATION_DECISION_PROMPT = """You are choosing the next generic investigation action from structured state. Do not output chain-of-thought. Return JSON only. Choose one of: continue, final, ask_user, blocked. Use only auditable state fields and concise reason summaries. If the latest reflection has should_continue=true, choose continue unless the investigation is blocked, budget is exhausted, or user input is required. Do not choose final while the latest reflection lists remaining gaps or recommended next actions that are required to complete the objective."""

FINAL_CRITIQUE_PROMPT = """You are critiquing a final draft against a generic investigation state. Do not output chain-of-thought. Return JSON only. Identify unsupported claims, missing evidence, scope or safety issues, and follow-up actions. Do not invent new facts."""

RUN_GUIDANCE_PROMPT = (
    "Run mode: {mode}. Work within the bounded investigation loop. "
    "Use tools only when useful and in scope. Do not expose chain-of-thought; "
    "final responses should summarize auditable findings and uncertainty."
)


@dataclass(frozen=True, slots=True)
class InvestigationPromptSet:
    """Prompt bundle used by the bounded investigation loop."""

    initial_plan: str
    step_reflection: str
    decision: str
    final_critique: str
    run_guidance: str

    def append_domain_guidance(self, guidance: str) -> "InvestigationPromptSet":
        appendix = guidance.strip()
        if not appendix:
            return self
        return replace(
            self,
            initial_plan=_append_guidance(self.initial_plan, appendix),
            step_reflection=_append_guidance(self.step_reflection, appendix),
            decision=_append_guidance(self.decision, appendix),
            final_critique=_append_guidance(self.final_critique, appendix),
            run_guidance=_append_guidance(self.run_guidance, appendix),
        )

    def render_run_guidance(self, *, mode: str) -> str:
        return self.run_guidance.replace("{mode}", mode)


def _append_guidance(prompt: str, guidance: str) -> str:
    return "\n\n".join([prompt.strip(), guidance])


DEFAULT_INVESTIGATION_PROMPTS = InvestigationPromptSet(
    initial_plan=INITIAL_PLAN_PROMPT,
    step_reflection=STEP_REFLECTION_PROMPT,
    decision=INVESTIGATION_DECISION_PROMPT,
    final_critique=FINAL_CRITIQUE_PROMPT,
    run_guidance=RUN_GUIDANCE_PROMPT,
)
