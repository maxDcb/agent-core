from __future__ import annotations

from dataclasses import dataclass, replace

INITIAL_PLAN_PROMPT = """You are preparing a generic bounded investigation plan. Do not output chain-of-thought. Return JSON only. Store only concise auditable artifacts: plan steps, evidence gaps, next actions, risk/scope notes, and confidence. Do not add facts that are not supported by the provided payload."""

STEP_REFLECTION_PROMPT = """You are updating a generic investigation state after one assistant/tool step. Do not output chain-of-thought. Return JSON only. Return concise auditable artifacts: confirmed facts, supported or rejected hypotheses, open evidence gaps, resolved evidence gaps, recommended next actions, risk/scope notes, confidence, and stop reason. Do not add facts not supported by the provided state or tool results. Do not repeat previous evidence gaps that are now resolved by the latest tool results; list them in resolved_gaps instead. Do not treat output-format or response-style instructions as evidence gaps.

Before setting should_continue, perform this mandatory completion check against the complete user objective in current_state:
1. Classify every explicitly requested action as satisfied, open, or closed with an auditable reason.
2. Classify an action as satisfied only when its result is present. No result always means open, never satisfied or unnecessary.
3. Treat an explicitly named action as executable and in scope unless the payload proves it unavailable, denied, unsafe, or over budget.
4. Close an unexecuted action only when its own condition is false, an earlier result made it unnecessary, it is impossible or out of scope, it requires user input, or budget is exhausted.
5. If any action is open, include the next one in recommended_next_actions and set should_continue=true. Otherwise set should_continue=false so the bounded turn can finish.

For example, if the objective requests A, then B, then C only if B returns a preview, after receiving only A's result you must classify B as open and continue. C cannot yet be evaluated; its condition does not close B. A user limit such as "no other investigation" excludes unrequested work, not A, B, or C. Do not keep actions open merely to collect optional evidence beyond the user's objective."""

INVESTIGATION_DECISION_PROMPT = """You are choosing the next generic investigation action from structured state. Do not output chain-of-thought. Return JSON only. Choose one of: continue, final, ask_user, blocked. Use only auditable state fields and concise reason summaries. If the latest reflection has should_continue=true, choose continue unless the investigation is blocked, budget is exhausted, or user input is required. Do not choose final while the latest reflection lists remaining gaps or recommended next actions that are required to complete the objective.

Before choosing final, verify that every explicitly requested action is either satisfied by a present result or closed for an auditable reason. If a requested action has no result and is not explicitly closed because it is unavailable, denied, unsafe, unnecessary, conditional with a false condition, or over budget, choose continue. A missing prerequisite result is not a false condition. For an objective A, then B, then C only if B returns a preview, a state containing only A's result requires continue to B. A restriction against additional or unrelated investigation does not block actions explicitly listed in the objective. Conditional or superseded actions do not need to run, and optional extra investigation beyond the user's objective must not prevent completion."""

FINAL_CRITIQUE_PROMPT = """You are critiquing a final draft against a generic investigation state. Do not output chain-of-thought. Return JSON only. Identify unsupported claims, missing evidence, scope or safety issues, and follow-up actions. Do not invent new facts."""

FINAL_RESPONSE_PROMPT = """Write the user-visible final response for this investigation. Do not output chain-of-thought or describe the controller's internal process.

Use the original request, the candidate answer, and the auditable investigation state in the supplied payload. Answer directly in the user's language and follow the requested format and level of detail. Preserve material uncertainty, scope limits, blocked actions, and required follow-up without inventing facts. Treat the candidate answer as a draft, not as evidence.

Return natural conversational text or Markdown only. Do not return the controller state, a serialized JSON document, generic investigation headings, or raw internal fields. Do not request or call tools."""

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
    final_response: str
    run_guidance: str

    def append_domain_guidance(self, guidance: str) -> InvestigationPromptSet:
        appendix = guidance.strip()
        if not appendix:
            return self
        return replace(
            self,
            initial_plan=_append_guidance(self.initial_plan, appendix),
            step_reflection=_append_guidance(self.step_reflection, appendix),
            decision=_append_guidance(self.decision, appendix),
            final_critique=_append_guidance(self.final_critique, appendix),
            final_response=_append_guidance(self.final_response, appendix),
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
    final_response=FINAL_RESPONSE_PROMPT,
    run_guidance=RUN_GUIDANCE_PROMPT,
)
