"""LLM prompt templates for CausalAudit narrative generation.

All prompts must:
1. Return only JSON — no markdown code fences, no prose outside the JSON.
2. Be robust to different LLM output styles (the caller strips/validates).
3. Provide enough context that the model understands the causal domain.
"""

from __future__ import annotations

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert ML reliability engineer specialising in causal inference "
    "and production model monitoring.  You analyse structural changes in causal "
    "graphs derived from deployed ML model inference logs.  Your explanations are "
    "precise, technically accurate, and actionable.  You always return valid JSON "
    "with no markdown, no code fences, and no text outside the JSON object."
)

# ── Drift explanation prompt ──────────────────────────────────────────────────

DRIFT_EXPLANATION_PROMPT = """\
You are an ML reliability expert. A causal drift analysis has found the following \
structural changes in a deployed ML model's feature relationships:

{breaks_description}

The model predicts: {target}
Baseline period: {baseline_period}
Current period: {current_period}

Explain in plain English what these causal breaks mean for model reliability, \
and suggest specific investigation steps. Return JSON:
{{"explanation": "...", "severity": "low|medium|high|critical", \
"recommended_actions": ["...", "..."]}}"""

# ── Root-cause analysis prompt ────────────────────────────────────────────────

ROOT_CAUSE_PROMPT = """\
You are an expert in ML model degradation and causal feature analysis.

A production ML model has experienced the following causal structural changes:

Degradation score: {degradation_score:.2f} / 1.00
Primary break: {primary_break_type} on edge '{primary_cause}' → '{primary_effect}'
  Baseline strength: {baseline_strength}
  Current strength: {current_strength}
  Explanation: {break_explanation}

All breaks detected:
{all_breaks_summary}

Model target variable: {target}

Based on these causal breaks, identify:
1. The most likely root cause of model degradation
2. Whether this is a data pipeline issue, population shift, or model staleness
3. The urgency level

Return only valid JSON (no markdown, no prose outside the JSON):
{{
  "root_cause_category": "data_pipeline|population_shift|model_staleness|confounding|unknown",
  "root_cause_explanation": "...",
  "urgency": "immediate|high|medium|low",
  "investigation_steps": ["step1", "step2", "step3"],
  "estimated_impact": "..."
}}"""

# ── Recommendations prompt ────────────────────────────────────────────────────

RECOMMENDATIONS_PROMPT = """\
You are an ML engineer advising a team whose production model has experienced causal drift.

Drift report summary:
- Degradation score: {degradation_score:.2f}
- Number of breaks: {n_breaks}
- Break types: {break_types_summary}
- Root cause: {root_cause}
- Target variable: {target}

Affected features (by severity):
{affected_features}

Generate a prioritised list of concrete, actionable recommendations for the team.
Consider: data pipeline fixes, feature engineering updates, retraining strategy,
monitoring improvements, and stakeholder communication.

Return only valid JSON:
{{
  "recommendations": [
    {{"priority": 1, "action": "...", "rationale": "...", "effort": "low|medium|high"}},
    ...
  ],
  "immediate_actions": ["..."],
  "long_term_actions": ["..."]
}}"""
