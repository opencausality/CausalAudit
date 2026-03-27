<div align="center">

# CausalAudit

**Causal drift detection for deployed ML models.**

When your model degrades, find out *which causal relationships broke* — not just that distributions shifted.

[![CI](https://github.com/opencausality/causalaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/opencausality/causalaudit/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is CausalAudit?

CausalAudit monitors deployed ML models for **causal drift** — structural changes in the
cause-and-effect relationships between input features and model labels. When accuracy drops,
it tells you *which causal link broke*, not just *that something changed*.

```
Historical inference logs  ──► Baseline causal graph
                                         │
                                         ▼ compare
Current inference logs     ──► Current causal graph
                                         │
                                         ▼
                              Drift report: which causal
                              relationships changed and why
```

### Key Features

- 🔍 **Causal, not correlational** — detects structural breaks in feature relationships, not just distribution shifts
- 📊 **Break classification** — NEW_EDGE, REMOVED_EDGE, WEAKENED, STRENGTHENED, PROXY_COLLAPSE
- 🧠 **Root cause attribution** — identifies which causal break most explains model degradation
- 💬 **LLM explanations** — optional natural language narrative of what changed and why
- 🏠 **Local-first** — Ollama default, no API key required
- 🔌 **Multi-provider** — Ollama, OpenAI, Anthropic, Groq, Mistral, Together AI

---

## Why Not Just Use Data Drift Monitoring?

Existing tools (Evidently AI, WhyLabs, Arize) detect *statistical drift* — changes in feature
distributions or prediction distributions. They answer: *"Did the input data change?"*

**CausalAudit answers a harder question: *"Did the causal mechanism change?"***

| Scenario | Data Drift Tools | CausalAudit |
|----------|-----------------|-------------|
| Income distribution shifted (new demographic) | ✅ Detects it | ✅ Detects it |
| Income still predicts approval (mechanism intact) | ❌ False alarm | ✅ No alert — mechanism unchanged |
| Policy change: income no longer used for decisions | ❌ Misses it | ✅ Detects: income→approval edge removed |
| Proxy variable corrupted in data pipeline | ❌ Misses the cause | ✅ Detects: PROXY_COLLAPSE on corrupted variable |
| Demographic shift invalidated training assumption | ⚠️ Distribution alert only | ✅ Shows exact causal break + affected downstream features |

### Concrete Example

**Scenario**: Your loan approval model worked well for 2 years. Last month accuracy dropped 8 points.

**Standard drift tool output:**
> ⚠️ Distribution drift detected: `income` feature KL divergence = 0.42

That's a statistical observation. It doesn't tell you whether to retrain on more income data,
remove income from the model, or investigate a data pipeline issue.

**CausalAudit output:**
```
Causal drift detected — 2 structural breaks found

PROXY_COLLAPSE  income → loan_approved
  Baseline strength:  0.73 (p=0.001)
  Current strength:   0.04 (p=0.847)  ← no longer significant
  Significance:       0.97 (critical)
  Explanation:        The causal link from income to approval has collapsed.
                      income is no longer predictive of the outcome.
                      Likely cause: policy change or income data corruption.

WEAKENED  credit_score → loan_approved
  Baseline strength:  0.81 (p=0.000)
  Current strength:   0.51 (p=0.003)  ← weakened but still significant
  Significance:       0.58 (medium)

Root cause: income data corruption or policy change affecting income-based rules.
Recommendation: Investigate income data pipeline. If policy changed, retrain
                excluding income or with new policy-aligned training data.
```

The answer isn't *better statistics* — it's a **structural diagnosis** that tells you exactly
what to fix and why.

---

## When to Use CausalAudit vs. Standard Drift Monitoring

| Use case | Standard drift tools | CausalAudit |
|----------|---------------------|-------------|
| Scheduled data quality checks | ✅ Fast and lightweight | Overkill |
| "My model accuracy dropped — why?" | ⚠️ Tells you *what* moved | ✅ Tells you *why* it moved |
| Policy or business rule changes | ❌ Misses mechanism change | ✅ Detects structural breaks |
| Data pipeline bugs affecting features | ⚠️ May catch distribution shift | ✅ Identifies broken causal link |
| Regulatory audit of model behavior | ❌ Correlation reports only | ✅ Causal structure evidence |

---

## Installation

```bash
# Using pip
pip install causalaudit

# Using uv (recommended)
uv add causalaudit

# With optional LLM explanation support
pip install "causalaudit[llm]"
```

**Requirements**: Python 3.10+, no API key needed for local Ollama mode.

---

## Quick Start

### 1. Build a baseline causal model from historical logs

```bash
causalaudit baseline \
  --data historical_inference.csv \
  --target loan_approved \
  --output baseline.json
```

Output:
```
Building baseline causal model...
  Discovered 4 significant causal relationships:
    income → loan_approved         (strength: 0.73, p=0.001)
    credit_score → loan_approved   (strength: 0.81, p=0.000)
    employment_years → loan_approved (strength: 0.44, p=0.023)
    age → loan_approved            (strength: 0.18, p=0.041)

Baseline saved to baseline.json
```

### 2. Audit current inference logs against baseline

```bash
causalaudit audit \
  --baseline baseline.json \
  --current current_inference.csv \
  --target loan_approved \
  --output drift_report.json
```

Output:
```
CausalAudit — Causal Drift Report
══════════════════════════════════

Model: loan_approval_v2
Baseline: 1000 samples (Jan 2026)
Current:  847 samples (Mar 2026)

Drift detected: YES — 2 causal breaks found

────────────────────────────────────────
PROXY_COLLAPSE  income → loan_approved
  Baseline: 0.730 (significant, p=0.001)
  Current:  0.041 (not significant, p=0.847)
  Change:   -0.689  ← CRITICAL

WEAKENED  credit_score → loan_approved
  Baseline: 0.810 (significant, p=0.000)
  Current:  0.514 (significant, p=0.003)
  Change:   -0.296  ← MEDIUM

────────────────────────────────────────
Degradation score: 0.74 (HIGH)
Root cause: income feature lost causal signal — data pipeline or policy change

Recommendations:
  1. Investigate income data pipeline for corruption or schema change
  2. Check if business policy changed income weighting in approvals
  3. If policy changed: retrain model with updated feature importance
  4. Monitor credit_score relationship — weakening trend may continue
```

### 3. Get an LLM explanation

```bash
causalaudit explain --report drift_report.json
```

Produces a plain-English narrative of what changed, why it matters, and what to do about it.

---

## CLI Reference

```bash
# Build baseline causal model
causalaudit baseline --data historical.csv --target LABEL --output baseline.json

# Audit: compare current logs to baseline
causalaudit audit --baseline baseline.json --current current.csv --target LABEL
causalaudit audit --baseline baseline.json --current current.csv --target LABEL --explain
causalaudit audit --baseline baseline.json --current current.csv --target LABEL --show  # visualize
causalaudit audit --baseline baseline.json --current current.csv --target LABEL --output report.json

# Auto-split a single file (first 70% = baseline, last 30% = current)
causalaudit audit --data all_logs.csv --target LABEL --split 0.7

# Explain an existing report using LLM
causalaudit explain --report drift_report.json

# Check LLM provider connectivity
causalaudit providers

# Start REST API server
causalaudit serve --port 8000
```

---

## Architecture

```
Inference Logs (CSV)
       │
       ▼
┌─────────────────────┐
│    Data Loader      │  ← Validates schema, checks min sample size
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Causal Discovery    │  ← Fisher's Z test, partial correlations
│  (independence.py)  │    PC-style structure learning
└─────────────────────┘
       │
       ▼ (CausalGraph)
┌─────────────────────┐
│  Drift Detector     │  ← Compare baseline vs current graphs
│  (detector.py)      │    Finds structural breaks per edge
└─────────────────────┘
       │
       ▼ (list[CausalBreak])
┌─────────────────────┐
│  Drift Classifier   │  ← Scores degradation severity
│  (classifier.py)    │    Identifies root cause break
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Attribution Engine │  ← Generates ranked recommendations
└─────────────────────┘
       │
       ▼ (DriftReport)
┌─────────────────────┐
│  LLM Explainer      │  ← Optional: natural language narrative
│  (optional)         │    via Ollama / any LiteLLM provider
└─────────────────────┘
```

---

## Configuration

Set in `.env` or as environment variables:

```env
# LLM provider (for --explain flag)
CAUSALAUDIT_LLM_PROVIDER=ollama
CAUSALAUDIT_LLM_MODEL=ollama/llama3.1

# Anthropic
CAUSALAUDIT_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
CAUSALAUDIT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Causal discovery settings
CAUSALAUDIT_SIGNIFICANCE_LEVEL=0.05        # p-value threshold for edge inclusion
CAUSALAUDIT_CONFIDENCE_THRESHOLD=0.5       # minimum edge strength to include
CAUSALAUDIT_BASELINE_WINDOW=1000           # minimum rows for reliable baseline

# Logging
CAUSALAUDIT_LOG_LEVEL=INFO
```

---

## REST API

Start the API server: `causalaudit serve --port 8000`

```
POST /audit
  Body: {"baseline_csv": "...", "current_csv": "...", "target_column": "label"}
  Returns: DriftReport JSON

POST /baseline
  Body: {"data_csv": "...", "target_column": "label"}
  Returns: CausalGraph JSON

GET /health
  Returns: {"status": "ok", "version": "0.1.0"}
```

Swagger UI available at `http://localhost:8000/docs`

---

## Data Model

```python
@dataclass
class CausalBreak:
    break_type: str       # "PROXY_COLLAPSE", "REMOVED_EDGE", "WEAKENED", "STRENGTHENED", "NEW_EDGE"
    cause: str
    effect: str
    baseline_strength: float | None
    current_strength: float | None
    significance: float   # 0–1 severity score
    explanation: str

@dataclass
class DriftReport:
    baseline_graph: CausalGraph
    current_graph: CausalGraph
    breaks: list[CausalBreak]
    degradation_score: float  # 0–1
    root_cause: str
    recommendations: list[str]
    model_used: str
```

---

## Supported Input Format

CausalAudit accepts CSV inference logs with:
- One column per feature (numeric)
- One label column (the prediction target)
- Optional timestamp column

```csv
timestamp,age,income,credit_score,employment_years,loan_approved
2026-01-01,35,75000,720,8,1
2026-01-01,28,45000,640,3,0
...
```

---

## Philosophy

CausalAudit is built on the principle that **model monitoring should be mechanistic, not statistical**.

- 🏠 **Local-first**: Ollama default — no cloud API required
- 🔓 **Open source**: All causal logic is MIT licensed
- 🚫 **No telemetry**: All analysis runs locally
- 🧠 **Causal, not correlational**: Structural breaks, not distribution alerts

---

## Contributing

CausalAudit is free for personal, educational, and research use.
If you're building production ML monitoring on top of CausalAudit, consider:
- Using a cloud LLM and contributing model improvements upstream
- Opening issues or PRs at the GitHub repository

*"Know why your model breaks, not just when."*
