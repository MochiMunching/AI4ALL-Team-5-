# Security Risk Assessment for Agentic AI Models

Auditing an automated **agent security risk-scoring system** with a four-model
triangulation pipeline. Rather than build a risk score from scratch, this project
*interrogates an existing one*: what drives it, whether its logic is coherent, where
it has blind spots, and whether access decisions actually follow it.

> AI4ALL — Group 05A

---

## Motivation

As AI agents become more autonomous, they access tools, use resources, and make
decisions with limited human input — creating a new class of security risk. Many
systems assign each agent action a **risk score** and an **access decision**
(Allowed / Blocked / Needs Human Approval). This project treats that scoring system
as something to be *audited*, not trusted by default, and asks how well it holds up.

## Research question

> **Can we predict the security risk score of an AI agent's action based on its
> autonomy level, permissions, requested tools, and resource access patterns?
> And can we decide whether an agent should be granted access based on that score?**

**Short answer (from this run):**

- **Predicting the score:** *partially.* The four named factors alone explain
  R² ≈ 0.48; the full feature set reaches R² ≈ 0.94, but the extra power comes from
  threat-detection signals the question doesn't name (prompt injection, data
  exfiltration risk, previous failed attempts).
- **Deciding access from the score:** *only for clear-cut cases.* The score alone
  predicts the decision at 76% accuracy but **cannot** identify the
  `Needs_Human_Approval` tier. Adding three signals recovers it (accuracy → 90%,
  macro-F1 0.51 → 0.83). **The score can gate allow-vs-block, but routing to a human
  needs more than the score.**

## Dataset

- **Name:** Agent AI Security Risk Dataset
- **Source:** Kaggle — [algozee/agentic-ai-security-risk-dataset](https://www.kaggle.com/datasets/algozee/agentic-ai-security-risk-dataset)
- **Size:** 2,200 simulated agent requests × 15 columns
- **Key fields:** `agent_role`, `agent_autonomy_level`, `user_role`,
  `requested_action`, `tool_requested`, `resource_type`, `resource_sensitivity`,
  `permission_match`, `prompt_injection_detected`, `data_exfiltration_risk`,
  `previous_failed_attempts`, `action_risk_score` (target), `access_decision`

**Leakage note:** `human_approval_required` and `access_decision` are *downstream* of
the score, so they are excluded from the models that predict it.

## Method — model triangulation

No single model answers every question about a black-box score, so four models each
answer one, and the value comes from reading them together.

| # | Model | Question it answers |
|---|-------|---------------------|
| 1 | XGBoost + SHAP | What drives the risk score? |
| 2 | Shallow decision tree (depth 3) | Can the scoring logic be written as human-readable rules? |
| 3 | Isolation Forest | Are there anomalous requests the score under-rates (blind spots)? |
| 4 | Logistic regression | Do access decisions follow the score consistently? |

The notebook then adds a **direct-answer section** (Section 5) for the research
question and a **robustness section** (Section 6): a linear baseline + 5-fold
cross-validation, a secondary-leakage check, class-imbalance handling, a
score-by-decision distribution plot, a "score + a few features" experiment, and
over-blocking profiling.

## Notebook walkthrough

Each section answers a specific question; here is what every part does and *why* it
is there.

### Setup & imports
Loads the data/plotting stack and the four model families, and fixes a single
`RANDOM_STATE` so every split, model, and anomaly flag is reproducible. The load cell
pulls the CSV directly from Kaggle (via `/kaggle/input` or `kagglehub`) so the notebook
runs without any manual file handling.

### 0. Quick EDA
**Purpose:** sanity-check the data before modeling — the score's distribution, the
balance of access decisions, and how each numeric feature correlates with the score.
This is also where leakage becomes *visible*: a downstream field like
`human_approval_required` shows an unusually high correlation, which is why it is
excluded later.

### Preprocessing
**Purpose:** turn raw rows into a model-ready matrix. Categoricals are one-hot encoded
(one column per category, which keeps SHAP attributions readable), the leaky columns
are dropped from the score models, and 25% of the data is held out as a test set to
measure generalisation.

### 1. XGBoost + SHAP — *what drives the score?*
**Purpose:** get the most accurate possible model of the score, then explain it. XGBoost
captures interactions; SHAP converts it into per-feature contributions measured in
score points. The resulting driver ranking is treated as the **ground truth** for what
the score responds to, and every later section is compared against it.

### 2. Shallow decision tree — *is the logic legible?*
**Purpose:** fit a deliberately tiny (depth-3) tree so the scoring logic can be read as
plain if/else rules. A lower R² than XGBoost is expected and fine — this model is a
**legibility check**, not a performance contender. If its splits match SHAP's top
drivers, the scoring logic is simple and coherent; if not, the score relies on
interactions that plain rules can't capture.

### 3. Isolation Forest — *where are the blind spots?*
**Purpose:** find requests whose *features* are unusual, without ever showing the model
the score. **Blind spots** are the dangerous combination — structurally anomalous
requests that nonetheless received a *low* score. Those are candidate gaps in the
scoring policy and are profiled to see what they have in common.

### 4. Logistic regression — *do decisions follow the score?*
**Purpose:** check whether enforcement is consistent with the score. Unlike Sections
1–3, this model *intentionally* uses the score as an input, because its question is
precisely whether decisions track it. It also surfaces **contradictions** — high-score
requests that were Allowed, and low-score requests that were Blocked — as an audit list.

### 5. Direct answer to the research question
**Purpose:** combine the pieces into a pointed, two-part verdict.
- **Part A (predictability):** trains on *only* the four factors the research question
  names and reports how much of the score they explain.
- **Part B (actionability):** uses the score *alone* to predict the access decision and
  compares it against the full-feature model.

### 6. Robustness, baselines & deeper analysis
**Purpose:** stress-test the results before drawing conclusions.
- **6.1 Baseline + cross-validation** — a linear baseline and 5-fold CV so the headline
  R² comes with error bars, not one lucky split.
- **6.2 Secondary-leakage check** — confirms the strong "threat" signals are genuine
  inputs, not the score in disguise.
- **6.3 Class-imbalance handling** — re-runs Model 4 with balanced class weights so the
  rare classes aren't drowned out by the majority.
- **6.4 Score-by-decision distribution** — visualises *why* a single score threshold
  can't separate the decision tiers.
- **6.5 Part B revisited** — quantifies exactly what routing to a human requires beyond
  the score.
- **6.6 Over-blocking profile** — characterises the Blocked-despite-low-score cases.
- **6.7 SHAP dependence plots** — shows the *shape* of the effect for the top drivers,
  not just their importance.
- **6.8 Save metrics** — writes `results_metrics.json` for the report/poster.

### Findings & conclusions
**Purpose:** the written-up results — the explicit answer to the research question, the
key findings, limitations (tied to the project's known data biases), and actionable
recommendations. This is the section a reader should start from to understand *what the
analysis concluded*.

## Key findings

- **The scoring logic is simple / near-linear.** A plain linear model reaches
  R² ≈ 0.92 vs XGBoost's 0.94 (CV 0.936 ± 0.004), and the depth-3 tree splits on the
  same variables SHAP ranks highest. The system behaves like a transparent weighted
  sum, not an opaque black box.
- **Top drivers:** permission mismatch and resource sensitivity dominate, followed by
  specific actions (`read_record`, `search_records`) and the threat signals.
- **No dangerous leniency, but notable over-blocking.** 0 requests Allowed with a high
  score (≥ 70); 98 Blocked despite a low score (≤ 30) — overwhelmingly benign reads on
  the *least* sensitive resources (`public_knowledge_base`, `support_ticket`).
- **Few blind spots.** Only 4 anomalous requests received a low score.
- **No secondary leakage.** Threat signals correlate only moderately (0.22–0.55) and
  removing them drops R² by just ~0.06 — legitimate inputs, not the score relabelled.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── build_notebook.py                    # generator: builds the analysis notebook
├── agent_security_risk_analysis.ipynb   # the analysis (run top-to-bottom)
├── data/                                 # place the Kaggle CSV here (gitignored)
└── results/                              # results_metrics.json + exported figures
```

> `build_notebook.py` is the source of truth — it programmatically assembles the
> notebook cell by cell, so the analysis diffs cleanly in Git. Regenerate the notebook
> with `python build_notebook.py`.

## Setup

```bash
python -m pip install -r requirements.txt
```

**`requirements.txt`:**

```
pandas
numpy
matplotlib
scikit-learn
xgboost
shap
nbformat
```

## Running the analysis

### On Kaggle (recommended)
1. Open the notebook in a Kaggle kernel.
2. Click **Add Input** and add the *Agent AI Security Risk Dataset*, **or** enable
   **Internet** in Settings (the loader will fall back to `kagglehub`).
3. Run all cells top to bottom.

### Locally
1. Download the dataset from the Kaggle link above into `data/`.
2. The load cell auto-detects the CSV (via `/kaggle/input`, `kagglehub`, or a local
   path); point it at `data/` if needed.
3. Run the notebook, or regenerate it first with `python build_notebook.py`.

## Results snapshot

Cell 6.8 writes `results_metrics.json`:

```json
{
  "n_rows": 2200,
  "n_features": 67,
  "xgb_test_r2": 0.937,
  "xgb_test_mae": 5.45,
  "xgb_cv_r2_mean": 0.936,
  "xgb_cv_r2_std": 0.004,
  "named_factors_r2": 0.481,
  "model4_accuracy": 0.953,
  "score_only_accuracy": 0.758,
  "over_blocked_count": 98
}
```

## Limitations

- **Label subjectivity** — the score and decisions were assigned by a prior process;
  the models learn *that* definition of risk, not ground truth.
- **Rare high-risk actions are under-represented** — genuine agentic misalignment is
  scarce, so performance on the most dangerous cases is the least certain.
- **Simulated, single-source data** — results would need revalidation on real logs.

## Team

Ruhi Shah · Donovon Mott · Deeksha Vaidyanathan · Dim Zuun · Mannat Kaur · Sangam Subedi

## References

- Madkour, N., et al. (2025). *Agentic AI Risk-Management Standards Profile.* UC Berkeley Center for Long-Term Cybersecurity.
- National Institute of Standards and Technology. (2023). *AI Risk Management Framework (AI RMF 1.0).*
- Lynch, A., et al. (2025). *Agentic Misalignment: How LLMs Could Be Insider Threats.* Anthropic.
- Christodorescu, M., et al. (2026). *Agent security is a systems problem.* arXiv.
- Chhabra, A., et al. (2026). *Agentic AI security: Threats, defenses, evaluation, and open challenges.* IEEE Access.
- Evtimov, I., et al. (2025). *WASP: Benchmarking web agent security against prompt injection attacks.* NeurIPS.

## License

Add a license of your choice (e.g. MIT) before publishing.
