"""Builds agent_security_risk_analysis.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Agent Security Risk Analysis — Model Triangulation Pipeline

Four models, each answering a distinct security question about `action_risk_score`:

| # | Model | Question |
|---|-------|----------|
| 1 | XGBoost + SHAP | What drives the risk score? |
| 2 | Shallow decision tree | Can the scoring logic be expressed as human-readable rules? |
| 3 | Isolation Forest | Are there anomalous requests the score under-rates (blind spots)? |
| 4 | Logistic regression | Do access decisions follow the risk score consistently? |

**Leakage note:** `human_approval_required` (corr ≈ 0.73) and `access_decision` are downstream of the score, so they are excluded from models predicting it.""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, classification_report
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

RANDOM_STATE = 42
df = pd.read_csv("agent_security_risk_scores.csv")
print(df.shape)
df.head()""")

md("## 0. Quick EDA")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["action_risk_score"], bins=30, edgecolor="k")
axes[0].set_title("action_risk_score distribution")
df["access_decision"].value_counts().plot.bar(ax=axes[1], title="access_decision")
plt.tight_layout(); plt.show()

corr = df.select_dtypes("number").corr()["action_risk_score"].drop("action_risk_score").sort_values()
corr.plot.barh(figsize=(7, 4), title="Correlation with action_risk_score")
plt.tight_layout(); plt.show()""")

md("""## Preprocessing

One-hot encode categoricals. Feature set for score models excludes the leaky columns.""")

code("""TARGET = "action_risk_score"
LEAKY = ["human_approval_required", "access_decision"]
cat_cols = ["agent_role", "user_role", "requested_action", "tool_requested", "resource_type"]

X = pd.get_dummies(df.drop(columns=[TARGET] + LEAKY), columns=cat_cols, drop_first=False)
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
print(f"{X.shape[1]} features, train={len(X_train)}, test={len(X_test)}")""")

md("""## 1. XGBoost + SHAP — what drives the score?""")

code("""xgb_model = xgb.XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
)
xgb_model.fit(X_train, y_train)
pred = xgb_model.predict(X_test)
print(f"R2  = {r2_score(y_test, pred):.3f}")
print(f"MAE = {mean_absolute_error(y_test, pred):.2f} points (score range 0-100)")""")

code("""explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, max_display=15)""")

code("""mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
top_drivers = mean_abs.sort_values(ascending=False).head(10)
print("Top 10 drivers (mean |SHAP|, in score points):")
print(top_drivers.round(2))""")

md("""## 2. Shallow decision tree — extract the scoring rules

A depth-3 tree fit to the same target. If its splits agree with SHAP's top drivers, the scoring logic is coherent.""")

code("""tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=RANDOM_STATE)
tree.fit(X_train, y_train)
print(f"Tree R2 (test) = {r2_score(y_test, tree.predict(X_test)):.3f}\\n")
print(export_text(tree, feature_names=list(X.columns), decimals=1))""")

md("""## 3. Isolation Forest — blind spots

Fit on features only (score excluded). Anomalous requests that nevertheless received a **low** risk score are potential gaps in the scoring system.""")

code("""iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=RANDOM_STATE)
anomaly = iso.fit_predict(X) == -1
df_a = df.assign(anomaly=anomaly)

print(f"Flagged {anomaly.sum()} anomalous requests ({anomaly.mean():.1%})")
print(f"Mean score  — anomalies: {df_a.loc[anomaly, TARGET].mean():.1f}  vs normal: {df_a.loc[~anomaly, TARGET].mean():.1f}")

blind_spots = df_a[anomaly & (df_a[TARGET] < 40)]
print(f"\\nPotential blind spots (anomalous but score < 40): {len(blind_spots)}")
blind_spots.sort_values(TARGET).head(10)""")

code("""# What do the blind spots have in common?
if len(blind_spots):
    for col in ["requested_action", "resource_type", "user_role", "access_decision"]:
        print(f"\\n{col}:")
        print(blind_spots[col].value_counts().head(5).to_string())""")

md("""## 4. Logistic regression — are decisions consistent with the score?

Multinomial logistic on `access_decision`. Then flag contradictions: high-score requests that were **Allowed** and low-score requests that were **Blocked**.""")

code("""feat_cols = [c for c in X.columns] + [TARGET]
X_dec = pd.get_dummies(df.drop(columns=["access_decision", "human_approval_required"]), columns=cat_cols)
y_dec = df["access_decision"]
scaler = StandardScaler()
X_dec_s = scaler.fit_transform(X_dec)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(X_dec_s, y_dec, test_size=0.25,
                                              random_state=RANDOM_STATE, stratify=y_dec)
logit = LogisticRegression(max_iter=2000, C=1.0)
logit.fit(Xd_tr, yd_tr)
print(classification_report(yd_te, logit.predict(Xd_te)))""")

code("""coefs = pd.DataFrame(logit.coef_.T, index=X_dec.columns, columns=logit.classes_)
print("Strongest coefficients per decision class:\\n")
for cls in logit.classes_:
    print(f"--- {cls} ---")
    print(coefs[cls].sort_values(key=abs, ascending=False).head(6).round(2).to_string(), "\\n")""")

code("""allowed_high = df[(df.access_decision == "Allowed") & (df[TARGET] >= 70)]
blocked_low = df[(df.access_decision == "Blocked") & (df[TARGET] <= 30)]
print(f"Contradictions:")
print(f"  Allowed despite score >= 70 : {len(allowed_high)}")
print(f"  Blocked despite score <= 30 : {len(blocked_low)}")
allowed_high.head(10)""")

md("""## Triangulated findings

Run all cells, then read across the models:

1. **Drivers (Model 1):** the SHAP ranking is the ground truth for what the score responds to.
2. **Coherence (Model 2):** compare the tree's split variables with SHAP's top drivers — agreement means the scoring logic is simple and consistent; disagreement means interactions the rules miss.
3. **Blind spots (Model 3):** anomalous-but-low-scored rows above are candidates for scoring-policy gaps — review them manually.
4. **Decision consistency (Model 4):** the contradiction tables show where enforcement diverges from the score; each `Allowed`-with-high-score row is a potential security exception worth auditing.""")

nb.cells = cells
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "agent_security_risk_analysis.ipynb")
print("written")
