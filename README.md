# Telecom Customer Churn Prediction

## Overview

Customer churn -- when a subscriber cancels or stops using a service -- is one of the most costly problems for telecom companies. Acquiring a new customer costs 5-25x more than retaining an existing one, so identifying at-risk customers *before* they leave is high-value. This project builds and evaluates a suite of machine learning classifiers to predict which customers are likely to churn, using the IBM Telco Customer Churn dataset. Seven models are compared, the best is tuned with GridSearchCV, and the final model is serialised for deployment via a standalone Python script (`predict.py`).

---

## Dataset

| Property | Value |
|---|---|
| Source | IBM Sample Dataset (Telco Customer Churn) |
| Rows | 7,043 customers |
| Columns | 21 (20 features + 1 target) |
| Target | `Churn` (Yes / No) |
| Class distribution | No Churn: 5,174 (73.5%) / Churn: 1,869 (26.5%) |

**Class imbalance note:** The dataset is imbalanced at roughly 3:1 in favour of non-churners. Models trained without correction tend to be biased toward predicting "No Churn". This project addresses imbalance via `class_weight='balanced'` for sklearn models and `scale_pos_weight` for XGBoost, plus a stratified train/test split.

**Key features:** tenure, Contract type, PaymentMethod, MonthlyCharges, InternetService, OnlineSecurity, TechSupport, PaperlessBilling, SeniorCitizen, and 9 other service and demographic columns.

**Preprocessing note:** `TotalCharges` was dropped due to high multicollinearity with `MonthlyCharges` (VIF = 13.99 vs 12.57); the feature adds little independent signal once MonthlyCharges is present.

---

## Repository Structure

```
.
+-- notebooks/
|   +-- Telecom Customer Churn Prediction Final.ipynb  # Full analysis notebook
+-- data/
|   +-- customer_churn_data.csv                        # Raw dataset
+-- docs/
|   +-- Telecom Customer Churn Prediction Final.html   # Static notebook export
|   +-- Telecom-Customer-Churn-Prediction.pptx         # Project slide deck
|   +-- Telecom_Churn_Prediction_Report.pdf            # Project report
+-- models/
|   +-- best_model.pkl                                 # Serialised final model
|   +-- scaler.pkl                                     # Fitted StandardScaler
|   +-- feature_columns.pkl                            # Post-OHE feature list (39 features)
+-- predict.py                                         # Standalone single-customer prediction script
+-- requirements.txt
+-- .gitignore
+-- REPO_AUDIT.md                                      # Pre-refactor code audit
```

---

## EDA Highlights

All numbers below are taken directly from notebook output cells.

**1. Churn rate is 26.5%.** The dataset has 5,174 non-churners and 1,869 churners -- a 3:1 imbalance that requires correction in modeling.

**2. Contract type is the strongest categorical predictor.** Month-to-month customers churn at a dramatically higher rate than one-year or two-year contract customers. Customers on long-term contracts rarely churn because switching costs are high.

**3. Tenure is the strongest continuous negative predictor** (correlation with churn = -0.352). The longer a customer has been with the company, the less likely they are to leave -- a classic loyalty effect.

**4. Monthly charges correlate positively with churn** (+0.193). Higher-paying customers are more sensitive to service dissatisfaction and more likely to compare alternatives.

**5. Electronic check users and paperless billing users churn more** (PaperlessBilling correlation = +0.192). This segment shows higher churn rates, possibly because it attracts less committed customers who are easier to switch.

---

## Models Compared

Seven classifiers were trained and evaluated on a held-out 20% test set (1,409 customers, stratified split, `random_state=42`). **F2 Score** (beta=2, double-weighting recall) is used as the primary ranking metric since missing a churner is more expensive than a false alarm.

| Model | Accuracy | Precision | Recall | F1 Score | F2 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Naive Bayes** | 0.6835 | 0.4490 | **0.8476** | 0.5870 | **0.7198** | 0.8082 |
| **Naive Bayes (Tuned)** | 0.6835 | 0.4490 | 0.8476 | 0.5870 | 0.7198 | 0.8082 |
| Logistic Regression | 0.7410 | 0.5079 | 0.7781 | 0.6146 | 0.7032 | **0.8382** |
| SVC | 0.7488 | 0.5181 | 0.7674 | 0.6185 | 0.7000 | 0.8188 |
| XGBoost | 0.7559 | 0.5316 | 0.6738 | 0.5943 | 0.6396 | 0.8173 |
| Random Forest | **0.7814** | **0.6100** | 0.4893 | 0.5430 | 0.5095 | 0.8105 |
| Decision Tree | 0.7303 | 0.4923 | 0.5134 | 0.5026 | 0.5090 | 0.6618 |
| K-Nearest Neighbors | 0.7473 | 0.5257 | 0.4920 | 0.5083 | 0.4984 | 0.7658 |

*GridSearchCV on Naive Bayes confirmed the default `var_smoothing=1e-9` is already optimal; the tuned row is included for completeness.*

---

## Final Model: Naive Bayes

**Chosen model:** Gaussian Naive Bayes (`models/best_model.pkl`)

**Primary justification (business):** In telecom churn, the cost asymmetry strongly favours maximising recall:

- A **False Negative** (missed churner) = lost subscription revenue with no chance to intervene -- potentially months or years of recurring income gone.
- A **False Positive** (loyal customer flagged as at-risk) = minor cost of a retention offer (a discount SMS or a brief call from customer service), easily absorbed by the retained revenue.

Naive Bayes achieves **84.8% recall on the churn class** -- it catches nearly 9 in 10 churners -- at the cost of lower precision (44.9%), meaning about half of flagged customers would have stayed anyway. For most telecom retention budgets this trade-off is acceptable.

**Alternative:** Logistic Regression achieves a higher ROC-AUC (0.8382 vs 0.8082) and better precision (0.508 vs 0.449). Teams with a constrained outreach budget who cannot afford many false alarms should consider Logistic Regression instead.

---

## How to Run

**Full analysis (notebook):**

```bash
git clone https://github.com/pavankalyanperla/Telecom-churn-prediction.git
cd Telecom-churn-prediction
pip install -r requirements.txt
jupyter notebook "notebooks/Telecom Customer Churn Prediction Final.ipynb"
```

**Quick single-customer prediction demo:**

```bash
python predict.py
```

This loads the saved model artefacts from `models/` and prints the churn prediction and probability for an example high-risk customer (month-to-month contract, fibre optic, electronic check, tenure = 2 months).

To predict a different customer, edit the `example_customer` dict in `predict.py`'s `__main__` block.

---

## Possible Next Steps

- **SMOTE oversampling:** Synthesise minority-class samples to address class imbalance more aggressively, potentially improving precision without sacrificing recall.
- **Feature importance analysis:** Use SHAP values or Logistic Regression coefficients to explain which features drive individual predictions -- useful for actionable retention strategies (e.g. targeting month-to-month customers with high monthly charges for a contract upgrade offer).
- **Threshold tuning:** The default 0.5 classification threshold is not optimal for imbalanced problems. Tuning the threshold for the specific business cost ratio of false negatives vs false positives can improve real-world performance.
- **API wrapper:** Wrap `predict.py` in a FastAPI or Flask endpoint so CRM systems can call the model in real time when a customer contacts support.
- **Model monitoring:** Track prediction distribution drift over time -- if the model starts producing very different churn rates than observed, it needs retraining on fresh data.
- **Ensemble stacking:** Combine Logistic Regression (high AUC, good calibration) and Naive Bayes (high recall) via a meta-learner to seek both high recall and higher precision simultaneously.
