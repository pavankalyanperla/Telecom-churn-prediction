"""
predict.py — Standalone churn prediction script.

Loads the trained model, StandardScaler, and feature column list from models/,
then predicts churn probability for a single customer record.

Usage:
    python predict.py

The __main__ block at the bottom contains a realistic example customer.
To predict a different customer, replace the values in `example_customer`.

Input format:
    A dict with the original CSV column names (before OHE and scaling).
    Missing columns are filled with 0 after one-hot encoding, which is the
    standard practice for unseen categories in a known feature space.

Required columns (raw, pre-processing):
    gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
    MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod, MonthlyCharges
    (TotalCharges is excluded — dropped during training due to multicollinearity)
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")


def load_artefacts():
    """Load the three required inference artefacts from models/."""
    model   = joblib.load(os.path.join(_MODELS_DIR, "best_model.pkl"))
    scaler  = joblib.load(os.path.join(_MODELS_DIR, "scaler.pkl"))
    columns = joblib.load(os.path.join(_MODELS_DIR, "feature_columns.pkl"))
    return model, scaler, columns


def preprocess(customer: dict, columns: list, scaler) -> pd.DataFrame:
    """
    Apply the same preprocessing as the training notebook:
    1. Binary label-encode known binary fields (same mapping as training).
    2. One-hot encode multi-category fields via pd.get_dummies.
    3. Align to the saved feature_columns (fill missing OHE columns with 0).
    4. Scale with the fitted StandardScaler.

    Parameters
    ----------
    customer : dict
        Raw customer record — column names and values as they appear in the CSV.
    columns : list
        The ordered list of feature column names after OHE and customerID removal
        (loaded from models/feature_columns.pkl).
    scaler : StandardScaler
        The fitted scaler (loaded from models/scaler.pkl).

    Returns
    -------
    pd.DataFrame — one row, scaled, aligned to training feature space.
    """
    df = pd.DataFrame([customer])

    # Drop TotalCharges if present (excluded during training)
    df.drop(columns=[c for c in ["TotalCharges", "customerID"] if c in df.columns],
            inplace=True, errors="ignore")

    # Binary label encoding — same mapping as sklearn LabelEncoder(fit on full dataset):
    binary_map = {
        "gender":           {"Female": 0, "Male": 1},
        "Partner":          {"No": 0, "Yes": 1},
        "Dependents":       {"No": 0, "Yes": 1},
        "PhoneService":     {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
        "Churn":            {"No": 0, "Yes": 1},
    }
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])

    # One-hot encode remaining object columns
    df = pd.get_dummies(df)

    # Align to training columns: add missing OHE columns as 0, drop extras
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    # Scale
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=columns
    )
    return df_scaled


def predict_churn(customer: dict) -> dict:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer : dict
        Raw customer record (see module docstring for required fields).

    Returns
    -------
    dict with keys:
        'prediction' : str — "Yes" or "No"
        'churn_probability' : float — probability of churn (class 1)
        'no_churn_probability' : float — probability of staying (class 0)
    """
    model, scaler, columns = load_artefacts()
    X = preprocess(customer, columns, scaler)
    prediction_label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {
        "prediction":          "Yes" if prediction_label == 1 else "No",
        "churn_probability":   round(float(proba[1]), 4),
        "no_churn_probability": round(float(proba[0]), 4),
    }


if __name__ == "__main__":
    # ── Example customer ──────────────────────────────────────────────────────
    # A real-looking record using values from the original CSV's category set.
    # This customer has several high-churn indicators:
    #   - Month-to-month contract (highest churn segment)
    #   - Electronic check payment (correlated with churn)
    #   - Paperless billing enabled
    #   - Short tenure (2 months — new customer)
    example_customer = {
        "gender":           "Female",
        "SeniorCitizen":    0,
        "Partner":          "No",
        "Dependents":       "No",
        "tenure":           2,
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  "Fiber optic",
        "OnlineSecurity":   "No",
        "OnlineBackup":     "No",
        "DeviceProtection": "No",
        "TechSupport":      "No",
        "StreamingTV":      "No",
        "StreamingMovies":  "No",
        "Contract":         "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod":    "Electronic check",
        "MonthlyCharges":   70.70,
    }

    print("=" * 55)
    print("  Telecom Churn Prediction -- Single Customer Demo")
    print("=" * 55)
    print("\nCustomer profile:")
    for k, v in example_customer.items():
        print(f"  {k:<20} {v}")

    result = predict_churn(example_customer)

    print("\nPrediction result:")
    print(f"  Churn:              {result['prediction']}")
    print(f"  Churn probability:  {result['churn_probability']:.1%}")
    print(f"  Retain probability: {result['no_churn_probability']:.1%}")
    print()
    if result["prediction"] == "Yes":
        print("  [!] High churn risk -- consider a retention offer.")
    else:
        print("  [OK] Low churn risk -- customer likely to stay.")
    print("=" * 55)
