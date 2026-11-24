"""
Prediction service — applies the Phase 7.1 final_preprocessor to raw CSV data,
runs inference with the selected model and threshold, and returns results.
Never refits the preprocessor.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score,
)

from app.config import RAW_INPUT_FEATURES, NON_FEATURE_COLS
from app.services.model_loader import ArtifactStore

logger = logging.getLogger(__name__)


def validate_and_extract_features(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Validates uploaded DataFrame, extracts feature columns and optional ground truth.
    Returns (X_raw, y_true_or_None, attack_cat_or_None).
    Raises ValueError with a user-friendly message on bad input.
    """
    if df.empty:
        raise ValueError("Uploaded file is empty.")

    # Extract optional labels before dropping
    y_true = None
    attack_cat = None
    if "label" in df.columns:
        y_true = pd.to_numeric(df["label"], errors="coerce")
    if "attack_cat" in df.columns:
        attack_cat = df["attack_cat"].astype(str)

    # Check required feature columns
    missing = [c for c in RAW_INPUT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing {len(missing)} required feature column(s): {missing[:10]}"
            + (" …" if len(missing) > 10 else "")
        )

    # Keep only the 42 raw input features (in correct order)
    X_raw = df[RAW_INPUT_FEATURES].copy()

    if X_raw.empty:
        raise ValueError("No data rows found after feature extraction.")

    return X_raw, y_true, attack_cat


def predict(
    store: ArtifactStore,
    df: pd.DataFrame,
    model_key: str,
    threshold: float,
) -> dict:
    """
    Full inference pipeline:
      1. validate & extract features
      2. transform with final_preprocessor (no refit)
      3. predict probabilities
      4. apply threshold
      5. optionally compute evaluation metrics if labels present
    """
    if model_key not in store.models:
        raise ValueError(f"Unknown model key: {model_key!r}. Valid: {list(store.models.keys())}")
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"Threshold must be in (0, 1), got {threshold}")

    X_raw, y_true, attack_cat = validate_and_extract_features(df)
    n_rows = len(X_raw)
    logger.info("Predicting %d rows with model=%s, threshold=%.2f", n_rows, model_key, threshold)

    # Transform (never refit)
    X_transformed = store.preprocessor.transform(X_raw)

    # Predict
    model = store.models[model_key]
    proba = model.predict_proba(X_transformed)[:, 1]
    predictions = (proba >= threshold).astype(int)

    n_attacks = int(predictions.sum())
    n_normal  = n_rows - n_attacks
    attack_pct = round(n_attacks / n_rows * 100, 2) if n_rows > 0 else 0.0

    # Build per-row results (top 200 for display; full data for export)
    rows = []
    for i in range(n_rows):
        row: dict = {
            "row_index": i,
            "probability": round(float(proba[i]), 6),
            "prediction": int(predictions[i]),
            "label_text": "Attack" if predictions[i] == 1 else "Normal",
        }
        if y_true is not None:
            row["true_label"] = int(y_true.iloc[i]) if not pd.isna(y_true.iloc[i]) else None
        if attack_cat is not None:
            row["attack_cat"] = str(attack_cat.iloc[i])
        rows.append(row)

    result = {
        "model_key": model_key,
        "threshold": threshold,
        "n_rows": n_rows,
        "n_attacks": n_attacks,
        "n_normal": n_normal,
        "attack_pct": attack_pct,
        "rows": rows,
        "has_ground_truth": y_true is not None,
        "evaluation": None,
    }

    # Evaluation (only if ground truth present and no NaN labels)
    if y_true is not None:
        valid_mask = ~y_true.isna()
        if valid_mask.sum() > 0:
            y_true_clean = y_true[valid_mask].astype(int).values
            preds_clean  = predictions[valid_mask]
            proba_clean  = proba[valid_mask]

            cm = confusion_matrix(y_true_clean, preds_clean)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            report = classification_report(y_true_clean, preds_clean, output_dict=True, zero_division=0)

            eval_metrics: dict = {
                "accuracy":          round(float(accuracy_score(y_true_clean, preds_clean)), 6),
                "attack_recall":     round(float(recall_score(y_true_clean, preds_clean, pos_label=1, zero_division=0)), 6),
                "attack_precision":  round(float(precision_score(y_true_clean, preds_clean, pos_label=1, zero_division=0)), 6),
                "attack_f1":         round(float(f1_score(y_true_clean, preds_clean, pos_label=1, zero_division=0)), 6),
                "macro_f1":          round(float(f1_score(y_true_clean, preds_clean, average="macro", zero_division=0)), 6),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "confusion_matrix": cm.tolist(),
                "n_labeled": int(valid_mask.sum()),
            }
            # AUC requires both classes
            if len(np.unique(y_true_clean)) > 1:
                eval_metrics["roc_auc"] = round(float(roc_auc_score(y_true_clean, proba_clean)), 6)
                eval_metrics["pr_auc"]  = round(float(average_precision_score(y_true_clean, proba_clean)), 6)

            result["evaluation"] = eval_metrics

    return result
