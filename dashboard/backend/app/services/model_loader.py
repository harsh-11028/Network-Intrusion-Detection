"""
Model loader — loads all Phase 7.1 final artifacts at startup.
Nothing here ever refits a preprocessor or retrains a model.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import (
    PREPROCESSOR_PATH, FEATURE_METADATA_PATH, MODEL_CONFIGURATIONS_PATH,
    MODEL_PATHS, THRESHOLD_SELECTION_CSV, THRESHOLD_ANALYSIS_CSV,
    VALIDATION_METRICS_JSON, TEST_METRICS_JSON, TEST_PREDICTIONS_NPZ,
    XGB_FEATURE_IMPORTANCE_CSV, MODEL_DISPLAY, RAW_INPUT_FEATURES,
)

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Singleton that holds all loaded Phase 7.1 artifacts."""

    def __init__(self):
        logger.info("Loading Phase 7.1 artifacts …")
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info("  ✓ Preprocessor loaded (%s features)", len(RAW_INPUT_FEATURES))

        with open(FEATURE_METADATA_PATH) as f:
            self.feature_metadata = json.load(f)
        logger.info("  ✓ Feature metadata: %d transformed features", self.feature_metadata["count"])

        self.models: dict = {}
        for key, path in MODEL_PATHS.items():
            self.models[key] = joblib.load(path)
            logger.info("  ✓ Model loaded: %s", key)

        # Threshold data
        self.threshold_summary = pd.read_csv(THRESHOLD_SELECTION_CSV)
        self.threshold_analysis = pd.read_csv(THRESHOLD_ANALYSIS_CSV)

        # Validation metrics
        with open(VALIDATION_METRICS_JSON) as f:
            self.validation_metrics = json.load(f)

        # Official test metrics
        with open(TEST_METRICS_JSON) as f:
            self.test_metrics = json.load(f)

        # Test predictions (y_true + probabilities)
        npz = np.load(TEST_PREDICTIONS_NPZ)
        self.test_predictions = {k: npz[k].tolist() for k in npz.files}

        # XGBoost feature importance
        self.xgb_importance = pd.read_csv(XGB_FEATURE_IMPORTANCE_CSV)

        # Pre-build threshold presets per model
        self.threshold_presets = self._build_threshold_presets()

        # Pre-compute sklearn feature importances
        self.feature_importances = self._compute_feature_importances()

        logger.info("All artifacts loaded successfully.")

    def _build_threshold_presets(self) -> dict:
        presets = {}
        df = self.threshold_summary
        for model_key in ["logistic_regression", "decision_tree", "random_forest", "xgboost"]:
            rows = df[df["model_key"] == model_key]
            model_presets = []
            for _, row in rows.iterrows():
                feasible = str(row.get("feasibility_note", "")).lower()
                is_infeasible = "infeasible" in feasible
                threshold = row.get("selected_threshold")
                val_th = _safe_float(threshold)
                if val_th is None:
                    is_infeasible = True

                model_presets.append({
                    "strategy": str(row["strategy"]),
                    "threshold": val_th,
                    "val_attack_recall": _safe_float(row.get("val_attack_recall")),
                    "val_attack_f1": _safe_float(row.get("val_attack_f1")),
                    "val_fpr": _safe_float(row.get("val_fpr")),
                    "feasibility_note": str(row.get("feasibility_note", "")),
                    "infeasible": is_infeasible,
                })
            presets[model_key] = model_presets
        return presets

    def _compute_feature_importances(self) -> dict:
        """Extract feature importances / coefficients from loaded models."""
        result = {}
        transformed_names = self.feature_metadata["feature_names"]

        # XGBoost — use the saved CSV (gain-based)
        xgb_df = self.xgb_importance.copy()
        if "feature" in xgb_df.columns and "importance" in xgb_df.columns:
            xgb_df = xgb_df.sort_values("importance", ascending=False)
            result["xgboost"] = xgb_df[["feature", "importance"]].head(50).to_dict("records")
        else:
            # Fallback: get from model directly
            xgb_model = self.models["xgboost"]
            if hasattr(xgb_model, "feature_importances_"):
                imps = xgb_model.feature_importances_
                pairs = sorted(zip(transformed_names, imps), key=lambda x: -x[1])
                result["xgboost"] = [{"feature": f, "importance": float(v)} for f, v in pairs[:50]]

        # Random Forest
        rf_model = self.models["random_forest"]
        if hasattr(rf_model, "feature_importances_"):
            imps = rf_model.feature_importances_
            pairs = sorted(zip(transformed_names, imps), key=lambda x: -x[1])
            result["random_forest"] = [{"feature": f, "importance": float(v)} for f, v in pairs[:50]]

        # Decision Tree
        dt_model = self.models["decision_tree"]
        if hasattr(dt_model, "feature_importances_"):
            imps = dt_model.feature_importances_
            pairs = sorted(zip(transformed_names, imps), key=lambda x: -x[1])
            result["decision_tree"] = [{"feature": f, "importance": float(v)} for f, v in pairs[:50]]

        # Logistic Regression — coefficients (absolute value as importance proxy)
        lr_model = self.models["logistic_regression"]
        if hasattr(lr_model, "coef_"):
            coefs = lr_model.coef_[0]
            pairs = sorted(zip(transformed_names, coefs), key=lambda x: -abs(x[1]))
            result["logistic_regression"] = [{"feature": f, "importance": float(v)} for f, v in pairs[:50]]

        return result


def _safe_float(val):
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


_store: ArtifactStore | None = None


def get_store() -> ArtifactStore:
    global _store
    if _store is None:
        _store = ArtifactStore()
    return _store
