"""
Configuration for the Network Intrusion Detection Dashboard backend.
All paths are resolved relative to the project root.
"""
import os
from pathlib import Path

# Project root is two levels above dashboard/backend/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Phase 7.1 final model artifacts (source of truth for inference)
FINAL_MODELS_DIR = PROJECT_ROOT / "outputs" / "phase7_1" / "final_models"
VALIDATION_DIR   = PROJECT_ROOT / "outputs" / "phase7_1" / "validation"
FINAL_TEST_DIR   = PROJECT_ROOT / "outputs" / "phase7_1" / "final_test"
MODELS_XGB_DIR   = PROJECT_ROOT / "outputs" / "models" / "xgboost"

PREPROCESSOR_PATH          = FINAL_MODELS_DIR / "final_preprocessor.joblib"
FEATURE_METADATA_PATH      = FINAL_MODELS_DIR / "feature_metadata.json"
MODEL_CONFIGURATIONS_PATH  = FINAL_MODELS_DIR / "model_configurations.json"

MODEL_PATHS = {
    "logistic_regression": FINAL_MODELS_DIR / "logistic_regression_final.joblib",
    "decision_tree":       FINAL_MODELS_DIR / "decision_tree_final.joblib",
    "random_forest":       FINAL_MODELS_DIR / "random_forest_final.joblib",
    "xgboost":             FINAL_MODELS_DIR / "xgboost_final.joblib",
}

THRESHOLD_SELECTION_CSV   = VALIDATION_DIR / "threshold_selection_summary.csv"
THRESHOLD_ANALYSIS_CSV    = VALIDATION_DIR / "threshold_analysis.csv"
VALIDATION_METRICS_JSON   = VALIDATION_DIR / "validation_metrics.json"
TEST_METRICS_JSON         = FINAL_TEST_DIR / "test_metrics.json"
TEST_PREDICTIONS_NPZ      = FINAL_TEST_DIR / "test_predictions.npz"
XGB_FEATURE_IMPORTANCE_CSV = MODELS_XGB_DIR / "feature_importance.csv"

# Raw 42 input feature names (order matters for the preprocessor)
RAW_INPUT_FEATURES = [
    "dur","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl",
    "sload","dload","sloss","dloss","sinpkt","dinpkt","sjit","djit",
    "swin","stcpb","dtcpb","dwin","tcprtt","synack","ackdat",
    "smean","dmean","trans_depth","response_body_len",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm",
    "ct_dst_sport_ltm","ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd",
    "ct_flw_http_mthd","ct_src_ltm","ct_srv_dst","is_sm_ips_ports",
    "proto","service","state",
]

NON_FEATURE_COLS = {"label", "attack_cat", "id", "ID", "Id"}

MODEL_DISPLAY = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "short": "LR",
        "description": "Linear baseline model. Fast and interpretable via coefficient weights.",
        "color": "#6366f1",
    },
    "decision_tree": {
        "name": "Decision Tree",
        "short": "DT",
        "description": "Single tree with discrete leaf probabilities. Note: threshold tuning has limited effect due to discrete output.",
        "color": "#f59e0b",
        "threshold_note": "Decision Tree outputs discrete leaf probabilities. Threshold changes between leaf boundaries have no effect on predictions.",
    },
    "random_forest": {
        "name": "Random Forest",
        "short": "RF",
        "description": "Ensemble of 300 trees. Strong balanced performer across all metrics.",
        "color": "#10b981",
    },
    "xgboost": {
        "name": "XGBoost",
        "short": "XGB",
        "description": "Gradient-boosted ensemble. Best overall test performance (ROC-AUC 0.9719, F1 0.8212). Recommended model.",
        "color": "#06b6d4",
        "recommended": True,
    },
}
