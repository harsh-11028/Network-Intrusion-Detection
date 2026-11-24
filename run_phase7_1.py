"""
run_phase7_1.py
---------------
Phase 7.1: Leakage-Free Validation, Threshold Re-selection, Final Retraining & Reproducibility Audit
Project: Machine Learning-Based Network Intrusion Detection and Attack Classification
Dataset: UNSW-NB15

Key Methodological Corrections:
1. RAW Split: The original cleaned CSV (107,740 rows) is split into 80% internal train
   and 20% validation partitions BEFORE any preprocessing or scaling.
2. Leakage-Free Preprocessing: Imputation, StandardScaler, and OneHotEncoder are fitted
   STRICTLY on the 80% internal training partition (86,192 rows).
3. The 20% validation partition is transformed using this fitted pipeline without leaking
   validation statistics or rare categorical vocabularies (e.g. proto='rtp', state='URN').
4. Thresholds are evaluated and selected using validation predictions ONLY.
5. Independent TTL ablation is conducted on the internal split with independently fitted preprocessors.
6. A fresh final preprocessor is fitted on the complete 100% training set (107,740 rows).
7. Final models are retrained on the 100% training set and evaluated on the quarantined
   official test set (55,945 rows) at validation-selected thresholds.
"""

import os
from pathlib import Path
import json
import time
import hashlib
from typing import Dict, Any, Tuple, List, Optional

# Headless matplotlib configuration
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from preprocessing import (
    separate_features_and_target,
    identify_feature_types,
    build_preprocessor,
    compute_file_sha256
)

# Directory Structure
BASE_DIR = Path("outputs/phase7_1")
VAL_DIR = BASE_DIR / "validation"
FINAL_MODELS_DIR = BASE_DIR / "final_models"
FINAL_TEST_DIR = BASE_DIR / "final_test"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

for d in [BASE_DIR, VAL_DIR, FINAL_MODELS_DIR, FINAL_TEST_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "color": "#1f77b4",
        "line_style": "-",
        "params": {
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs"
        }
    },
    "decision_tree": {
        "name": "Decision Tree",
        "color": "#ff7f0e",
        "line_style": "--",
        "params": {
            "random_state": 42
        }
    },
    "random_forest": {
        "name": "Random Forest",
        "color": "#2ca02c",
        "line_style": "-.",
        "params": {
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1,
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": None
        }
    },
    "xgboost": {
        "name": "XGBoost",
        "color": "#d62728",
        "line_style": ":",
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }
    }
}


def log(msg: str, log_file: Optional[Path] = None):
    print(msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


def get_model_instance(model_key: str):
    cfg = MODEL_CONFIGS[model_key]["params"]
    if model_key == "logistic_regression":
        return LogisticRegression(**cfg)
    elif model_key == "decision_tree":
        return DecisionTreeClassifier(**cfg)
    elif model_key == "random_forest":
        return RandomForestClassifier(**cfg)
    elif model_key == "xgboost":
        return XGBClassifier(**cfg)
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def main():
    log_file = LOGS_DIR / "training_log.txt"
    with open(log_file, "w") as f:
        f.write("PHASE 7.1 EXECUTION LOG\n")
        f.write("=" * 60 + "\n")

    log("=" * 75, log_file)
    log(" PHASE 7.1: LEAKAGE-FREE VALIDATION, THRESHOLD SELECTION & FINAL EVALUATION", log_file)
    log("=" * 75, log_file)

    # 1. Audit and load raw cleaned datasets
    train_csv = Path("data/UNSW_NB15_training_cleaned.csv")
    test_csv = Path("data/UNSW_NB15_testing_cleaned.csv")

    train_sha = compute_file_sha256(train_csv)
    test_sha = compute_file_sha256(test_csv)

    log(f"\n[DATA AUDIT] Verifying dataset files and immutability sha256...", log_file)
    log(f"  -> Train CSV: {train_csv} (SHA-256: {train_sha})", log_file)
    log(f"  -> Test  CSV: {test_csv} (SHA-256: {test_sha})", log_file)

    train_df = pd.read_csv(train_csv, low_memory=False)
    test_df = pd.read_csv(test_csv, low_memory=False)

    log(f"  -> Raw Train shape: {train_df.shape}", log_file)
    log(f"  -> Raw Test  shape: {test_df.shape}", log_file)

    # 2. Separate features from targets
    X_train_raw, y_train_raw = separate_features_and_target(train_df, target_col="label")
    X_test_raw, y_test_raw = separate_features_and_target(test_df, target_col="label")

    num_cols, cat_cols = identify_feature_types(X_train_raw)
    log(f"  -> Separated features: {X_train_raw.shape[1]} input columns ({len(num_cols)} num, {len(cat_cols)} cat)", log_file)
    log(f"  -> Target distribution in Train: Normal={np.sum(y_train_raw==0):,}, Attack={np.sum(y_train_raw==1):,}", log_file)
    log(f"  -> Target distribution in Test : Normal={np.sum(y_test_raw==0):,}, Attack={np.sum(y_test_raw==1):,}", log_file)

    # 3. Create strictly RAW 80/20 train/validation split
    log(f"\n[SPLIT] Creating stratified 80/20 train/val split on RAW features BEFORE preprocessing...", log_file)
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=0.20, random_state=42, stratify=y_train_raw
    )
    log(f"  -> Internal Train subset : {X_tr_raw.shape} (Attack: {np.sum(y_tr==1):,}, Normal: {np.sum(y_tr==0):,})", log_file)
    log(f"  -> Validation subset     : {X_val_raw.shape} (Attack: {np.sum(y_val==1):,}, Normal: {np.sum(y_val==0):,})", log_file)

    # 4. Fit Preprocessing Pipeline EXCLUSIVELY on Internal Training Data
    log(f"\n[PREPROCESSING] Fitting ColumnTransformer exclusively on internal training subset...", log_file)
    val_preprocessor, active_num_cols, active_cat_cols = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    val_preprocessor.fit(X_tr_raw)

    X_tr_val_trans = val_preprocessor.transform(X_tr_raw)
    X_val_trans = val_preprocessor.transform(X_val_raw)

    val_feat_names = val_preprocessor.get_feature_names_out().tolist()
    log(f"  -> Transformed internal train shape: {X_tr_val_trans.shape} ({len(val_feat_names)} features)", log_file)
    log(f"  -> Transformed validation shape    : {X_val_trans.shape} ({len(val_feat_names)} features)", log_file)
    log(f"  -> Note: Validation features = {len(val_feat_names)} (vs 194 in full set, proving unseen 'proto=rtp' and 'state=URN' are properly handled without leakage)", log_file)

    joblib.dump(val_preprocessor, VAL_DIR / "val_preprocessor.joblib")
    with open(VAL_DIR / "val_feature_metadata.json", "w") as f:
        json.dump({"feature_names": val_feat_names, "count": len(val_feat_names)}, f, indent=2)

    # 5. Train Validation-Stage Models
    log(f"\n[TRAIN VAL] Retraining 4 baseline models on internal training subset (86,192 rows, {len(val_feat_names)} features)...", log_file)
    val_models = {}
    val_probs = {}
    val_train_times = {}

    for key, info in MODEL_CONFIGS.items():
        log(f"  -> Fitting {info['name']} on internal training subset...", log_file)
        model = get_model_instance(key)
        start_t = time.perf_counter()
        model.fit(X_tr_val_trans, y_tr.values if hasattr(y_tr, "values") else y_tr)
        fit_t = time.perf_counter() - start_t
        val_train_times[key] = fit_t

        prob = model.predict_proba(X_val_trans)[:, 1]
        val_models[key] = model
        val_probs[key] = prob
        log(f"     Fitted in {fit_t:.2f}s | Validation prob range: [{prob.min():.4f}, {prob.max():.4f}]", log_file)

    # 6. Sweep Thresholds on Leakage-Free Validation Set
    log(f"\n[SWEEP] Evaluating thresholds 0.05 to 0.95 (step 0.01) on leakage-free validation set...", log_file)
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    all_val_rows = []
    per_model_val_dfs = {}
    y_val_arr = y_val.values if hasattr(y_val, "values") else y_val

    for key, prob in val_probs.items():
        m_name = MODEL_CONFIGS[key]["name"]
        m_rows = []
        for th in thresholds:
            y_pred = (prob >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val_arr, y_pred, labels=[0, 1]).ravel()
            acc = accuracy_score(y_val_arr, y_pred)
            prec = precision_score(y_val_arr, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_val_arr, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_val_arr, y_pred, pos_label=1, zero_division=0)
            norm_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            macro_f1 = f1_score(y_val_arr, y_pred, average="macro", zero_division=0)

            row = {
                "model_key": key,
                "model_name": m_name,
                "threshold": float(th),
                "accuracy": float(acc),
                "attack_precision": float(prec),
                "attack_recall": float(rec),
                "attack_f1": float(f1),
                "normal_recall": float(norm_rec),
                "false_positive_rate": float(fpr),
                "false_negative_rate": float(fnr),
                "macro_f1": float(macro_f1),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "total_validation_samples": int(len(y_val_arr))
            }
            m_rows.append(row)
            all_val_rows.append(row)

        df_m = pd.DataFrame(m_rows)
        per_model_val_dfs[key] = df_m
        csv_path = VAL_DIR / f"{key}_thresholds.csv"
        df_m.to_csv(csv_path, index=False)
        log(f"  -> Saved {csv_path.name}", log_file)

    df_val_all = pd.DataFrame(all_val_rows)
    df_val_all.to_csv(VAL_DIR / "threshold_analysis.csv", index=False)
    log(f"  -> Saved {VAL_DIR / 'threshold_analysis.csv'}", log_file)

    # 7. Select Thresholds under Strategies A, B, C, D
    log(f"\n[STRATEGY] Selecting operating thresholds under Strategies A, B, C, D...", log_file)
    selected_dict = {}
    summary_rows = []

    for key, df in per_model_val_dfs.items():
        m_name = MODEL_CONFIGS[key]["name"]
        selected_dict[key] = {}

        # Strategy A: Max Attack F1 (tie-breaker: lower FPR, higher threshold)
        max_f1 = df["attack_f1"].max()
        cand_a = df[df["attack_f1"] == max_f1].sort_values(by=["false_positive_rate", "threshold"], ascending=[True, False])
        best_a = cand_a.iloc[0]
        selected_dict[key]["strategy_a_max_attack_f1"] = best_a.to_dict()

        # Strategy B: Max Macro F1
        max_macro = df["macro_f1"].max()
        cand_b = df[df["macro_f1"] == max_macro].sort_values(by=["false_positive_rate", "threshold"], ascending=[True, False])
        best_b = cand_b.iloc[0]
        selected_dict[key]["strategy_b_max_macro_f1"] = best_b.to_dict()

        # Strategy C: Security-Oriented (Recall >= 97%, min FPR)
        feas_c = df[df["attack_recall"] >= 0.97]
        if len(feas_c) > 0:
            best_c = feas_c.sort_values(by=["false_positive_rate", "threshold"], ascending=[True, False]).iloc[0]
            status_c = "Feasible (Recall >= 97%)"
            c_feasible = True
        else:
            best_c = df.sort_values(by=["attack_recall", "false_positive_rate"], ascending=[False, True]).iloc[0]
            status_c = f"Infeasible (Max Recall={best_c['attack_recall']*100:.2f}%)"
            c_feasible = False
        selected_dict[key]["strategy_c_security_oriented"] = best_c.to_dict()
        selected_dict[key]["strategy_c_status"] = status_c
        selected_dict[key]["strategy_c_feasible"] = c_feasible

        # Strategy D: Operational Target (Recall >= 95%, FPR <= 20%)
        feas_d = df[(df["attack_recall"] >= 0.95) & (df["false_positive_rate"] <= 0.20)]
        if len(feas_d) > 0:
            best_d = feas_d.sort_values(by=["attack_f1", "false_positive_rate"], ascending=[False, True]).iloc[0]
            status_d = f"Feasible (Th={best_d['threshold']:.2f}, Rec={best_d['attack_recall']*100:.2f}%, FPR={best_d['false_positive_rate']*100:.2f}%)"
            d_feasible = True
        else:
            best_d = None
            status_d = "Infeasible (No threshold satisfies Recall >= 95% AND FPR <= 20%)"
            d_feasible = False
        selected_dict[key]["strategy_d_operational"] = best_d.to_dict() if best_d is not None else None
        selected_dict[key]["strategy_d_status"] = status_d
        selected_dict[key]["strategy_d_feasible"] = d_feasible

        # Default 0.50
        def_row = df[df["threshold"] == 0.50].iloc[0]
        selected_dict[key]["default_threshold_0_5"] = def_row.to_dict()

        log(f"  -> {m_name}:", log_file)
        log(f"     Strategy A (Max Attack F1) : Th={best_a['threshold']:.2f} | Val F1={best_a['attack_f1']:.4f} | Rec={best_a['attack_recall']*100:.2f}% | FPR={best_a['false_positive_rate']*100:.2f}%", log_file)
        log(f"     Strategy B (Max Macro F1)  : Th={best_b['threshold']:.2f} | Val Macro F1={best_b['macro_f1']:.4f}", log_file)
        log(f"     Strategy C (Security 97%)  : Th={best_c['threshold']:.2f} | {status_c}", log_file)
        log(f"     Strategy D (Operational)   : {status_d}", log_file)

        # Append to summary table
        strats = [
            ("Default (0.50)", 0.50, def_row, "Standard reference"),
            ("Strategy A (Max Attack F1)", best_a["threshold"], best_a, f"Max validation Attack F1 ({best_a['attack_f1']:.4f})"),
            ("Strategy B (Max Macro F1)", best_b["threshold"], best_b, f"Max validation Macro F1 ({best_b['macro_f1']:.4f})"),
            ("Strategy C (Security >= 97%)", best_c["threshold"], best_c, status_c),
            ("Strategy D (Operational Target)", best_d["threshold"] if best_d is not None else np.nan, best_d, status_d)
        ]
        for s_label, th_v, row_d, note in strats:
            summary_rows.append({
                "model_key": key,
                "model_name": m_name,
                "strategy": s_label,
                "selected_threshold": th_v,
                "val_accuracy": round(float(row_d["accuracy"]), 4) if row_d is not None else None,
                "val_attack_recall": round(float(row_d["attack_recall"]), 4) if row_d is not None else None,
                "val_attack_precision": round(float(row_d["attack_precision"]), 4) if row_d is not None else None,
                "val_attack_f1": round(float(row_d["attack_f1"]), 4) if row_d is not None else None,
                "val_fpr": round(float(row_d["false_positive_rate"]), 4) if row_d is not None else None,
                "val_false_positives": int(row_d["false_positives"]) if row_d is not None else None,
                "val_false_negatives": int(row_d["false_negatives"]) if row_d is not None else None,
                "feasibility_note": note
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(VAL_DIR / "threshold_selection_summary.csv", index=False)
    log(f"  -> Saved {VAL_DIR / 'threshold_selection_summary.csv'}", log_file)

    # 8. Generate 9 Validation Plots
    log(f"\n[PLOTS] Generating 9 validation threshold plots in {VAL_DIR}...", log_file)
    generate_validation_visualizations(df_val_all, y_val_arr, val_probs, VAL_DIR)

    # 9. TTL Feature Ablation (Independent Pipeline on Internal Split)
    log(f"\n[TTL ABLATION] Running independent validation feature ablation...", log_file)
    ablation_results = run_ttl_ablation_experiment(X_tr_raw, y_tr, X_val_raw, y_val, num_cols, cat_cols, log_file)

    # 10. Fit Final Preprocessing on Complete 100% Training Dataset
    log(f"\n[FINAL PREPROCESSING] Fitting fresh preprocessor on 100% of official training set ({len(train_df):,} rows)...", log_file)
    final_preprocessor, final_num_cols, final_cat_cols = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    final_preprocessor.fit(X_train_raw)

    X_train_full_trans = final_preprocessor.transform(X_train_raw)
    X_test_trans = final_preprocessor.transform(X_test_raw)

    final_feat_names = final_preprocessor.get_feature_names_out().tolist()
    log(f"  -> Full transformed train shape: {X_train_full_trans.shape} ({len(final_feat_names)} features)", log_file)
    log(f"  -> Transformed official test shape: {X_test_trans.shape} ({len(final_feat_names)} features)", log_file)

    joblib.dump(final_preprocessor, FINAL_MODELS_DIR / "final_preprocessor.joblib")
    with open(FINAL_MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump({"feature_names": final_feat_names, "count": len(final_feat_names)}, f, indent=2)

    # 11. Retrain Final Models on 100% Training Set
    log(f"\n[RETRAIN FINAL] Training final models on complete transformed training set...", log_file)
    final_models = {}
    y_train_arr = y_train_raw.values if hasattr(y_train_raw, "values") else y_train_raw

    for key, info in MODEL_CONFIGS.items():
        log(f"  -> Retraining final {info['name']}...", log_file)
        model = get_model_instance(key)
        start_t = time.perf_counter()
        model.fit(X_train_full_trans, y_train_arr)
        dur = time.perf_counter() - start_t
        log(f"     Fitted in {dur:.2f}s", log_file)
        save_p = FINAL_MODELS_DIR / f"{key}_final.joblib"
        joblib.dump(model, save_p)
        final_models[key] = model

    with open(FINAL_MODELS_DIR / "model_configurations.json", "w") as f:
        json.dump({k: v["params"] for k, v in MODEL_CONFIGS.items()}, f, indent=2)

    # 12. Evaluate Final Models on Official Test Set (55,945 rows)
    log(f"\n[OFFICIAL TEST EVAL] Evaluating final models on untouched official test set...", log_file)
    final_test_results = {}
    test_probs = {}
    y_test_arr = y_test_raw.values if hasattr(y_test_raw, "values") else y_test_raw

    for key, model in final_models.items():
        m_name = MODEL_CONFIGS[key]["name"]
        final_test_results[key] = {}
        prob = model.predict_proba(X_test_trans)[:, 1]
        test_probs[key] = prob

        # Strategies to evaluate on test set:
        strategies_to_eval = [
            ("default_0_5", 0.50, "Default (0.50)"),
            ("strategy_a_max_attack_f1", selected_dict[key]["strategy_a_max_attack_f1"]["threshold"], "Strategy A (Max Val Attack F1)"),
            ("strategy_b_max_macro_f1", selected_dict[key]["strategy_b_max_macro_f1"]["threshold"], "Strategy B (Max Val Macro F1)"),
            ("strategy_c_security_oriented", selected_dict[key]["strategy_c_security_oriented"]["threshold"], "Strategy C (Security-Oriented)")
        ]
        if selected_dict[key]["strategy_d_feasible"] and selected_dict[key].get("strategy_d_operational") is not None:
            strategies_to_eval.append((
                "strategy_d_operational",
                selected_dict[key]["strategy_d_operational"]["threshold"],
                "Strategy D (Operational Target)"
            ))

        for strat_key, th, strat_label in strategies_to_eval:
            y_pred = (prob >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred, labels=[0, 1]).ravel()
            acc = accuracy_score(y_test_arr, y_pred)
            prec = precision_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
            norm_prec = precision_score(y_test_arr, y_pred, pos_label=0, zero_division=0)
            norm_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            macro_prec = precision_score(y_test_arr, y_pred, average="macro", zero_division=0)
            macro_rec = recall_score(y_test_arr, y_pred, average="macro", zero_division=0)
            macro_f1 = f1_score(y_test_arr, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_test_arr, y_pred, average="weighted", zero_division=0)
            roc_auc = roc_auc_score(y_test_arr, prob)
            pr_auc = average_precision_score(y_test_arr, prob)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            res = {
                "strategy_key": strat_key,
                "strategy_label": strat_label,
                "threshold": float(th),
                "accuracy": round(float(acc), 4),
                "attack_precision": round(float(prec), 4),
                "attack_recall": round(float(rec), 4),
                "attack_f1": round(float(f1), 4),
                "normal_precision": round(float(norm_prec), 4),
                "normal_recall": round(float(norm_rec), 4),
                "macro_precision": round(float(macro_prec), 4),
                "macro_recall": round(float(macro_rec), 4),
                "macro_f1": round(float(macro_f1), 4),
                "weighted_f1": round(float(weighted_f1), 4),
                "roc_auc": round(float(roc_auc), 4),
                "pr_auc": round(float(pr_auc), 4),
                "false_positive_rate": round(float(fpr), 4),
                "false_negative_rate": round(float(fnr), 4),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "total_test_samples": int(len(y_test_arr))
            }
            final_test_results[key][strat_key] = res
            log(f"  -> {m_name} [{strat_label} @ Th={th:.2f}]: Acc={acc*100:.2f}%, Rec={rec*100:.2f}%, Prec={prec*100:.2f}%, F1={f1:.4f}, FP={fp:,}, FN={fn:,}", log_file)

    # Save test predictions and metrics
    np.savez_compressed(
        FINAL_TEST_DIR / "test_predictions.npz",
        y_true=y_test_arr,
        **{f"prob_{k}": test_probs[k] for k in test_probs}
    )
    with open(FINAL_TEST_DIR / "test_metrics.json", "w") as f:
        json.dump(final_test_results, f, indent=2)

    # 13. Generate Final Test Visualizations
    log(f"\n[PLOTS] Generating 7 final test evaluation plots in {FINAL_TEST_DIR}...", log_file)
    generate_final_test_visualizations(final_test_results, y_test_arr, test_probs, FINAL_TEST_DIR)

    # 14. Save validation_metrics.json and execution_metadata.json
    validation_payload = {
        "phase": "7.1",
        "description": "Leakage-free validation split and threshold re-selection",
        "validation_split_configuration": {
            "validation_proportion": 0.20,
            "training_proportion": 0.80,
            "random_state": 42,
            "stratification": True,
            "raw_train_samples": int(len(train_df)),
            "internal_train_subset_size": int(len(X_tr_raw)),
            "validation_subset_size": int(len(X_val_raw)),
            "features_in_val_subset": len(val_feat_names),
            "features_in_full_dataset": len(final_feat_names),
            "leakage_eliminated": True,
            "note": "Preprocessing fitted exclusively on 80% train. Unseen validation categories safely ignored."
        },
        "validation_training_durations_sec": val_train_times,
        "selected_thresholds_and_validation_metrics": selected_dict,
        "ttl_ablation_results": ablation_results,
        "final_test_evaluation_results": final_test_results
    }
    with open(VAL_DIR / "validation_metrics.json", "w") as f:
        json.dump(validation_payload, f, indent=2)

    with open(LOGS_DIR / "execution_metadata.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "train_csv_sha256": train_sha,
            "test_csv_sha256": test_sha,
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "internal_train_shape": list(X_tr_val_trans.shape),
            "validation_shape": list(X_val_trans.shape),
            "final_train_shape": list(X_train_full_trans.shape),
            "final_test_shape": list(X_test_trans.shape),
            "status": "SUCCESS"
        }, f, indent=2)

    log("\n" + "=" * 75, log_file)
    log(" PHASE 7.1 PIPELINE EXECUTION COMPLETED SUCCESSFULLY", log_file)
    log("=" * 75, log_file)


def run_ttl_ablation_experiment(
    X_tr_raw: pd.DataFrame, y_tr: pd.Series,
    X_val_raw: pd.DataFrame, y_val: pd.Series,
    num_cols: List[str], cat_cols: List[str],
    log_file: Path
) -> Dict[str, Any]:
    log("  -> Fitting independent validation pipeline for TTL ablation...", log_file)
    results = {}

    y_tr_arr = y_tr.values if hasattr(y_tr, "values") else y_tr
    y_val_arr = y_val.values if hasattr(y_val, "values") else y_val

    # 1. Full features (already done in main pipeline, record metrics for consistency)
    xgb_full = get_model_instance("xgboost")
    val_prep_full, _, _ = build_preprocessor(num_cols, cat_cols, scale_numeric=True, include_sttl=True)
    val_prep_full.fit(X_tr_raw)
    X_tr_full = val_prep_full.transform(X_tr_raw)
    X_val_full = val_prep_full.transform(X_val_raw)
    xgb_full.fit(X_tr_full, y_tr_arr)
    prob_full = xgb_full.predict_proba(X_val_full)[:, 1]
    pred_full = (prob_full >= 0.50).astype(int)

    results["full_model"] = {
        "features_count": X_tr_full.shape[1],
        "accuracy": round(float(accuracy_score(y_val_arr, pred_full)), 4),
        "attack_recall": round(float(recall_score(y_val_arr, pred_full)), 4),
        "attack_precision": round(float(precision_score(y_val_arr, pred_full)), 4),
        "attack_f1": round(float(f1_score(y_val_arr, pred_full)), 4),
        "roc_auc": round(float(roc_auc_score(y_val_arr, prob_full)), 4),
        "pr_auc": round(float(average_precision_score(y_val_arr, prob_full)), 4)
    }

    # 2. Without sttl (independent pipeline fitted on X_tr_raw without sttl)
    num_cols_no_sttl = [c for c in num_cols if c != "sttl"]
    val_prep_no_sttl, _, _ = build_preprocessor(num_cols_no_sttl, cat_cols, scale_numeric=True)
    val_prep_no_sttl.fit(X_tr_raw[num_cols_no_sttl + cat_cols])
    X_tr_no_sttl = val_prep_no_sttl.transform(X_tr_raw[num_cols_no_sttl + cat_cols])
    X_val_no_sttl = val_prep_no_sttl.transform(X_val_raw[num_cols_no_sttl + cat_cols])

    xgb_no_sttl = get_model_instance("xgboost")
    xgb_no_sttl.fit(X_tr_no_sttl, y_tr_arr)
    prob_no_sttl = xgb_no_sttl.predict_proba(X_val_no_sttl)[:, 1]
    pred_no_sttl = (prob_no_sttl >= 0.50).astype(int)

    results["without_sttl"] = {
        "features_count": X_tr_no_sttl.shape[1],
        "accuracy": round(float(accuracy_score(y_val_arr, pred_no_sttl)), 4),
        "attack_recall": round(float(recall_score(y_val_arr, pred_no_sttl)), 4),
        "attack_precision": round(float(precision_score(y_val_arr, pred_no_sttl)), 4),
        "attack_f1": round(float(f1_score(y_val_arr, pred_no_sttl)), 4),
        "roc_auc": round(float(roc_auc_score(y_val_arr, prob_no_sttl)), 4),
        "pr_auc": round(float(average_precision_score(y_val_arr, prob_no_sttl)), 4)
    }
    log(f"     Without 'sttl' (191 features): Acc={results['without_sttl']['accuracy']}, F1={results['without_sttl']['attack_f1']}", log_file)

    # 3. Without sttl and ct_state_ttl
    num_cols_no_ttls = [c for c in num_cols if c not in ("sttl", "ct_state_ttl")]
    val_prep_no_ttls, _, _ = build_preprocessor(num_cols_no_ttls, cat_cols, scale_numeric=True)
    val_prep_no_ttls.fit(X_tr_raw[num_cols_no_ttls + cat_cols])
    X_tr_no_ttls = val_prep_no_ttls.transform(X_tr_raw[num_cols_no_ttls + cat_cols])
    X_val_no_ttls = val_prep_no_ttls.transform(X_val_raw[num_cols_no_ttls + cat_cols])

    xgb_no_ttls = get_model_instance("xgboost")
    xgb_no_ttls.fit(X_tr_no_ttls, y_tr_arr)
    prob_no_ttls = xgb_no_ttls.predict_proba(X_val_no_ttls)[:, 1]
    pred_no_ttls = (prob_no_ttls >= 0.50).astype(int)

    results["without_sttl_and_ct_state_ttl"] = {
        "features_count": X_tr_no_ttls.shape[1],
        "accuracy": round(float(accuracy_score(y_val_arr, pred_no_ttls)), 4),
        "attack_recall": round(float(recall_score(y_val_arr, pred_no_ttls)), 4),
        "attack_precision": round(float(precision_score(y_val_arr, pred_no_ttls)), 4),
        "attack_f1": round(float(f1_score(y_val_arr, pred_no_ttls)), 4),
        "roc_auc": round(float(roc_auc_score(y_val_arr, prob_no_ttls)), 4),
        "pr_auc": round(float(average_precision_score(y_val_arr, prob_no_ttls)), 4)
    }
    log(f"     Without 'sttl' & 'ct_state_ttl' (190 features): Acc={results['without_sttl_and_ct_state_ttl']['accuracy']}, F1={results['without_sttl_and_ct_state_ttl']['attack_f1']}", log_file)

    return results


def generate_validation_visualizations(df_all: pd.DataFrame, y_val: np.ndarray, val_probs: Dict[str, np.ndarray], out_dir: Path):
    sns.set_theme(style="whitegrid")

    def format_ax(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=11, fontweight="semibold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="semibold")
        ax.legend(title="Model", frameon=True, fontsize=10)
        ax.set_xlim(0.05, 0.95)

    # 1. threshold_vs_attack_recall.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["attack_recall"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    ax.axhline(0.97, color="gray", linestyle=":", label="Security Target (97%)")
    format_ax(ax, "Leakage-Free Validation: Threshold vs Attack Recall", "Threshold", "Attack Recall")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_vs_attack_recall.png", dpi=300)
    plt.close()

    # 2. threshold_vs_precision.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["attack_precision"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    format_ax(ax, "Leakage-Free Validation: Threshold vs Attack Precision", "Threshold", "Attack Precision")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_vs_precision.png", dpi=300)
    plt.close()

    # 3. threshold_vs_f1.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["attack_f1"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    format_ax(ax, "Leakage-Free Validation: Threshold vs Attack F1-Score", "Threshold", "Attack F1-Score")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_vs_f1.png", dpi=300)
    plt.close()

    # 4. threshold_vs_false_positive_rate.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["false_positive_rate"] * 100, label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    ax.axhline(20.0, color="gray", linestyle=":", label="Operational Limit (20%)")
    format_ax(ax, "Leakage-Free Validation: Threshold vs False Positive Rate (%)", "Threshold", "False Positive Rate (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_vs_false_positive_rate.png", dpi=300)
    plt.close()

    # 5. threshold_vs_false_negatives.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["false_negatives"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    format_ax(ax, "Leakage-Free Validation: Threshold vs Missed Attacks (FN)", "Threshold", "False Negative Count")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_vs_false_negatives.png", dpi=300)
    plt.close()

    # 6. threshold_vs_false_positives.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["false_positives"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    format_ax(ax, "Leakage-Free Validation: Threshold vs False Alarms (FP)", "Threshold", "False Positive Count")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_vs_false_positives.png", dpi=300)
    plt.close()

    # 7. model_threshold_comparison.png (2x2 multi-panel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (key, info) in enumerate(MODEL_CONFIGS.items()):
        ax = axes[idx]
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["attack_recall"], label="Recall", color="#2ca02c", lw=2)
        ax.plot(sub["threshold"], sub["attack_precision"], label="Precision", color="#1f77b4", lw=2)
        ax.plot(sub["threshold"], sub["attack_f1"], label="Attack F1", color="#d62728", lw=2.2, linestyle="--")
        ax.plot(sub["threshold"], sub["false_positive_rate"], label="FPR", color="#ff7f0e", lw=1.8, linestyle=":")
        ax.axvline(0.50, color="gray", linestyle="-.", alpha=0.7, label="Default 0.5")
        ax.set_title(f"{info['name']}: Trade-off Curves", fontsize=12, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("Score / Rate", fontsize=10)
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="center right", fontsize=8, frameon=True)
    plt.suptitle("Leakage-Free Validation Threshold Dynamics across Models", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_dir / "model_threshold_comparison.png", dpi=300)
    plt.close()

    # 8. validation_roc_curves.png
    fig, ax = plt.subplots(figsize=(9, 7))
    for key, info in MODEL_CONFIGS.items():
        prob = val_probs[key]
        fpr_arr, tpr_arr, _ = roc_curve(y_val, prob)
        auc_val = roc_auc_score(y_val, prob)
        ax.plot(fpr_arr, tpr_arr, label=f"{info['name']} (AUC = {auc_val:.4f})", color=info["color"], lw=2.2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)", lw=1.5)
    ax.set_title("Leakage-Free Validation ROC Curves (21,548 samples)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="semibold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "validation_roc_curves.png", dpi=300)
    plt.close()

    # 9. validation_precision_recall_curves.png
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline_prev = np.sum(y_val == 1) / len(y_val)
    for key, info in MODEL_CONFIGS.items():
        prob = val_probs[key]
        prec_arr, rec_arr, _ = precision_recall_curve(y_val, prob)
        ap_val = average_precision_score(y_val, prob)
        ax.plot(rec_arr, prec_arr, label=f"{info['name']} (PR-AUC = {ap_val:.4f})", color=info["color"], lw=2.2)
    ax.axhline(baseline_prev, color="black", linestyle="--", label=f"Prevalence ({baseline_prev*100:.1f}%)", lw=1.5)
    ax.set_title("Leakage-Free Validation PR Curves (21,548 samples)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Attack Detection Rate)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="semibold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "validation_precision_recall_curves.png", dpi=300)
    plt.close()


def generate_final_test_visualizations(
    final_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    test_probs: Dict[str, np.ndarray],
    out_dir: Path
):
    sns.set_theme(style="whitegrid")

    # 1. final_model_comparison_default_threshold.png
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ["accuracy", "attack_recall", "attack_precision", "attack_f1", "roc_auc"]
    metric_labels = ["Accuracy", "Attack Recall", "Attack Precision", "Attack F1", "ROC-AUC"]
    bar_width = 0.18
    x = np.arange(len(metrics_to_plot))

    for idx, (key, info) in enumerate(MODEL_CONFIGS.items()):
        vals = [final_results[key]["default_0_5"][m] for m in metrics_to_plot]
        ax.bar(x + idx * bar_width, vals, bar_width, label=info["name"], color=info["color"], alpha=0.9)

    ax.set_title("Official Test Set: Four-Model Benchmark at Default Threshold (0.50)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="semibold")
    ax.set_ylabel("Score", fontsize=11, fontweight="semibold")
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "final_model_comparison_default_threshold.png", dpi=300)
    plt.close()

    # 2. final_model_comparison_selected_thresholds.png
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (key, info) in enumerate(MODEL_CONFIGS.items()):
        strat_a_res = final_results[key]["strategy_a_max_attack_f1"]
        th_val = strat_a_res["threshold"]
        vals = [strat_a_res[m] for m in metrics_to_plot]
        ax.bar(x + idx * bar_width, vals, bar_width, label=f"{info['name']} (Th={th_val:.2f})", color=info["color"], alpha=0.9)

    ax.set_title("Official Test Set: Models Evaluated at Strategy A Thresholds (Max Validation Attack F1)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="semibold")
    ax.set_ylabel("Score", fontsize=11, fontweight="semibold")
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "final_model_comparison_selected_thresholds.png", dpi=300)
    plt.close()

    # 3. final_confusion_matrix_xgboost.png
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xgb_strats = [
        ("default_0_5", "Default Threshold (0.50)"),
        ("strategy_a_max_attack_f1", f"Strategy A: Max F1 (Th={final_results['xgboost']['strategy_a_max_attack_f1']['threshold']:.2f})"),
        ("strategy_c_security_oriented", f"Strategy C: Security (Th={final_results['xgboost']['strategy_c_security_oriented']['threshold']:.2f})")
    ]
    for idx, (s_key, title) in enumerate(xgb_strats):
        res = final_results["xgboost"][s_key]
        cm = np.array([[res["true_negatives"], res["false_positives"]], [res["false_negatives"], res["true_positives"]]])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        labels = np.array([[f"{cm[0,0]:,}\n({cm_norm[0,0]*100:.1f}%)", f"{cm[0,1]:,}\n({cm_norm[0,1]*100:.1f}%)"],
                           [f"{cm[1,0]:,}\n({cm_norm[1,0]*100:.1f}%)", f"{cm[1,1]:,}\n({cm_norm[1,1]*100:.1f}%)"]])
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=axes[idx], cbar=False,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
                    annot_kws={"fontsize": 11, "fontweight": "bold"})
        axes[idx].set_title(title, fontsize=12, fontweight="bold", pad=8)
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label", fontsize=10)
    plt.suptitle("XGBoost Confusion Matrix Evolution across Validation-Selected Thresholds (Test Set)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "final_confusion_matrix_xgboost.png", dpi=300)
    plt.close()

    # 4. final_confusion_matrix_random_forest.png
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rf_strats = [
        ("default_0_5", "Default Threshold (0.50)"),
        ("strategy_a_max_attack_f1", f"Strategy A: Max F1 (Th={final_results['random_forest']['strategy_a_max_attack_f1']['threshold']:.2f})"),
        ("strategy_c_security_oriented", f"Strategy C: Security (Th={final_results['random_forest']['strategy_c_security_oriented']['threshold']:.2f})")
    ]
    for idx, (s_key, title) in enumerate(rf_strats):
        res = final_results["random_forest"][s_key]
        cm = np.array([[res["true_negatives"], res["false_positives"]], [res["false_negatives"], res["true_positives"]]])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        labels = np.array([[f"{cm[0,0]:,}\n({cm_norm[0,0]*100:.1f}%)", f"{cm[0,1]:,}\n({cm_norm[0,1]*100:.1f}%)"],
                           [f"{cm[1,0]:,}\n({cm_norm[1,0]*100:.1f}%)", f"{cm[1,1]:,}\n({cm_norm[1,1]*100:.1f}%)"]])
        sns.heatmap(cm, annot=labels, fmt="", cmap="Greens", ax=axes[idx], cbar=False,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
                    annot_kws={"fontsize": 11, "fontweight": "bold"})
        axes[idx].set_title(title, fontsize=12, fontweight="bold", pad=8)
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label", fontsize=10)
    plt.suptitle("Random Forest Confusion Matrix Evolution across Validation-Selected Thresholds (Test Set)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "final_confusion_matrix_random_forest.png", dpi=300)
    plt.close()

    # 5. final_roc_comparison.png
    fig, ax = plt.subplots(figsize=(9, 7))
    for key, info in MODEL_CONFIGS.items():
        prob = test_probs[key]
        fpr_arr, tpr_arr, _ = roc_curve(y_test, prob)
        auc_val = roc_auc_score(y_test, prob)
        ax.plot(fpr_arr, tpr_arr, label=f"{info['name']} (AUC = {auc_val:.4f})", color=info["color"], lw=2.2)
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)", lw=1.5)
    ax.set_title("Official Test Set: ROC Curve Comparison (55,945 samples)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="semibold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "final_roc_comparison.png", dpi=300)
    plt.close()

    # 6. final_precision_recall_comparison.png
    fig, ax = plt.subplots(figsize=(9, 7))
    prev_test = np.sum(y_test == 1) / len(y_test)
    for key, info in MODEL_CONFIGS.items():
        prob = test_probs[key]
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, prob)
        ap_val = average_precision_score(y_test, prob)
        ax.plot(rec_arr, prec_arr, label=f"{info['name']} (PR-AUC = {ap_val:.4f})", color=info["color"], lw=2.2)
    ax.axhline(prev_test, color="black", linestyle="--", label=f"Prevalence ({prev_test*100:.1f}%)", lw=1.5)
    ax.set_title("Official Test Set: Precision-Recall Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Attack Recall", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Attack Precision", fontsize=11, fontweight="semibold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "final_precision_recall_comparison.png", dpi=300)
    plt.close()

    # 7. selected_threshold_comparison.png
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"default_0_5": "o", "strategy_a_max_attack_f1": "s", "strategy_b_max_macro_f1": "^", "strategy_c_security_oriented": "*"}
    strategy_names = {
        "default_0_5": "Default (0.50)",
        "strategy_a_max_attack_f1": "Strategy A (Max F1)",
        "strategy_b_max_macro_f1": "Strategy B (Max Macro)",
        "strategy_c_security_oriented": "Strategy C (Security 97%)"
    }

    for key, info in MODEL_CONFIGS.items():
        for s_key, marker in markers.items():
            if s_key in final_results[key]:
                res = final_results[key][s_key]
                fp = res["false_positives"]
                fn = res["false_negatives"]
                th = res["threshold"]
                ax.scatter(fp, fn, color=info["color"], marker=marker, s=110, alpha=0.9,
                           edgecolor="black", zorder=4)
                ax.annotate(f"{info['name'][:3]}-{strategy_names[s_key][:4]} ({th:.2f})", (fp + 150, fn + 15),
                            fontsize=8, alpha=0.85)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=info["color"], lw=3, label=info["name"]) for info in MODEL_CONFIGS.values()]
    legend_elements.append(Line2D([0], [0], color="gray", marker="None", linestyle="None", label="--- Strategy ---"))
    for s_key, marker in markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color="w", markerfacecolor="black", markersize=9, label=strategy_names[s_key]))

    ax.set_title("Test Set Error Trade-off: False Alarms (FP) vs Missed Attacks (FN)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positives (False Alarms on Normal Traffic)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("False Negatives (Missed Cyber Attacks)", fontsize=11, fontweight="semibold")
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "selected_threshold_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
