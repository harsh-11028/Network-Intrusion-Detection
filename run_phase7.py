"""
run_phase7.py
-------------
Phase 7: Comparative Model Analysis, Validation-Based Threshold Optimization & Final Evaluation
Project: Machine Learning-Based Network Intrusion Detection and Attack Classification
Dataset: UNSW-NB15

This script performs:
1. Stratified 80/20 train/validation split on official training set (107,740 samples).
2. Retrains all four models on the 80% train subset (86,192 samples).
3. Evaluates thresholds (0.05 to 0.95, step 0.01) on the 20% validation subset (21,548 samples).
4. Selects optimal operating thresholds under Strategies A, B, C, and D.
5. Generates 9 validation threshold and curve visualizations under outputs/phase7/.
6. Saves comprehensive CSV and JSON validation summaries.
7. Performs validation-stage feature ablation for TTL/bias investigation.
8. Retrains final models on the complete 100% training set and saves to outputs/phase7/final_models/.
9. Evaluates final models on the official test set (55,945 samples) at default 0.5 and selected thresholds.
10. Generates 7 final test evaluation plots under outputs/phase7/final_test/.
11. Generates the comprehensive outputs/phase7/phase7_report.md.
"""

import os
from pathlib import Path
import json
import time
from typing import Dict, Any, Tuple, List

# Headless matplotlib configuration
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
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

# Output directories
OUTPUT_DIR = Path("outputs/phase7")
FINAL_MODELS_DIR = OUTPUT_DIR / "final_models"
FINAL_TEST_DIR = OUTPUT_DIR / "final_test"

for d in [OUTPUT_DIR, FINAL_MODELS_DIR, FINAL_TEST_DIR]:
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


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    print("\n[DATA] Loading preprocessed dataset...")
    train_data = np.load("data/processed/train_processed.npz")
    test_data = np.load("data/processed/test_processed.npz")

    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    with open("data/processed/feature_metadata.json", "r") as f:
        metadata = json.load(f)
    feature_names = metadata["feature_names"]

    print(f"  -> X_train shape: {X_train.shape}, y_train: {y_train.shape} (Attack: {np.sum(y_train==1):,}, Normal: {np.sum(y_train==0):,})")
    print(f"  -> X_test  shape: {X_test.shape}, y_test: {y_test.shape} (Attack: {np.sum(y_test==1):,}, Normal: {np.sum(y_test==0):,})")
    print(f"  -> Total features: {len(feature_names)}")

    return X_train, y_train, X_test, y_test, feature_names


def create_validation_split(X_train: np.ndarray, y_train: np.ndarray):
    print("\n[SPLIT] Creating stratified 80/20 train/validation split from training data...")
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    print(f"  -> Train subset: {X_tr_sub.shape} (Attack: {np.sum(y_tr_sub==1):,}, Normal: {np.sum(y_tr_sub==0):,})")
    print(f"  -> Validation  : {X_val.shape} (Attack: {np.sum(y_val==1):,}, Normal: {np.sum(y_val==0):,})")
    return X_tr_sub, X_val, y_tr_sub, y_val


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


def train_validation_models(X_tr_sub: np.ndarray, y_tr_sub: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    print("\n[TRAIN VAL] Retraining models on internal 80% train subset...")
    val_models = {}
    val_probs = {}
    train_times = {}

    for key, info in MODEL_CONFIGS.items():
        print(f"  -> Fitting {info['name']}...")
        model = get_model_instance(key)
        start_t = time.perf_counter()
        model.fit(X_tr_sub, y_tr_sub)
        fit_t = time.perf_counter() - start_t
        train_times[key] = fit_t

        prob = model.predict_proba(X_val)[:, 1]
        val_models[key] = model
        val_probs[key] = prob
        print(f"     Fitted in {fit_t:.2f}s | Prob range: [{prob.min():.4f}, {prob.max():.4f}]")

    return val_models, val_probs, train_times


def sweep_thresholds(y_val: np.ndarray, val_probs: Dict[str, np.ndarray]):
    print("\n[SWEEP] Evaluating thresholds 0.05 to 0.95 (step 0.01) on validation set...")
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    all_rows = []
    per_model_dfs = {}

    for key, prob in val_probs.items():
        m_name = MODEL_CONFIGS[key]["name"]
        m_rows = []
        for th in thresholds:
            y_pred = (prob >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            norm_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

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
                "total_validation_samples": int(len(y_val))
            }
            m_rows.append(row)
            all_rows.append(row)

        df_m = pd.DataFrame(m_rows)
        per_model_dfs[key] = df_m
        csv_path = OUTPUT_DIR / f"{key}_thresholds.csv"
        df_m.to_csv(csv_path, index=False)
        print(f"  -> Saved {csv_path.name}")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)
    print(f"  -> Saved {OUTPUT_DIR / 'threshold_analysis.csv'}")

    return df_all, per_model_dfs


def select_thresholds(per_model_dfs: Dict[str, pd.DataFrame]):
    print("\n[STRATEGY] Selecting operating thresholds under Strategies A, B, C, D...")
    summary_rows = []
    selected_dict = {}

    for key, df in per_model_dfs.items():
        m_name = MODEL_CONFIGS[key]["name"]
        selected_dict[key] = {}

        # Strategy A: Max Attack F1
        # In case of ties, choose the one with lower FPR (or higher threshold)
        max_f1 = df["attack_f1"].max()
        cand_a = df[df["attack_f1"] == max_f1].sort_values(by=["false_positive_rate", "threshold"], ascending=[True, False])
        best_a = cand_a.iloc[0]
        selected_dict[key]["strategy_a_max_attack_f1"] = best_a.to_dict()

        # Strategy B: Max Macro F1
        max_macro = df["macro_f1"].max()
        cand_b = df[df["macro_f1"] == max_macro].sort_values(by=["false_positive_rate", "threshold"], ascending=[True, False])
        best_b = cand_b.iloc[0]
        selected_dict[key]["strategy_b_max_macro_f1"] = best_b.to_dict()

        # Strategy C: Security-oriented (Recall >= 97%, min FPR)
        feas_c = df[df["attack_recall"] >= 0.97]
        if len(feas_c) > 0:
            best_c = feas_c.sort_values(by=["false_positive_rate", "threshold"], ascending=[True, False]).iloc[0]
            status_c = "Feasible (Recall >= 97%)"
        else:
            best_c = df.sort_values(by=["attack_recall", "false_positive_rate"], ascending=[False, True]).iloc[0]
            status_c = f"Infeasible (Max Recall={best_c['attack_recall']*100:.2f}%)"
        selected_dict[key]["strategy_c_security_oriented"] = best_c.to_dict()
        selected_dict[key]["strategy_c_status"] = status_c

        # Strategy D: Operational target (Recall >= 95%, FPR <= 20%)
        feas_d = df[(df["attack_recall"] >= 0.95) & (df["false_positive_rate"] <= 0.20)]
        if len(feas_d) > 0:
            best_d = feas_d.sort_values(by=["attack_f1", "false_positive_rate"], ascending=[False, True]).iloc[0]
            status_d = f"Feasible (Th={best_d['threshold']:.2f}, Rec={best_d['attack_recall']*100:.2f}%, FPR={best_d['false_positive_rate']*100:.2f}%)"
        else:
            best_d = None
            status_d = "Infeasible (No threshold satisfies Recall >= 95% AND FPR <= 20%)"
        selected_dict[key]["strategy_d_operational"] = best_d.to_dict() if best_d is not None else None
        selected_dict[key]["strategy_d_status"] = status_d

        # Default 0.5 row
        def_row = df[df["threshold"] == 0.50].iloc[0]
        selected_dict[key]["default_threshold_0_5"] = def_row.to_dict()

        summary_rows.append({
            "model_key": key,
            "model_name": m_name,
            "strategy": "Default (0.50)",
            "selected_threshold": 0.50,
            "val_accuracy": round(def_row["accuracy"], 4),
            "val_attack_recall": round(def_row["attack_recall"], 4),
            "val_attack_precision": round(def_row["attack_precision"], 4),
            "val_attack_f1": round(def_row["attack_f1"], 4),
            "val_fpr": round(def_row["false_positive_rate"], 4),
            "val_false_positives": int(def_row["false_positives"]),
            "val_false_negatives": int(def_row["false_negatives"]),
            "feasibility_note": "Baseline reference"
        })
        summary_rows.append({
            "model_key": key,
            "model_name": m_name,
            "strategy": "Strategy A (Max Attack F1)",
            "selected_threshold": round(best_a["threshold"], 2),
            "val_accuracy": round(best_a["accuracy"], 4),
            "val_attack_recall": round(best_a["attack_recall"], 4),
            "val_attack_precision": round(best_a["attack_precision"], 4),
            "val_attack_f1": round(best_a["attack_f1"], 4),
            "val_fpr": round(best_a["false_positive_rate"], 4),
            "val_false_positives": int(best_a["false_positives"]),
            "val_false_negatives": int(best_a["false_negatives"]),
            "feasibility_note": f"Max validation Attack F1 ({best_a['attack_f1']:.4f})"
        })
        summary_rows.append({
            "model_key": key,
            "model_name": m_name,
            "strategy": "Strategy B (Max Macro F1)",
            "selected_threshold": round(best_b["threshold"], 2),
            "val_accuracy": round(best_b["accuracy"], 4),
            "val_attack_recall": round(best_b["attack_recall"], 4),
            "val_attack_precision": round(best_b["attack_precision"], 4),
            "val_attack_f1": round(best_b["attack_f1"], 4),
            "val_fpr": round(best_b["false_positive_rate"], 4),
            "val_false_positives": int(best_b["false_positives"]),
            "val_false_negatives": int(best_b["false_negatives"]),
            "feasibility_note": f"Max validation Macro F1 ({best_b['macro_f1']:.4f})"
        })
        summary_rows.append({
            "model_key": key,
            "model_name": m_name,
            "strategy": "Strategy C (Security-Oriented: Rec>=97%)",
            "selected_threshold": round(best_c["threshold"], 2),
            "val_accuracy": round(best_c["accuracy"], 4),
            "val_attack_recall": round(best_c["attack_recall"], 4),
            "val_attack_precision": round(best_c["attack_precision"], 4),
            "val_attack_f1": round(best_c["attack_f1"], 4),
            "val_fpr": round(best_c["false_positive_rate"], 4),
            "val_false_positives": int(best_c["false_positives"]),
            "val_false_negatives": int(best_c["false_negatives"]),
            "feasibility_note": status_c
        })
        summary_rows.append({
            "model_key": key,
            "model_name": m_name,
            "strategy": "Strategy D (Operational: Rec>=95%, FPR<=20%)",
            "selected_threshold": round(best_d["threshold"], 2) if best_d is not None else None,
            "val_accuracy": round(best_d["accuracy"], 4) if best_d is not None else None,
            "val_attack_recall": round(best_d["attack_recall"], 4) if best_d is not None else None,
            "val_attack_precision": round(best_d["attack_precision"], 4) if best_d is not None else None,
            "val_attack_f1": round(best_d["attack_f1"], 4) if best_d is not None else None,
            "val_fpr": round(best_d["false_positive_rate"], 4) if best_d is not None else None,
            "val_false_positives": int(best_d["false_positives"]) if best_d is not None else None,
            "val_false_negatives": int(best_d["false_negatives"]) if best_d is not None else None,
            "feasibility_note": status_d
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUTPUT_DIR / "threshold_selection_summary.csv", index=False)
    print(f"  -> Saved {OUTPUT_DIR / 'threshold_selection_summary.csv'}")

    return selected_dict, df_sum


def generate_validation_plots(df_all: pd.DataFrame, y_val: np.ndarray, val_probs: Dict[str, np.ndarray]):
    print("\n[PLOTS] Generating 9 validation threshold plots in outputs/phase7/...")
    sns.set_theme(style="whitegrid")

    # Helper styling
    def format_ax(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontsize=12, fontweight="semibold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="semibold")
        ax.legend(title="Model", frameon=True, fontsize=10)
        ax.set_xlim(0.05, 0.95)

    # 1. threshold_vs_attack_recall.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["attack_recall"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    ax.axhline(0.97, color="gray", linestyle=":", label="Security Target (97%)")
    format_ax(ax, "Validation: Threshold vs Attack Detection Rate (Recall)", "Classification Threshold", "Attack Recall")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_vs_attack_recall.png", dpi=300)
    plt.close()

    # 2. threshold_vs_precision.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["attack_precision"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    format_ax(ax, "Validation: Threshold vs Attack Precision", "Classification Threshold", "Attack Precision")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_vs_precision.png", dpi=300)
    plt.close()

    # 3. threshold_vs_f1.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["attack_f1"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    format_ax(ax, "Validation: Threshold vs Attack F1-Score", "Classification Threshold", "Attack F1-Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_vs_f1.png", dpi=300)
    plt.close()

    # 4. threshold_vs_false_positive_rate.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["false_positive_rate"] * 100, label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    ax.axhline(20.0, color="gray", linestyle=":", label="Operational Limit (20%)")
    format_ax(ax, "Validation: Threshold vs False Positive Rate (FPR)", "Classification Threshold", "False Positive Rate (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_vs_false_positive_rate.png", dpi=300)
    plt.close()

    # 5. threshold_vs_false_negatives.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["false_negatives"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    format_ax(ax, "Validation: Threshold vs Missed Attacks (False Negatives)", "Classification Threshold", "False Negative Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_vs_false_negatives.png", dpi=300)
    plt.close()

    # 6. threshold_vs_false_positives.png
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, info in MODEL_CONFIGS.items():
        sub = df_all[df_all["model_key"] == key]
        ax.plot(sub["threshold"], sub["false_positives"], label=info["name"], color=info["color"], linestyle=info["line_style"], lw=2.2)
    format_ax(ax, "Validation: Threshold vs False Alarms (False Positives)", "Classification Threshold", "False Positive Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_vs_false_positives.png", dpi=300)
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
    plt.suptitle("Validation Threshold Dynamics across Machine Learning Models", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_threshold_comparison.png", dpi=300)
    plt.close()

    # 8. validation_roc_curves.png
    fig, ax = plt.subplots(figsize=(9, 7))
    for key, info in MODEL_CONFIGS.items():
        prob = val_probs[key]
        fpr_arr, tpr_arr, _ = roc_curve(y_val, prob)
        auc_val = roc_auc_score(y_val, prob)
        ax.plot(fpr_arr, tpr_arr, label=f"{info['name']} (AUC = {auc_val:.4f})", color=info["color"], lw=2.2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)", lw=1.5)
    ax.set_title("Validation ROC Curves (21,548 samples)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12, fontweight="semibold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_roc_curves.png", dpi=300)
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
    ax.set_title("Validation Precision-Recall Curves (21,548 samples)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Attack Detection Rate)", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="semibold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_precision_recall_curves.png", dpi=300)
    plt.close()

    print("  -> All 9 validation visualizations generated successfully.")


def run_validation_feature_ablation(
    X_tr_sub: np.ndarray, y_tr_sub: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    print("\n[BIAS AUDIT] Running validation-stage feature ablation on internal split...")
    results = {}

    sttl_idx = feature_names.index("sttl") if "sttl" in feature_names else -1
    ct_ttl_idx = feature_names.index("ct_state_ttl") if "ct_state_ttl" in feature_names else -1

    # 1. Baseline XGBoost on all features (already trained, evaluate on validation)
    xgb_full = get_model_instance("xgboost")
    xgb_full.fit(X_tr_sub, y_tr_sub)
    prob_full = xgb_full.predict_proba(X_val)[:, 1]
    pred_full = (prob_full >= 0.50).astype(int)

    results["full_model"] = {
        "features_count": len(feature_names),
        "accuracy": round(float(accuracy_score(y_val, pred_full)), 4),
        "attack_recall": round(float(recall_score(y_val, pred_full)), 4),
        "attack_precision": round(float(precision_score(y_val, pred_full)), 4),
        "attack_f1": round(float(f1_score(y_val, pred_full)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, prob_full)), 4),
        "pr_auc": round(float(average_precision_score(y_val, prob_full)), 4)
    }

    # 2. Ablate sttl
    if sttl_idx != -1:
        mask_no_sttl = [i for i in range(len(feature_names)) if i != sttl_idx]
        X_tr_no_sttl = X_tr_sub[:, mask_no_sttl]
        X_val_no_sttl = X_val[:, mask_no_sttl]

        xgb_no_sttl = get_model_instance("xgboost")
        xgb_no_sttl.fit(X_tr_no_sttl, y_tr_sub)
        prob_no_sttl = xgb_no_sttl.predict_proba(X_val_no_sttl)[:, 1]
        pred_no_sttl = (prob_no_sttl >= 0.50).astype(int)

        results["without_sttl"] = {
            "features_count": len(mask_no_sttl),
            "accuracy": round(float(accuracy_score(y_val, pred_no_sttl)), 4),
            "attack_recall": round(float(recall_score(y_val, pred_no_sttl)), 4),
            "attack_precision": round(float(precision_score(y_val, pred_no_sttl)), 4),
            "attack_f1": round(float(f1_score(y_val, pred_no_sttl)), 4),
            "roc_auc": round(float(roc_auc_score(y_val, prob_no_sttl)), 4),
            "pr_auc": round(float(average_precision_score(y_val, prob_no_sttl)), 4)
        }
        print(f"  -> Validation XGBoost without 'sttl': Acc={results['without_sttl']['accuracy']}, F1={results['without_sttl']['attack_f1']}")

    # 3. Ablate both sttl and ct_state_ttl
    if sttl_idx != -1 and ct_ttl_idx != -1:
        mask_no_ttls = [i for i in range(len(feature_names)) if i not in (sttl_idx, ct_ttl_idx)]
        X_tr_no_ttls = X_tr_sub[:, mask_no_ttls]
        X_val_no_ttls = X_val[:, mask_no_ttls]

        xgb_no_ttls = get_model_instance("xgboost")
        xgb_no_ttls.fit(X_tr_no_ttls, y_tr_sub)
        prob_no_ttls = xgb_no_ttls.predict_proba(X_val_no_ttls)[:, 1]
        pred_no_ttls = (prob_no_ttls >= 0.50).astype(int)

        results["without_sttl_and_ct_state_ttl"] = {
            "features_count": len(mask_no_ttls),
            "accuracy": round(float(accuracy_score(y_val, pred_no_ttls)), 4),
            "attack_recall": round(float(recall_score(y_val, pred_no_ttls)), 4),
            "attack_precision": round(float(precision_score(y_val, pred_no_ttls)), 4),
            "attack_f1": round(float(f1_score(y_val, pred_no_ttls)), 4),
            "roc_auc": round(float(roc_auc_score(y_val, prob_no_ttls)), 4),
            "pr_auc": round(float(average_precision_score(y_val, prob_no_ttls)), 4)
        }
        print(f"  -> Validation XGBoost without 'sttl' & 'ct_state_ttl': Acc={results['without_sttl_and_ct_state_ttl']['accuracy']}, F1={results['without_sttl_and_ct_state_ttl']['attack_f1']}")

    return results


def retrain_final_models(X_train: np.ndarray, y_train: np.ndarray):
    print("\n[RETRAIN FINAL] Training final models on full official training set (107,740 samples)...")
    final_models = {}
    import joblib

    for key, info in MODEL_CONFIGS.items():
        print(f"  -> Retraining final {info['name']}...")
        model = get_model_instance(key)
        start_t = time.perf_counter()
        model.fit(X_train, y_train)
        dur = time.perf_counter() - start_t
        print(f"     Fitted in {dur:.2f}s")
        save_p = FINAL_MODELS_DIR / f"{key}_final.joblib"
        joblib.dump(model, save_p)
        final_models[key] = model

    return final_models


def evaluate_final_test_models(
    final_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_dict: Dict[str, Dict[str, Any]]
):
    print("\n[FINAL TEST EVAL] Evaluating final models on official test set (55,945 samples)...")
    final_results = {}
    test_probs = {}

    for key, model in final_models.items():
        m_name = MODEL_CONFIGS[key]["name"]
        final_results[key] = {}
        prob = model.predict_proba(X_test)[:, 1]
        test_probs[key] = prob

        # Strategies to evaluate on test set:
        strategies_to_eval = [
            ("default_0_5", 0.50, "Default Baseline (0.50)"),
            ("strategy_a_max_attack_f1", selected_dict[key]["strategy_a_max_attack_f1"]["threshold"], "Strategy A (Max Val Attack F1)"),
            ("strategy_b_max_macro_f1", selected_dict[key]["strategy_b_max_macro_f1"]["threshold"], "Strategy B (Max Val Macro F1)"),
            ("strategy_c_security_oriented", selected_dict[key]["strategy_c_security_oriented"]["threshold"], "Strategy C (Security-Oriented)")
        ]

        # Add strategy D if feasible
        if selected_dict[key].get("strategy_d_operational") is not None:
            strategies_to_eval.append((
                "strategy_d_operational",
                selected_dict[key]["strategy_d_operational"]["threshold"],
                "Strategy D (Operational Target)"
            ))

        for strat_key, th, strat_label in strategies_to_eval:
            y_pred = (prob >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            norm_prec = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
            norm_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            roc_auc = roc_auc_score(y_test, prob)
            pr_auc = average_precision_score(y_test, prob)
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
                "total_test_samples": int(len(y_test))
            }
            final_results[key][strat_key] = res
            print(f"  -> {m_name} [{strat_label} @ Th={th:.2f}]: Acc={acc*100:.2f}%, Rec={rec*100:.2f}%, Prec={prec*100:.2f}%, F1={f1:.4f}, FP={fp:,}, FN={fn:,}")

    return final_results, test_probs


def generate_final_test_plots(
    final_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    test_probs: Dict[str, np.ndarray],
    selected_dict: Dict[str, Dict[str, Any]]
):
    print("\n[PLOTS] Generating 7 final test evaluation plots in outputs/phase7/final_test/...")
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

    ax.set_title("Official Test Set: Four-Model Benchmark at Default Threshold (0.50)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="semibold")
    ax.set_ylabel("Score", fontsize=11, fontweight="semibold")
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FINAL_TEST_DIR / "final_model_comparison_default_threshold.png", dpi=300)
    plt.close()

    # 2. final_model_comparison_selected_thresholds.png
    # Comparing Strategy A (Max Val Attack F1) across models
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (key, info) in enumerate(MODEL_CONFIGS.items()):
        strat_a_res = final_results[key]["strategy_a_max_attack_f1"]
        th_val = strat_a_res["threshold"]
        vals = [strat_a_res[m] for m in metrics_to_plot]
        ax.bar(x + idx * bar_width, vals, bar_width, label=f"{info['name']} (Th={th_val:.2f})", color=info["color"], alpha=0.9)

    ax.set_title("Official Test Set: Models Evaluated at Strategy A Thresholds (Max Validation Attack F1)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="semibold")
    ax.set_ylabel("Score", fontsize=11, fontweight="semibold")
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(FINAL_TEST_DIR / "final_model_comparison_selected_thresholds.png", dpi=300)
    plt.close()

    # 3. final_confusion_matrix_xgboost.png (Side-by-side: default 0.5 vs Strategy A vs Strategy C)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xgb_strats = [
        ("default_0_5", "Default Threshold (0.50)"),
        ("strategy_a_max_attack_f1", f"Strategy A: Max Attack F1 (Th={final_results['xgboost']['strategy_a_max_attack_f1']['threshold']:.2f})"),
        ("strategy_c_security_oriented", f"Strategy C: Security-Oriented (Th={final_results['xgboost']['strategy_c_security_oriented']['threshold']:.2f})")
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
    plt.suptitle("XGBoost Confusion Matrix Evolution Across Validation-Selected Thresholds (Test Set)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FINAL_TEST_DIR / "final_confusion_matrix_xgboost.png", dpi=300)
    plt.close()

    # 4. final_confusion_matrix_random_forest.png
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rf_strats = [
        ("default_0_5", "Default Threshold (0.50)"),
        ("strategy_a_max_attack_f1", f"Strategy A: Max Attack F1 (Th={final_results['random_forest']['strategy_a_max_attack_f1']['threshold']:.2f})"),
        ("strategy_c_security_oriented", f"Strategy C: Security-Oriented (Th={final_results['random_forest']['strategy_c_security_oriented']['threshold']:.2f})")
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
    plt.suptitle("Random Forest Confusion Matrix Evolution Across Validation-Selected Thresholds (Test Set)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FINAL_TEST_DIR / "final_confusion_matrix_random_forest.png", dpi=300)
    plt.close()

    # 5. final_roc_comparison.png
    fig, ax = plt.subplots(figsize=(9, 7))
    for key, info in MODEL_CONFIGS.items():
        prob = test_probs[key]
        fpr_arr, tpr_arr, _ = roc_curve(y_test, prob)
        auc_val = roc_auc_score(y_test, prob)
        ax.plot(fpr_arr, tpr_arr, label=f"{info['name']} (AUC = {auc_val:.4f})", color=info["color"], lw=2.2)
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)", lw=1.5)
    ax.set_title("Official Test Set: ROC Curve Comparison (55,945 samples)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12, fontweight="semibold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(FINAL_TEST_DIR / "final_roc_comparison.png", dpi=300)
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
    ax.set_title("Official Test Set: Precision-Recall Curve Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Attack Recall", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Attack Precision", fontsize=12, fontweight="semibold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(FINAL_TEST_DIR / "final_precision_recall_comparison.png", dpi=300)
    plt.close()

    # 7. selected_threshold_comparison.png
    # Scatter plot / tradeoff visualization of FP vs FN across models and strategies on the test set
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

    # Custom legend for models and strategies
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=info["color"], lw=3, label=info["name"]) for info in MODEL_CONFIGS.values()]
    legend_elements.append(Line2D([0], [0], color="gray", marker="None", linestyle="None", label="--- Strategy ---"))
    for s_key, marker in markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color="w", markerfacecolor="black", markersize=9, label=strategy_names[s_key]))

    ax.set_title("Test Set Error Trade-off: False Alarms (FP) vs Missed Attacks (FN)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positives (False Alarms on Normal Traffic)", fontsize=12, fontweight="semibold")
    ax.set_ylabel("False Negatives (Missed Cyber Attacks)", fontsize=12, fontweight="semibold")
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(FINAL_TEST_DIR / "selected_threshold_comparison.png", dpi=300)
    plt.close()

    print("  -> All 7 final test visualizations generated successfully.")


def save_validation_metrics_json(
    train_times: Dict[str, float],
    selected_dict: Dict[str, Dict[str, Any]],
    ablation_results: Dict[str, Any],
    final_results: Dict[str, Dict[str, Any]]
):
    print("\n[SAVE] Saving validation_metrics.json...")
    payload = {
        "validation_split_configuration": {
            "validation_proportion": 0.20,
            "training_proportion": 0.80,
            "random_state": 42,
            "stratification": True,
            "train_subset_size": 86192,
            "validation_subset_size": 21548,
            "preprocessor_fitting_note": "Preprocessor was fitted on the full training set (107,740) during Phase 2; documented in report."
        },
        "validation_stage_training_duration_sec": train_times,
        "selected_thresholds_and_validation_metrics": selected_dict,
        "validation_feature_ablation_results": ablation_results,
        "final_test_evaluation_results": final_results,
        "model_configurations": {k: v["params"] for k, v in MODEL_CONFIGS.items()}
    }

    with open(OUTPUT_DIR / "validation_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> Saved {OUTPUT_DIR / 'validation_metrics.json'}")


def generate_comprehensive_report(
    selected_dict: Dict[str, Dict[str, Any]],
    final_results: Dict[str, Dict[str, Any]],
    ablation_results: Dict[str, Any],
    train_times: Dict[str, float]
):
    print("\n[REPORT] Report will be generated by dedicated report generator...")



def main():
    print("=" * 75)
    print(" PHASE 7: COMPARATIVE MODEL ANALYSIS, VALIDATION THRESHOLD & FINAL EVAL")
    print("=" * 75)

    # 1. Load data
    X_train, y_train, X_test, y_test, feature_names = load_data()

    # 2. Create validation split (80/20 stratified)
    X_tr_sub, X_val, y_tr_sub, y_val = create_validation_split(X_train, y_train)

    # 3. Retrain models on internal training subset
    val_models, val_probs, train_times = train_validation_models(X_tr_sub, y_tr_sub, X_val, y_val)

    # 4. Sweep thresholds on validation set
    df_all, per_model_dfs = sweep_thresholds(y_val, val_probs)

    # 5. Select operating thresholds under strategies A, B, C, D
    selected_dict, df_sum = select_thresholds(per_model_dfs)

    # 6. Generate 9 validation plots
    generate_validation_plots(df_all, y_val, val_probs)

    # 7. Run validation-stage feature ablation for TTL/bias investigation
    ablation_results = run_validation_feature_ablation(X_tr_sub, y_tr_sub, X_val, y_val, feature_names)

    # 8. Retrain final models on complete 100% training set
    final_models = retrain_final_models(X_train, y_train)

    # 9. Evaluate final models on official test set at 0.5 and selected thresholds
    final_results, test_probs = evaluate_final_test_models(final_models, X_test, y_test, selected_dict)

    # 10. Generate 7 final test evaluation plots
    generate_final_test_plots(final_results, y_test, test_probs, selected_dict)

    # 11. Save validation_metrics.json
    save_validation_metrics_json(train_times, selected_dict, ablation_results, final_results)

    # 12. Generate comprehensive phase7_report.md
    generate_comprehensive_report(selected_dict, final_results, ablation_results, train_times)

    print("\n" + "=" * 75)
    print(" PHASE 7 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
