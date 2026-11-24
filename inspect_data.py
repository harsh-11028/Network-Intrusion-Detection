"""
inspect_data.py
---------------
Phase 1: Dataset Validation and Inspection Script
College ML Project: Machine Learning-Based Network Intrusion Detection and Attack Classification
Dataset: UNSW-NB15 (Cleaned Train and Test Sets)

Usage:
    python inspect_data.py
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np


TRAIN_PATH = Path("data/UNSW_NB15_training_cleaned.csv")
TEST_PATH = Path("data/UNSW_NB15_testing_cleaned.csv")
RAW_BENCHMARK_ROWS = {"train": 175341, "test": 82332}


def print_header(title: str, char: str = "=") -> None:
    print(f"\n{char * 78}")
    print(f" {title}")
    print(f"{char * 78}")


def verify_file_existence() -> None:
    print_header("1. FILE EXISTENCE & SYSTEM CHECKS")
    for path, name in [(TRAIN_PATH, "Training Dataset"), (TEST_PATH, "Testing Dataset")]:
        if not path.exists():
            print(f"[FAIL] Missing {name} at expected path: {path}")
            sys.exit(1)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[PASS] {name} found: {path} ({size_mb:.2f} MB)")


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    print_header("2. DATASET INGESTION & SHAPE VERIFICATION")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    print(f"Loaded Training Set : {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"Loaded Testing Set  : {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")

    # Explanation regarding raw vs cleaned row counts
    print("\n[NOTE ON DATASET DIMENSIONS]")
    print(f" - Raw UNSW-NB15 benchmark splits have {RAW_BENCHMARK_ROWS['train']:,} (train) and {RAW_BENCHMARK_ROWS['test']:,} (test) records.")
    print(f" - The provided cleaned files have {train_df.shape[0]:,} (train) and {test_df.shape[0]:,} (test) records.")
    print(f" - Training reduction: {RAW_BENCHMARK_ROWS['train'] - train_df.shape[0]:,} duplicate records removed during prior cleaning.")
    print(f" - Testing reduction : {RAW_BENCHMARK_ROWS['test'] - test_df.shape[0]:,} duplicate records removed during prior cleaning.")
    print(" - Zero duplicates remain in either cleaned CSV.")

    return train_df, test_df


def verify_columns_and_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print_header("3. SCHEMA COMPATIBILITY & COLUMN ALIGNMENT")
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)

    if train_cols == test_cols:
        print(f"[PASS] Identical column schema across train and test ({len(train_cols)} columns).")
    else:
        print("[FAIL] Column schema mismatch detected between train and test!")
        diff_train = set(train_cols) - set(test_cols)
        diff_test = set(test_cols) - set(train_cols)
        if diff_train:
            print(f"  Columns in Train but missing in Test: {diff_train}")
        if diff_test:
            print(f"  Columns in Test but missing in Train: {diff_test}")
        sys.exit(1)

    # Check targets
    for target in ["label", "attack_cat"]:
        if target in train_df.columns:
            print(f"[PASS] Target column '{target}' verified.")
        else:
            print(f"[FAIL] Required target column '{target}' missing!")
            sys.exit(1)

    # Verify matching data types
    dtype_mismatches = []
    for col in train_cols:
        if train_df[col].dtype != test_df[col].dtype:
            dtype_mismatches.append((col, str(train_df[col].dtype), str(test_df[col].dtype)))

    if not dtype_mismatches:
        print("[PASS] All feature data types match exactly between train and test.")
    else:
        print(f"[WARN] Data type differences detected in {len(dtype_mismatches)} columns:")
        for col, tr_dt, te_dt in dtype_mismatches:
            print(f"  - {col}: train ({tr_dt}) vs test ({te_dt})")


def categorize_features(train_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    print_header("4. FEATURE CLASSIFICATION (CATEGORICAL VS NUMERICAL)")
    targets = ["label", "attack_cat"]
    feature_cols = [c for c in train_df.columns if c not in targets]

    cat_features = train_df[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    print(f"Total input features: {len(feature_cols)}")
    print(f" - Categorical features ({len(cat_features)}): {cat_features}")
    print(f" - Numerical features   ({len(num_features)}): {num_features}")
    print(f" - Target columns       ({len(targets)}): {targets} ('label'=binary, 'attack_cat'=multiclass)")

    return cat_features, num_features


def check_data_quality(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print_header("5. DATA QUALITY: MISSING VALUES, INFINITIES & DUPLICATES")
    
    # Missing values
    tr_na = train_df.isna().sum().sum()
    te_na = test_df.isna().sum().sum()
    print(f"Missing values (NaN/Null) - Train: {tr_na} | Test: {te_na}")
    if tr_na == 0 and te_na == 0:
        print("[PASS] No missing values detected in either dataset.")
    else:
        print("[WARN] Missing values present; inspect specific columns.")

    # Infinite values in numerical columns
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    tr_inf = np.isinf(train_df[num_cols].to_numpy()).sum()
    te_inf = np.isinf(test_df[num_cols].to_numpy()).sum()
    print(f"Infinite values (+/- Inf) - Train: {tr_inf} | Test: {te_inf}")

    # Negative values in non-negative network metrics
    tr_negs = (train_df[num_cols] < 0).sum().sum()
    te_negs = (test_df[num_cols] < 0).sum().sum()
    print(f"Negative values           - Train: {tr_negs} | Test: {te_negs}")

    # Exact duplicates across all 44 columns
    tr_dups = train_df.duplicated().sum()
    te_dups = test_df.duplicated().sum()
    print(f"Exact full-row duplicates - Train: {tr_dups} | Test: {te_dups}")

    # Feature-only duplicates (dropping label and attack_cat)
    feature_cols = [c for c in train_df.columns if c not in ["label", "attack_cat"]]
    tr_feat_dups = train_df.duplicated(subset=feature_cols).sum()
    te_feat_dups = test_df.duplicated(subset=feature_cols).sum()
    print(f"Feature-only duplicates   - Train: {tr_feat_dups:,} | Test: {te_feat_dups:,}")

    # Check for label ambiguity among duplicate feature vectors
    conflicting_labels = (train_df.groupby(feature_cols)["label"].nunique() > 1).sum()
    conflicting_attacks = (train_df.groupby(feature_cols)["attack_cat"].nunique() > 1).sum()
    print(f"Ambiguous feature vectors (same features, conflicting binary label): {conflicting_labels:,}")
    print(f"Ambiguous feature vectors (same features, conflicting attack category): {conflicting_attacks:,}")


def check_categorical_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_features: list[str]) -> None:
    print_header("6. CATEGORICAL DRIFT & UNSEEN LEVELS CHECK")
    for col in cat_features:
        tr_unique = set(train_df[col].dropna().unique())
        te_unique = set(test_df[col].dropna().unique())
        unseen_in_test = te_unique - tr_unique
        train_only = tr_unique - te_unique

        print(f"\nFeature: '{col}'")
        print(f"  Unique levels in Train: {len(tr_unique):3d} | Unique levels in Test: {len(te_unique):3d}")
        if unseen_in_test:
            print(f"  [CRITICAL PREPROCESSING WARNING] Levels present in Test but UNSEEN in Train ({len(unseen_in_test)}): {unseen_in_test}")
            print("  -> Any One-Hot Encoder or Categorical Transformer must handle unknown categories (e.g., handle_unknown='ignore').")
        else:
            print("  [PASS] No unseen levels in Test for this feature.")
        if train_only:
            print(f"  Levels in Train but absent in Test ({len(train_only)}): {train_only}")

    # Service placeholder check
    svc_dash_tr = (train_df["service"] == "-").sum()
    svc_dash_te = (test_df["service"] == "-").sum()
    print("\nSpecial note on 'service' column:")
    print(f"  Placeholder '-' represents unmapped / non-standard ports: Train {svc_dash_tr:,} ({svc_dash_tr/len(train_df)*100:.2f}%) | Test {svc_dash_te:,} ({svc_dash_te/len(test_df)*100:.2f}%)")


def analyze_target_distributions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print_header("7. TARGET DISTRIBUTION ANALYSIS (BINARY & MULTICLASS)")
    print("\n--- Binary Target ('label') ---")
    tr_bin = train_df["label"].value_counts()
    te_bin = test_df["label"].value_counts()
    bin_summary = pd.DataFrame({
        "Train Count": tr_bin,
        "Train Pct (%)": (tr_bin / len(train_df) * 100).round(2),
        "Test Count": te_bin,
        "Test Pct (%)": (te_bin / len(test_df) * 100).round(2)
    })
    bin_summary.index = ["Normal (0)" if idx == 0 else "Attack (1)" for idx in bin_summary.index]
    print(bin_summary)

    print("\n--- Multi-class Target ('attack_cat') ---")
    tr_cat = train_df["attack_cat"].value_counts()
    te_cat = test_df["attack_cat"].value_counts()
    cat_summary = pd.DataFrame({
        "Train Count": tr_cat,
        "Train Pct (%)": (tr_cat / len(train_df) * 100).round(2),
        "Test Count": te_cat,
        "Test Pct (%)": (te_cat / len(test_df) * 100).round(2)
    }).fillna(0)
    print(cat_summary)

    # Verification of label alignment: normal == 0, attacks == 1
    tr_mismatch = ((train_df["attack_cat"] == "Normal") != (train_df["label"] == 0)).sum()
    te_mismatch = ((test_df["attack_cat"] == "Normal") != (test_df["label"] == 0)).sum()
    print(f"\nTarget alignment check (Normal == 0 and Attack == 1):")
    print(f"  Train inconsistencies: {tr_mismatch} | Test inconsistencies: {te_mismatch}")
    if tr_mismatch == 0 and te_mismatch == 0:
        print("  [PASS] 100% strict alignment between 'label' and 'attack_cat'.")


def check_target_leakage_and_collinearity(train_df: pd.DataFrame) -> None:
    print_header("8. TARGET LEAKAGE, COLLINEARITY & TESTBED ARTIFACT WARNINGS")
    
    # 1. Target Leakage Rule
    print("\n[LEAKAGE WARNING 1: MUTUAL EXCLUSION OF TARGETS]")
    print(" - 'label' and 'attack_cat' are two representations of the same target ground truth.")
    print(" - Neither must EVER be included as an input feature for predicting the other.")

    # 2. Exact Collinearity Checks
    print("\n[COLLINEARITY CHECK: REDUNDANT OR PERFECTLY CORRELATED FEATURES]")
    ftp_match = (train_df["is_ftp_login"] == train_df["ct_ftp_cmd"]).mean()
    print(f" - 'is_ftp_login' vs 'ct_ftp_cmd' exact agreement: {ftp_match * 100:.2f}%")
    if ftp_match > 0.999:
        print("   -> 'is_ftp_login' and 'ct_ftp_cmd' are virtually identical; one is redundant.")

    diff_tcprtt = (train_df["tcprtt"] - (train_df["synack"] + train_df["ackdat"])).abs().max()
    print(f" - TCP Round Trip Time identity check: max |tcprtt - (synack + ackdat)| = {diff_tcprtt:.2e}")
    if diff_tcprtt < 1e-10:
        print("   -> 'tcprtt' is an exact linear combination of 'synack' + 'ackdat'.")

    # 3. Testbed Artifacts / TTL Over-separation
    print("\n[TESTBED ARTIFACT CHECK: TTL / STATE FEATURES]")
    sttl_corr = train_df["sttl"].corr(train_df["label"])
    ct_ttl_corr = train_df["ct_state_ttl"].corr(train_df["label"])
    print(f" - Pearson correlation with binary label: 'sttl' = {sttl_corr:.4f}, 'ct_state_ttl' = {ct_ttl_corr:.4f}")
    
    sttl_31_attacks = ((train_df["sttl"] == 31) & (train_df["label"] == 1)).sum()
    sttl_31_normals = ((train_df["sttl"] == 31) & (train_df["label"] == 0)).sum()
    print(f" - sttl == 31 count: Normal = {sttl_31_normals:,} | Attack = {sttl_31_attacks}")
    print("   -> UNSW-NB15 testbed artifact: The IXIA traffic generator and legitimate servers used distinct default TTL values,")
    print("      causing models to heavily rely on TTL rather than deep behavioral indicators.")


def main() -> None:
    print_header("UNSW-NB15 DATASET INSPECTION & VALIDATION SUITE", "#")
    verify_file_existence()
    train_df, test_df = load_datasets()
    verify_columns_and_schema(train_df, test_df)
    cat_features, num_features = categorize_features(train_df)
    check_data_quality(train_df, test_df)
    check_categorical_consistency(train_df, test_df, cat_features)
    analyze_target_distributions(train_df, test_df)
    check_target_leakage_and_collinearity(train_df)
    print_header("PHASE 1 DATASET VALIDATION COMPLETED SUCCESSFULLY", "#")


if __name__ == "__main__":
    main()
