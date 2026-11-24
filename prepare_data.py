"""
prepare_data.py
---------------
Phase 2: Preprocessing Execution, Validation, and Artifact Generation Script
College ML Project: Machine Learning-Based Network Intrusion Detection and Attack Classification
Dataset: UNSW-NB15

Usage:
    python prepare_data.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

from preprocessing import (
    compute_file_sha256,
    load_raw_data,
    separate_features_and_target,
    identify_feature_types,
    build_preprocessor,
    check_data_integrity_and_overlap,
    save_processed_artifacts,
    TARGET_COLUMNS
)

TRAIN_PATH = Path("data/UNSW_NB15_training_cleaned.csv")
TEST_PATH = Path("data/UNSW_NB15_testing_cleaned.csv")
PROCESSED_DATA_DIR = Path("data/processed")
PREPROCESSING_OUTPUT_DIR = Path("outputs/preprocessing")


def print_section(title: str, char: str = "=") -> None:
    print(f"\n{char * 78}")
    print(f" {title}")
    print(f"{char * 78}")


def generate_preprocessing_report(
    integrity_info: dict,
    feature_names: list,
    active_num_cols: list,
    active_cat_cols: list,
    X_train_shape: tuple,
    X_test_shape: tuple,
    hash_train: str,
    hash_test: str
) -> None:
    """Generate outputs/preprocessing/preprocessing_report.md explaining decisions and checks."""
    PREPROCESSING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = PREPROCESSING_OUTPUT_DIR / "preprocessing_report.md"

    report_content = f"""# Machine Learning Preprocessing Pipeline Report (Phase 2)

**Project Title:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Dataset:** UNSW-NB15  
**Target:** Binary Classification (`label = 0` for Normal, `label = 1` for Attack)  
**Environment:** macOS Apple Silicon, Python 3.13, Scikit-learn, Pandas, NumPy  

---

## 1. Executive Summary

Phase 2 builds a leakage-safe, reusable machine learning preprocessing pipeline using Scikit-learn's `ColumnTransformer` and `Pipeline` architectures.

The pipeline ingests raw training and testing datasets, separates input features from targets, protects against target leakage, imputes missing values, standardizes numerical features, and applies one-hot encoding with safeguards for unseen test categories.

All preprocessing parameters (mean, standard deviation, imputer statistics, one-hot category bins) are **strictly fitted on the training split only** and then applied to transform both training and testing datasets.

---

## 2. Feature & Target Isolation (Zero Target Leakage)

To eliminate target leakage:
1. **Target Separation:** `label` was extracted as the target series $y$.
2. **Multi-class Ground Truth Exclusion:** The `attack_cat` column represents the same ground truth as `label`. Including `attack_cat` in binary intrusion detection is a catastrophic 100% target leak. It was strictly excluded from feature matrix $X$.
3. **ID Removal:** Any identifier columns (e.g. `id`) were excluded.
4. **Target Absence Verification:** Transformed feature matrices were verified to contain 0 instances of `label` or `attack_cat`.

---

## 3. Feature Typing & Transformation Architecture

The raw feature matrix contains **42 input features** partitioned automatically into:
- **39 Numerical Features:** Continuous network rates, packet counts, byte volumes, jitter, window sizes, and connection metrics.
- **3 Categorical Features:** Transport protocol (`proto`), application service (`service`), and TCP connection state (`state`).

### 3.1 Numerical Pipeline
- **Imputation:** `SimpleImputer(strategy='median')`
  - Median imputation is robust to the extreme positive skewness documented during Phase 1 EDA.
- **Scaling:** `StandardScaler()`
  - Centers features to zero mean and unit variance.
  - Vital for distance-based and gradient-based algorithms (such as Logistic Regression).
  - Configurable in `preprocessing.py` via `scale_numeric=False` if tree-only unscaled features are desired.

### 3.2 Categorical Pipeline
- **Imputation:** `SimpleImputer(strategy='most_frequent')`
- **Encoding:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
  - Encodes categories into binary indicator vectors.
  - **Unseen Level Protection:** In Phase 1 EDA, the testing set was found to contain connection states (`state = 'CLO'`, `'ACC'`) that never appeared in the training set. Setting `handle_unknown='ignore'` guarantees that unseen categories in test data evaluate to an all-zero vector, avoiding runtime exceptions and preventing dimension mismatch.

### 3.3 Dimension Expansion Summary

| Metric | Training Split | Testing Split |
|---|---|---|
| **Raw Input Rows** | {integrity_info['train_rows']:,} | {integrity_info['test_rows']:,} |
| **Input Feature Columns** | 42 | 42 |
| **Active Numerical Columns** | {len(active_num_cols)} | {len(active_num_cols)} |
| **Active Categorical Columns** | {len(active_cat_cols)} | {len(active_cat_cols)} |
| **One-Hot Encoded Columns** | 155 categories (133 proto + 13 service + 9 state) | 155 categories |
| **Transformed Output Shape** | **{X_train_shape[0]:,} rows × {X_train_shape[1]} columns** | **{X_test_shape[0]:,} rows × {X_test_shape[1]} columns** |
| **Binary Target Vector Shape** | {X_train_shape[0]:,} | {X_test_shape[0]:,} |

---

## 4. Data Integrity & Leakage Verification

### 4.1 Strict Train/Test Separation
- The official benchmark split was strictly preserved. No rows were merged or randomly split.
- The preprocessor was fitted exclusively on `X_train`.

### 4.2 Overlap Analysis (Reported, Not Deleted)
Network intrusion datasets often feature identical network flow profiles across test and train environments:
- **Exact 44-Column Row Matches:** **{integrity_info['exact_full_row_overlap']:,} rows** appear in both train and test.
- **Unique Feature Vector Matches (ignoring targets):** **{integrity_info['feature_vector_overlap']:,} unique feature patterns** exist in both train and test.
- **Action Taken:** In accordance with the benchmarking protocol, these rows were **not silently deleted**, ensuring standard reproducibility.

### 4.3 Original Dataset Immutability
To guarantee non-destructive execution, SHA-256 hashes of original files were verified before and after execution:
- `UNSW_NB15_training_cleaned.csv`: `{hash_train}` (Verified Unaltered)
- `UNSW_NB15_testing_cleaned.csv`: `{hash_test}` (Verified Unaltered)

---

## 5. TTL Bias & Ablation Preparation

During Phase 1, Source Time-To-Live (`sttl`) showed strong separation (Pearson correlation = 0.61) due to testbed hardware differences (IXIA traffic generator vs. native workstations).

In Phase 2:
- `sttl` is retained in the baseline pipeline to maintain feature completeness.
- The pipeline function `build_preprocessor(..., include_sttl=False)` was tested and validated.
- In Phase 3, this parameter allows seamless comparison between models trained with all features (194 features) versus models trained without `sttl` (193 features) to measure how heavily models rely on synthetic testbed artifacts.

---

## 6. Generated Processed Artifacts

All processed arrays and serialized pipeline components are stored in `data/processed/`:
1. `train_processed.npz`: Compressed archive containing `X` (float32) and `y` (int64) for training.
2. `test_processed.npz`: Compressed archive containing `X` (float32) and `y` (int64) for testing.
3. `preprocessor.joblib`: Serialized Scikit-learn ColumnTransformer fitted on training data.
4. `feature_metadata.json`: Complete ordered feature names and configuration metadata.

---

## 7. Reusability for Downstream Models (Phase 3)

The resulting artifacts can be loaded directly in Phase 3 using:
```python
import numpy as np
import joblib

# Load preprocessed arrays instantly
train_data = np.load("data/processed/train_processed.npz")
X_train, y_train = train_data["X"], train_data["y"]

test_data = np.load("data/processed/test_processed.npz")
X_test, y_test = test_data["X"], test_data["y"]

# Load pipeline for new flow inference
pipeline = joblib.load("data/processed/preprocessor.joblib")
```
This enables zero-overhead model training for **Logistic Regression**, **Decision Trees**, and **Random Forests**.
"""
    report_file.write_text(report_content, encoding="utf-8")
    print(f"[REPORT] Preprocessing documentation saved to {report_file}")


def main() -> None:
    print_section("PHASE 2: MACHINE LEARNING PREPROCESSING PIPELINE", "#")

    # 1. Compute initial SHA256 hashes to guarantee file immutability
    print_section("1. ORIGINAL CSV FILE IMMUTABILITY CHECK (PRE-RUN)")
    initial_train_hash = compute_file_sha256(TRAIN_PATH)
    initial_test_hash = compute_file_sha256(TEST_PATH)
    print(f"Training CSV SHA256: {initial_train_hash}")
    print(f"Testing CSV SHA256 : {initial_test_hash}")
    print("[PASS] Initial checksums recorded.")

    # 2. Ingest datasets
    print_section("2. INGESTING BENCHMARK DATASETS")
    train_df, test_df = load_raw_data(TRAIN_PATH, TEST_PATH)
    print(f"Loaded Raw Training: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"Loaded Raw Testing : {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")

    # 3. Feature and target separation (Strict Target Isolation)
    print_section("3. FEATURE & TARGET SEPARATION")
    X_train, y_train = separate_features_and_target(train_df, target_col="label")
    X_test, y_test = separate_features_and_target(test_df, target_col="label")

    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_test shape : {X_test.shape} | y_test shape : {y_test.shape}")

    # Confirm neither label nor attack_cat is present in X
    for target in TARGET_COLUMNS:
        assert target not in X_train.columns, f"LEAKAGE DETECTED: {target} found in X_train!"
        assert target not in X_test.columns, f"LEAKAGE DETECTED: {target} found in X_test!"
    print("[PASS] Target isolation confirmed: 'label' and 'attack_cat' strictly excluded from features.")

    # 4. Feature typing
    print_section("4. AUTOMATED FEATURE CLASSIFICATION")
    num_cols, cat_cols = identify_feature_types(X_train)
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical features   ({len(num_cols)}): {len(num_cols)} features identified")
    assert len(cat_cols) == 3, f"Expected 3 categorical features, got {len(cat_cols)}"
    assert len(num_cols) == 39, f"Expected 39 numerical features, got {len(num_cols)}"
    print("[PASS] All 42 input features correctly and unambiguously typed.")

    # 5. Data integrity and overlap verification
    print_section("5. DATA INTEGRITY & OVERLAP REPORT")
    integrity_info = check_data_integrity_and_overlap(train_df, test_df, X_train, X_test)
    print(f"Total training records          : {integrity_info['train_rows']:,}")
    print(f"Total testing records           : {integrity_info['test_rows']:,}")
    print(f"Exact 44-column matching rows   : {integrity_info['exact_full_row_overlap']:,} rows")
    print(f"Unique feature vector matches   : {integrity_info['feature_vector_overlap']:,} patterns")
    if integrity_info["unseen_test_categories"]:
        print(f"Unseen categories in test data  : {integrity_info['unseen_test_categories']}")
    print("[PASS] Overlap documented. No rows silently deleted; benchmark split fully preserved.")

    # 6. Pipeline construction and fitting
    print_section("6. PIPELINE FIT (TRAINING SET ONLY)")
    preprocessor, active_num_cols, active_cat_cols = build_preprocessor(
        num_cols,
        cat_cols,
        scale_numeric=True,
        include_sttl=True
    )

    print("Fitting ColumnTransformer strictly on X_train...")
    preprocessor.fit(X_train)
    print("[PASS] Preprocessor fitted exclusively on training data.")

    # 7. Transformation of train and test
    print_section("7. DATA TRANSFORMATION & SHAPE VALIDATION")
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    print(f"Transformed X_train shape: {X_train_trans.shape} (Expected: (107740, 194))")
    print(f"Transformed X_test shape : {X_test_trans.shape} (Expected: (55945, 194))")
    print(f"Total output feature names: {len(feature_names)}")

    assert X_train_trans.shape == (107740, 194), f"Unexpected X_train shape: {X_train_trans.shape}"
    assert X_test_trans.shape == (55945, 194), f"Unexpected X_test shape: {X_test_trans.shape}"
    print("[PASS] Transformed dimensions match expected schema exactly.")

    # 8. Unseen category handling test
    print_section("8. UNSEEN CATEGORY SAFETY VALIDATION")
    # In Phase 1, state='CLO' was in test but not train
    clo_mask = X_test["state"] == "CLO"
    clo_count = clo_mask.sum()
    print(f"Test records with unseen state='CLO': {clo_count}")
    state_feature_indices = [idx for idx, name in enumerate(feature_names) if name.startswith("state_")]
    state_vals_for_clo = X_test_trans[clo_mask.to_numpy()][:, state_feature_indices]
    assert np.all(state_vals_for_clo == 0.0), "Unseen category was not encoded as zeros!"
    print("[PASS] Unseen category 'CLO' safely encoded as all-zero vector without errors.")

    # 9. TTL Ablation pipeline check
    print_section("9. TTL BIAS ABLATION PREPARATION CHECK")
    ablation_preprocessor, ab_num_cols, ab_cat_cols = build_preprocessor(
        num_cols,
        cat_cols,
        scale_numeric=True,
        include_sttl=False
    )
    ablation_preprocessor.fit(X_train)
    X_train_ab = ablation_preprocessor.transform(X_train)
    X_test_ab = ablation_preprocessor.transform(X_test)
    print(f"Ablation pipeline (without 'sttl') transformed shape: {X_train_ab.shape}")
    assert X_train_ab.shape == (107740, 193), f"Unexpected ablation shape: {X_train_ab.shape}"
    assert "sttl" not in ablation_preprocessor.get_feature_names_out(), "sttl still present in ablation!"
    print("[PASS] TTL ablation preprocessor verified (193 features ready for Phase 3 comparison).")

    # 10. Check file immutability post-run
    print_section("10. VERIFYING ORIGINAL CSV IMMUTABILITY (POST-RUN)")
    post_train_hash = compute_file_sha256(TRAIN_PATH)
    post_test_hash = compute_file_sha256(TEST_PATH)
    assert initial_train_hash == post_train_hash, "Training CSV hash changed! File was modified!"
    assert initial_test_hash == post_test_hash, "Testing CSV hash changed! File was modified!"
    print(f"Training CSV Post-Run SHA256: {post_train_hash} (Matches Initial)")
    print(f"Testing CSV Post-Run SHA256 : {post_test_hash} (Matches Initial)")
    print("[PASS] Original CSV files are 100% untouched and preserved.")

    # 11. Save processed artifacts
    print_section("11. SAVING PREPROCESSED ARTIFACTS")
    metadata = {
        "dataset": "UNSW-NB15",
        "phase": 2,
        "target": "label (binary: 0=Normal, 1=Attack)",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "raw_features_count": len(num_cols) + len(cat_cols),
        "active_numerical_features": active_num_cols,
        "active_categorical_features": active_cat_cols,
        "scaled_numeric": True,
        "sttl_included": True
    }
    save_processed_artifacts(
        PROCESSED_DATA_DIR,
        X_train_trans,
        y_train.to_numpy(),
        X_test_trans,
        y_test.to_numpy(),
        feature_names,
        preprocessor,
        metadata
    )

    # 12. Generate report
    print_section("12. GENERATING PREPROCESSING REPORT")
    generate_preprocessing_report(
        integrity_info,
        feature_names,
        active_num_cols,
        active_cat_cols,
        X_train_trans.shape,
        X_test_trans.shape,
        post_train_hash,
        post_test_hash
    )

    print_section("PHASE 2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY", "#")


if __name__ == "__main__":
    main()
