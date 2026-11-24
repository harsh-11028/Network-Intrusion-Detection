"""
preprocessing.py
----------------
Phase 2: Machine Learning Preprocessing Module
College ML Project: Machine Learning-Based Network Intrusion Detection and Attack Classification
Dataset: UNSW-NB15

This module provides reusable, leakage-safe preprocessing functions and Scikit-learn
pipelines for binary network intrusion detection classification.

Key design principles:
  1. Strict target isolation: 'label' and 'attack_cat' are never included in input features.
  2. Training-only parameter estimation: Scaler, Encoder, and Imputer parameters are fitted
     strictly on X_train.
  3. Safe categorical encoding: OneHotEncoder uses handle_unknown='ignore' so unseen categories
     in test data (e.g., state='CLO', 'ACC') do not cause pipeline failures.
  4. Configurable TTL bias experimentation: Supports building pipelines with or without 'sttl'
     for investigating synthetic testbed artifacts.
  5. Reusable across downstream models: Supports both scaled features (for Logistic Regression)
     and unscaled features (for Decision Trees and Random Forests).
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Known target columns and identifiers to exclude from feature matrices
TARGET_COLUMNS = ["label", "attack_cat"]
ID_COLUMNS = ["id", "ID", "Id"]


def compute_file_sha256(filepath: Path) -> str:
    """Compute the SHA-256 checksum of a file to verify immutability."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_raw_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training and testing CSV datasets into DataFrames without modifying them.
    
    Returns:
        (train_df, test_df)
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing dataset not found: {test_path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    return train_df, test_df


def separate_features_and_target(
    df: pd.DataFrame,
    target_col: str = "label",
    additional_drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features from target variable while eliminating target leakage.
    
    Args:
        df: Input DataFrame.
        target_col: The primary target column to extract ('label' for binary classification).
        additional_drop_cols: Extra columns to drop from feature set (e.g., ['attack_cat']).
        
    Returns:
        (X, y): Feature DataFrame and Target Series.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {df.columns.tolist()}")

    # Extract target
    y = df[target_col].copy()

    # Determine all columns to exclude from X
    cols_to_drop = [target_col]
    
    # Always drop alternative target representations to prevent leakage
    for alt_target in TARGET_COLUMNS:
        if alt_target in df.columns and alt_target not in cols_to_drop:
            cols_to_drop.append(alt_target)

    # Always drop ID columns if present
    for id_col in ID_COLUMNS:
        if id_col in df.columns and id_col not in cols_to_drop:
            cols_to_drop.append(id_col)

    # Any user-specified additional columns
    if additional_drop_cols:
        for col in additional_drop_cols:
            if col in df.columns and col not in cols_to_drop:
                cols_to_drop.append(col)

    X = df.drop(columns=cols_to_drop).copy()
    return X, y


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Automatically and accurately classify columns into numerical and categorical features.
    
    Ensures numerical columns are not misclassified as categorical.
    
    Returns:
        (num_cols, cat_cols): Lists of numerical and categorical column names.
    """
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Safety check: confirm no overlap and all columns accounted for
    all_identified = set(cat_cols + num_cols)
    if len(all_identified) != len(X.columns):
        unaccounted = set(X.columns) - all_identified
        raise ValueError(f"Columns could not be categorized into num/cat: {unaccounted}")

    return num_cols, cat_cols


def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    scale_numeric: bool = True,
    include_sttl: bool = True
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a Scikit-learn ColumnTransformer preprocessing pipeline.
    
    Numerical pipeline:
      - SimpleImputer(strategy='median')
      - StandardScaler() (if scale_numeric=True)
      
    Categorical pipeline:
      - SimpleImputer(strategy='most_frequent')
      - OneHotEncoder(handle_unknown='ignore', sparse_output=False)
      
    Args:
        num_cols: List of numerical column names.
        cat_cols: List of categorical column names.
        scale_numeric: Whether to apply StandardScaler to numerical features.
                       Set True for Logistic Regression/SVM; can be False for Decision Trees/Random Forest.
        include_sttl: Whether to include 'sttl' (Source TTL) in numerical features.
                      Set False to prepare an ablation experiment for TTL testbed bias.
                      
    Returns:
        (preprocessor, active_num_cols, active_cat_cols)
    """
    active_num_cols = [c for c in num_cols]
    if not include_sttl and "sttl" in active_num_cols:
        active_num_cols.remove("sttl")

    active_cat_cols = [c for c in cat_cols]

    # Construct numerical pipeline
    num_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(steps=num_steps)

    # Construct categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, active_num_cols),
            ("cat", cat_pipeline, active_cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor, active_num_cols, active_cat_cols


def check_data_integrity_and_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Perform rigorous integrity and overlap checks between training and testing sets.
    Reports findings without dropping or modifying any rows.
    
    Returns:
        Dictionary containing summary statistics of integrity and overlap.
    """
    # 1. Exact full-row duplicates across train and test
    exact_overlap = pd.merge(train_df, test_df, how="inner")
    exact_overlap_count = len(exact_overlap)

    # 2. Feature-only unique vector overlap
    train_unique_feats = X_train.drop_duplicates()
    test_unique_feats = X_test.drop_duplicates()
    feat_overlap = pd.merge(train_unique_feats, test_unique_feats, how="inner")
    feat_overlap_count = len(feat_overlap)

    # 3. Categorical levels check
    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    unseen_categories: Dict[str, List[str]] = {}
    for col in cat_cols:
        tr_unique = set(X_train[col].dropna().unique())
        te_unique = set(X_test[col].dropna().unique())
        unseen = list(te_unique - tr_unique)
        if unseen:
            unseen_categories[col] = unseen

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "exact_full_row_overlap": exact_overlap_count,
        "feature_vector_overlap": feat_overlap_count,
        "unseen_test_categories": unseen_categories,
        "train_has_target_in_X": any(t in X_train.columns for t in TARGET_COLUMNS),
        "test_has_target_in_X": any(t in X_test.columns for t in TARGET_COLUMNS),
    }


def save_processed_artifacts(
    output_dir: Path,
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    X_test_trans: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    preprocessor: ColumnTransformer,
    metadata: Dict[str, Any]
) -> None:
    """
    Save transformed arrays, fitted preprocessor, and feature metadata for Phase 3 model training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save compressed numpy archives (fast loading, minimal disk footprint)
    train_npz_path = output_dir / "train_processed.npz"
    np.savez_compressed(
        train_npz_path,
        X=X_train_trans.astype(np.float32),
        y=y_train.astype(np.int64)
    )

    test_npz_path = output_dir / "test_processed.npz"
    np.savez_compressed(
        test_npz_path,
        X=X_test_trans.astype(np.float32),
        y=y_test.astype(np.int64)
    )

    # Save fitted preprocessor
    pipeline_path = output_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, pipeline_path)

    # Save feature metadata
    meta_path = output_dir / "feature_metadata.json"
    full_meta = {
        **metadata,
        "feature_count": len(feature_names),
        "feature_names": feature_names
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)

    print(f"[SAVE] Transformed training set saved: {train_npz_path} ({train_npz_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"[SAVE] Transformed testing set saved : {test_npz_path} ({test_npz_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"[SAVE] Fitted pipeline saved        : {pipeline_path}")
    print(f"[SAVE] Feature metadata saved       : {meta_path}")
