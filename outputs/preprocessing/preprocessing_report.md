# Machine Learning Preprocessing Pipeline Report (Phase 2)

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
| **Raw Input Rows** | 107,740 | 55,945 |
| **Input Feature Columns** | 42 | 42 |
| **Active Numerical Columns** | 39 | 39 |
| **Active Categorical Columns** | 3 | 3 |
| **One-Hot Encoded Columns** | 155 categories (133 proto + 13 service + 9 state) | 155 categories |
| **Transformed Output Shape** | **107,740 rows × 194 columns** | **55,945 rows × 194 columns** |
| **Binary Target Vector Shape** | 107,740 | 55,945 |

---

## 4. Data Integrity & Leakage Verification

### 4.1 Strict Train/Test Separation
- The official benchmark split was strictly preserved. No rows were merged or randomly split.
- The preprocessor was fitted exclusively on `X_train`.

### 4.2 Overlap Analysis (Reported, Not Deleted)
Network intrusion datasets often feature identical network flow profiles across test and train environments:
- **Exact 44-Column Row Matches:** **940 rows** appear in both train and test.
- **Unique Feature Vector Matches (ignoring targets):** **1,302 unique feature patterns** exist in both train and test.
- **Action Taken:** In accordance with the benchmarking protocol, these rows were **not silently deleted**, ensuring standard reproducibility.

### 4.3 Original Dataset Immutability
To guarantee non-destructive execution, SHA-256 hashes of original files were verified before and after execution:
- `UNSW_NB15_training_cleaned.csv`: `a82f16c0d1788560dfe5a927782e06ceffd4d2b54daa7821013aa8ac72d93d04` (Verified Unaltered)
- `UNSW_NB15_testing_cleaned.csv`: `8ed1055914edd482b581a3782dfd04f08be45bc9b472931d0889d4358622ce66` (Verified Unaltered)

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
