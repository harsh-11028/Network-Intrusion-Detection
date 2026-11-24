# Phase 7.1: Reproducibility & Data Integrity Audit

**Project:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Phase:** 7.1  
**Dataset:** UNSW-NB15  
**Audit Date:** November 2026  
**Auditor:** Senior ML Engineer & Data Science Auditor  

---

## 1. Immutability & File Checksums

The integrity of the original raw cleaned datasets was verified via cryptographic hashing:

| Dataset File | File Size | Row Count | Column Count | SHA-256 Checksum | Immutability Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `data/UNSW_NB15_training_cleaned.csv` | 15.2 MB | 107,740 | 44 | `a82f16c0d1788560dfe5a927782e06ceffd4d2b54daa7821013aa8ac72d93d04` | **VERIFIED (Unchanged)** |
| `data/UNSW_NB15_testing_cleaned.csv` | 7.9 MB | 55,945 | 44 | `8ed1055914edd482b581a3782dfd04f08be45bc9b472931d0889d4358622ce66` | **VERIFIED (Unchanged)** |

---

## 2. Partitioning & Dimension Verification

| Pipeline Stage | Input Partition | Dimensions Before Transformation | Dimensions After Transformation | Feature Count | Label Distribution (Normal / Attack) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Validation Stage** | Internal Train (80%) | (86,192, 42) | (86,192, 192) | 192 | 41,512 / 44,680 (51.84% Attack) |
| **Validation Stage** | Validation Subset (20%) | (21,548, 42) | (21,548, 192) | 192 | 10,378 / 11,170 (51.84% Attack) |
| **Final Retraining** | Official Training (100%) | (107,740, 42) | (107,740, 194) | 194 | 51,890 / 55,850 (51.84% Attack) |
| **Final Evaluation** | Official Testing (100%) | (55,945, 42) | (55,945, 194) | 194 | 34,206 / 21,739 (38.86% Attack) |

---

## 3. Data Leakage Checklist

- [x] **Target Isolation:** `label` (binary target) and `attack_cat` (multi-class label) were separated immediately upon loading and were never present in input feature matrices $X$.
- [x] **Split Timing:** Train/validation split was executed on RAW features before instantiating or fitting any imputer, scaler, or encoder.
- [x] **Validation Parameter Independence:** `val_preprocessor` was fitted strictly on the 86,192 internal training samples. Validation samples were transformed via `.transform()` with `handle_unknown='ignore'`.
- [x] **Test Set Quarantine:** The official test set (55,945 samples) was quarantined during validation model training, threshold sweeps, and threshold selection. Test metrics were computed once as a final verification.
- [x] **Random State Consistency:** `random_state=42` was applied across `train_test_split`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `XGBClassifier`.
- [x] **Probability Calibration & Alignment:** All predicted probability arrays were verified to represent posterior probabilities for class `1` (Attack), with continuous range $[0.0000, 1.0000]$.
- [x] **Confusion Matrix Consistency:** All confusion matrix cell counts ($TN + FP + FN + TP$) strictly sum to:
  - Validation: $21,548$
  - Test: $55,945$

---

## 4. Train/Test Overlap & Duplicate Reconciliation

- **Exact 44-Column Row Matches:** **940 rows** appear in both train and test.
- **Shared Unique 42-Feature Patterns:** **1,302 unique patterns** exist symmetrically across both datasets.
- **Overlapping Test Records:** **2,030 test records** (3.63%) belong to those 1,302 patterns.
- **Novel Non-Overlapping Test Records:** **53,915 records** (96.37%) are strictly unique and unseen.
- **Label Contradictions:** **181 test records** match training feature vectors but have opposite binary labels (inherent Bayes optimal noise in statistical flow profiles).
- **Academic Protocol:** In accordance with established benchmark standards, no test records were deleted or altered. Previous phases proved that removing overlapping records resulted in less than a $0.35\%$ drop in accuracy, demonstrating authentic generalization.
