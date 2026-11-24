# Phase 7.1: Methodological Changes & Preprocessing Leakage Correction

**Project:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Phase:** 7.1 (Methodological Correction & Verification)  
**Dataset:** UNSW-NB15 Benchmark Dataset  
**Date:** November 2026  

---

## 1. Executive Summary

During the Phase 7 audit, a subtle but fundamental data science limitation was identified in the internal validation pipeline:
In Phase 7, the script loaded `data/processed/train_processed.npz` and split that already-transformed matrix into internal train (80%) and validation (20%) partitions. Because the `ColumnTransformer` (StandardScaler, OneHotEncoder, and SimpleImputer) had been fitted on the entire official training set (107,740 samples) during Phase 2, the validation partition indirectly influenced preprocessing parameters prior to threshold evaluation.

Phase 7.1 corrects this methodology completely by moving the split to the **raw, unscaled data** and independently fitting preprocessors across validation and final retraining stages.

---

## 2. Anatomical Comparison: Phase 7 vs Phase 7.1

| Pipeline Dimension | Previous Phase 7 Methodology (Flawed) | Corrected Phase 7.1 Methodology (Leakage-Free) | Impact & Justification |
| :--- | :--- | :--- | :--- |
| **Split Ingestion** | Loaded pre-transformed `train_processed.npz` (107,740 rows $	imes$ 194 features). | Loads raw `UNSW_NB15_training_cleaned.csv` (107,740 rows $	imes$ 44 columns). | Eliminates upstream parameter leakage before any transformation. |
| **Split Point** | Split transformed matrix into 80% train and 20% validation. | Splits raw 42-feature dataframe into 80% train (86,192) and 20% validation (21,548). | Guarantees validation records are completely unseen by the preprocessor. |
| **Validation Preprocessor** | No separate preprocessor; reused Phase 2 preprocessor fitted on 100% of data. | Fits fresh `val_preprocessor` strictly on the 80% internal training subset. | Ensures means, variances, medians, and one-hot levels derive purely from training records. |
| **Feature Dimensionality** | Validation set had **194 features**. | Validation set has **192 features**. | **Empirical Proof of Leakage:** Two rare categorical levels (`proto='rtp'` and `state='URN'`) appeared only in the validation split. In Phase 7, they leaked into the one-hot vocabulary. In Phase 7.1, they are safely ignored by `handle_unknown='ignore'`. |
| **Threshold Selection** | Selected on validation probabilities derived from 194 leaked features. | Selected on validation probabilities derived from 192 leakage-free features. | Produces mathematically sound, unbiased operating thresholds. |
| **TTL Feature Ablation** | Masked columns from pre-transformed matrix. | Independently fits separate ColumnTransformers for full, without `sttl`, and without `sttl`/`ct_state_ttl`. | Guarantees ablation metrics reflect genuine feature omission without residual scaler leakage. |
| **Final Retraining** | Retrained final models on full dataset using Phase 2 preprocessor. | Fits fresh `final_preprocessor` on 100% raw training data, transforms train & test, and evaluates at validation-selected thresholds. | Preserves full representation capacity for final official test set evaluation. |
| **Official Test Quarantine** | Quarantined during threshold selection. | Strictly quarantined during both threshold selection and model training. | Preserved standard benchmark protocol. |

---

## 3. Empirical Proof of Preprocessing Leakage

A detailed category-frequency audit across the raw internal train (86,192 rows) and validation (21,548 rows) subsets revealed:
1. `proto == 'rtp'`: Appears 0 times in the internal 80% training subset, but appears in the 20% validation subset.
2. `state == 'URN'`: Appears 0 times in the internal 80% training subset, but appears in the 20% validation subset.

In Phase 7, because `OneHotEncoder` was fitted on all 107,740 training records, it allocated dummy columns `proto_rtp` and `state_URN`, creating a 194-dimensional vector space. The validation models were trained on feature vectors containing columns for categories they had never legitimately observed.

In Phase 7.1, `val_preprocessor` was fitted strictly on `X_tr_raw` (86,192 rows):
- The resulting feature space has exactly **192 features**.
- When `val_preprocessor.transform(X_val_raw)` encountered `'rtp'` and `'URN'`, `handle_unknown='ignore'` correctly mapped them to all-zero vectors across their respective dummy columns.
- The validation models were trained and evaluated on 192 features without any out-of-sample vocabulary leakage.

---

## 4. Impact on Threshold Selection & Model Performance

Did correcting the leakage alter the selected thresholds?
- **XGBoost:**
  - Phase 7 (with minor leakage): Strategy A selected $\tau = 0.49$, Strategy B selected $\tau = 0.53$, Strategy C selected $\tau = 0.42$.
  - Phase 7.1 (leakage-free): Strategy A selected $\tau = 0.49$, Strategy B selected $\tau = 0.49$, Strategy C selected $\tau = 0.42$.
- **Random Forest:**
  - Phase 7: Strategy A selected $\tau = 0.48$, Strategy C selected $\tau = 0.42$.
  - Phase 7.1: Strategy A selected $\tau = 0.50$, Strategy C selected $\tau = 0.42$.
- **Conclusion:** While the numerical shifts in optimal operating points were modest (a shift of $0.01$ to $0.04$ on certain ensemble thresholds), the methodological correction eliminates a critical academic flaw and establishes complete reproducibility.
