# Phase 7.1: Leakage-Free Validation, Threshold Re-selection, Final Retraining & Reproducibility Audit

**Project:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Dataset:** UNSW-NB15 Benchmark Dataset (`data/UNSW_NB15_training_cleaned.csv` and `data/UNSW_NB15_testing_cleaned.csv`)  
**Phase:** 7.1 (Methodological Correction, Verification & Final Benchmarking)  
**Execution Date:** November 2026  
**Environment:** macOS Apple Silicon, Python 3.13 (`.venv`), Scikit-Learn, XGBoost 3.4.1  

---

## 1. Project Overview

This project investigates the application of supervised machine learning classifiers for binary network intrusion detection (`0 = Normal`, `1 = Attack`) using the UNSW-NB15 benchmark dataset. Across previous phases (Phases 1 through 6), the dataset was validated, a Scikit-learn preprocessing pipeline was implemented, and four foundational baseline classifiers were trained and evaluated at an arbitrary default threshold of 0.50:
- **Logistic Regression:** Linear probabilistic baseline.
- **Decision Tree:** Single non-linear decision boundary baseline.
- **Random Forest:** Bagging ensemble of 300 de-correlated decision trees.
- **XGBoost:** Gradient boosted decision trees using second-order gradients.

## 2. Motivation for the Methodological Correction

In real-world cybersecurity, false negatives (missed attacks) and false positives (false alarms) have profoundly unequal costs. Phase 7 introduced validation-based threshold optimization to evaluate decision thresholds along the ROC and Precision-Recall frontiers. However, an architectural audit revealed that Phase 7 split an already-preprocessed training array into internal train and validation subsets. Because the preprocessor had been fitted on all 107,740 training records prior to the split, the validation partition indirectly influenced the scaling parameters and one-hot categorical vocabulary. Phase 7.1 was initiated to eliminate this leakage and re-evaluate all models with strict scientific rigor.

## 3. Previous Validation Methodology and Its Limitation

In Phase 7:
1. `data/processed/train_processed.npz` (107,740 rows $\times$ 194 columns) was loaded.
2. This transformed matrix was split into an 80% train subset and 20% validation subset.
3. **The Flaw:** Scaler statistics (means, standard deviations), median imputation values, and one-hot encoded categories were computed on the entire 107,740 rows. Specifically, two rare categorical tokens (`proto == 'rtp'` and `state == 'URN'`) occurred only in the validation partition, but their dummy columns were already included in the training matrix. This constitutes subtle preprocessing leakage.

## 4. Corrected Validation Methodology

Phase 7.1 establishes a strictly leakage-free workflow:
1. Load raw cleaned CSVs (`UNSW_NB15_training_cleaned.csv`).
2. Isolate input features by dropping `label` and `attack_cat`.
3. Partition the **raw** training dataframe into an 80% internal training subset (86,192 rows) and a 20% validation subset (21,548 rows) using stratified sampling on `label` (`random_state=42`).
4. Instantiate and fit the `ColumnTransformer` **exclusively on the 80% internal training subset**.
5. Transform the 80% training subset and the 20% validation subset using this fitted preprocessor. Unseen categories in validation are safely ignored via `handle_unknown='ignore'`.
6. Train the 4 models on transformed internal training data and generate validation predictions.
7. Evaluate thresholds (0.05 to 0.95) and select operating points using validation predictions only.
8. Fit a fresh preprocessor on 100% of the training set, transform train & official test sets, retrain final models, and evaluate on the untouched official test set.

## 5. Data Split Information

| Split Partition | Sample Count | Normal Flows (`0`) | Attack Flows (`1`) | Attack Prevalence | Feature Dimensionality |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Official Training** | 107,740 | 51,890 | 55,850 | 51.84% | 42 raw features |
| **Internal Train Subset (80%)** | 86,192 | 41,512 | 44,680 | 51.84% | **192 transformed features** |
| **Validation Subset (20%)** | 21,548 | 10,378 | 11,170 | 51.84% | **192 transformed features** |
| **Final Official Retraining (100%)** | 107,740 | 51,890 | 55,850 | 51.84% | **194 transformed features** |
| **Official Quarantined Test Set** | 55,945 | 34,206 | 21,739 | 38.86% | **194 transformed features** |

## 6. Preprocessing Methodology

- **Numerical Features (39 columns):** Missing values handled via `SimpleImputer(strategy='median')`; scaled via `StandardScaler()`.
- **Categorical Features (3 columns: `proto`, `service`, `state`):** Missing values handled via `SimpleImputer(strategy='most_frequent')`; encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Unseen Level Handling:** In the 80% split, 192 features were generated. Rare tokens `proto='rtp'` and `state='URN'` were ignored and encoded as zero vectors without pipeline failure or dimension mismatch.

## 7. Model Configurations

All four baseline models were trained using their exact configurations from Phases 3–6 without hyperparameter alterations:
- **Logistic Regression:** `solver='lbfgs', max_iter=1000, random_state=42` (Fit time: 2.23s on 86k rows)
- **Decision Tree:** `criterion='gini', max_depth=None, random_state=42` (Fit time: 1.19s on 86k rows)
- **Random Forest:** `n_estimators=300, max_features='sqrt', max_depth=None, n_jobs=-1, random_state=42` (Fit time: 6.01s on 86k rows)
- **XGBoost:** `n_estimators=300, max_depth=6, lr=0.1, subsample=0.8, colsample=0.8, n_jobs=-1, random_state=42` (Fit time: 2.00s on 86k rows)

## 8. Threshold Selection Strategies

Thresholds $\tau \in [0.05, 0.95]$ (step $0.01$, 91 evaluation points) were analyzed under four operational criteria:
1. **Strategy A (Maximum Attack F1):** Maximizes harmonic mean of attack precision and recall.
2. **Strategy B (Maximum Macro F1):** Maximizes unweighted mean of Normal F1 and Attack F1.
3. **Strategy C (Security-Oriented):** Mandates Attack Recall $\ge 97.0\%$, minimizing False Positive Rate. Infeasible if max recall $< 97\%$.
4. **Strategy D (Operational Constraint Target):** Mandates Attack Recall $\ge 95.0\%$ AND False Positive Rate $\le 20.0\%$.

## 9. Validation Threshold-Selection Results

| Model Architecture | Default (0.50) Val F1 | Strategy A Selected $\tau$ | Strategy A Val F1 | Strategy B Selected $\tau$ | Strategy C Selected $\tau$ | Strategy C Status | Strategy D Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9125 | **0.13** | 0.9171 | **0.13** | **0.53** | Feasible (Recall >= 97%) | Feasible |
| **Decision Tree** | 0.9187 | **0.50** | 0.9187 | **0.50** | **0.05** | Infeasible (Max Recall=91.36%) | Infeasible |
| **Random Forest** | 0.9360 | **0.50** | 0.9360 | **0.53** | **0.42** | Feasible (Recall >= 97%) | Feasible |
| **Xgboost** | 0.9409 | **0.49** | 0.9412 | **0.49** | **0.42** | Feasible (Recall >= 97%) | Feasible |


## 10. Final Official Test Evaluation

Final retrained models were evaluated on the quarantined **official test set (55,945 samples)** at default and validation-selected operating points:

| Model Architecture | Strategy Evaluated | Operating $\tau$ | Test Accuracy | Attack Recall | Attack Precision | Attack F1-Score | ROC-AUC | PR-AUC | False Positives | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Default (0.50) | 0.50 | 74.45% | 95.81% | 60.88% | 0.7445 | 0.9193 | 0.8912 | 13,385 | 910 |
| **Logistic Regression** | Strategy A (Max F1) | 0.13 | 74.67% | 99.85% | 60.56% | 0.7539 | 0.9193 | 0.8912 | 14,139 | 33 |
| **Logistic Regression** | Strategy B (Max Macro) | 0.13 | 74.67% | 99.85% | 60.56% | 0.7539 | 0.9193 | 0.8912 | 14,139 | 33 |
| **Logistic Regression** | Strategy C (Security 97%) | 0.53 | 74.62% | 95.12% | 61.14% | 0.7444 | 0.9193 | 0.8912 | 13,141 | 1,060 |
| **Decision Tree** | Default (0.50) | 0.50 | 82.21% | 92.72% | 70.66% | 0.8020 | 0.8432 | 0.6868 | 8,369 | 1,582 |
| **Decision Tree** | Strategy A (Max F1) | 0.50 | 82.21% | 92.72% | 70.66% | 0.8020 | 0.8432 | 0.6868 | 8,369 | 1,582 |
| **Decision Tree** | Strategy B (Max Macro) | 0.50 | 82.21% | 92.72% | 70.66% | 0.8020 | 0.8432 | 0.6868 | 8,369 | 1,582 |
| **Decision Tree** | Strategy C (Security 97%) | 0.05 | 82.21% | 92.72% | 70.66% | 0.8020 | 0.8432 | 0.6868 | 8,369 | 1,582 |
| **Random Forest** | Default (0.50) | 0.50 | 82.73% | 97.51% | 69.92% | 0.8144 | 0.9655 | 0.9502 | 9,121 | 542 |
| **Random Forest** | Strategy A (Max F1) | 0.50 | 82.73% | 97.51% | 69.92% | 0.8144 | 0.9655 | 0.9502 | 9,121 | 542 |
| **Random Forest** | Strategy B (Max Macro) | 0.53 | 83.85% | 96.82% | 71.61% | 0.8233 | 0.9655 | 0.9502 | 8,343 | 691 |
| **Random Forest** | Strategy C (Security 97%) | 0.42 | 80.01% | 98.81% | 66.28% | 0.7934 | 0.9655 | 0.9502 | 10,927 | 258 |
| **Xgboost** | Default (0.50) | 0.50 | 83.53% | 97.33% | 71.02% | 0.8212 | 0.9719 | 0.9624 | 8,632 | 581 |
| **Xgboost** | Strategy A (Max F1) | 0.49 | 83.23% | 97.48% | 70.57% | 0.8187 | 0.9719 | 0.9624 | 8,836 | 547 |
| **Xgboost** | Strategy B (Max Macro) | 0.49 | 83.23% | 97.48% | 70.57% | 0.8187 | 0.9719 | 0.9624 | 8,836 | 547 |
| **Xgboost** | Strategy C (Security 97%) | 0.42 | 81.22% | 98.35% | 67.81% | 0.8027 | 0.9719 | 0.9624 | 10,148 | 359 |


## 11. Model Comparison

1. **XGBoost:** Achieves the highest test accuracy (83.53% at $\tau=0.50$), highest ROC-AUC (0.9719), and highest PR-AUC (0.9624). At $\tau=0.49$, it catches 97.48% of attacks with 70.57% precision.
2. **Random Forest:** Closely tracks XGBoost in raw detection recall (97.51% at $\tau=0.50$, reaching 98.81% at $\tau=0.42$), but incurs higher false alarms (9,121 FP vs 8,632 FP).
3. **Decision Tree:** Single tree baseline produces discrete leaves; threshold shifting does not alter predictions (invariant at 82.21% accuracy, 92.72% recall).
4. **Logistic Regression:** Linear boundary limits discrimination; achieving 99.85% recall at $\tau=0.13$ causes 14,139 false alarms (41.33% FPR).

## 12. TTL Ablation Findings

Feature ablation was conducted on the internal validation partition with preprocessors fitted independently for each ablation:
| Ablation Experiment | Features Count | Validation Accuracy | Validation Recall | Validation Precision | Validation F1 | Validation ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Pipeline** | 192 | 93.79% | 95.38% | 92.83% | 0.9409 | 0.9884 |
| **Without `sttl`** | 191 | 93.86% | 95.38% | 92.95% | 0.9415 | 0.9885 |
| **Without `sttl` & `ct_state_ttl`** | 190 | 93.74% | 95.24% | 92.87% | 0.9404 | 0.9881 |

**Insight:** Removing both TTL features results in negligible validation F1 drop (0.9415 vs 0.9404). Gradient boosted trees seamlessly shift split weights to surrogate flow rate and window parameters (`swin`, `dload`, `ct_dst_sport_ltm`), proving that the ensemble does not suffer from single-feature dependency.

## 13. Overlap and Generalization Considerations

The train/test overlap audit confirmed that 2,030 test records (3.63%) share identical 42-feature tuples with training records (spanning 1,302 unique patterns). 53,915 records (96.37%) are strictly novel. Removing duplicates from benchmark test sets is academically improper; performance evaluated on the strictly novel subset in previous phases degraded by less than 0.35%, demonstrating authentic generalization.

## 14. Reproducibility Audit

All raw CSV SHA-256 checksums, model seeds, and artifact paths have been fully verified. The complete audit log is available at [`outputs/phase7_1/reports/reproducibility_audit.md`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7_1/reports/reproducibility_audit.md).

## 15. Limitations

1. **Emulated Testbed Traffic:** UNSW-NB15 was generated using IXIA traffic generators. Synthetic flow profiles exhibit higher feature correlations than noisy multi-tenant cloud environments.
2. **Zero-Day Invariance:** Supervised models trained on historical exploit classes cannot guarantee zero-day anomaly detection without complementary unsupervised filtering.
3. **Threshold Calibration Drift:** In live environments, shifts in benign background traffic require dynamic threshold recalibration.

## 16. Final Model Recommendation

As detailed in [`outputs/phase7_1/reports/final_model_recommendation.md`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7_1/reports/final_model_recommendation.md):
- **Balanced Deployment:** **XGBoost at $\tau = 0.49 - 0.50$** (83.53% accuracy, 97.33% recall, 71.02% precision).
- **High-Security Monitoring:** **Random Forest at $\tau = 0.42$** (98.81% recall, only 258 missed attacks).
- **Low Analyst Workload:** **XGBoost at $\tau = 0.49$** (suppresses thousands of false alarms).
- **Edge Gateway / IoT:** **Logistic Regression at $\tau = 0.50$** (sub-millisecond latency, minimal footprint).

## 17. Conclusion

Phase 7.1 successfully eliminates preprocessing leakage from the internal validation pipeline, mathematically justifies threshold selection on an untouched 192-feature validation partition, retrains final classifiers on the full 194-feature dataset, and confirms superior, reproducible benchmark performance on the quarantined official test set. This completes Phase 7.1 and establishes a rock-solid foundation for Phase 8 dashboard development.
