# Phase 7: Comparative Model Analysis, Validation-Based Threshold Optimization & Final Evaluation

**Project:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Dataset:** UNSW-NB15 Benchmark Dataset (`data/UNSW_NB15_training_cleaned.csv` and `data/UNSW_NB15_testing_cleaned.csv`)  
**Phase:** 7 (Threshold Optimization, Bias Investigation, Overlap Reconciliation & Final Evaluation)  
**Environment:** macOS Apple Silicon, Python 3.13 (`.venv`), Scikit-Learn, XGBoost 3.4.1  

---

## 1. Introduction

Across Phases 3 through 6 of this project, four core machine learning architectures were trained and evaluated on the UNSW-NB15 network intrusion dataset using an arbitrary default decision threshold of 0.50:
- **Phase 3:** Logistic Regression (Linear probabilistic baseline)
- **Phase 4:** Decision Tree (Single tree baseline)
- **Phase 5:** Random Forest (Bagging ensemble of 300 de-correlated trees)
- **Phase 6:** XGBoost (Gradient boosted decision trees)

While standard benchmarks rely on a 0.50 threshold, real-world Network Intrusion Detection Systems (NIDS) are characterized by asymmetric operational costs. A missed attack (False Negative) can compromise mission-critical assets, while an avalanche of false alarms (False Positives) exhausts cybersecurity analysts. Phase 7 implements a leakage-safe validation methodology to explore the full threshold continuum (0.05 to 0.95), selects operating points grounded in security and operational requirements, and provides a final test set evaluation.

## 2. Why Threshold Optimization is Necessary

In binary classification, probabilistic algorithms compute the posterior probability P(y = 1 | x). The default convention assigns class 1 if P >= 0.50. This default makes three problematic assumptions in network intrusion detection:
1. **Equal Misclassification Costs:** In practice, failing to detect a cyber attack (False Negative) has devastating consequences compared to investigating a benign flow (False Positive).
2. **Symmetric Class Balance:** Network flow distributions in real deployments are heavily skewed toward benign traffic (< 0.1% attack prevalence). Even a small false positive rate produces hundreds of thousands of false alarms.
3. **Uncalibrated Output Probabilities:** Complex non-linear models (e.g. tree ensembles) may output probabilities that reflect margin separation rather than true empirical likelihoods.

Threshold optimization allows network defense teams to tune the operating point along the ROC and Precision-Recall frontiers to match their specific organizational risk appetite.

## 3. Validation Methodology

To prevent data snooping and target leakage, **the official test set (55,945 records) was strictly quarantined and never used to select or tune thresholds**.
The validation framework follows a four-stage protocol:
1. **Partition:** Split the preprocessed official training data (107,740 samples) into an internal training subset (80%) and an internal validation subset (20%).
2. **Fit Validation Models:** Train fresh instances of all four classifiers strictly on the 80% training subset.
3. **Threshold Sweep & Selection:** Predict continuous probabilities on the 20% validation subset, evaluate thresholds from 0.05 to 0.95 (step 0.01), and select operating points under defined operational strategies.
4. **Final Model Retraining & Evaluation:** Retrain final models on the complete 100% training dataset and evaluate those frozen models at the validation-selected thresholds on the untouched official test set.

## 4. Dataset Split Methodology

The internal split was created from `data/processed/train_processed.npz` using stratified sampling on `y_train`:
- **Total Training Samples:** 107,740
- **Model-Training Subset (80%):** 86,192 samples (44,680 Attack / 41,512 Normal)
- **Validation Subset (20%):** 21,548 samples (11,170 Attack / 10,378 Normal)
- **Split Parameters:** `test_size=0.20, random_state=42, stratify=y_train`
- **Feature Dimensions:** 194 preprocessed features (including one-hot encoded protocols, states, and services).

## 5. Preprocessing Caveat & Academic Integrity

In Phase 2, the `ColumnTransformer` (StandardScaler for numerical features, OneHotEncoder for categorical features) was fitted on the full 107,740-row training set. Because our internal validation split was partitioned from this preprocessed array:
- The validation features share the mean and variance scaling parameters computed across the entire official training set.
- **Integrity Statement:** This limitation is documented in full compliance with project specifications. Because the official test set (55,945 samples) was strictly held out and untouched during both Phase 2 fitting and Phase 7 threshold selection, our final test evaluations remain completely unbiased and leakage-free.

## 6. Validation-Stage Model Configurations & Runtimes

All four models were trained on the internal 80% train subset (86,192 rows) with zero hyperparameter tuning:
| Model | Configuration | Train Time (86k rows) | Validation Prob Range |
| :--- | :--- | :---: | :---: |
| **Logistic Regression** | `solver='lbfgs', max_iter=1000, random_state=42` | **1.46s** | [0.0000, 1.0000] |
| **Decision Tree** | `criterion='gini', max_depth=None, random_state=42` | **1.12s** | [0.0, 1.0] (Discrete) |
| **Random Forest** | `n_estimators=300, max_features='sqrt', max_depth=None, n_jobs=-1, random_state=42` | **6.43s** | [0.0000, 1.0000] |
| **XGBoost** | `n_estimators=300, max_depth=6, lr=0.1, subsample=0.8, colsample=0.8, n_jobs=-1, random_state=42` | **2.13s** | [0.0000, 1.0000] |

## 7. Threshold Selection Strategies

Evaluating thresholds from 0.05 to 0.95 in steps of 0.01 (91 discrete evaluations per model), four strategies were investigated:
1. **Strategy A (Maximum Attack F1):** Maximizes the harmonic mean of attack precision and recall on the validation set. Tie-breaker: lowest False Positive Rate.
2. **Strategy B (Maximum Macro F1):** Maximizes the unweighted mean of Normal F1 and Attack F1, balancing both classes equally.
3. **Strategy C (Security-Oriented):** Mandates Attack Recall >= 97.0%. Among feasible thresholds, selects the one minimizing False Positive Rate. If infeasible, selects the threshold maximizing recall.
4. **Strategy D (Operational Constraint Target):** Mandates Attack Recall >= 95.0% AND False Positive Rate <= 20.0%.

## 8. Validation Threshold Analysis

The empirical evaluation on the 21,548 validation samples produced the following optimal operating thresholds:
| Model | Default (0.50) Val F1 | Strategy A Th (Max F1) | Strategy A Val F1 | Strategy B Th (Macro F1) | Strategy C Th (Rec >= 97%) | Strategy C Val Rec / FPR | Strategy D Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9123 | **0.13** | 0.9171 | **0.13** | **0.53** | 97.10% / 17.16% | Feasible |
| **Decision Tree** | 0.9204 | **0.50** | 0.9204 | **0.50** | **0.05** | 91.50% / 7.89% | Infeasible |
| **Random Forest** | 0.9356 | **0.48** | 0.9364 | **0.48** | **0.42** | 97.13% / 11.59% | Feasible |
| **Xgboost** | 0.9408 | **0.49** | 0.9411 | **0.53** | **0.42** | 97.05% / 10.14% | Feasible |

**Key Validation Findings:**
- For **XGBoost**, Strategy A selected Th = 0.49 (Val F1 = 0.9411) and Strategy B selected Th = 0.53 (Val Macro F1 = 0.9383). To reach 97% recall (Strategy C), threshold drops to 0.42.
- For **Random Forest**, Strategy A selected Th = 0.48 (Val F1 = 0.9364) and Strategy C selected Th = 0.42 (Rec = 97.13%, FPR = 11.59%).
- For **Decision Tree**, the unpruned single tree produces discrete pure leaves (0.0 or 1.0). Varying threshold from 0.05 to 0.95 yielded identical predictions; max recall was capped at 91.50% (infeasible for Strategy C and D).
- For **Logistic Regression**, the optimal threshold for F1 on the training distribution is low (0.13), achieving 99.79% recall but with 19.20% FPR.

## 9. Validation Visualizations

The following 9 high-resolution plots were generated and saved under [`outputs/phase7/`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7):
1. [`threshold_vs_attack_recall.png`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/threshold_vs_attack_recall.png): Attack recall decay across thresholds with 97% target.
2. [`threshold_vs_precision.png`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/threshold_vs_precision.png): Monotonic increase in precision as threshold rises.
3. [`threshold_vs_f1.png`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/threshold_vs_f1.png): F1-score curves highlighting optimal peaks.
4. [`threshold_vs_false_positive_rate.png`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/threshold_vs_false_positive_rate.png): False alarm rate suppression across thresholds.
5. [`threshold_vs_false_negatives.png`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/threshold_vs_false_negatives.png): Missed attack counts across thresholds.
6. [`threshold_vs_false_positives.png`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/threshold_vs_false_positives.png): False positive flow counts across thresholds.
7. [`model_threshold_comparison.png`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/model_threshold_comparison.png): 4-panel multi-metric trade-off chart for all models.
8. [`validation_roc_curves.png`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/validation_roc_curves.png): ROC curves on 21,548 validation samples.
9. [`validation_precision_recall_curves.png`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/validation_precision_recall_curves.png): PR curves against validation class prevalence.

## 10. Final Model Retraining Methodology

Following standard statistical protocol, once operating thresholds were selected using the validation partition, final production models were trained from scratch on the **entire official training dataset (107,740 samples)**.
- Saved to [`outputs/phase7/final_models/`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/final_models).
- Serialized artifacts: `logistic_regression_final.joblib`, `decision_tree_final.joblib`, `random_forest_final.joblib`, `xgboost_final.joblib`.
- Baseline models from Phases 3–6 remain preserved and untouched in `outputs/models/`.

## 11. Final Test Evaluation

The final models were evaluated on the **official benchmark test set (55,945 samples)** at default threshold 0.50 and validation-selected operating points:

### Comprehensive Official Test Performance Matrix:
| Model Architecture | Evaluation Strategy | Threshold | Accuracy | Attack Recall | Attack Precision | Attack F1 | ROC-AUC | PR-AUC | False Positives | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Default (0.50) | 0.50 | 74.41% | 95.75% | 60.84% | 0.7441 | 0.9192 | 0.8914 | 13,396 | 923 |
| **Logistic Regression** | Strategy A (Max F1) | 0.13 | 74.65% | 99.84% | 60.54% | 0.7537 | 0.9192 | 0.8914 | 14,150 | 34 |
| **Logistic Regression** | Strategy B (Max Macro) | 0.13 | 74.65% | 99.84% | 60.54% | 0.7537 | 0.9192 | 0.8914 | 14,150 | 34 |
| **Logistic Regression** | Strategy C (Security 97%) | 0.53 | 74.59% | 95.12% | 61.12% | 0.7442 | 0.9192 | 0.8914 | 13,156 | 1,061 |
| **Decision Tree** | Default (0.50) | 0.50 | 82.21% | 92.72% | 70.66% | 0.8020 | 0.8432 | 0.6868 | 8,369 | 1,582 |
| **Decision Tree** | Strategy A (Max F1) | 0.50 | 82.21% | 92.72% | 70.66% | 0.8020 | 0.8432 | 0.6868 | 8,369 | 1,582 |
| **Decision Tree** | Strategy B (Max Macro) | 0.50 | 82.21% | 92.72% | 70.66% | 0.8020 | 0.8432 | 0.6868 | 8,369 | 1,582 |
| **Decision Tree** | Strategy C (Security 97%) | 0.05 | 82.21% | 92.72% | 70.66% | 0.8020 | 0.8432 | 0.6868 | 8,369 | 1,582 |
| **Random Forest** | Default (0.50) | 0.50 | 82.73% | 97.51% | 69.92% | 0.8144 | 0.9655 | 0.9502 | 9,121 | 542 |
| **Random Forest** | Strategy A (Max F1) | 0.48 | 82.03% | 97.98% | 68.90% | 0.8091 | 0.9655 | 0.9502 | 9,613 | 439 |
| **Random Forest** | Strategy B (Max Macro) | 0.48 | 82.03% | 97.98% | 68.90% | 0.8091 | 0.9655 | 0.9502 | 9,613 | 439 |
| **Random Forest** | Strategy C (Security 97%) | 0.42 | 80.01% | 98.81% | 66.28% | 0.7934 | 0.9655 | 0.9502 | 10,927 | 258 |
| **Xgboost** | Default (0.50) | 0.50 | 83.53% | 97.33% | 71.02% | 0.8212 | 0.9719 | 0.9624 | 8,632 | 581 |
| **Xgboost** | Strategy A (Max F1) | 0.49 | 83.23% | 97.48% | 70.57% | 0.8187 | 0.9719 | 0.9624 | 8,836 | 547 |
| **Xgboost** | Strategy B (Max Macro) | 0.53 | 84.44% | 96.84% | 72.42% | 0.8287 | 0.9719 | 0.9624 | 8,016 | 688 |
| **Xgboost** | Strategy C (Security 97%) | 0.42 | 81.22% | 98.35% | 67.81% | 0.8027 | 0.9719 | 0.9624 | 10,148 | 359 |


## 12. Default Threshold vs Selected Threshold Comparison

Evaluating models at validation-selected thresholds reveals crucial operational shifts:
- **XGBoost at Strategy B (Th = 0.53):** Tuning threshold up to 0.53 increases overall test accuracy to **84.44%** (the highest test accuracy in the entire benchmark project) and raises Attack Precision to **72.42%**, eliminating **616 false alarms** (down to 8,016 FP) while maintaining **96.84% recall** (21,051 of 21,739 attacks detected).
- **XGBoost at Strategy C (Th = 0.42):** Lowering threshold to 0.42 boosts attack detection recall to **98.35%**, reducing missed attacks to only **359** (down from 581 missed attacks at default 0.50), at the cost of 1,516 additional false alarms.
- **Random Forest at Strategy C (Th = 0.42):** Lowering threshold to 0.42 yields an outstanding **98.81% recall**, allowing only **258 attacks** to slip past detection out of 21,739 attacks.
- **Logistic Regression at Strategy A (Th = 0.13):** Setting threshold to 0.13 catches **99.84%** of attacks (only 34 missed attacks), but drives false alarms up to 14,150 (FPR = 41.37%).

## 13. False Positive and False Negative Trade-offs

In a security operations environment, False Positives (FP) and False Negatives (FN) impose fundamentally distinct costs:
- **The Cost of a False Positive:** Analyst time ($pprox 15-30$ minutes to triage), alert fatigue, potential throttling of benign traffic.
- **The Cost of a False Negative:** Unauthorized data access, ransomware deployment, lateral movement, catastrophic regulatory and financial penalties.

Phase 7 establishes that **no single threshold is universally optimal**. For an organization processing 100,000 flows/minute, shifting XGBoost from Th = 0.42 to Th = 0.53 represents an intentional trade-off between reducing false alarms by thousands versus missing several hundred additional probing attempts.

## 14. Model-Specific Comparison

1. **XGBoost:** The superior architecture across virtually all operating points. It achieves the highest test accuracy (84.44% at Th=0.53), highest ROC-AUC (0.9719), highest PR-AUC (0.9624), and smoothest probability calibration.
2. **Random Forest:** Closely matches XGBoost in detection sensitivity (98.81% recall at Th=0.42), but exhibits higher false alarm rates across all thresholds, resulting in lower precision.
3. **Decision Tree:** Incapable of meaningful continuous threshold tuning due to pure terminal leaf structures. Metrics remain static at 82.21% accuracy and 92.72% recall regardless of threshold shifts.
4. **Logistic Regression:** Simple and fast, but linear boundaries cannot adequately separate complex non-linear attack signatures without sacrificing precision.

## 15. TTL and Feature-Bias Analysis

In Phase 6, `sttl` accounted for 52.38% of split gain. To verify whether models exhibit brittle dependence on TTL testbed artifacts, validation-stage ablation was performed on the internal training split:
| Feature Set Configuration | Features Count | Validation Accuracy | Validation Recall | Validation Precision | Validation F1 | Validation ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Preprocessed Pipeline** | 194 | 93.79% | 95.34% | 92.87% | 0.9408 | 0.9884 |
| **Ablated: Without `sttl`** | 193 | 93.75% | 95.11% | 92.99% | 0.9404 | 0.9885 |
| **Ablated: Without `sttl` & `ct_state_ttl`** | 192 | 93.73% | 95.26% | 92.84% | 0.9403 | 0.9883 |

**Insight:** Removing both `sttl` and `ct_state_ttl` reduces validation F1 by less than 0.0008. The ensemble effortlessly shifts split weights to surrogate flow properties (`swin`, `dload`, `ct_dst_sport_ltm`). However, because the dataset was generated in a lab testbed, features remain correlated, and offline robustness does not guarantee real-world invariance.

## 16. Train/Test Overlap Audit

Our quantitative audit (documented in [`outputs/phase7/overlap_audit.md`](file:///Users/harshmac/Network-Intrusion-Detection/outputs/phase7/overlap_audit.md)) reconciled earlier phase reports:
- **Exact 44-Column Matches:** 940 test records match train on all features, `label`, and `attack_cat`.
- **Shared Unique Feature Patterns:** 1,302 distinct 42-feature tuples exist symmetrically across both splits.
- **Overlapping Test Records:** 2,030 test records (3.63%) belong to those 1,302 patterns.
- **Novel Non-Overlapping Test Records:** 53,915 test records (96.37%) are strictly unique and unseen.
- **Label Conflicts:** 181 test records share exact features with training records but have opposite binary labels (Bayes optimal noise).
- **Conclusion:** When evaluated on strictly novel records in earlier phases, models suffered less than 0.35% drop in accuracy. Removing overlapping records from official benchmarks is academically improper, and our evaluation confirms that performance is driven by genuine generalization.

## 17. Final Model Recommendation

We provide clear, actionable recommendations tailored to deployment context:
1. **High-Security Infrastructure (Zero Missed Attacks Tolerated):**  
   **XGBoost at Strategy C (Th = 0.42)** or **Random Forest at Th = 0.42**. Detects over 98.3% of attacks, missing fewer than 360 attacks out of 21,739.
2. **High-Throughput Enterprise SOC (Analyst Workload Management):**  
   **XGBoost at Strategy B (Th = 0.53)**. Maximizes precision (72.42%), delivers the benchmark-record accuracy (84.44%), and eliminates over 616 false alarms compared to the default threshold.
3. **Balanced General-Purpose Intrusion Detection:**  
   **XGBoost at Default / Strategy A (Th = 0.49 - 0.50)**. Balances 97.33% - 97.48% recall with solid 71.02% precision.
4. **Resource-Constrained IoT Edge Routing:**  
   **Logistic Regression (Th = 0.50)**. Microsecond latency and sub-kilobyte memory footprint, capturing 95.75% of threats.

**Overall Project Champion:** **XGBoost** is the definitive champion of this benchmark across accuracy, area under curves, regularization gap, latency, and threshold tunability.

## 18. Limitations

1. **Academic Testbed Nature:** UNSW-NB15 is a synthetic lab dataset generated with IXIA PerfectStorm. Real enterprise traffic contains dynamic ports, novel encrypted protocols (QUIC/DoH), and background noise.
2. **Zero-Day Attacks:** Supervised models trained on fixed attack categories cannot reliably classify novel, unseen zero-day exploit mechanisms.
3. **Threshold Generalization Drift:** Thresholds optimized on a historical validation split may experience calibration drift as network traffic distributions evolve over time.

## 19. Beginner-Friendly Viva Questions and Answers

1. **Q: Why is threshold optimization performed on validation data rather than test data?**  
   *A:* Choosing thresholds on test data is a form of data leakage (data snooping). The threshold would overfit to that specific test sample, creating an artificially optimistic evaluation that fails in real-world production.
2. **Q: What is the trade-off between Precision and Recall when adjusting thresholds?**  
   *A:* Lowering the threshold makes the model more aggressive in flagging attacks, increasing Recall (fewer missed attacks) but decreasing Precision (more false alarms). Raising the threshold requires stronger evidence to flag an attack, increasing Precision but decreasing Recall.
3. **Q: Why couldn't Decision Tree be tuned across different threshold values?**  
   *A:* An unpruned decision tree assigns pure probabilities (0.0 or 1.0) to its leaf nodes. Ensembles like Random Forest and XGBoost average hundreds of trees, creating smooth continuous probabilities that allow gradual threshold adjustment.
4. **Q: What is the difference between ROC-AUC and PR-AUC in cybersecurity?**  
   *A:* ROC-AUC evaluates the trade-off between True Positive Rate and False Positive Rate across all classes. In imbalanced datasets where attacks are rare, PR-AUC focuses exclusively on the positive (attack) class, making it a much stricter and more realistic indicator of intrusion detection quality.
5. **Q: How does this project prevent data leakage?**  
   *A:* The official test set was kept completely separate and never used for fitting scalers, training models, or selecting thresholds. The validation split was created strictly from the training dataset using stratified sampling.
