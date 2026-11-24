# XGBoost Classifier Evaluation & Four-Model Benchmark Report (Phase 6)

**Project Title:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Model:** eXtreme Gradient Boosting Classifier (`xgboost 3.4.1`)  
**Comparison Baselines:** Logistic Regression (Phase 3), Decision Tree (Phase 4), Random Forest (Phase 5)  
**Dataset:** UNSW-NB15 Cleaned Benchmark  
**Target:** Binary Classification (`0 = Normal`, `1 = Attack`)  
**Environment:** macOS Apple Silicon, Python 3.13, XGBoost, Scikit-learn, NumPy, Matplotlib  

---

## 1. Executive Summary

In Phase 6, we implemented, trained, and rigorously evaluated an **eXtreme Gradient Boosting (XGBoost) Classifier** for binary network intrusion detection.

The model was trained strictly on the preprocessed training dataset (`107,740` samples $	imes$ `194` features) and evaluated on the official testing dataset (`55,945` samples $	imes$ `194` features).

### Benchmark Highlights:
1. **Highest Overall Accuracy:** XGBoost achieves **83.53% test accuracy**, outperforming Logistic Regression (74.41%), Decision Tree (82.46%), and Random Forest (82.84%).
2. **Highest Overall Discriminative Power:** Delivers **ROC-AUC of 0.9719** and **PR-AUC of 0.9624**, setting the project's highest global separation benchmark.
3. **Top F1-Score:** Reaches **0.8212 Attack F1-Score** and **0.8343 Macro F1-Score**.
4. **Controlled Overfitting:** Unlike unpruned Decision Trees and Random Forests (which memorized the training set to 99.79%), XGBoost regularized boosting restricted training accuracy to **95.56%**, cutting the generalization gap to **12.03%** (vs. 16.94% for Random Forest).

---

## 2. Why XGBoost Is Suitable for Intrusion Detection

1. **Sequential Gradient Correction:** Unlike Random Forest (which averages independent trees in parallel), XGBoost trains trees sequentially: each new tree targets the gradient residuals (errors) of the previous ensemble, specifically refining the boundary between subtle malicious probing and normal background traffic.
2. **Second-Order Loss Optimization:** Uses Newton-Raphson second-order Taylor expansion (gradients and Hessians) to achieve faster, more stable loss convergence on complex tabular distributions.
3. **Built-in Regularization:** Penalizes tree complexity via L1/L2 leaf weights and tree-depth limits (`max_depth=6`), directly combating the over-memorization observed in deep unpruned trees.
4. **Computational Efficiency on High Dimensions:** Fast histogram-based binning and multithreaded tree construction completed 300 boosting rounds in just **2.30 seconds**.

---

## 3. Dataset Information & Pipeline Reuse

- **Training Split:** `107,740` rows $	imes$ `194` transformed features.
- **Testing Split:** `55,945` rows $	imes$ `194` transformed features.
- **Features Breakdown:** 39 numerical continuous network metrics + 155 one-hot encoded categories across `proto` (133), `service` (13), and `state` (9).
- **Target Ground Truth:** Binary indicator `label` (`0 = Normal`, `1 = Attack`).
- **Zero Target Leakage:** Neither `label` nor `attack_cat` was present in the feature matrix $X$.
- **Zero Ingestion Leakage:** All imputer medians, standard scaler means, and one-hot encoder categories were fitted exclusively on training data during Phase 2.

---

## 4. Model Architecture & Training Configuration

The model was initialized with the official Phase 6 baseline configuration without test-set tuning:
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
```
- **XGBoost Version:** `3.4.1`
- **Training Duration:** **2.30 seconds**
- **Inference Speed:** **0.0709 seconds** across 55,945 flows (~13 microseconds per network flow).

---

## 5. Comprehensive Four-Model Performance Benchmark

All figures below represent actual recorded results on the identical official UNSW-NB15 test set:

| Metric | Logistic Regression (Phase 3) | Decision Tree (Phase 4) | Random Forest (Phase 5) | XGBoost (Phase 6) | Best Model |
|---|---|---|---|---|---|
| **Overall Accuracy** | 74.41% | 82.46% | 82.84% | **83.53%** | **XGBoost (83.53%)** |
| **Attack Detection Rate (Recall)** | 95.75% | 92.67% | **97.45%** | 97.33% | **Random Forest (97.45%)** / XGB close (97.33%) |
| **Attack Precision** | 60.84% | **71.02%** | 70.08% | **71.02%** | **XGBoost & Decision Tree (71.02%)** |
| **Attack F1-Score** | 0.7441 | 0.8041 | 0.8153 | **0.8212** | **XGBoost (0.8212)** |
| **Normal Recall (Specificity)** | 60.84% | **75.97%** | 73.56% | 74.76% | **Decision Tree (75.97%)** |
| **Normal Precision** | 95.75% | 94.22% | **97.84%** | 97.78% | **Random Forest (97.84%)** / XGB close (97.78%) |
| **Macro Average F1** | 0.7441 | 0.8227 | 0.8276 | **0.8343** | **XGBoost (0.8343)** |
| **ROC-AUC Score** | 0.9192 | 0.8432 | 0.9655 | **0.9719** | **XGBoost (0.9719)** |
| **PR-AUC Score** | 0.8914 | 0.6868 | 0.9502 | **0.9624** | **XGBoost (0.9624)** |
| **False Positives (False Alarms)** | 13,396 flows | **8,219 flows** | 9,043 flows | 8,632 flows | **Decision Tree (8,219)** |
| **False Negatives (Missed Attacks)** | 923 flows | 1,594 flows | **555 flows** | 581 flows | **Random Forest (555)** / XGB close (581) |
| **Training Duration** | 1.80s | **1.59s** | 9.72s | 2.45s | **Decision Tree (1.59s)** / XGB (2.45s) |

---

## 6. Confusion Matrix Breakdown & Security Interpretation

```
                       Predicted Normal (0)    Predicted Attack (1)
Actual Normal (0)      TN = 25,574 (74.76%)    FP =  8,632 (25.24%)   [Support: 34,206]
Actual Attack (1)      FN =    581 ( 2.67%)    TP = 21,158 (97.33%)   [Support: 21,739]
```

### Operational Security Insights:
1. **Exceptional Detection Capability:** XGBoost catches **21,158 out of 21,739 attacks** (**97.33% Attack Recall**). Only 581 malicious flows bypassed detection, preserving enterprise network perimeter defense.
2. **False Alarm Reduction:** XGBoost generated **8,632 false alarms**, which is **4,764 fewer false alarms than Logistic Regression** and **411 fewer false alarms than Random Forest**.
3. **The Precision/Recall Compromise:** Random Forest missed 26 fewer attacks (555 vs. 581) but generated 411 more false alarms (9,043 vs. 8,632). XGBoost achieves a higher F1-score (**0.8212 vs. 0.8153**) by maintaining virtually identical recall while filtering out hundreds of false alarms.

---

## 7. Visualizations & Multi-Model Analysis

All figures are saved under `outputs/models/xgboost/`:

1. **`confusion_matrix.png`**:
   - Heatmap illustrating the distribution of TN, FP, FN, and TP flows.
2. **`roc_curve.png`**:
   - Comparative 4-model ROC curves. The red XGBoost curve achieves the highest overall area (**0.9719**), dominating all previous models.
3. **`precision_recall_curve.png`**:
   - Comparative 4-model PR curves. Shows that XGBoost sustains superior precision at high recall thresholds (**PR-AUC = 0.9624**).
4. **`feature_importance.png`**:
   - Bar chart of the top 20 features ranked by normalized Gain.
5. **`model_comparison.png`**:
   - Multi-bar benchmark showing Accuracy, Attack Precision, Attack Recall, Attack F1, ROC-AUC, and PR-AUC across all four models.
6. **`xgboost_probability_distribution.png`**:
   - Dual-density distribution visualizing predicted probabilities $P(y=1)$ for actual Normal vs. Attack flows. Demonstrates clear bimodal separation with minimal mass around the 0.5 decision threshold.

---

## 8. Feature Importance Analysis (Gain Metric)

### Top 10 Most Important Features in XGBoost:

| Rank | Feature Name | Gain Importance (%) | Domain Description |
|---|---|---|---|
| 1 | `sttl` | 52.38% |
| 2 | `ct_state_ttl` | 12.13% |
| 3 | `swin` | 4.10% |
| 4 | `is_sm_ips_ports` | 3.38% |
| 5 | `proto_tcp` | 2.06% |
| 6 | `dload` | 1.97% |
| 7 | `dttl` | 1.92% |
| 8 | `ct_dst_sport_ltm` | 1.92% |
| 9 | `ct_srv_dst` | 1.31% |
| 10 | `state_CON` | 1.28% |

### What Feature Importance Represents:
- The importance metric reported here is **Gain**: the fractional contribution of each feature to the model based on the total gain (reduction in log-loss) this feature brings across all trees.
- **Important Caveat:** High feature importance indicates that the model finds the feature useful for separating classes within the dataset's distribution. **It does not imply causal relationship** in real-world network attacks.

---

## 9. Overfitting Investigation

| Metric | Training Split | Testing Split | Generalization Gap |
|---|---|---|---|
| **Accuracy** | **95.56%** | **83.53%** | **12.03%** |
| **Number of Estimators** | 300 | 300 | Sequential boosted rounds |
| **Tree Max Depth** | 6 | 6 | Constrained depth |

### Discussion on Overfitting:
- Both the single Decision Tree (Phase 4) and Random Forest (Phase 5) memorized the training set to **99.79%**, yielding generalization gaps of **17.33%** and **16.94%**.
- XGBoost achieved a training accuracy of **95.56%**, resulting in a smaller generalization gap of **12.03%**.
- **Why?** Constrained tree depth (`max_depth=6`), shrinkage (`learning_rate=0.1`), and random subsampling (`subsample=0.8`, `colsample_bytree=0.8`) prevent individual trees from building single-sample leaf partitions.
- While a 12% gap indicates moderate variance, it demonstrates noticeably superior regularization compared to unrestricted tree ensembles.

---

## 10. TTL Bias & Ablation Analysis

During Phase 1, Source Time-To-Live (`sttl`) showed strong correlation due to testbed hardware generator configurations.

We trained and evaluated an independent XGBoost model on the dataset without `sttl` (193 features):

| Configuration | Features | Accuracy | Attack Recall | Attack Precision | Attack F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Baseline (With `sttl`)** | 194 | **83.53%** | **97.33%** | 71.02% | **0.8212** | **0.9719** | **0.9624** |
| **Ablation (Without `sttl`)** | 193 | **83.56%** | 97.16% | **71.11%** | **0.8212** | 0.9712 | 0.9615 |

### Conclusion on TTL Bias:
- Without `sttl`, XGBoost accuracy is **83.56%** (a slight 0.03% increase), precision rises to **71.11%**, and F1 is completely unchanged at **0.8212**.
- **What this proves:** XGBoost does not rely on `sttl` as an essential artifact. When `sttl` is removed, gradient boosting readily finds alternative discriminating features across network throughput, packet sizes, and TCP state transitions.
- **What this does not prove:** It does not prove that other features in the dataset are free from testbed artifacts; it only proves that the model does not depend on the specific TTL anomaly.

---

## 11. Train/Test Overlap Investigation

- **Methodology:** We matched the 42 input feature vectors of the testing set against all unique training feature vectors.
- **Findings:** Out of 55,945 testing records, **2,030 records (3.63%)** share identical feature vectors with the training set, while **53,915 records (96.37%)** are strictly novel.

### Evaluation on Strictly Novel Test Records:

| Test Evaluation Subset | Record Count | Accuracy | Attack Precision | Attack Recall | Attack F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Full Official Test Set** | 55,945 | **83.53%** | 71.02% | **97.33%** | **0.8212** | **0.9719** | **0.9624** |
| **Novel Non-Overlapping Test Set** | 53,915 | 83.43% | **70.17%** | 97.18% | 0.8150 | 0.9704 | 0.9582 |

- **Verification:** Performance on novel samples drops by only **0.10% in accuracy** and **0.0015 in ROC-AUC**. This confirms that the model's high benchmark metrics are driven by authentic generalization rather than memorized overlapping samples.

---

## 12. Limitations & Deployment Caveats

1. **Lab Testbed Distribution Shift:** UNSW-NB15 was generated in an emulated cyber-range testbed using the IXIA PerfectStorm generator. Live production traffic will feature newer protocols, dynamic port allocations, and emerging zero-day exploit variants not present in this benchmark.
2. **False Positives in Enterprise Networks:** While 8,632 false positives on 55k flows is a major reduction compared to Logistic Regression, in an enterprise network handling millions of flows per minute, 8,632 false alarms would overwhelm human analysts. Threshold tuning or secondary anomaly filters will be required.
3. **No Claim of Live Deployment Readiness:** This project benchmarks offline academic intrusion detection performance and does not claim real-time operational deployment readiness.

---

## 13. Viva / Interview Questions & Explanations (Beginner-Friendly)

1. **Q: Why does XGBoost outperform Random Forest on this dataset?**  
   *A:* Random Forest builds independent trees that vote equally. If many trees make the same mistake on difficult edge cases, the ensemble still errs. XGBoost builds trees sequentially, where each subsequent tree focuses specifically on the residual errors of the previous trees, creating tighter decision boundaries for ambiguous network flows.

2. **Q: Why did XGBoost train faster than Random Forest?**  
   *A:* In Random Forest, 300 deep trees were grown to unrestricted depth (depth 42 with 5,616 leaves each). In XGBoost, tree depth was constrained to `max_depth=6` (at most 64 leaves per tree), greatly reducing tree construction operations.

3. **Q: What is the difference between Gini Importance in Random Forest and Gain in XGBoost?**  
   *A:* Random Forest Gini importance measures how much each feature decreases impurity across random bootstrap samples. XGBoost Gain measures the average improvement in the gradient loss function brought by splits using that feature.

4. **Q: Why did we test without `sttl`?**  
   *A:* In academic testbeds, traffic generators often use synthetic default TTL values that don't reflect real-world internet routing. Removing `sttl` tests whether our model is genuinely learning intrusion patterns or just memorizing the generator's operating system settings.

---

## 14. Persisted Artifacts

All outputs are saved and organized under `outputs/models/xgboost/`:
- `xgboost_model.joblib`: Serialized Scikit-learn/XGBoost model.
- `metrics.json`: Machine-readable evaluation metrics, hyperparameters, and full 4-model comparison.
- `test_predictions.npz`: Test predictions (`y_true`, `y_pred`, `y_prob`).
- `feature_importance.csv`: All 194 features ranked by Gain importance.
- `confusion_matrix.png`: Confusion matrix heatmap.
- `roc_curve.png`: Comparative ROC curve (LR vs DT vs RF vs XGB).
- `precision_recall_curve.png`: Comparative PR curve (LR vs DT vs RF vs XGB).
- `feature_importance.png`: Top 20 features bar chart.
- `model_comparison.png`: Metric comparison chart across all four models.
- `xgboost_probability_distribution.png`: Dual probability density plot.
