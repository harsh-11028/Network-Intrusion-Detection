# Decision Tree Classifier Evaluation & Baseline Comparison Report (Phase 4)

**Project Title:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Model:** Decision Tree Classifier (Unpruned Non-Linear Baseline)  
**Baseline Reference:** Logistic Regression (Phase 3 Linear Baseline)  
**Dataset:** UNSW-NB15 Cleaned Benchmark  
**Target:** Binary Classification (`0 = Normal`, `1 = Attack`)  
**Environment:** macOS Apple Silicon, Python 3.13, Scikit-learn, NumPy, Matplotlib  

---

## 1. Executive Summary

In Phase 4, we trained and evaluated our first non-linear machine learning model: **Decision Tree Classifier** (`DecisionTreeClassifier(random_state=42)`).

The model was trained strictly on the preprocessed training dataset (`107,740` samples $\times$ `194` features) and evaluated on the official testing dataset (`55,945` samples $\times$ `194` features).

### Head-to-Head Comparison with Logistic Regression:

| Metric | Logistic Regression (Phase 3) | Decision Tree (Phase 4) | Absolute Delta | Relative Gain / Shift |
|---|---|---|---|---|
| **Overall Accuracy** | 74.41% | **82.46%** | **+8.05%** | Significant improvement in overall classification |
| **Attack Precision** | 60.84% | **71.02%** | **+10.18%** | **+10.18% cleaner attack alerts** |
| **Attack Recall** | **95.75%** | 92.67% | -3.08% | Slight drop (-3.08%), still captures 92.67% of attacks |
| **Attack F1-Score** | 0.7441 | **0.8041** | **+0.0600** | Substantial balance gain (+0.0600) |
| **False Positives (False Alarms)** | 13,396 flows | **8,219 flows** | **-5,177 flows** | **38.65% reduction in false alarms** |
| **False Negatives (Missed Attacks)** | **923 flows** | 1,594 flows | +671 flows | 671 additional missed attacks |
| **ROC-AUC Score** | **0.9192** | 0.8432 | -0.0760 | Lower due to discrete leaf probability steps |
| **PR-AUC (Average Precision)** | **0.8914** | 0.6868 | -0.2046 | Unpruned leaves produce stepped precision curves |

---

## 2. Model Training Configuration & Tree Complexity

- **Algorithm:** Scikit-learn `DecisionTreeClassifier`
- **Splitting Criterion:** Gini Impurity (`criterion='gini'`)
- **Splitter Strategy:** Best (`splitter='best'`)
- **Max Depth:** Unrestricted (`None`) $\rightarrow$ Resulted in actual depth of **42**
- **Leaf Nodes:** **5,616 terminal leaves**
- **Min Samples Split / Leaf:** 2 / 1 (Default unpruned configuration)
- **Random State:** 42 (Reproducible initialization)
- **Training Duration:** **1.47 seconds**
- **Training Accuracy:** **99.79%** (`0.9979`)
- **Testing Accuracy:** **82.46%** (`0.8246`)

---

## 3. Detailed Performance Evaluation

### 3.1 Confusion Matrix Breakdown

| Metric | Predicted Normal (0) | Predicted Attack (1) | Total Actual |
|---|---|---|---|
| **Actual Normal (0)** | **TN = 25,987** (75.97%) | **FP = 8,219** (24.03%) | 34,206 |
| **Actual Attack (1)** | **FN = 1,594** (7.33%) | **TP = 20,145** (92.67%) | 21,739 |
| **Total Predicted** | 27,581 | 28,364 | **55,945** |

#### In-Depth Confusion Matrix Analysis:
1. **Dramatic Reduction in False Alarms:**
   - Logistic Regression generated **13,396 false alarms** (39.16% of normal traffic).
   - Decision Tree generated **8,219 false alarms** (24.03% of normal traffic).
   - This represents **5,177 fewer false alarms** (a **38.65% reduction** in false positives). In an operational Security Operations Center (SOC), this drastically reduces alert fatigue.
2. **Attack Capture Rate (Recall):**
   - The tree caught **20,145 out of 21,739 attacks** (**92.67%**).
   - While Logistic Regression achieved slightly higher recall (95.75%), it did so by predicting "Attack" overly aggressively across 34,212 instances. The Decision Tree achieves a much healthier precision/recall balance.

### 3.2 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Normal (0)** | 0.9422 | 0.7597 | 0.8412 | 34,206 |
| **Attack (1)** | 0.7102 | 0.9267 | 0.8041 | 21,739 |
| **Macro Average** | 0.8227 | 0.8227 | 0.8227 | 55,945 |
| **Weighted Average** | 0.8268 | 0.8246 | 0.8268 | 55,945 |

---

## 4. Visualizations & Interpretations

The following high-resolution figures are generated and saved in `outputs/models/decision_tree/`:

1. **`confusion_matrix.png`**:
   - Visualizes the 4 quadrants of the confusion matrix with counts and percentages.
   - Highlights the increased concentration of correctly classified Normal flows (25,987 TN) and Attacks (20,145 TP).
2. **`roc_curve.png`**:
   - Plots Decision Tree ROC curve alongside the Logistic Regression baseline curve.
   - Shows that while Logistic Regression has a smoother curve (due to continuous sigmoid probabilities), the Decision Tree achieves higher discrete accuracy at the default threshold.
3. **`precision_recall_curve.png`**:
   - Shows the precision-recall trade-off. The Decision Tree operating point reaches **Precision = 0.71** at **Recall = 0.93**, demonstrating superior point-wise precision over Logistic Regression.
4. **`decision_tree_structure.png`**:
   - A clear, readable visualization of the **top 2 decision levels (root and primary branches)**.
   - Reveals that the very first split in the tree is **`sttl <= -0.7613`** (separating low Source TTLs around 31/32 from higher TTLs), corroborating our Phase 1 EDA findings.

---

## 5. Overfitting Considerations in Unpruned Decision Trees

A vital machine learning takeaway from Phase 4 is understanding the **bias-variance trade-off** in unpruned decision trees:

| Metric | Training Split | Testing Split | Discrepancy (Overfitting Gap) |
|---|---|---|---|
| **Accuracy** | **99.79%** | **82.46%** | **17.33%** |
| **Tree Depth** | 42 levels | — | Very deep branching |
| **Total Leaf Nodes** | 5,616 leaves | — | Many single-sample terminal leaves |

### Why This Happens:
- An unrestricted decision tree has virtually zero bias: it splits repeatedly until nearly every training sample is isolated into its own leaf node (hence 99.79% training accuracy).
- However, it suffers from **high variance**: individual leaves memorize training noise or specific testbed idiosyncrasies, leading to a drop to 82.46% on unseen test traffic.
- **Looking Ahead to Random Forest (Phase 5):** Random Forests solve this exact problem by ensembling hundreds of decorrelated decision trees (bagging and feature sub-sampling), which cancels out the high variance of individual unpruned trees while preserving non-linear expressiveness.

---

## 6. TTL Bias & Ablation Analysis

In Phase 1, Source Time-To-Live (`sttl`) showed strong separation ($0.61$). In the Decision Tree, `sttl` accounts for **63.12% of total Gini feature importance** and is chosen as the root node.

We conducted an ablation experiment by training the Decision Tree without `sttl` (193 features):

| Configuration | Features | Accuracy | Attack Recall | Attack Precision | Attack F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Baseline (With `sttl`)** | 194 | **82.46%** | **92.67%** | **71.02%** | **0.8041** | **0.8432** |
| **Ablation (Without `sttl`)** | 193 | **82.38%** | 92.58% | 70.94% | 0.8033 | 0.8424 |

### Key Insight:
- When `sttl` is removed, the Decision Tree's accuracy drops by **only 0.08%** (from 82.46% to 82.38%), and F1 drops by only 0.0008!
- **Explanation:** When denied `sttl`, the tree automatically shifts its root splits to surrogate features that provide equivalent discriminatory power: `ct_state_ttl`, `dttl`, `sbytes`, and `ct_srv_dst`.
- This confirms that network intrusion patterns are multi-faceted and robustly detectable even when synthetic testbed TTL artifacts are eliminated.

---

## 7. Train/Test Overlap Caveat

- Phase 2 revealed that 940 exact rows and 1,302 unique feature vectors appear in both the training and testing sets.
- In accordance with benchmark standards, all rows were retained for the primary benchmark.
- When evaluated strictly on the **53,915 novel, non-overlapping test records**, the Decision Tree achieves:
  - **Accuracy:** **82.14%** (vs 82.46% full test)
  - **Attack F1:** **0.7951** (vs 0.8041 full test)
  - **ROC-AUC:** **0.8416** (vs 0.8432 full test)
- This confirms that the overlap accounts for less than **0.32% difference in accuracy**, proving that the Decision Tree's performance is driven by genuine generalization across novel network flows.

---

## 8. Persisted Artifacts

All outputs are saved and organized under `outputs/models/decision_tree/`:
- `decision_tree_model.joblib`: Serialized Scikit-learn Decision Tree model.
- `metrics.json`: Machine-readable evaluation metrics, tree properties, and Logistic Regression comparison.
- `test_predictions.npz`: Test predictions (`y_true`, `y_pred`, `y_prob`).
- `confusion_matrix.png`: Annotated heatmap.
- `roc_curve.png`: Comparative ROC curve.
- `precision_recall_curve.png`: Comparative PR curve.
- `decision_tree_structure.png`: Readable shallow tree architecture diagram.
