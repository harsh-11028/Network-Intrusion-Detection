# Train/Test Overlap Methodology Audit: UNSW-NB15

**Project:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Phase:** Phase 7 — Comparative Model Analysis & Final Evaluation  
**Dataset:** UNSW-NB15 (`UNSW_NB15_training_cleaned.csv` and `UNSW_NB15_testing_cleaned.csv`)  
**Scope:** Rigorous verification and methodological audit of train/test data overlap across raw features, processed representations, and target relationships.

---

## 1. Executive Summary

In previous project phases (Phases 2 through 6), various metrics were reported concerning data overlap between the official training set (107,740 samples) and testing set (55,945 samples):
- Phase 2 reported **940 exact rows** and **1,302 unique feature vectors**.
- Phases 3, 4, 5, and 6 reported evaluating models on **53,915 novel test records** (corresponding to **2,030 overlapping test records**).

This audit resolves how these figures were derived, reconciles the terminology, analyzes label consistency across identical feature vectors, evaluates one-to-many match multiplicity, and examines the impact of one-hot encoding and feature scaling.

---

## 2. Quantitative Overlap Breakdown

Our systematic audit of the raw and processed datasets establishes the following definitive counts:

| Overlap Category | Scope / Definition | Matching Count | Percentage of Test Set (55,945) |
| :--- | :--- | :---: | :---: |
| **Exact 44-Column Row Matches** | Identical 42 features + identical binary `label` + identical multi-class `attack_cat` | **940 rows** | **1.68%** |
| **Exact 43-Column Matches** | Identical 42 features + identical binary `label` (ignoring `attack_cat` subcategory) | **1,849 rows** | **3.31%** |
| **Feature-Only Test Matches** | Test records whose 42 raw feature values match at least one training record (ignoring target) | **2,030 rows** | **3.63%** |
| **Novel Test Records** | Test records whose 42 raw feature values appear nowhere in the training dataset | **53,915 rows** | **96.37%** |
| **Unique Feature Patterns in Train** | Deduplicated 42-feature tuples across the 107,740 training records | **101,040 patterns** | — |
| **Unique Feature Patterns in Test** | Deduplicated 42-feature tuples across the 55,945 testing records | **53,946 patterns** | — |
| **Shared Unique Feature Patterns** | Intersection of unique 42-feature patterns between train and test | **1,302 patterns** | — |
| **Label Contradictions (Conflicts)** | Test records matching a training feature vector but assigned an **opposite binary label** | **181 rows** | **0.32%** |
| **Processed 194-Feature Matches** | Matching 194-dimensional vectors in `test_processed.npz` (rounded to 6 decimals) | **2,034 rows** | **3.64%** |

---

## 3. Reconciliation of Previous Numbers

The apparent discrepancies across earlier reports are fully explained by differences in the unit of measurement:
1. **1,302 vs 2,030:**
   - **`1,302`** is the count of **unique, deduplicated feature patterns** shared between the two sets:  
     $$\vert \text{UniqueFeatures}_{\text{train}} \cap \text{UniqueFeatures}_{\text{test}} \vert = 1,302$$
   - **`2,030`** is the total number of **individual test rows** whose feature vector belongs to that shared set of 1,302 patterns.
   - Because multiple network flows in the test set share identical features (multiplicity), 1,302 distinct patterns account for 2,030 test instances.
2. **940 vs 1,849 vs 2,030:**
   - **`940`** represents exact matches across **all 44 columns**, requiring identity in both the binary label and the specific attack category text (e.g., `'Exploits'`, `'Generic'`, `'Fuzzers'`).
   - **`1,849`** represents matches across the 42 features plus the binary `label` (`0` vs `1`), allowing minor differences in the textual `attack_cat` annotation (e.g. slight reclassification between attack subcategories).
   - **`2,030`** represents feature-only matches (42 columns), which includes 1,849 label-concordant records and 181 label-contradicting records.

---

## 4. Multiplicity & One-to-Many Relationships

Network intrusion datasets naturally record repetitive flows from standard protocols (e.g., DNS queries, ARP broadcasts, keep-alive heartbeats, automated SYN scans).
- In the training set, several feature patterns repeat dozens or hundreds of times.
- When an identical connection profile appears in both training and testing, a single training pattern matches multiple test records, and multiple training records may match a single test record.
- Specifically, the 1,302 shared unique patterns correspond to:
  - **4,461 training rows** (4.14% of the training set).
  - **2,030 testing rows** (3.63% of the testing set).
- This confirms that repetitive network behavior exists symmetrically in both splits.

---

## 5. Label Contradiction Analysis (Ground-Truth Inconsistencies)

An essential finding of this audit is that identical network flow profiles do not always have identical labels:
- **181 test records** share exact 42-feature tuples with records in the training set, but possess the **opposite binary label**:
  - Test records labeled **Attack (`1`)** whose exact feature vector is labeled **Normal (`0`)** in training.
  - Test records labeled **Normal (`0`)** whose exact feature vector is labeled **Attack (`1`)** in training.
- **Root Cause in Network Data:**
  - Common service requests (such as standard ICMP echo requests, basic HTTP GETs, or DNS resolutions) can be generated by benign user activity or by an attacker conducting reconnaissance. Without payload inspection or deeper host contextual features, pure flow-level statistical metrics cannot disambiguate benign vs malicious intent for identical minimal flows.
- **Impact on Supervised Classifiers:**
  - Any model evaluated on these 181 conflicting samples faces Bayes optimal error; a deterministic function cannot correctly predict contradictory labels for identical inputs.
  - However, because these 181 rows constitute only **0.32%** of the test set, their net distortion on benchmark accuracy and F1 is negligible.

---

## 6. Raw vs Preprocessed Vector Space Matches

When evaluating overlap on the preprocessed 194-dimensional feature arrays (`train_processed.npz` and `test_processed.npz`):
- Applying `StandardScaler` to numerical columns and `OneHotEncoder` (with `handle_unknown='ignore'`) to categorical columns maps raw records to continuous floating-point vectors.
- Exact byte-level floating point comparisons can fail due to tiny precision artifacts.
- When rounded to 6 decimal places:
  - **2,034 test rows** match training vectors in processed space.
  - The slight difference (+4 rows compared to raw 2,030) arises from test categories that were unseen during training and consequently encoded as all-zeros across the one-hot columns, collapsing rare test categories into the same zero-vector representation as missing/unseen training tokens.

---

## 7. Methodological Conclusions & Academic Standards

1. **Protocol Adherence:** In benchmark evaluations on established datasets like UNSW-NB15, overlapping test samples must **never be silently deleted** from the official benchmark test set. The official test split (55,945 samples) must remain the standard benchmark baseline to enable fair comparison with published literature.
2. **Generalization Integrity:** The fact that 96.37% (53,915 samples) of the test dataset is completely novel—and that all evaluated models (LR, DT, RF, XGBoost) experience less than a **0.35% drop in performance** when evaluated strictly on the novel partition—proves that model performance is driven by genuine structural learning rather than memorization of overlapping vectors.
3. **Real-World Reality:** Deduplication on a testbed dataset does not prove real-world generalization. True network environments experience non-stationary traffic distributions, concept drift, zero-day vulnerabilities, and topological differences that no offline split can completely simulate.
