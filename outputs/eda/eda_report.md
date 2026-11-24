# Exploratory Data Analysis & Dataset Validation Report

**Project Title:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Dataset:** UNSW-NB15 Cleaned Benchmark (`UNSW_NB15_training_cleaned.csv` & `UNSW_NB15_testing_cleaned.csv`)  
**Phase:** Phase 1 — Dataset Validation and Exploratory Data Analysis  
**Environment:** macOS Apple Silicon, Python 3.13, Pandas, NumPy, Matplotlib, Seaborn  

---

## 1. Executive Summary

This report documents the dataset validation, quality inspection, and exploratory data analysis (EDA) conducted on the cleaned UNSW-NB15 network intrusion detection benchmark dataset.

The primary objective of Phase 1 is to verify the integrity, schema compatibility, feature distributions, potential target leakage risks, and synthetic artifacts of the datasets **prior** to any model training or preprocessing pipeline construction.

### Key Validation Outcomes:
1. **Schema Consistency:** Both training and testing sets share an identical schema of **44 columns** (42 input features and 2 target variables: `label` and `attack_cat`).
2. **Cleanliness:** Zero missing values (`NaN`/`null`), zero infinite values, and zero negative numbers were detected across both datasets.
3. **Target Alignment:** The binary target `label` and multi-class target `attack_cat` exhibit **100% strict ground-truth alignment** (`Normal` <=> `0`, all attack classes <=> `1`).
4. **Dimensions:**
   - **Training:** 107,740 rows × 44 columns
   - **Testing:** 55,945 rows × 44 columns
   *(Discrepancy with raw benchmark dimensions of 175,341 / 82,332 rows is fully accounted for by 67,601 and 26,387 duplicate records removed during prior cleaning).*

---

## 2. Dataset Dimensions & Benchmark Clarification

A crucial finding during dataset inspection involves row counts relative to the raw published UNSW-NB15 splits:

| Dataset Split | Cleaned CSV Rows | Raw Benchmark Rows | Removed Duplicates | Cleaned CSV File Size |
|---|---|---|---|---|
| **Training Set** | **107,740** | 175,341 | 67,601 (38.55%) | 20.84 MB |
| **Testing Set** | **55,945** | 82,332 | 26,387 (32.05%) | 10.74 MB |
| **Total** | **163,685** | 257,673 | 93,988 (36.48%) | 31.58 MB |

The cleaned CSV files in `data/` represent a clean, deduplicated version of the UNSW-NB15 benchmark. Removing exact duplicate network records prevents models from artificially inflating accuracy by memorizing repeated network flow records. Zero exact row duplicates exist in either file.

---

## 3. Target Distribution Analysis

### 3.1 Binary Target (`label`)
The binary target indicates whether a network connection is legitimate (`0 = Normal`) or malicious (`1 = Attack`).

| Split | Normal (0) Count | Normal (%) | Attack (1) Count | Attack (%) | Total Flows |
|---|---|---|---|---|---|
| **Training Set** | 51,890 | 48.16% | 55,850 | **51.84%** | 107,740 |
| **Testing Set** | 34,206 | **61.14%** | 21,739 | 38.86% | 55,945 |

**Important Note on Distribution Shift:**
In the training set, attacks form a slight majority (51.84%), whereas in the testing set, normal traffic forms the majority (61.14%). This distribution shift reflects real-world operational environments where test distributions may differ from lab attack generation setups.

### 3.2 Multi-Class Target (`attack_cat`)
The dataset categorizes malicious activity into 9 attack categories alongside `Normal`:

| Attack Category | Train Count | Train (%) | Test Count | Test (%) |
|---|---|---|---|---|
| Normal | 51,890 | 48.16% | 34,206 | 61.14% |
| Exploits | 19,844 | 18.42% | 7,609 | 13.60% |
| Fuzzers | 16,150 | 14.99% | 4,838 | 8.65% |
| Reconnaissance | 7,522 | 6.98% | 2,703 | 4.83% |
| Generic | 4,181 | 3.88% | 3,657 | 6.54% |
| DoS | 3,806 | 3.53% | 1,718 | 3.07% |
| Analysis | 1,594 | 1.48% | 446 | 0.80% |
| Backdoor | 1,535 | 1.42% | 346 | 0.62% |
| Shellcode | 1,091 | 1.01% | 378 | 0.68% |
| Worms | 127 | 0.12% | 44 | 0.08% |

**Observations on Imbalance:**
- High frequency: `Exploits` (18.42%) and `Fuzzers` (14.99%) represent the vast majority of malicious samples.
- Severe minority classes: `Worms` represents only **0.12%** of training flows (127 samples) and **0.08%** of testing flows (44 samples). `Shellcode` (1.01%), `Backdoor` (1.42%), and `Analysis` (1.48%) also exhibit high class scarcity.
- Standard accuracy will be misleading for multi-class classification; macro-averaged F1-score, balanced precision/recall, and confusion matrices will be required in Phase 2.

---

## 4. Feature Space Categorization

The dataset comprises **42 input features** and **2 target columns**:

### 4.1 Categorical Features (3 features)
1. `proto` (Protocol): 133 unique values in Train, 131 in Test (e.g., `tcp`, `udp`, `unas`, `arp`, `ospf`, `sctp`).
2. `service` (Application Service): 13 unique values in Train and Test (e.g., `http`, `dns`, `smtp`, `ftp`, `ftp-data`, `ssh`).
   - Over **60.28%** of training records have `service = '-'`, denoting unmapped or non-standard ports.
3. `state` (TCP / Connection State): 9 unique values in Train, 7 in Test (e.g., `FIN`, `INT`, `CON`, `REQ`, `RST`).

### 4.2 Numerical Features (39 features)
- **Time and Rate:** `dur`, `rate`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `tcprtt`, `synack`, `ackdat`.
- **Packet and Volume:** `spkts`, `dpkts`, `sbytes`, `dbytes`, `sload`, `dload`, `sloss`, `dloss`, `smean`, `dmean`.
- **Window and TCP Parameters:** `swin`, `dwin`, `stcpb`, `dtcpb`.
- **TTL Metrics:** `sttl`, `dttl`.
- **Application & Content:** `trans_depth`, `response_body_len`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `is_sm_ips_ports`.
- **Connection Counts (Aggregated Traffic):** `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_src_ltm`, `ct_srv_dst`.

---

## 5. Critical Findings: Target Leakage, Collinearity & Testbed Artifacts

Our deep inspection uncovered several critical domain-specific properties that must guide preprocessing and model design:

### 5.1 Mutual Exclusion of Targets (Target Leakage Rule)
- `label` and `attack_cat` are two forms of the exact same label.
- When training a binary classifier, `attack_cat` must be strictly excluded from input features.
- When training a multi-class classifier, `label` must be strictly excluded.
- Failure to drop the corresponding target column will cause 100% artificial target leakage.

### 5.2 Perfect Collinearity & Redundancy
1. **`is_ftp_login` and `ct_ftp_cmd`:**
   - In the training set, these two columns are **100% identical** across all 107,740 rows.
   - In the testing set, they are 99.99% identical.
   - Retaining both introduces perfect multicollinearity.
2. **`tcprtt` Mathematical Identity:**
   - The TCP Round Trip Time feature obeys: `tcprtt = synack + ackdat` (maximum floating-point deviation < 1e-15).
   - `tcprtt` is an exact linear combination of two other features.
3. **Packet Loss and Packet Counts:**
   - Correlation(`spkts`, `sloss`) = 0.974
   - Correlation(`dpkts`, `dloss`) = 0.978
   - Correlation(`swin`, `dwin`) = 0.987

### 5.3 Testbed Artifacts: TTL Bimodal Separation
- The Source Time-To-Live (`sttl`) feature exhibits a Pearson correlation of **0.6107** with `label`.
- Further inspection reveals:
  - When `sttl = 31`, there are **39,123 Normal flows and 0 Attack flows**.
  - When `sttl = 254`, there are **42,050 Attack flows and 9,988 Normal flows**.
- **Explanation:** In the UNSW-NB15 cyber-range testbed, attacks generated using the IXIA PerfectStorm traffic generator used specific operating system TTL configurations (often 254/255 or 62/64) that differed systematically from the normal traffic servers (TTL 31/32).
- Models that rely blindly on `sttl` may learn synthetic testbed hardware characteristics rather than genuine behavioral attack patterns.

### 5.4 Categorical Drift & Unseen Test Categories
- The `state` feature contains two categories in the testing set that **never appear** in the training set: `CLO` and `ACC`.
- Conversely, four categories appear only in the training set: `URN`, `no`, `ECO`, `PAR`.
- In `proto`, two protocols (`icmp`, `rtp`) appear in train but not test.
- **Preprocessing Requirement:** In Phase 2, any categorical encoder (e.g., `OneHotEncoder`) must be configured with `handle_unknown='ignore'` to avoid crashing on unseen test categories.

### 5.5 Extreme Skewness in Network Volumes
Features such as `trans_depth` (skewness ~136), `response_body_len` (~60), `sbytes` (~35), and `sload` (~26) feature heavy positive skewness with long power-law tails. A logarithmic transformation (e.g., `log(1 + x)`) or robust scaling will be necessary for distance-based or linear models.

---

## 6. Generated Visualizations Catalogue

All high-resolution figures have been saved to `outputs/eda/`:

1. **`01_binary_target_distribution.png`**: Side-by-side comparison of normal vs attack flow records and percentages in both Training and Testing splits.
2. **`02_attack_category_distribution.png`**: Multi-class distribution across all 10 categories, visually exposing severe class imbalance (e.g., Exploits/Fuzzers vs Worms/Shellcode).
3. **`03_numerical_feature_distributions.png`**: 3x3 grid of density distributions for 9 key numerical features separated by class (Normal vs Attack) with log-scale transformations where applicable.
4. **`04_correlation_matrix.png`**: Annotated Pearson correlation heatmap of top discriminative numerical features and their inter-correlations.
5. **`05_service_and_proto_distribution.png`**: Breakdown of attack proportions across top application services and transport protocols.
6. **`06_testbed_ttl_artifact_deepdive.png`**: Deep dive into `sttl` and `ct_state_ttl`, visualizing the synthetic testbed clustering artifacts.

---

## 7. Concrete Recommendations for Phase 2 (Preprocessing & Feature Engineering)

Based on our Phase 1 empirical findings, the following actions are recommended for Phase 2:
1. **Target Separation:** Formulate separate binary target vectors (`y_binary = df['label']`) and multi-class target vectors (`y_multi = df['attack_cat']`). Exclude both from feature matrix `X`.
2. **Redundant Feature Pruning:** Drop `ct_ftp_cmd` (redundant with `is_ftp_login`) and consider dropping `tcprtt` (redundant with `synack + ackdat`).
3. **Encoding Strategy:** Use One-Hot Encoding with `handle_unknown='ignore'` for `state`, `service`, and `proto` (or frequency encoding for high-cardinality `proto`).
4. **Log Transformation:** Apply `log1p` transformation to heavy-tailed continuous metrics (`dur`, `sbytes`, `dbytes`, `sload`, `dload`, `rate`, `sjit`, `djit`).
5. **Evaluation Metric Selection:** Do not rely on plain classification accuracy due to the multi-class imbalance. Use Precision, Recall, Macro-F1, PR-AUC, and Confusion Matrices.
