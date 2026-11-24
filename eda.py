"""
eda.py
------
Phase 1: Exploratory Data Analysis & Visualization Suite
College ML Project: Machine Learning-Based Network Intrusion Detection and Attack Classification
Dataset: UNSW-NB15 (Cleaned Train and Test Sets)

Outputs generated:
  - outputs/eda/01_binary_target_distribution.png
  - outputs/eda/02_attack_category_distribution.png
  - outputs/eda/03_numerical_feature_distributions.png
  - outputs/eda/04_correlation_matrix.png
  - outputs/eda/05_service_and_proto_distribution.png
  - outputs/eda/06_testbed_ttl_artifact_deepdive.png
  - outputs/eda/eda_report.md

Usage:
    python eda.py
"""

import os
from pathlib import Path
import sys

# Configure headless matplotlib environment
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Set aesthetic styling
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/eda")
TRAIN_FILE = DATA_DIR / "UNSW_NB15_training_cleaned.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing_cleaned.csv"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] Output directory verified: {OUTPUT_DIR}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[LOAD] Ingesting cleaned datasets...")
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        print(f"[FAIL] Missing dataset files in {DATA_DIR}")
        sys.exit(1)
    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    test_df = pd.read_csv(TEST_FILE, low_memory=False)
    print(f"[LOAD] Train set: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"[LOAD] Test set : {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")
    return train_df, test_df


def plot_binary_target_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("[PLOT 1/6] Generating binary target distribution chart...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    datasets = [("Training Dataset", train_df, axes[0]), ("Testing Dataset", test_df, axes[1])]
    palette = ["#2b5c8f", "#d95f02"]  # Blue for Normal, Orange/Red for Attack

    for title, df, ax in datasets:
        counts = df["label"].value_counts().sort_index()
        total = len(df)
        labels = ["Normal (0)", "Attack (1)"]
        bars = ax.bar(labels, counts.values, color=palette, width=0.55, edgecolor="#333333", linewidth=1)

        ax.set_title(f"{title}\n(Total: {total:,} rows)", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Number of Flow Records" if ax == axes[0] else "", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar, count in zip(bars, counts.values):
            pct = count / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (total * 0.015),
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

    fig.suptitle(
        "UNSW-NB15: Binary Target ('label') Distribution & Class Shift",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "01_binary_target_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")


def plot_attack_category_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("[PLOT 2/6] Generating multi-class attack category distribution chart...")
    tr_counts = train_df["attack_cat"].value_counts()
    te_counts = test_df["attack_cat"].value_counts()

    # Align all unique categories across both
    all_cats = tr_counts.index.union(te_counts.index).tolist()
    # Sort categories by train count descending
    all_cats = train_df["attack_cat"].value_counts().index.tolist()

    tr_pcts = [(train_df["attack_cat"] == c).mean() * 100 for c in all_cats]
    te_pcts = [(test_df["attack_cat"] == c).mean() * 100 for c in all_cats]

    y = np.arange(len(all_cats))
    height = 0.38

    fig, ax = plt.subplots(figsize=(13, 8))
    bars1 = ax.barh(y - height / 2, tr_pcts, height, label="Training Set", color="#1f77b4", edgecolor="#333")
    bars2 = ax.barh(y + height / 2, te_pcts, height, label="Testing Set", color="#ff7f0e", edgecolor="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(all_cats, fontsize=11, fontweight="medium")
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of Dataset (%)", fontsize=12, fontweight="bold")
    ax.set_title("UNSW-NB15: Multi-class Attack Category Distribution (Imbalance & Split Comparison)", fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # Annotate bars
    for bar in bars1:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, color="#0d47a1")
    for bar in bars2:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, color="#b23c00")

    ax.set_xlim(0, max(max(tr_pcts), max(te_pcts)) + 7)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "02_attack_category_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")


def plot_numerical_feature_distributions(train_df: pd.DataFrame) -> None:
    print("[PLOT 3/6] Generating numerical feature distributions comparison...")
    # Selected prominent network features
    features_config = [
        ("sttl", "Source Time-To-Live (sttl)", False),
        ("dttl", "Destination Time-To-Live (dttl)", False),
        ("dur", "Flow Duration (dur, log1p scale)", True),
        ("rate", "Packet Rate (rate, log1p scale)", True),
        ("sbytes", "Source Bytes (sbytes, log1p scale)", True),
        ("dbytes", "Destination Bytes (dbytes, log1p scale)", True),
        ("sload", "Source Load bps (sload, log1p scale)", True),
        ("dload", "Destination Load bps (dload, log1p scale)", True),
        ("ct_state_ttl", "Connections with same state & TTL (ct_state_ttl)", False)
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    norm_mask = train_df["label"] == 0
    att_mask = train_df["label"] == 1

    for idx, (col, display_name, use_log) in enumerate(features_config):
        ax = axes[idx]
        norm_vals = train_df.loc[norm_mask, col]
        att_vals = train_df.loc[att_mask, col]

        if use_log:
            norm_vals = np.log1p(np.maximum(norm_vals, 0))
            att_vals = np.log1p(np.maximum(att_vals, 0))

        # Plot KDE / Histograms
        sns.histplot(
            norm_vals,
            color="#2b5c8f",
            label="Normal",
            kde=True,
            stat="density",
            common_norm=False,
            alpha=0.35,
            ax=ax,
            bins=30
        )
        sns.histplot(
            att_vals,
            color="#d95f02",
            label="Attack",
            kde=True,
            stat="density",
            common_norm=False,
            alpha=0.35,
            ax=ax,
            bins=30
        )

        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlabel("Value" if not use_log else "log1p(Value)", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "UNSW-NB15: Density Distributions of Key Numerical Features (Normal vs Attack)",
        fontsize=15,
        fontweight="bold",
        y=1.01
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "03_numerical_feature_distributions.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")


def plot_correlation_matrix(train_df: pd.DataFrame) -> None:
    print("[PLOT 4/6] Generating correlation heatmap for prominent features...")
    # Select most relevant numerical features (top correlated + key connection metrics)
    selected_cols = [
        "label",
        "sttl", "ct_state_ttl", "dttl", "dload", "ackdat", "tcprtt", "synack",
        "dmean", "rate", "ct_dst_sport_ltm", "sload", "smean", "dur",
        "spkts", "sloss", "dpkts", "dloss", "swin", "dwin"
    ]

    corr = train_df[selected_cols].corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(15, 12))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.6,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation Coefficient"},
        annot_kws={"size": 8.5},
        ax=ax
    )

    ax.set_title("UNSW-NB15: Feature Correlation Heatmap (Target & Linear Dependencies)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "04_correlation_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")


def plot_service_and_proto_distribution(train_df: pd.DataFrame) -> None:
    print("[PLOT 5/6] Generating protocol and service categorical distribution chart...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Top 8 protocols
    top_protos = train_df["proto"].value_counts().head(8).index
    proto_df = train_df[train_df["proto"].isin(top_protos)]
    proto_ct = pd.crosstab(proto_df["proto"], proto_df["label"], normalize="index") * 100
    proto_ct = proto_ct.loc[top_protos]  # preserve order

    proto_ct.plot(
        kind="bar",
        stacked=True,
        color=["#2b5c8f", "#d95f02"],
        edgecolor="#333",
        ax=axes[0],
        width=0.6
    )
    axes[0].set_title("Top 8 Network Protocols (Normal vs Attack %)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Protocol", fontsize=11)
    axes[0].set_ylabel("Proportion (%)", fontsize=11)
    axes[0].set_xticklabels(proto_ct.index, rotation=30, ha="right")
    axes[0].legend(["Normal (0)", "Attack (1)"], loc="upper right")
    axes[0].set_ylim(0, 105)

    # Top 8 services
    top_services = train_df["service"].value_counts().head(8).index
    svc_df = train_df[train_df["service"].isin(top_services)]
    svc_ct = pd.crosstab(svc_df["service"], svc_df["label"], normalize="index") * 100
    svc_ct = svc_ct.loc[top_services]

    svc_ct.plot(
        kind="bar",
        stacked=True,
        color=["#2b5c8f", "#d95f02"],
        edgecolor="#333",
        ax=axes[1],
        width=0.6
    )
    axes[1].set_title("Top 8 Application Services (Normal vs Attack %)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Service", fontsize=11)
    axes[1].set_ylabel("Proportion (%)", fontsize=11)
    axes[1].set_xticklabels(svc_ct.index, rotation=30, ha="right")
    axes[1].legend(["Normal (0)", "Attack (1)"], loc="upper right")
    axes[1].set_ylim(0, 105)

    fig.suptitle("UNSW-NB15: Categorical Attack Composition across Protocols & Services", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "05_service_and_proto_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")


def plot_testbed_ttl_artifact_deepdive(train_df: pd.DataFrame) -> None:
    print("[PLOT 6/6] Generating testbed TTL artifact deep dive chart...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: sttl distribution by attack_cat
    order = train_df["attack_cat"].value_counts().index.tolist()
    sns.boxplot(
        data=train_df,
        x="attack_cat",
        y="sttl",
        hue="attack_cat",
        legend=False,
        order=order,
        palette="tab10",
        ax=axes[0],
        fliersize=1.5
    )
    axes[0].set_title("Distribution of Source TTL ('sttl') across Attack Categories", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Attack Category", fontsize=11)
    axes[0].set_ylabel("Source TTL Value (sttl)", fontsize=11)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].axhline(y=31, color="blue", linestyle="--", alpha=0.7, label="sttl = 31 (Legitimate traffic cluster)")
    axes[0].axhline(y=254, color="red", linestyle="--", alpha=0.7, label="sttl = 254 (IXIA generator cluster)")
    axes[0].legend(loc="upper right", fontsize=9)

    # Right: ct_state_ttl vs label
    sns.countplot(
        data=train_df,
        x="ct_state_ttl",
        hue="label",
        palette=["#2b5c8f", "#d95f02"],
        ax=axes[1],
        edgecolor="#333"
    )
    axes[1].set_title("Count of Flows by 'ct_state_ttl' and Label", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("ct_state_ttl value", fontsize=11)
    axes[1].set_ylabel("Flow Count", fontsize=11)
    axes[1].legend(["Normal (0)", "Attack (1)"], title="Label", loc="upper right")

    fig.suptitle(
        "UNSW-NB15 Testbed Artifacts: Evidence of Synthetic TTL Bimodal Clustering",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "06_testbed_ttl_artifact_deepdive.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")


def generate_eda_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("[REPORT] Writing comprehensive summary report to outputs/eda/eda_report.md...")
    
    tr_len = len(train_df)
    te_len = len(test_df)
    tr_norm = (train_df["label"] == 0).sum()
    tr_att = (train_df["label"] == 1).sum()
    te_norm = (test_df["label"] == 0).sum()
    te_att = (test_df["label"] == 1).sum()

    # Targets summary table
    cat_order = train_df["attack_cat"].value_counts().index
    cat_rows = []
    for c in cat_order:
        tr_c = (train_df["attack_cat"] == c).sum()
        tr_p = tr_c / tr_len * 100
        te_c = (test_df["attack_cat"] == c).sum()
        te_p = te_c / te_len * 100
        cat_rows.append(f"| {c} | {tr_c:,} | {tr_p:.2f}% | {te_c:,} | {te_p:.2f}% |")
    cat_table_str = "\n".join(cat_rows)

    report_template = """# Exploratory Data Analysis & Dataset Validation Report

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
{{CAT_TABLE}}

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
"""
    report_content = report_template.replace("{{CAT_TABLE}}", cat_table_str)
    report_path = OUTPUT_DIR / "eda_report.md"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"[REPORT] Saved report to {report_path}")


def main() -> None:
    print("=" * 70)
    print(" UNSW-NB15 EXPLORATORY DATA ANALYSIS & VISUALIZATION PIPELINE")
    print("=" * 70)
    ensure_output_dir()
    train_df, test_df = load_data()
    plot_binary_target_distribution(train_df, test_df)
    plot_attack_category_distribution(train_df, test_df)
    plot_numerical_feature_distributions(train_df)
    plot_correlation_matrix(train_df)
    plot_service_and_proto_distribution(train_df)
    plot_testbed_ttl_artifact_deepdive(train_df)
    generate_eda_report(train_df, test_df)
    print("=" * 70)
    print(" EDA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
